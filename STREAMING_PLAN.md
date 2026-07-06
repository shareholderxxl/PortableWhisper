# STREAMING_PLAN.md — VAD-basierte Chunked Transcription

> **Scope:** Tier 1 (Backend-Streaming, UI unverändert)
> **Branch:** `phase-1`
> **Status:** Umsetzung begonnen

## Zielsetzung

Aktuell transkribiert die App Audio erst **nach**dem Stop-Knopf (`POST /stop`). Ziel ist es, bereits **während** der Aufnahme in Echtzeit-Chunks zu transkribieren. Der Nutzer bemerkt nur, dass der Text beim Stop fast sofort da ist.

---

## Architektur

### Heute
```
POST /start → InputStream → audio_queue (nur sammeln, keine Transkription)
                                ↓
POST /stop  → drain_all() → 1× transcribe_audio() → Text

Wartezeit: volle Audio-Dauer + Modell-Laden (1-3s) + Transkription (X s)
```

### Morgen
```
POST /start → InputStream → audio_queue
                                ↓
                    streaming_worker (asyncio Task, 10 Hz)
                                ↓
                    rolling_buffer + Energie-VAD
                                ↓
                    Bei Pause ≥ 400ms + Sprache ≥ 1s → transcribe_chunk()
                                ↓
                    partial_transcript += chunk_text
                                ↓
POST /stop  → Worker canceln → Rest flushen → partial_transcript

Wartezeit: ~0.5–1 s (nur Rest-Flush)
```

---

## Konfiguration (getroffene Entscheidungen)

| Entscheidung | Wert |
|-------------|------|
| Scope | Tier 1 (keine UI-Änderungen) |
| VAD | Energie-basiert (nicht Silero/faster-whisper VAD) |
| VAD RMS-Schwelle | 0.02 (Sprache typ. 0.05–0.3) |
| Mindest-Sprechdauer vor Pause | 1.0 s (Halluzinations-Schutz) |
| Mindest-Pausendauer | 400 ms |
| Polling-Intervall Worker | 100 ms |
| Modell-Lade-Strategie | Lazy parallel (Aufnahme sofort, Laden im Hintergrund) |
| Fallback bei Fehler | Automatisch auf Batch-Modus |

---

## Änderungen pro Datei

### 1. `backend/audio_capture.py`
- **Neu:** `drain_available() -> Optional[np.ndarray]` — non-blocking Queue-Drain
- **Neu:** `peek_latest_chunk() -> Optional[float]` — nur Energiewert ohne Queue zu berühren

### 2. `backend/whisper_engine.py`
- **Neu:** `transcribe_chunk(audio, language, task) -> Dict` — optimiert für kurze Chunks
  - Mindest-Check: < 1 s → skipped
  - `condition_on_previous_text=False` (Halluzinations-Schutz)
  - Input sanitization wie transcribe_audio()

### 3. `backend/main.py`
- **Neue Globals:**
  - `streaming_task: Optional[asyncio.Task]`
  - `model_load_task: Optional[asyncio.Task]`
  - `partial_transcript: str = ""`
  - `streaming_lock: asyncio.Lock`
  - `streaming_failed: bool = False`
  - `streaming_buffer: List[np.ndarray]`
  - `accumulated_rms_buffer: List[float]`
  - `last_speech_time: float`
  - `in_speech: bool`
- **Geändert `/start`:** Modell-Laden als Background Task, Streaming-Worker starten
- **Neu `_load_model_async()`:** Lädt Modell in Executor
- **Neu `_streaming_worker()`:** 10 Hz Polling mit Energie-VAD und Chunk-Transkription
- **Geändert `/stop`:** Worker canceln, Rest flushen, Fallback-Logik
- **Geändert `/cancel`:** Worker canceln, Zustand zurücksetzen

### 4. `frontend/dist/recording.html`
- Keine Änderungen (Tier 1)

---

## Risiken & Mitigation

| Risiko | Mitigation |
|--------|-----------|
| Silero-VAD entfernt Speech | Keine Silero-VAD — eigene energie-basierte |
| Halluzination bei kurzen Chunks | Min 1 s Sprache vor Transkription |
| CTranslate2 nicht thread-safe | streaming_lock serialisiert |
| Wort an Chunk-Grenze abgeschnitten | VAD schneidet nur bei natürlichen Pausen |
| Nutzer stop innerhalb 1 s (kein Chunk fertig) | Batch-Fallback in /stop |
| Modell-Laden schlägt fehl | streaming_failed = true → Fallback |
| Sehr langer Monolog (>30 s) | Hard-Cap: Chunk bei 25 s erzwingen |

---

## Test-Cases (manuell)

1. **Normal:** 5s sprechen → Pause → 5s sprechen → Stop → 2 Chunks, Text sofort
2. **Kurz:** < 1s sprechen → Stop → Batch-Fallback
3. **Lang:** 60s durchgehend → mehrere Chunks via 25s-Cap
4. **Cancel:** Abbrechen → kein Text, kein Crash
5. **Modell fehlt:** data/models/default/ leer → Fehlermeldung
6. **Schnell-Start/Stop:** Race-Condition-Sicherheit
