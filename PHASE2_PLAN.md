# Phase 2: Migrationsplan — faster-whisper → Parakeet TDT v3 + onnx-asr

> **Status:** Geplant — noch nicht in Umsetzung.
> **Branch:** `phase-2` (wird von `main` abgezweigt, nicht von `phase-1`).
> **Ziel:** Austausch der STT-Engine für 30-40× schnellere Transkription auf CPU
> bei besserer deutscher Qualität.

---

## 1. Zusammenfassung

| Aspekt | Phase 1 (aktuell) | Phase 2 (Ziel) |
|--------|-------------------|----------------|
| **STT-Engine** | faster-whisper (CTranslate2) | onnx-asr (ONNX Runtime) |
| **Modell** | Whisper large-v3-turbo (CTranslate2) | NVIDIA Parakeet TDT 0.6B v3 (ONNX) |
| **Architektur** | Encoder-Decoder (30s-Fenster) | RNN-Transducer/TDT (Frame-für-Frame) |
| **CPU RTFx** | ~0,3-1× Echtzeit | **36× Echtzeit** |
| **Deutsch WER** | ~7-8% | **5,04%** |
| **Abhängigkeiten** | ctranslate2, faster-whisper | onnxruntime, onnx-asr, numpy |
| **Binary-Größe** | ~120 MB | ~100 MB (geschätzt, kein ctranslate2) |
| **PyTorch** | Nicht benötigt | Nicht benötigt |
| **Zeichensetzung** | Nur via Prompt | **Automatisch** |
| **Streaming** | Nicht möglich | Architektur-bedingt möglich (Phase 2C) |

---

## 2. Modell

### 2.1 Parakeet TDT 0.6B v3

* **HuggingFace (Original):** `nvidia/parakeet-tdt-0.6b-v3`
* **HuggingFace (ONNX konvertiert):** `istupakov/parakeet-tdt-0.6b-v3-onnx`
* **Parameter:** 600M
* **Sprachen:** 25 EU-Sprachen (inkl. Deutsch, Englisch, Französisch, etc.)
* **Automatische Spracherkennung:** Ja (erkennt Sprache automatisch)
* **Lizenz:** CC-BY-4.0 (kommerziell nutzbar)
* **Format:** ONNX (encoder.onnx + decoder.onnx + vocab.txt)
* **Größe:** ~1,2 GB (fp32), ~600 MB (int8 quantisiert verfügbar)

### 2.2 ONNX-Modell-Varianten

| Variante | HF Repo | Größe | Qualität | Bemerkung |
|----------|---------|-------|----------|-----------|
| fp32 (Standard) | `istupakov/parakeet-tdt-0.6b-v3-onnx` | ~1,2 GB | Beste | Empfohlen für CPU |
| fp16 | `ako101/parakeet-tdt-0.6b-v3-sherpa-onnx-fp16` | ~600 MB | Sehr gut | Kleinere Binary |
| int8 | `nasedkinpv/parakeet-tdt-0.6b-v3-onnx-int8` | ~350 MB | Gut | Für ressourcenlimitierte Geräte |
| int4 | `efederici/parakeet-tdt-0.6b-v3-onnx-int4` | ~200 MB | Akzeptabel | Experimentell |

**Empfehlung:** fp32 für maximale Qualität (Modell lädt via onnx-asr automatisch
von HF, sobald `load_model("nemo-parakeet-tdt-0.6b-v3")` aufgerufen wird).

---

## 3. onnx-asr Bibliothek

### 3.1 Übersicht

* **Repository:** https://github.com/istupakov/onnx-asr
* **PyPI:** `onnx-asr`
* **Lizenz:** MIT
* **Abhängigkeiten:** numpy, onnxruntime, huggingface-hub (optional)
* **Python:** 3.10 – 3.14
* **Plattformen:** Windows, Linux, macOS (x86 + ARM)
* **Execution Provider:** CPU, CUDA, TensorRT, **DirectML**, CoreML, ROCm, WebGPU

### 3.2 API (für unsere Integration relevant)

```python
import onnx_asr

# Modell laden (einmalig, lädt automatisch von HuggingFace)
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")

# Transkription (Audio als numpy float32 Array, 16kHz, mono)
result = model.recognize(audio_array)
# Returns: {"text": "Das ist ein Test.", ...}

# Mit Timestamps
result = model.recognize(audio_array, timestamps=True)
# Returns: {"text": "...", "timestamps": [...]}

# Mit VAD (für lange Audio >30s)
result = model.recognize(audio_array, vad={"model": "silero"})
```

### 3.3 DirectML-Aktivierung

```python
# Statt onnxruntime: onnxruntime-directml installieren
# pip install onnx-asr[directml,hub]

# onnx-asr wählt automatisch den besten verfügbaren Provider:
# 1. DmlExecutionProvider (GPU/NPU via DirectML)
# 2. CPUExecutionProvider (Fallback)

# Provider manuell wählen (in unserer Engine):
import onnxruntime as ort
providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(model_path, providers=providers)
```

---

## 4. Backend-Änderungen

### 4.1 Neue Datei: `backend/parakeet_engine.py`

Ersetzt `whisper_engine.py` als primäre STT-Engine. Die Klasse implementiert
dieselbe Schnittstelle wie `WhisperEngine`, sodass `main.py` minimal geändert
werden muss.

```python
"""
Parakeet TDT STT Engine
Handles model loading and transcription using onnx-asr + Parakeet TDT v3
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

try:
    import onnx_asr
    ONNX_ASR_AVAILABLE = True
    logger.info("✅ onnx-asr is available")
except ImportError as e:
    ONNX_ASR_AVAILABLE = False
    logger.warning(f"⚠️ onnx-asr not available: {e}")


class ParakeetEngine:
    """Parakeet TDT speech-to-text engine via onnx-asr"""

    def __init__(
        self,
        model_name: str = "nemo-parakeet-tdt-0.6b-v3",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.is_loaded = False
        self._original_device = device

    def load_model(self) -> bool:
        if not ONNX_ASR_AVAILABLE:
            logger.error("❌ onnx-asr is not installed!")
            return False

        if self.is_loaded:
            logger.info("Model already loaded")
            return True

        try:
            logger.info(f"📥 Loading Parakeet TDT model: {self.model_name}")
            self.model = onnx_asr.load_model(self.model_name)
            self.is_loaded = True
            logger.info(f"✅ Parakeet model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def is_model_downloaded(self, model_name: str = None) -> bool:
        # onnx-asr nutzt HF-Cache — Prüfung ob Modell vorhanden
        # (Implementierung folgt)
        return True  # Vereinfacht für ersten Entwurf

    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        if not self.is_loaded:
            if not self.load_model():
                return {"success": False, "error": "Failed to load model", "text": ""}

        try:
            logger.info(f"🎙️ Transcribing with Parakeet TDT...")

            # Audio vorbereiten
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()

            # Transkription (Parakeet erkennt Sprache automatisch)
            result = self.model.recognize(audio_data)
            text = result.get("text", "").strip()

            logger.info(f"✅ Transcription complete: {text[:100]}")

            return {
                "success": True,
                "text": text,
                "segments": [],  # Parakeet liefert keine Segmente wie Whisper
                "language": "auto",  # Parakeet erkennt automatisch
                "language_probability": 1.0,
                "duration": len(audio_data) / 16000,
            }
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return {"success": False, "error": str(e), "text": ""}

    def transcribe_chunk(self, audio_data: np.ndarray) -> Dict:
        """Für inkrementelle Transkription (falls später benötigt)."""
        return self.transcribe_audio(audio_data)
```

### 4.2 Änderungen in `backend/main.py`

Minimaler Eingriff — nur die Engine-Klasse wird ausgetauscht:

```python
# ALT (Phase 1):
from whisper_engine import WhisperEngine
whisper_engine = WhisperEngine(model_size="default", device="auto")

# NEU (Phase 2):
from parakeet_engine import ParakeetEngine
whisper_engine = ParakeetEngine(model_name="nemo-parakeet-tdt-0.6b-v3", device="auto")
```

Alle anderen Endpunkte (`/start`, `/stop`, `/health`, `/load_model_async`)
bleiben unverändert, da sie die Engine über die gemeinsame Schnittstelle
(`is_loaded`, `load_model()`, `transcribe_audio()`) ansprechen.

**Wichtig:** Die globale Variable heißt weiterhin `whisper_engine` (für
minimale Diff-Größe), ist aber eine `ParakeetEngine`-Instanz.

### 4.3 Änderungen in `backend/requirements.txt`

```diff
 # Phase 2: onnx-asr + Parakeet TDT v3
-faster-whisper>=1.0.0
-ctranslate2>=4.0.0
+onnx-asr>=0.11.0
+onnxruntime>=1.20.0
+# Optional für GPU/NPU (Phase 2B):
+# onnxruntime-directml>=1.20.0  # ersetzt onnxruntime
```

### 4.4 Änderungen am PyInstaller Spec (`whisper-backend.spec`)

```diff
 hiddenimports=[
     'uvicorn',
     'fastapi',
-    'faster_whisper',
-    'ctranslate2',
+    'onnx_asr',
+    'onnxruntime',
     'numpy',
     'huggingface_hub',
 ]

# ONNX Runtime DLLs einsammeln:
+import onnxruntime
+onnxruntime_dir = os.path.dirname(onnxruntime.__file__)
+binaries += Tree(onnxruntime_dir, prefix='onnxruntime')
```

---

## 5. Pfad-Management

### 5.1 Modell-Speicherort

Parakeet ONNX-Modelle werden von `onnx-asr` automatisch im HuggingFace-Cache
gespeichert. Dieser wird bereits durch `path_redirect.py` umgeleitet:

```
%LOCALAPPDATA%/PortableWhisper/models/
  └── models--istupakov--parakeet-tdt-0.6b-v3-onnx/
      └── snapshots/
          └── <hash>/
              ├── encoder.onnx
              ├── decoder.onnx
              └── vocab.txt
```

Keine Änderungen an `path_redirect.py` nötig — die HF-Cache-Umleitung
(HF_HOME, HUGGINGFACE_HUB_CACHE) ist bereits aktiv.

### 5.2 Option: Manuelles Modell (wie Phase 1 "default")

Für Offline-Nutzung ohne Internet beim ersten Start kann das Modell auch
vorab nach `data/models/parakeet/` kopiert werden. `parakeet_engine.py`
prüft dann diesen Pfad zuerst, bevor der HF-Cache genutzt wird.

---

## 6. Migration: Schritt-für-Schritt

### Schritt 1: Branch erstellen
```bash
git checkout main
git checkout -b phase-2
```

### Schritt 2: Abhängigkeiten installieren
```bash
cd backend
pip install onnx-asr[cpu,hub]
# Test:
python -c "import onnx_asr; m = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3'); print('OK')"
```

### Schritt 3: `parakeet_engine.py` erstellen
* Neue Datei wie in Abschnitt 4.1 beschrieben.
* Unit-Tests: Audio-Array rein → Text raus.

### Schritt 4: `main.py` anpassen
* Import von `parakeet_engine` statt `whisper_engine`.
* `/health`-Endpoint: `model_status`-Logik anpassen (kein `is_model_downloaded`
  für HF-Cache-Modelle nötig).

### Schritt 5: Smart Pre-Load anpassen
* `/load_model_async` nutzt bereits die gemeinsame Schnittstelle.
* Es muss nur die Modell-Referenz geändert werden.

### Schritt 6: PyInstaller Spec aktualisieren
* `hiddenimports` und `binaries` wie in Abschnitt 4.4.
* Test: `pyinstaller whisper-backend.spec` lokal auf Windows.

### Schritt 7: Frontend anpassen (minimal)
* `index.html`: Modell-Auswahl-Dropdown entfernen oder auf "Parakeet" fixieren.
* `lib.rs`: `selected_model` Default auf `"nemo-parakeet-tdt-0.6b-v3"`.

### Schritt 8: UI-Geräteauswahl reaktivieren (Phase 2B)
* `index.html`: Auto/CPU/GPU Dropdown sichtbar machen.
* `lib.rs`: Device-Auswahl an Backend weitergeben.
* Backend: `onnxruntime-directml` statt `onnxruntime` installieren.

### Schritt 9: Build via GitHub Actions
* `.github/workflows/build-windows.yml`: `requirements.txt` neu installieren.
* PyInstaller spec in CI aktualisieren.

### Schritt 10: Benchmark & Validierung
* Gleiche Test-Audios mit Phase 1 (Whisper) und Phase 2 (Parakeet) transkribieren.
* WER, Latenz, RAM-Verbrauch vergleichen.
* Erwartung: 30-40× schnellere Transkription, bessere deutsche Qualität.

---

## 7. Risiken & Mitigation

| Risiko | Wahrscheinlichkeit | Auswirkung | Mitigation |
|--------|-------------------|------------|------------|
| onnx-asr inkompatibel mit PyInstaller | Mittel | Hoch | Frühzeitig PyInstaller-Test, `--collect-all onnxruntime` |
| Parakeet ONNX-Modell zu groß für Bundle | Niedrig | Mittel | int8-Quantisierung verwenden (~350 MB) |
| onnx-asr API ändert sich | Niedrig | Mittel | Version pinnen (`onnx-asr==0.11.0`) |
| DirectML auf Ziel-Hardware nicht verfügbar | Mittel | Niedrig | Graceful Fallback auf CPU (bereits implementiert) |
| Deutsche Qualität schlechter als erwartet | Niedrig | Hoch | A/B-Test mit Phase 1, Rollback möglich |
| HF-Cache-Download beim ersten Start fehlschlägt | Mittel | Mittel | Modell vorab in ZIP bundleln (Offline-Modus) |

---

## 8. Rollback-Plan

Phase 2 wird auf separatem Branch entwickelt. Bei Problemen:

1. `phase-1` Branch bleibt voll funktionsfähig (Whisper + CTranslate2).
2. Phase 2-Änderungen sind isoliert auf `phase-2` Branch.
3. Bei kritischen Problemen: `phase-1` als `main` beibehalten.
4. `whisper_engine.py` wird NICHT gelöscht, nur durch `parakeet_engine.py`
   ergänzt. Beide Engines können über einen Konfigurations-Switch coexistieren.

---

## 9. Zeitschätzung

| Schritt | Aufwand |
|---------|---------|
| Schritt 1-3: Engine + Tests | 1 Tag |
| Schritt 4-5: main.py Integration | 0,5 Tag |
| Schritt 6: PyInstaller Spec | 0,5 Tag |
| Schritt 7: Frontend minimal | 0,5 Tag |
| Schritt 8: DirectML (optional) | 1 Tag |
| Schritt 9-10: CI + Benchmark | 1 Tag |
| **Gesamt** | **3,5 - 4,5 Tage** |

---

## 10. Referenzen

* onnx-asr: https://github.com/istupakov/onnx-asr
* Parakeet TDT v3 (Original): https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
* Parakeet TDT v3 (ONNX): https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx
* sherpa-onnx (Streaming-Alternative): https://github.com/k2-fsa/sherpa-onnx
* onnxruntime-directml: https://pypi.org/project/onnxruntime-directml/
* Parakeet Technical Report: https://arxiv.org/abs/2509.14128
