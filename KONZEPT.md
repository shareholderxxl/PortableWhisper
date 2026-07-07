# Master-Prompt: PortableWhisper Refactoring, NPU Optimization & Handy-UI Integration

> **Status (Phase 1):** In Umsetzung auf Branch `phase-1`. Details und
> Entscheidungen siehe `PHASE1_PLAN.md` und `AGENTS.md`.
> **Status (Phase 2):** Geplant — Migrationsplan in `PHASE2_PLAN.md`.
> Phase 2 wechselt von Whisper (faster-whisper/CTranslate2) zu
> NVIDIA Parakeet TDT v3 via `onnx-asr` (ONNX Runtime).
> Wichtige Abweichungen vom Original-Prompt:
> - Standard-Modell (Phase 1): `default` (manuell in `data/models/default/`).
> - Standard-Modell (Phase 2): `parakeet-tdt-0.6b-v3` (ONNX, 25 Sprachen).
> - Backend-Port: 8765 (statt 8000).
> - PyInstaller **ohne torch** (Phase 1: ctranslate2, Phase 2: onnxruntime).
> - Build-Artefakt: nur portables ZIP (kein MSI/NSIS), gebaut via GitHub Actions.

## Rolle & Ziel
Du bist ein erfahrener Systems Engineer und AI Developer. Unser Ziel ist es, das kompakte Repository `PortableWhisper` (Tauri-Frontend + Python-FastAPI-Backend) zu klonen und in eine kommerziell nutzbare, komplett portable STT-Anwendung ohne Admin-Rechte zu verwandeln. 

Die Benutzeroberfläche für die Textkorrektur soll vom Open-Source-Projekt `cjpais/handy` inspiriert sein, die Ausführung des LLMs erfolgt jedoch – anders als bei Handy – vollkommen autark und lokal direkt in unserem eigenen, gebündelten Python-Backend (Zero-Dependency für den Endnutzer).

Wir arbeiten das Projekt strikt in drei aufeinander aufbauenden Phasen ab. Beginne heute ausschließlich mit **Phase 1**.

---

## PHASE 1: Zero-Dependency & Portabilität (Fokus: HEUTE)

1. **Ziel:** Die App muss als reines ZIP-Archiv laufen. Beim Entpacken und Starten der `.exe` dürfen keine Admin-Rechte (UAC-Prompt) und keine installierten Abhängigkeiten (kein Python, kein Node.js) benötigt werden.
2. **Standard-Modell:** Nutze für die Transkription das spezialisierte deutsche Modell `primeline/whisper-large-v3-german`.
3. **Backend-Kompilierung:** Nutze `PyInstaller` im Python-Verzeichnis, um das FastAPI-Backend mitsamt dem Modell in eine eigenständige Binärdatei (`whisper-backend.exe`) zu "freezen".
4. **Tauri-Sidecar:** Integriere diese `.exe` als Tauri-Sidecar, sodass sie beim Start der Haupt-App im Hintergrund mitgestartet und beim Schließen sauber beendet wird.
5. **Pfad-Management:** Biege alle Dateioperationen (Modellgewichte, temporäre Audio-Chunks) so um, dass sie relativ im Benutzerverzeichnis unter `%LOCALAPPDATA%/DeinApp_Name/` abgelegt werden. Niemals in geschützte Systemordner schreiben!
6. **Tauri-Konfiguration:** Stelle das Build-Target in der `tauri.conf.json` so ein, dass beim Befehl `tauri build` ein portables ZIP-Archiv erzeugt wird.

---

## PHASE 2: Parakeet TDT v3 + onnx-asr (Mittelfristig)

> **Entwicklung:** Die Umsetzung erfolgt auf einem **separaten GitHub-Branch** (z. B. `phase-2`).  
> **Wichtig:** Phase 2 wird NICHT in `phase-1` entwickelt. Der Branch `phase-2` wird vom `main`-Branch (oder dem stabilen Stand vor Phase-1-Änderungen) abgezweigt.
> **Details:** Siehe `PHASE2_PLAN.md` für den vollständigen Migrationsplan.

### Motivation: Warum Parakeet statt Whisper

Whisper (Encoder-Decoder, 30s-Fenster) ist für kurze Diktate zu langsam: auf CPU
dauert selbst die Transkription von 2 Wörtern 3-4 Sekunden. NVIDIA Parakeet TDT
0.6B v3 (FastConformer-TDT, RNN-Transducer) löst dieses Problem fundamental:

| Metrik | Whisper large-v3-turbo | Parakeet TDT v3 |
|--------|----------------------|-----------------|
| Architektur | Encoder-Decoder (30s-Fenster) | RNN-T/TDT (Frame-für-Frame) |
| Deutsch WER (Fleurs) | ~7-8% | **5,04%** |
| CPU RTFx | ~0,3-1× | **36×** |
| Parameter | ~809M | 600M |
| Zeichensetzung | Nur via Prompt | **Automatisch** |
| Streaming-fähig | Nein | Ja (2s-Chunks möglich) |
| Lizenz | MIT | CC-BY-4.0 (kommerziell) |

Parakeet ist auf CPU bereits **30-40× schneller** als Whisper und bietet **bessere
deutsche Qualität**. GPU/NPU-Beschleunigung ist optional, nicht zwingend nötig.

### Phase 2A: onnx-asr + Parakeet TDT v3 auf CPU (primäres Ziel)

1. **Engine-Wechsel:** Ersetze `faster-whisper`/`ctranslate2` durch
   [`onnx-asr`](https://github.com/istupakov/onnx-asr) (pure Python, minimale
   Abhängigkeiten, MIT-Lizenz).
2. **Modell:** `nvidia/parakeet-tdt-0.6b-v3` als ONNX-Format.
   Vorgefertigte ONNX-Version: `istupakov/parakeet-tdt-0.6b-v3-onnx` (HF).
3. **Laufzeit:** `onnxruntime` (CPU) — kein PyTorch, kein CTranslate2.
4. **Vorteile:**
   * 36× RTFx auf CPU → Transkription praktisch instantan (~0,14s für 5s Audio).
   * Bessere deutsche Qualität als Whisper (5,04% vs ~7-8% WER).
   * Automatische Großschreibung und Zeichensetzung (kein Post-Processing nötig).
   * Pure Python → einfache PyInstaller-Integration.
   * Keine schweren Abhängigkeiten (numpy + onnxruntime only).

### Phase 2B: DirectML für GPU/NPU-Beschleunigung (optional)

1. **GPU-Beschleunigung:** Installiere `onnxruntime-directml` statt `onnxruntime`.
   Aktiviere `DmlExecutionProvider` für AMD Radeon, Intel Arc, NVIDIA GPUs.
2. **NPU (XDNA 2):** XDNA 2 NPUs (AMD Ryzen AI 300) sind via DirectML
   theoretisch ansprechbar, aber die Treiber-Unterstützung für ASR-Modelle ist
   noch experimentell. Bei 36× RTFx auf CPU ist NPU-Beschleunigung für ASR
   **nicht zwingend erforderlich**.
3. **UI-Geräteauswahl:** Die bestehende Auto/CPU/GPU-Auswahl in `index.html`
   wird reaktiviert. NPU als experimentelle Option hinzufügen.
4. **Fallback-Hierarchie:** NPU → GPU → CPU (jeweils mit Graceful Degradation).

### Phase 2C: Streaming (optional, erst bei Bedarf)

1. **Live-Text während Aufnahme:** Für echtes Streaming (wie Handy) müsste
   `sherpa-onnx` statt `onnx-asr` verwendet werden (unterstützt 2s-Chunk-
   Streaming mit Parakeet). sherpa-onnx unterstützt derzeit noch kein DirectML
   (Issue #3194 offen).
2. **Aufwand vs. Nutzen:** Bei 36× RTFx auf CPU dauert die Batch-Transkription
   von 5s Audio nur 0,14s — Streaming bringt hier keinen spürbaren Mehrwert.
   Streaming wird nur relevant, wenn Live-Text während des Sprechens angezeigt
   werden soll.

---

## PHASE 3: Lokale LLM-Korrekturschicht (Handy-Inspired UI)

1. **UI nach dem Vorbild von Handy:** Integriere im Tauri-Frontend unter den Einstellungen einen Tab "Post-Processing". Der Nutzer kann dort eigene, benannte Prompts erstellen (z. B. "Protokoll-Stil", "Standard-Korrektur") und den exakten System-Prompt-Text in einem Textfeld editieren.
2. **Autarke Inferenz:** Anders als bei Handy (das Ollama voraussetzt) binden wir ein ultrakleines lokales LLM (Qwen 3 oder Gemma 4 2B) direkt in unser bestehendes Python-Backend ein (Ausführung via ONNX-Runtime).
3. **Workflow:** Nach der Parakeet-Transkription wird der Rohtext automatisch an die interne LLM-Engine übergeben, anhand des im UI gewählten Prompts bereinigt (Entfernen von Stotterern, Grammatikkorrektur) und erst dann in die Windows-Zwischenablage injiziert.
   > **Hinweis:** Parakeet liefert bereits automatische Zeichensetzung und
   > Großschreibung, sodass der LLM-Korrekturbedarf geringer ist als bei Whisper.

---

## DEINE ERSTE AUFGABE (Phase 1)

1. Analysiere das geklonte `PortableWhisper`-Repository.
2. Zeige mir, wie wir die `tauri.conf.json` anpassen müssen, um das Python-Backend als Sidecar zu registrieren und die App als portables ZIP zu packen.
3. Erkläre mir, welche PyInstaller-Konfiguration nötig ist, um das Modell `primeline/whisper-large-v3-german` beim ersten App-Start sauber in den lokalen AppData-Ordner des Nutzers herunterzuladen.

**Gib mir zunächst nur den Umsetzungsplan für diese erste Aufgabe aus.**