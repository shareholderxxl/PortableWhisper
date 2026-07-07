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
* **Format:** ONNX (encoder-model.int4.onnx + decoder_joint-model.int8.onnx + vocab.txt)
* **Größe:** 391 MB (int4, Default), 670 MB (int8 Fallback), ~2,55 GB (fp32)

### 2.2 ONNX-Modell-Varianten

| Variante | HF Repo | Größe | WER (Englisch) | Bemerkung |
|----------|---------|-------|----------------|-----------|
| **int4 hybrid** ⭐ | `efederici/parakeet-tdt-0.6b-v3-onnx-int4` | **391 MB** | **1,67%** | **Default — beste Size/Speed/Quality** |
| int8 (istupakov) | `istupakov/parakeet-tdt-0.6b-v3-onnx` | 670 MB | 1,67% | Fallback, breiteste GPU-Kompatibilität |
| fp32 | `istupakov/parakeet-tdt-0.6b-v3-onnx` | ~2,55 GB | 1,72% | Maximale Präzision, unnötig groß |
| fp16 | `ako101/parakeet-tdt-0.6b-v3-sherpa-onnx-fp16` | ~600 MB | — | Nur für sherpa-onnx optimiert |

> **Benchmark (LibriSpeech test-clean, aus efederici Model Card):**
> Alle drei Varianten (fp32, int8, int4) erreichen praktisch identische WER
> (~1,67%). int4 ist überraschend **17% schneller** als int8 und hat **halbe
> Degradation** bei 39% kleinerer Größe. Der int4-Hybrid-Ansatz quantisiert
> nur große Line/MatMul-Layer (87,5% der Gewichte), während kleine Conv-Layer
> in fp32 bleiben.

**Entscheidung: int4 als Default für Phase 2A.** Fallback auf int8 per
Konfigurationsparameter (nur eine Zeile Code-Änderung).

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

# Modell aus lokalem Verzeichnis laden (primär — portabel, offline):
model = onnx_asr.load_model("/pfad/zu/data/models/default")

# Modell aus HuggingFace laden (Fallback — falls lokal nicht vorhanden):
model = onnx_asr.load_model("efederici/parakeet-tdt-0.6b-v3-onnx-int4")

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

Modell-Speicherort (wie Phase 1): data/models/default/
  ├── encoder-model.int4.onnx       (373 MB, int4 Encoder)
  ├── decoder_joint-model.int8.onnx  (18 MB, int8 Decoder+Joint)
  ├── nemo128.int8.onnx              (41 KB, Mel-Preprocessor)
  ├── vocab.txt                      (92 KB, SentencePiece)
  └── config.json                    (97 B)
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

# Dateien, die ein gültiges lokales Parakeet-Modell enthalten muss.
_REQUIRED_PARAKEET_FILES = [
    "encoder-model.int4.onnx",      # Encoder (int4 oder int8)
    "decoder_joint-model.int8.onnx", # Decoder + Joint
    "vocab.txt",
]


class ParakeetEngine:
    """Parakeet TDT speech-to-text engine via onnx-asr"""

    def __init__(
        self,
        model_name: str = "default",
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
            # ── Strategie 1: Lokales Modell aus data/models/default/ ──
            default_dir = self._get_default_models_dir()
            if default_dir and self._is_local_model_present(default_dir):
                logger.info(f"📥 Loading Parakeet from local directory: {default_dir}")
                self.model = onnx_asr.load_model(str(default_dir))
                self.is_loaded = True
                logger.info(f"✅ Parakeet model loaded (local, offline)")
                return True

            # ── Strategie 2: HF-Fallback (Auto-Download) ──
            # Reihenfolge: int4 → int8 → fp32
            model_candidates = [
                "efederici/parakeet-tdt-0.6b-v3-onnx-int4",   # int4 (bevorzugt)
                "istupakov/parakeet-tdt-0.6b-v3-onnx",          # int8 (fallback)
            ]
            for hf_name in model_candidates:
                try:
                    logger.info(f"📥 Loading Parakeet from HuggingFace: {hf_name}")
                    self.model = onnx_asr.load_model(hf_name)
                    self.is_loaded = True
                    logger.info(f"✅ Parakeet model loaded (HF: {hf_name})")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ {hf_name} failed: {e}, trying next...")

            logger.error("❌ All model loading strategies failed")
            return False

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def _get_default_models_dir(self) -> Optional[Path]:
        """Liefert data/models/default/ — identisch zu Phase 1 WhisperEngine."""
        try:
            from runtime_hooks.path_redirect import DEFAULT_MODELS_DIR
            return DEFAULT_MODELS_DIR
        except ImportError:
            return None

    def _is_local_model_present(self, model_dir: Path) -> bool:
        """Prüft ob alle erforderlichen Parakeet-Dateien vorhanden sind."""
        return all((model_dir / f).exists() for f in _REQUIRED_PARAKEET_FILES)

    def is_model_downloaded(self, model_name: str = None) -> bool:
        """Prüft ob das Modell lokal verfügbar ist (für /health Endpoint)."""
        default_dir = self._get_default_models_dir()
        if default_dir:
            return self._is_local_model_present(default_dir)
        return False

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
whisper_engine = ParakeetEngine(model_name="default", device="auto")
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

### 5.1 Modell-Speicherort: `data/models/default/` (wie Phase 1)

Das Parakeet-Modell wird — identisch zum Whisper-Modell in Phase 1 — im
portablen Verzeichnis `data/models/default/` neben der `.exe` abgelegt.
Dies gewährleistet:

* **Offline-Betrieb:** Kein Internet/Download nötig
* **Portabilität:** Alles liegt neben der `.exe` (USB-Stick-tauglich)
* **Konsistenz:** Gleicher Pfad wie Phase 1, keine UI-Änderungen nötig
* **Updates:** Neue App-Version = nur `.exe` austauschen, Modell bleibt

```
PortableWhisper/
├── PortableWhisper.exe
├── binaries/
│   └── whisper-backend.exe
└── data/
    └── models/
        └── default/                         ← Parakeet int4 Dateien
            ├── encoder-model.int4.onnx       (373 MB)
            ├── decoder_joint-model.int8.onnx  (18 MB)
            ├── nemo128.int8.onnx              (41 KB)
            ├── vocab.txt                      (92 KB)
            └── config.json                    (97 B)
```

### 5.2 Modell-Lade-Strategie

`ParakeetEngine.load_model()` prüft in folgender Reihenfolge:

1. **Lokales Verzeichnis** (`data/models/default/`):
   * Prüft ob `encoder-model.int4.onnx` + `decoder_joint-model.int8.onnx` +
     `vocab.txt` vorhanden sind
   * Falls ja → `onnx_asr.load_model(str(default_dir))` (offline)
2. **HuggingFace-Fallback** (falls lokal nichts gefunden):
   * `efederici/parakeet-tdt-0.6b-v3-onnx-int4` (int4, 391 MB)
   * `istupakov/parakeet-tdt-0.6b-v3-onnx` (int8, 670 MB, letzter Fallback)

### 5.3 Keine Änderungen an `path_redirect.py`

Die bereits aktiven Umleitungen reichen aus:
```python
DEFAULT_MODELS_DIR = APP_DIR / "models" / "default"  # bereits definiert
os.environ["HF_HOME"] = str(MODELS_DIR)              # bereits gesetzt
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR) # bereits gesetzt
```

`ParakeetEngine` nutzt `DEFAULT_MODELS_DIR` aus `path_redirect.py` — derselbe
Pfad, den `WhisperEngine` in Phase 1 nutzt.

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
# Test (mit lokalem Modell):
python -c "import onnx_asr; m = onnx_asr.load_model('data/models/default'); print('OK')"
```

### Schritt 3: `parakeet_engine.py` erstellen
* Neue Datei wie in Abschnitt 4.1 beschrieben.
* Unit-Tests: Audio-Array rein → Text raus.
* Lokales Pfad-Laden aus `data/models/default/` testen.

### Schritt 4: `main.py` anpassen
* Import von `parakeet_engine` statt `whisper_engine`.
* `/health`-Endpoint: nutzt `ParakeetEngine.is_model_downloaded()` (prüft
  lokale Dateien in `data/models/default/`).

### Schritt 5: Smart Pre-Load anpassen
* `/load_model_async` nutzt bereits die gemeinsame Schnittstelle.
* ParakeetEngine lädt aus `data/models/default/` (offline) oder HF (fallback).

### Schritt 6: PyInstaller Spec aktualisieren
* `hiddenimports` und `binaries` wie in Abschnitt 4.4.
* Test: `pyinstaller whisper-backend.spec` lokal auf Windows.

### Schritt 7: Frontend anpassen (minimal)
* `index.html`: Modell-Auswahl-Dropdown entfernen oder auf "Parakeet" fixieren.
* `lib.rs`: `selected_model` Default bleibt `"default"` (gleicher Pfad wie Phase 1).

### Schritt 8: Modell im portable ZIP verteilen
* int4-Modell-Dateien (391 MB) nach `data/models/default/` kopieren.
* Entweder im Build-Workflow (GitHub Actions) oder manuell durch User.

### Schritt 9: UI-Geräteauswahl reaktivieren (Phase 2B)
* `index.html`: Auto/CPU/GPU Dropdown sichtbar machen.
* `lib.rs`: Device-Auswahl an Backend weitergeben.
* Backend: `onnxruntime-directml` statt `onnxruntime` installieren.

### Schritt 10: Build via GitHub Actions
* `.github/workflows/build-windows.yml`: `requirements.txt` neu installieren.
* PyInstaller spec in CI aktualisieren.

### Schritt 11: Benchmark & Validierung
* Gleiche Test-Audios mit Phase 1 (Whisper) und Phase 2 (Parakeet) transkribieren.
* WER, Latenz, RAM-Verbrauch vergleichen.
* Erwartung: 30-40× schnellere Transkription, bessere deutsche Qualität.

---

## 7. Risiken & Mitigation

| Risiko | Wahrscheinlichkeit | Auswirkung | Mitigation |
|--------|-------------------|------------|------------|
| onnx-asr inkompatibel mit PyInstaller | Mittel | Hoch | Frühzeitig PyInstaller-Test, `--collect-all onnxruntime` |
| onnx-asr kann nicht aus lokalem Pfad laden | Mittel | Hoch | Direkt via ONNX Runtime laden (onnx-asr Source als Referenz) |
| int4 `MatMulNBits` auf mancher Hardware unsupported | Niedrig | Mittel | Fallback auf int8 (`istupakov/...-onnx`, nur eine Zeile) |
| onnx-asr API ändert sich | Niedrig | Mittel | Version pinnen (`onnx-asr==0.11.0`) |
| DirectML auf Ziel-Hardware nicht verfügbar (Phase 2B) | Mittel | Niedrig | Graceful Fallback auf CPU (bereits implementiert) |
| Deutsche Qualität schlechter als erwartet | Niedrig | Hoch | A/B-Test mit Phase 1, Rollback möglich |

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
| Schritt 6-7: PyInstaller + Frontend | 0,5 Tag |
| Schritt 8: Modell ins ZIP integrieren | 0,5 Tag |
| Schritt 9: DirectML (optional) | 1 Tag |
| Schritt 10-11: CI + Benchmark | 1 Tag |
| **Gesamt** | **4 - 5 Tage** |

---

## 10. Referenzen

* onnx-asr: https://github.com/istupakov/onnx-asr
* Parakeet TDT v3 (Original): https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
* Parakeet TDT v3 **int4** (Default): https://huggingface.co/efederici/parakeet-tdt-0.6b-v3-onnx-int4
* Parakeet TDT v3 **int8** (Fallback): https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx
* sherpa-onnx (Streaming-Alternative): https://github.com/k2-fsa/sherpa-onnx
* onnxruntime-directml: https://pypi.org/project/onnxruntime-directml/
* Parakeet Technical Report: https://arxiv.org/abs/2509.14128
* Benchmark-Quelle (int4 vs int8 vs fp32): https://huggingface.co/efederici/parakeet-tdt-0.6b-v3-onnx-int4
