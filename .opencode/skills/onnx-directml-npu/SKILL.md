# ONNX Runtime + DirectML — NPU-Beschleunigung für Windows STT

## Ziel
Das Whisper-Backend wird von PyTorch auf **onnxruntime-directml** umgestellt. Dadurch werden NPUs von Intel (Core Ultra) und AMD (Ryzen AI) unter Windows 11 nativ und stromsparend angesteuert. Modell: `onnx-community/whisper-large-v3-turbo-german-ONNX`.

---

## 1. Abhängigkeiten

### 1.1 requirements-onnx.txt

```
onnxruntime-directml>=1.20.0
huggingface-hub>=0.30.0
numpy>=1.26.0
soundfile>=0.13.0
librosa>=0.10.0
fastapi>=0.115.0
uvicorn>=0.34.0
```

**Wichtig:** `torch` wird **nicht** benötigt — das spart ~1-2 GB in der PyInstaller-Binär!

### 1.2 PyInstaller-Hidden-Imports (angepasst)

```python
# statt 'torch', 'whisper':
hiddenimports=[
    'uvicorn',
    'fastapi',
    'huggingface_hub',
    'onnxruntime',
    'numpy',
    'soundfile',
    'librosa',
]
```

---

## 2. Modell-Laden mit ONNX Runtime

### 2.1 DirectML-Session erstellen

```python
import onnxruntime as ort
import numpy as np
from pathlib import Path

def create_dml_session(model_path: str) -> ort.InferenceSession:
    """Erstellt eine ONNX-Runtime-Session mit DirectML-Execution-Provider."""
    available_providers = ort.get_available_providers()
    print(f"[ONNX] Verfügbare Provider: {available_providers}")

    # DirectML bevorzugen, Fallback auf CPU
    if "DmlExecutionProvider" in available_providers:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_cpu_mem_arena = False
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            provider_options=[
                {"device_id": "0"},  # Standard-NPU
                {},                   # CPU-Fallback
            ],
        )
        print("[ONNX] ✅ DirectML-Session aktiv — läuft auf NPU")
    else:
        print("[ONNX] ⚠️ DmlExecutionProvider nicht verfügbar — Fallback auf CPU")
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    return session
```

### 2.2 Whisper-ONNX-Modell laden

```python
from huggingface_hub import hf_hub_download

MODEL_REPO = "onnx-community/whisper-large-v3-turbo-german-ONNX"
MODEL_FILENAME = "model.onnx"  # oder genauen Dateinamen prüfen

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILENAME,
    cache_dir=model_cache_dir,
)

session = create_dml_session(model_path)
```

---

## 3. NPU-Konfiguration im Detail

### 3.1 Intel Core Ultra NPU (Intel AI Boost)

```python
intel_options = {
    "device_id": "0",
    "enable_graph_capture": "1",     # Graph-Capture für wiederholte Inferenz
    "disable_winml_device_pairing": "1",  # WinML nicht verwenden
}
```

Intel NPUs erscheinen im Geräte-Manager als "Intel AI Boost". Der `DmlExecutionProvider` spricht sie automatisch an, wenn sie die DirectML 1.20+ API unterstützen.

### 3.2 AMD Ryzen AI NPU (XDNA)

```python
amd_options = {
    "device_id": "0",
    "disable_winml_device_pairing": "1",
    "dml_extension": "1",  # AMD-spezifische DirectML-Extensions aktivieren
}
```

AMD NPUs (Ryzen AI 300/7000 Serie) benötigen die aktuelle `onnxruntime-directml` (≥1.20). Der Treiber heißt "AMD NPU Driver".

### 3.3 NPU-Erkennung zur Laufzeit

```python
import subprocess, json

def detect_npu() -> dict:
    """Ermittelt, welche NPU im System verfügbar ist."""
    result = {
        "has_dml": "DmlExecutionProvider" in ort.get_available_providers(),
        "npu_type": None,
        "device_name": None,
    }

    if not result["has_dml"]:
        return result

    # Über Windows Management Instrumentation (WMI)
    try:
        output = subprocess.check_output(
            'wmic path Win32_VideoController get Name /format:csv',
            shell=True, text=True, timeout=5
        )
        for line in output.splitlines():
            if "intel" in line.lower() and ("npu" in line.lower() or "ai boost" in line.lower()):
                result["npu_type"] = "intel"
                result["device_name"] = line.split(",")[-1].strip()
            elif "amd" in line.lower() and ("npu" in line.lower() or "x dna" in line.lower()):
                result["npu_type"] = "amd"
                result["device_name"] = line.split(",")[-1].strip()
    except Exception:
        pass

    return result
```

---

## 4. Transkription mit ONNX

```python
import librosa

def transcribe_audio(file_path: str, session: ort.InferenceSession) -> str:
    """Transkribiert eine Audiodatei mit dem ONNX-Whisper-Modell."""
    # Audio laden (16kHz, mono)
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    audio = audio.astype(np.float32)

    # Whisper-ONNX erwartet: (batch, samples) oder (samples,)
    # Normierung auf [-1, 1] (librosa liefert das bereits)

    # Input-Name aus dem ONNX-Modell
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Inferenz
    ort_inputs = {input_name: np.expand_dims(audio, axis=0)}
    ort_outputs = session.run([output_name], ort_inputs)

    # Decoding (je nach ONNX-Modell-Architektur anpassen)
    tokens = np.argmax(ort_outputs[0], axis=-1)[0]
    # HuggingFace Tokenizer fürs Decoding nutzen
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "onnx-community/whisper-large-v3-turbo-german-ONNX",
        cache_dir=model_cache_dir,
    )
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text
```

---

## 5. FastAPI-Endpunkt

```python
from fastapi import FastAPI, UploadFile, File
import tempfile

app = FastAPI()
session = None

@app.on_event("startup")
async def startup():
    global session
    model_path = get_model_path()
    session = create_dml_session(model_path)
    print(f"[PortableWhisper] ONNX-Backend gestartet. NPU: {detect_npu()}")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Empfängt eine Audiodatei, transkribiert und gibt Text zurück."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = transcribe_audio(tmp_path, session)
        return {"text": text, "status": "ok"}
    except Exception as e:
        return {"text": "", "status": "error", "detail": str(e)}
    finally:
        os.unlink(tmp_path)
```

---

## 6. PyInstaller mit ONNX

### 6.1 Spec-Anpassungen für ONNX

```python
# In whisper-backend.spec:
# ONNX Runtime bringt eigene DLLs mit — diese müssen eingebunden werden

binaries = [
    # DirectML-DLLs automatisch einsammeln
    (r"C:\Program Files\...\onnxruntime.dll", "."),
]

# Oder: hook für onnxruntime
hiddenimports += [
    'onnxruntime.capi._pybind_state',
    'onnxruntime.capi.onnxruntime_pybind11_state',
]
```

**Einfacher:** `pyinstaller` mit `--collect-all onnxruntime`:

```powershell
pyinstaller --collect-all onnxruntime --collect-all onnxruntime-directml whisper-backend.spec
```

---

## 7. Wichtige Fallstricke

| Problem | Lösung |
|---------|--------|
| **DmlExecutionProvider nicht verfügbar** | Windows 11 + aktueller Grafiktreiber nötig. Fallback auf CPU. |
| **ONNX-Modell zu groß** | `whisper-large-v3-turbo` ist optimiert — ~1,5 GB statt ~3 GB PyTorch |
| **PyInstaller + ONNX = große Binary** | `--exclude torch matplotlib scipy` spart Gigabytes |
| **NPU wird nicht erkannt** | Prüfen: `ort.get_available_providers()` listet nur verfügbare Provider |
| **Modell-Filename unbekannt** | `hf_hub_download` mit `filename=model.onnx` oder Ordner-Inhalt listen |
| **Fehlerhafte Transkription** | ONNX-Modell-Decoding kann abweichen → Tokenizer von HuggingFace verwenden |

---

## 8. Verifikation

```powershell
# 1. Verfügbare Provider prüfen
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Sollte "DmlExecutionProvider" enthalten

# 2. Smoke-Test mit kurzer Audiodatei
curl -X POST -F "file=@test.wav" http://127.0.0.1:8765/transcribe
# Erwartet: {"text": "...", "status": "ok"}

# 3. Leistungsvergleich
# PyTorch (CPU): ~X Sekunden
# ONNX (CPU):   ~Y Sekunden (20-30% schneller)
# ONNX + NPU:   ~Z Sekunden (2-5x schneller, viel niedrigerer Stromverbrauch)
```