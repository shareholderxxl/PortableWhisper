"""
Pfad-Management für PortableWhisper.

Lenkt alle Dateioperationen (Modelle, temporäre Audio-Chunks, Logs, GPU-Libs,
HF-Cache) in den direkten App-Ordner auf eine flache Struktur:
    <App_Root>/model/     - ONNX-Modell (vom User manuell platziert)
    <App_Root>/temp/      - temporäre Audio-Chunks
    <App_Root>/logs/      - Backend-Logs
    <App_Root>/gpu_libs/  - GPU-Bibliotheken (Optional)
    <App_Root>/cache/     - HuggingFace-Cache (runtime)

Wird so früh wie möglich importiert (Top of main.py), damit die Umgebungs-
variablen für den HuggingFace-Cache gesetzt sind, BEVOR huggingface_hub /
faster_whisper importiert werden.

Dieselbe Datei dient als PyInstaller-Runtime-Hook (läuft dann noch vor
main.py) — siehe whisper-backend.spec.
"""

import os
import sys
import json
from pathlib import Path

APP_NAME = "PortableWhisper"


def get_appdata_dir() -> Path:
    """Ermittelt das Datenverzeichnis im portablen App-Ordner."""
    if getattr(sys, "frozen", False):
        # Im portablen Modus liegt das Backend in: <App_Root>/binaries/whisper-backend.exe
        # Das Hauptverzeichnis liegt somit zwei Ebenen darüber.
        app_root = Path(sys.executable).parent.parent
    else:
        # Im Entwicklungsmodus liegt das Skript in: <App_Root>/backend/runtime_hooks/path_redirect.py
        # Das Hauptverzeichnis liegt somit drei Ebenen darüber.
        app_root = Path(__file__).parent.parent.parent

    return app_root


# Basisverzeichnis (direkt im App-Ordner, eine Ebene tief)
APP_DIR: Path = get_appdata_dir()

# Einzelne Sub-Verzeichnisse (flache Struktur, nur eine Ebene unter App_Root)
MODELS_DIR: Path = APP_DIR / "model"
DEFAULT_MODELS_DIR: Path = APP_DIR / "model"
TEMP_DIR: Path = APP_DIR / "temp"
LOGS_DIR: Path = APP_DIR / "logs"
GPU_LIBS_DIR: Path = APP_DIR / "gpu_libs"
CACHE_DIR: Path = APP_DIR / "cache"

# Zentrale Konfigurationsdatei (fuer persistente Settings, z.B. Phase 3B).
CONFIG_FILE: Path = APP_DIR / "config.json"

# Verzeichnisse sicher anlegen (idempotent)
for _d in (MODELS_DIR, DEFAULT_MODELS_DIR, TEMP_DIR, LOGS_DIR, GPU_LIBS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# HuggingFace-Cache wird absichtlich NICHT mehr auf model/ umgeleitet.
# Die App laedt das Modell nur manuell aus model/ (PW_ALLOW_HF_DOWNLOAD
# ist Opt-In); ohne Download soll model/ ausschliesslich die ONNX-Dateien
# enthalten. Ein HF-Download (Opt-In) landet im Standard-Systemcache.
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
# Temp-Verzeichnis für Audio-Chunks ebenfalls umleiten
os.environ.setdefault("TMPDIR", str(TEMP_DIR))


# PyInstaller windowed mode (console=False) setzt sys.stdout/stderr auf None,
# was print()/logging (uvicorn) zum Absturz bringt. Noop-Sink bereitstellen.
if getattr(sys, "stdout", None) is None:
    sys.stdout = open(os.devnull, "w")
if getattr(sys, "stderr", None) is None:
    sys.stderr = open(os.devnull, "w")


def get_models_dir() -> Path:
    """Modelle-Verzeichnis (für whisper_engine.get_models_dir Kompatibilität)."""
    return MODELS_DIR


def get_default_models_dir() -> Path:
    """Verzeichnis für das vom User manuell platzierte Standardmodell."""
    return DEFAULT_MODELS_DIR


def get_gpu_libs_dir() -> Path:
    """GPU-Libs-Verzeichnis (für gpu_manager.get_gpu_libs_dir Kompatibilität)."""
    return GPU_LIBS_DIR


def get_temp_dir() -> Path:
    """Temporärer Audio-Chunk-Speicher."""
    return TEMP_DIR


def get_logs_dir() -> Path:
    """Log-Verzeichnis."""
    return LOGS_DIR


def load_config() -> dict:
    """Liest die zentrale config.json. Bei Fehler/fehlend -> leeres Dict."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        # Kein logging hier (path_redirect laeuft sehr frueh); stillschweigend.
        pass
    return {}


def save_config(data: dict) -> None:
    """Schreibt die zentrale config.json atomar (temp + replace)."""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_suffix(CONFIG_FILE.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data if isinstance(data, dict) else {}, f, indent=2, ensure_ascii=False)
        os.replace(tmp, CONFIG_FILE)
    except OSError:
        pass
