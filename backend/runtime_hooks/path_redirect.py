"""
Pfad-Management für Whisper4Windows.

Lenkt alle Dateioperationen (Modelle, temporäre Audio-Chunks, Logs, GPU-Libs,
HF-Cache) in ein einziges Benutzerverzeichnis unter
    %LOCALAPPDATA%/Whisper4Windows/   (Windows)
    ~/.local/share/Whisper4Windows/   (Linux/macOS)

Wird so früh wie möglich importiert (Top of main.py), damit die Umgebungs-
variablen für den HuggingFace-Cache gesetzt sind, BEVOR huggingface_hub /
faster_whisper importiert werden.

Dieselbe Datei dient als PyInstaller-Runtime-Hook (läuft dann noch vor
main.py) — siehe whisper-backend.spec.
"""

import os
import sys
from pathlib import Path

APP_NAME = "Whisper4Windows"


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
    
    return app_root / "data"


# Basisverzeichnis
APP_DIR: Path = get_appdata_dir()

# Einzelne Sub-Verzeichnisse
MODELS_DIR: Path = APP_DIR / "models"
TEMP_DIR: Path = APP_DIR / "temp"
LOGS_DIR: Path = APP_DIR / "logs"
GPU_LIBS_DIR: Path = APP_DIR / "gpu_libs"
CACHE_DIR: Path = APP_DIR / "cache"

# Verzeichnisse sicher anlegen (idempotent)
for _d in (MODELS_DIR, TEMP_DIR, LOGS_DIR, GPU_LIBS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# HuggingFace-Cache umleiten — MUSS vor jedem HF-Import gesetzt sein.
os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))
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


def get_gpu_libs_dir() -> Path:
    """GPU-Libs-Verzeichnis (für gpu_manager.get_gpu_libs_dir Kompatibilität)."""
    return GPU_LIBS_DIR


def get_temp_dir() -> Path:
    """Temporärer Audio-Chunk-Speicher."""
    return TEMP_DIR


def get_logs_dir() -> Path:
    """Log-Verzeichnis."""
    return LOGS_DIR
