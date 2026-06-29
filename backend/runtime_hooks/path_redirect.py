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
    """Ermittelt das lokale App-Datenverzeichnis des Benutzers."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            userprofile = os.environ.get("USERPROFILE")
            if userprofile:
                base = os.path.join(userprofile, "AppData", "Local")
            else:
                base = os.path.expanduser("~")
    else:
        base = os.path.join(os.environ.get("HOME", os.path.expanduser("~")),
                            ".local", "share")
    return Path(base) / APP_NAME


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
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR / "transformers"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
# Temp-Verzeichnis für Audio-Chunks ebenfalls umleiten
os.environ.setdefault("TMPDIR", str(TEMP_DIR))


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
