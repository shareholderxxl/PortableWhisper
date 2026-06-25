# Tauri Portable Sidecar — Whisper4Windows

## Ziel
Das FastAPI-Backend (mit Whisper-Modell `primeline/whisper-large-v3-german`) wird via **PyInstaller** in eine eigenständige `whisper-backend.exe` gefreezt und als **Tauri-Sidecar** registriert. Die App wird als portables ZIP-Archiv ohne Admin-Rechte ausgeliefert.

---

## 1. PyInstaller-Konfiguration

### 1.1 Spezifikationsdatei: `whisper-backend.spec`

Die `.spec`-Datei muss im Root des Python-Backend-Ordners liegen. Wichtig: Das Modell wird **nicht** in die `.exe` eingebettet, sondern beim ersten App-Start per `huggingface_hub` nach `%LOCALAPPDATA%/Whisper4Windows/models/` heruntergeladen.

```python
# whisper-backend.spec
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],  # Einstiegspunkt der FastAPI-App
    pathex=[],
    binaries=[],
    datas=[],  # Modell-Daten NICHT hier einbetten!
    hiddenimports=[
        'uvicorn',
        'fastapi',
        'huggingface_hub',
        'torch',
        'whisper',
        'soundfile',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'scipy', 'notebook',
        'jupyter', 'ipython', 'test', 'tests',
    ],
    noarchive=False,
    optimize=2,  # Größenoptimierung
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='whisper-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,            # UPX-Kompression für kleinere Binaries
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,       # Kein Konsolenfenster im Hintergrund
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

### 1.2 Build-Befehl (Windows)

```powershell
# Im Python-Backend-Verzeichnis
pyinstaller whisper-backend.spec
```

Ergebnis: `dist/whisper-backend/whisper-backend.exe`

---

## 2. Pfad-Management: Alles nach %LOCALAPPDATA%

### 2.1 Runtime-Hook für Pfad-Umleitung

Erstelle `runtime_hooks/path_redirect.py`:

```python
"""Lenkt alle Dateioperationen auf %LOCALAPPDATA%/Whisper4Windows/ um."""
import os
import sys
from pathlib import Path

APP_NAME = "Whisper4Windows"

def _get_appdata_dir() -> Path:
    """Ermittelt den lokalen AppData-Ordner des Benutzers."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            base = os.path.join(os.environ["USERPROFILE"], "AppData", "Local")
    else:
        base = os.path.join(os.environ["HOME"], ".local", "share")
    return Path(base) / APP_NAME

# Modelle-Cache
MODELS_DIR = _get_appdata_dir() / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Temporäre Audio-Chunks
TEMP_DIR = _get_appdata_dir() / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Logs
LOGS_DIR = _get_appdata_dir() / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
```

### 2.2 HuggingFace-Cache umleiten

```python
import os

# Vor huggingface_hub-Import setzen!
os.environ["HF_HOME"] = str(MODELS_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR / "hub")
os.environ["XDG_CACHE_HOME"] = str(_get_appdata_dir() / "cache")
```

### 2.3 Modell-Download beim ersten Start

```python
from huggingface_hub import snapshot_download
import os

MODEL_ID = "primeline/whisper-large-v3-german"
model_path = os.environ["HUGGINGFACE_HUB_CACHE"]

if not os.listdir(model_path):
    print(f"[Whisper4Windows] Lade Modell {MODEL_ID} herunter...")
    snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=model_path,
        local_files_only=False,
    )
    print("[Whisper4Windows] Modell-Download abgeschlossen.")
```

---

## 3. Tauri-Sidecar-Integration

### 3.1 Sidecar in `tauri.conf.json` registrieren

```json
{
  "bundle": {
    "externalBin": [
      "binaries/whisper-backend.exe"
    ],
    "windows": {
      "wix": null,
      "nsis": null
    }
  },
  "plugins": {
    "shell": {
      "sidecar": true,
      "scope": [
        {
          "name": "binaries/whisper-backend",
          "sidecar": true,
          "args": true
        }
      ]
    }
  }
}
```

**Wichtig:** Der Sidecar-Pfad in `externalBin` verwendet **keine** Dateiendung — Tauri sucht automatisch nach `whisper-backend.exe` auf Windows.

### 3.2 Sidecar-Kopie ins Tauri-Binary-Verzeichnis

Nach dem PyInstaller-Build muss `whisper-backend.exe` nach `src-tauri/binaries/` kopiert werden:

```powershell
# Von der Tauri-Projektwurzel aus
copy-item .\backend\dist\whisper-backend\whisper-backend.exe .\src-tauri\binaries\
```

Das Verzeichnis `src-tauri/binaries/` muss ggf. angelegt werden.

### 3.3 Sidecar aus Rust starten/beenden

In `src-tauri/src/main.rs` oder `lib.rs`:

```rust
use tauri_plugin_shell::ShellExt;

// Beim App-Start
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Sidecar starten (Backend läuft z.B. auf Port 8765)
            let sidecar_command = app.shell()
                .sidecar("binaries/whisper-backend")
                .expect("Sidecar binary not found")
                .args(["--port", "8765"]);
            let (mut _rx, _child) = sidecar_command
                .spawn()
                .expect("Failed to spawn sidecar");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 3.4 Cargo.toml — benötigte Plugins

```toml
[dependencies]
tauri = { version = "2", features = [] }
tauri-plugin-shell = "2"
```

---

## 4. Portables ZIP-Archiv

### 4.1 Tauri-Konfiguration für ZIP-Build

In `tauri.conf.json`:

```json
{
  "bundle": {
    "active": true,
    "targets": ["nsis", "msi"],
    "windows": {
      "wix": null,
      "nsis": {
        "installMode": "currentUser"
      }
    }
  }
}
```

Für ein reines ZIP (ohne Installer) kann ein Build-Skript verwendet werden:

```powershell
# build-portable.ps1 — nach `tauri build`
$releaseDir = "src-tauri/target/release"
$appName = "Whisper4Windows"
$outputZip = "$appName-portable.zip"

Compress-Archive -Path @(
    "$releaseDir/$appName.exe",
    "src-tauri/binaries/whisper-backend.exe",
    "src-tauri/resources/*"
) -DestinationPath $outputZip -Force

Write-Host "Portables ZIP erstellt: $outputZip"
```

### 4.2 Wichtige Pfad-Regel

```text
nach Entpacken:
Whisper4Windows/
├── Whisper4Windows.exe        # Tauri-Hauptapp
├── binaries/
│   └── whisper-backend.exe    # Sidecar (vom Tauri gestartet)
└── resources/                 # Optional
```

---

## 5. Wichtige Fallstricke

| Problem | Lösung |
|---------|--------|
| **Modell zu groß für PyInstaller** | Modell **nicht** einbetten, sondern via `huggingface_hub` on-demand laden |
| **UAC-Prompt beim Start** | Kein Installer — portables ZIP. Sidecar startet ohne Admin-Rechte |
| **Sidecar wird nicht gefunden** | `externalBin` ohne `.exe`-Endung; Kopie nach `src-tauri/binaries/` |
| **HuggingFace verbindet nicht** | Benötigt Internet beim ersten Start; Modell-Cache in `%LOCALAPPDATA%` |
| **Port-Konflikt** | Sidecar auf `127.0.0.1:8765` (localhost only, kein externer Zugriff) |
| **PyInstaller zu groß** | `UPX=True`, `excludes` mit `tkinter, matplotlib, scipy` etc. |

---

## 6. Verifikation

Nach dem Build:
1. ZIP entpacken → `Whisper4Windows.exe` ohne Admin starten
2. Tauri startet `whisper-backend.exe` als Sidecar (unsichtbar)
3. Backend lädt `primeline/whisper-large-v3-german` bei Bedarf nach `%LOCALAPPDATA%/Whisper4Windows/models/hub/`
4. API unter `http://127.0.0.1:8765` erreichbar
5. Kein UAC-Prompt, kein separates Python/Node.js nötig
