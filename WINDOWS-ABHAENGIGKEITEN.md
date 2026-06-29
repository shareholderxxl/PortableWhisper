# Windows-Abhängigkeiten für den Build

Dieses Projekt wird unter **Windows** gebaut. Der Linux-Server dient nur zur Vorbereitung und Code-Analyse.

> **Hinweis (Phase 1):** Der primäre Build-Pfad ist jetzt die GitHub-Actions-
> Pipeline `.github/workflows/build-windows.yml` (Runner `windows-latest`). Die
> unten genannten Tools werden dort automatisch eingerichtet. Eine lokale
> Installation ist nur für manuelle Builds / Fehlersuche nötig.

## 1. Rust & MSVC Build Tools

Tauri benötigt einen C++-Compiler unter Windows — den **MSVC Build Tools** (Microsoft Visual C++).

### Installation

```powershell
# 1. Rust via rustup installieren
winget install Rustlang.Rustup
# oder: https://rustup.rs/ herunterladen und ausführen

# 2. MSVC Build Tools installieren
winget install Microsoft.VisualStudio.2022.BuildTools
```

Alternativ: **Visual Studio 2022 Community** mit Workload „Desktop development with C++".

### Prüfen

```powershell
rustc --version        # z. B. 1.86.0
cargo --version        # z. B. 1.86.0
```

### Rust-Target (automatisch)

```
stable-x86_64-pc-windows-msvc
```

---

## 2. Python 3.11 & PyInstaller

Das Python-Backend wird mit **PyInstaller** in eine eigenständige `whisper-backend.exe` gefreezt.

### Installation

```powershell
# Python 3.11 installieren (nicht 3.13 — PyInstaller-Kompatibilität)
winget install Python.Python.3.11

# Paketmanager
python -m ensurepip --upgrade

# Abhängigkeiten installieren (ohne torch)
cd backend
pip install -r requirements-portable.txt  # ohne torch, mit huggingface_hub + pyinstaller

# PyInstaller
pip install pyinstaller

# Optional: UPX-Kompression (kleinere .exe)
# https://github.com/upx/upx/releases — upx.exe in den PATH kopieren
```

### Prüfen

```powershell
python --version               # 3.11.x
pyinstaller --version          # z. B. 6.12.0
```

---

## 3. WebView2 Runtime

Tauri benötigt die **WebView2 Runtime** zum Rendern des Frontends.

- **Windows 11** → ✅ bereits vorinstalliert
- **Windows 10** → meist vorinstalliert; falls nicht:
  ```powershell
  # https://developer.microsoft.com/en-us/microsoft-edge/webview2/
  # Oder via winget:
  winget install Microsoft.EdgeWebView2Runtime
  ```

---

## 4. Kurz-Checkliste für den Build-Rechner

```text
☐ Rust 1.86+ (rustup)         → rustc --version
☐ MSVC Build Tools            → cl.exe vorhanden im PATH
☐ Python 3.11                 → python --version
☐ pip                         → pip --version
☐ PyInstaller                 → pyinstaller --version
☐ Node.js 22+                 → node --version
☐ npm                         → npm --version
☐ WebView2 Runtime            → wird von Tauri benötigt
☐ Git                         → git --version
☐ Optional: UPX               → upx --version
```

---

## Hinweis

Alle genannten Tools sind **frei und öffentlich verfügbar**. Keine Lizenzkosten.
