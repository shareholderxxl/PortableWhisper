# GitHub Actions — Windows Build Pipeline

## Ziel
Automatisierter Build von PortableWhisper als portables ZIP-Archiv. Die Pipeline:
1. Baut das Python-Backend mit PyInstaller → `whisper-backend.exe`
2. Baut das Tauri-Frontend + Sidecar → `PortableWhisper.exe`
3. Packt alles in ein portables ZIP
4. Lädt es als GitHub Release / Artifact hoch

---

## 1. GitHub Actions Workflow

### `.github/workflows/build-windows.yml`

```yaml
name: Build Windows Portable

on:
  push:
    tags:
      - 'v*'              # Nur bei getaggten Releases
  workflow_dispatch:       # Manuell auslösbar

jobs:
  build-windows:
    runs-on: windows-latest
    defaults:
      run:
        shell: powershell

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      # ============================================
      # STEP 1: Python-Backend mit PyInstaller bauen
      # ============================================
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install Python Dependencies
        working-directory: backend
        run: |
          pip install -r requirements.txt
          pip install pyinstaller

      - name: Build Backend with PyInstaller
        working-directory: backend
        run: |
          pyinstaller --clean --onefile `
            --name whisper-backend `
            --console `
            --add-data "runtime_hooks;runtime_hooks" `
            --collect-all onnxruntime `
            --collect-all huggingface_hub `
            --hidden-import uvicorn `
            --hidden-import fastapi `
            --hidden-import numpy `
            --hidden-import soundfile `
            --hidden-import librosa `
            main.py

      # ============================================
      # STEP 2: Sidecar ins Tauri-Projekt kopieren
      # ============================================
      - name: Copy Sidecar to Tauri Binary Dir
        run: |
          New-Item -ItemType Directory -Force -Path src-tauri/binaries
          Copy-Item backend/dist/whisper-backend.exe src-tauri/binaries/

      # ============================================
      # STEP 3: Tauri-Frontend bauen
      # ============================================
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'npm'

      - name: Install Frontend Dependencies
        run: npm ci

      - name: Build Frontend
        run: npm run build

      # ============================================
      # STEP 4: Tauri-App bauen (Rust + Frontend)
      # ============================================
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Cache Rust Dependencies
        uses: Swatinem/rust-cache@v2
        with:
          workspaces: src-tauri

      - name: Build Tauri App
        run: npm run tauri build
        env:
          TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}
          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_KEY_PASSWORD }}

      # ============================================
      # STEP 5: Portables ZIP erstellen
      # ============================================
      - name: Build Portable ZIP
        run: |
          $appName = "PortableWhisper"
          $version = "${{ github.ref_name }}"
          $releaseDir = "src-tauri/target/release"
          $output = "${{ runner.temp }}\$appName-$version-portable.zip"

          Compress-Archive -Path @(
            "$releaseDir\PortableWhisper.exe",
            "src-tauri\binaries\whisper-backend.exe",
            "README.md",
            "LICENSE"
          ) -DestinationPath $output -Force

          echo "ZIP created: $output"
          echo "ZIP_PATH=$output" >> $env:GITHUB_ENV

      # ============================================
      # STEP 6: Hochladen
      # ============================================
      - name: Upload Portable ZIP as Artifact
        uses: actions/upload-artifact@v4
        with:
          name: PortableWhisper-portable
          path: ${{ env.ZIP_PATH }}
          retention-days: 30

      - name: Upload to GitHub Release
        if: startsWith(github.ref, 'refs/tags/')
        uses: softprops/action-gh-release@v2
        with:
          files: ${{ env.ZIP_PATH }}
          generate_release_notes: true
```

---

## 2. Zusätzliche Workflows

### Schnell-Build (PR/Commit — ohne Signing)

```yaml
# .github/workflows/build-check.yml
name: Build Check

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-check:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Quick Backend Build Check
        working-directory: backend
        run: |
          pip install -r requirements.txt
          python -c "import fastapi; import uvicorn; import onnxruntime; print('ONNX Provider:', ort.get_available_providers())"
      - uses: actions/setup-node@v4
        with:
          node-version: '22'
      - name: Quick Frontend Build Check
        run: |
          npm ci
          npm run build
```

---

## 3. Secrets & Environment

In den GitHub-Repository-Settings hinterlegen:

| Secret | Zweck |
|--------|-------|
| `TAURI_SIGNING_PRIVATE_KEY` | Optional: Tauri-Updater-Signing |
| `TAURI_SIGNING_KEY_PASSWORD` | Passwort für den Signing-Key |

---

## 4. Muon-Installer (Alternative zu ZIP)

Falls später ein Installer gewünscht wird:

```yaml
# Ergänzung im Build-Step
- name: Build NSIS Installer
  run: |
    # Tauri baut standardmäßig MSI/NSIS mit
    # Ausgabe in src-tauri/target/release/bundle/nsis/
    Compress-Archive -Path @(
      "src-tauri/target/release/bundle/nsis/*.exe"
    ) -DestinationPath "${{ runner.temp }}\$appName-$version-installer.zip"
```

---

## 5. Wichtige Fallstricke

| Problem | Lösung |
|---------|--------|
| **PyInstaller + ONNX = großer Binary** | `--exclude torch matplotlib scipy`; `--collect-all` nur für onnxruntime + huggingface_hub |
| **Tauri-Build schlägt fehl** | Rust-Toolchain muss `stable` sein; `windows-latest` hat MSVC-Build-Tools |
| **Sidecar-Pfad falsch** | `externalBin` in `tauri.conf.json` ohne `.exe`; Binary in `src-tauri/binaries/` |
| **GitHub Actions Cache** | Rust-Cache via `Swatinem/rust-cache` spart ~10 Minuten pro Build |
| **Release nur bei Tags** | `workflow_dispatch` für manuelle Builds; `on: push: tags: v*` für Releases |
| **Windows-Code-Signing** | Optional via Azure Key Vault; benötigt `TAURI_SIGNING_PRIVATE_KEY` |

---

## 6. Manuelles Auslösen

```bash
# Release taggen
git tag v1.0.0
git push origin v1.0.0
# → Workflow startet automatisch

# Oder via GitHub WebUI: Actions → Build Windows Portable → Run workflow
```