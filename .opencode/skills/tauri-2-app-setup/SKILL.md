# Tauri v2 App Setup — Projektstruktur & Konfiguration

## Ziel
Die Tauri v2-Desktop-App wird korrekt initialisiert, konfiguriert und gebaut. Enthält die Standard-Projektstruktur, `tauri.conf.json`, `Cargo.toml`, Build-Targets und Frontend-Einbindung.

---

## 1. Projektstruktur

```
PortableWhisper/
├── src/                          # Tauri-Frontend (React/TypeScript)
│   ├── App.tsx                   # Hauptkomponente
│   ├── App.css
│   ├── main.tsx
│   ├── components/
│   │   ├── TranscribeButton.tsx
│   │   ├── AudioVisualizer.tsx
│   │   ├── SettingsPanel.tsx
│   │   └── PostProcessingTab.tsx  # Phase 3
│   └── styles/
├── src-tauri/                    # Tauri-Rust-Backend
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── capabilities/
│   │   └── default.json
│   ├── binaries/                 # Hier landet die PyInstaller-.exe
│   │   └── whisper-backend.exe   (kompiliert & kopiert)
│   ├── src/
│   │   ├── lib.rs
│   │   └── main.rs
│   └── icons/
├── backend/                      # Python-FastAPI-Backend
│   ├── main.py
│   ├── whisper_engine.py
│   ├── path_redirect.py
│   ├── requirements.txt
│   ├── whisper-backend.spec      (PyInstaller)
│   └── runtime_hooks/
├── .opencode/
│   └── skills/                   (diese Skills)
├── package.json
├── tsconfig.json
├── vite.config.ts
└── index.html
```

---

## 2. `tauri.conf.json` — Minimal-Konfiguration

```json
{
  "$schema": "https://raw.githubusercontent.com/tauri-apps/tauri/dev/crates/tauri-config-schema/schema.json",
  "productName": "PortableWhisper",
  "version": "1.0.0",
  "identifier": "com.whisper4windows.app",
  "build": {
    "frontendDist": "../dist",
    "devUrl": "http://localhost:5173",
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build"
  },
  "app": {
    "title": "PortableWhisper",
    "windows": [
      {
        "title": "PortableWhisper",
        "width": 800,
        "height": 600,
        "resizable": true,
        "fullscreen": false,
        "center": true
      }
    ],
    "security": {
      "csp": null
    }
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ],
    "externalBin": [
      "binaries/whisper-backend"
    ]
  }
}
```

## 3. `Cargo.toml` — Abhängigkeiten

```toml
[package]
name = "whisper4windows"
version = "1.0.0"
edition = "2021"

[lib]
name = "whisper4windows_lib"
crate-type = ["lib", "cdylib", "staticlib"]

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = [] }
tauri-plugin-shell = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

## 4. `capabilities/default.json` — Berechtigungen

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "Standard-Berechtigungen für PortableWhisper",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "shell:allow-open",
    "shell:allow-execute",
    "shell:allow-spawn",
    {
      "identifier": "shell:allow-execute",
      "allow": [
        {
          "name": "binaries/whisper-backend",
          "sidecar": true,
          "args": true
        }
      ]
    }
  ]
}
```

## 5. Frontend-Initialisierung

```bash
# Tauri 2 + React + TypeScript erstellen
npm create tauri-app@latest PortableWhisper -- --template react-ts
cd PortableWhisper
npm install
```

Danach:
- `src/` mit React-Komponenten befüllen
- `src-tauri/tauri.conf.json` wie oben anpassen
- `src-tauri/Cargo.toml` erweitern

## 6. Build-Befehl

```bash
# Entwicklungsmodus
npm run tauri dev

# Release-Build
npm run tauri build
```

Ergebnis liegt in `src-tauri/target/release/bundle/`.

## 7. Wichtige Fallstricke

| Problem | Lösung |
|---------|--------|
| **Tauri v1 vs v2 API-Unterschiede** | Immer `@tauri-apps/api` v2 und `tauri-plugin-*` v2 verwenden |
| **Sidecar wird nicht gefunden** | `externalBin` OHNE `.exe`-Endung; Binary liegt in `src-tauri/binaries/` |
| **Capabilities fehlen** | Tauri v2 benötigt explizite `permissions` in `capabilities/*.json` |
| **npm create tauri-app hängt** | `npm create tauri-app@latest` ohne `--yes` — interaktiv bestätigen |