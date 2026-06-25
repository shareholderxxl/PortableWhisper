# Windows Hotkey & System Tray — Hintergrund-STT

## Ziel
Die Tauri-App registriert einen **globalen Hotkey** (z. B. `Ctrl+Shift+M`) zum Starten/Stoppen der Aufnahme und minimiert sich ins **System Tray** (Desktop-Taskleiste). So läuft die STT im Hintergrund und wird per Tastenkürzel aktiviert.

---

## 1. Plugin-Abhängigkeiten

### `Cargo.toml`

```toml
[dependencies]
tauri-plugin-global-shortcut = "2"
tauri = { version = "2", features = ["tray-icon"] }
```

### Frontend (npm)

```bash
npm install @tauri-apps/plugin-global-shortcut
```

---

## 2. Rust-Seite: Hotkey + Tray registrieren

In `src-tauri/src/lib.rs`:

```rust
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    Manager,
};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Code, Modifiers, Shortcut};

#[tauri::command]
fn start_recording(app: tauri::AppHandle) {
    // Signal ans Frontend/Python-Backend: Aufnahme starten
    let _ = app.emit("recording-started", ());
}

#[tauri::command]
fn stop_recording(app: tauri::AppHandle) {
    let _ = app.emit("recording-stopped", ());
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Globalen Hotkey registrieren: Ctrl+Shift+M
            let shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::KeyM);

            app.global_shortcut().register(shortcut).unwrap_or_else(|e| {
                eprintln!("Hotkey registration failed: {}", e);
            });

            // Hotkey-Event-Handler
            app.on_shortcut_event(move |_app, event, _shortcut| {
                if event.state == tauri_plugin_global_shortcut::ShortcutState::Pressed {
                    // Toggle: Aufnahme starten/stoppen
                    let _ = _app.emit("toggle-recording", ());
                }
            });

            // System Tray erstellen
            let quit = MenuItem::with_id(app, "quit", "Beenden", true, None::<&str>)?;
            let show = MenuItem::with_id(app, "show", "Anzeigen", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &quit])?;

            let _tray = TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "quit" => {
                        app.exit(0);
                    }
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    _ => {}
                })
                .build(app)?;

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![start_recording, stop_recording])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

---

## 3. Frontend-Seite: Events empfangen

In `src/App.tsx`:

```typescript
import { listen } from '@tauri-apps/api/event';

useEffect(() => {
    // Auf Hotkey-Event vom Rust-Backend lauschen
    const unlistenToggle = listen('toggle-recording', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });

    const unlistenStart = listen('recording-started', () => {
        setIsRecording(true);
    });

    const unlistenStop = listen('recording-stopped', () => {
        setIsRecording(false);
        // Transkriptionsergebnis anzeigen
    });

    return () => {
        unlistenToggle.then(fn => fn());
        unlistenStart.then(fn => fn());
        unlistenStop.then(fn => fn());
    };
}, []);
```

---

## 4. Capabilities erweitern

In `src-tauri/capabilities/default.json`:

```json
{
  "permissions": [
    "core:default",
    "shell:allow-open",
    "global-shortcut:allow-register",
    "global-shortcut:allow-unregister",
    "global-shortcut:allow-is-registered",
    "global-shortcut:allow-register-all",
    "global-shortcut:allow-unregister-all"
  ]
}
```

---

## 5. Fensterverhalten beim Schließen

Damit die App beim Schließen ins Tray minimiert wird (nicht beendet):

In `src-tauri/src/main.rs`:

```rust
fn main() {
    whisper4windows_lib::run()
}
```

In `tauri.conf.json`:

```json
{
  "app": {
    "windows": [
      {
        "closeBehavior": "minimize"
      }
    ]
  }
}
```

In `lib.rs` beim Tray-Event:

```rust
// Zusätzlich: Fenster ausblenden statt schließen
app.on_window_event(|window, event| {
    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
        api.prevent_close();  // Fenster nicht wirklich schließen
        let _ = window.hide();  // Nur verstecken
    }
});
```

---

## 6. Auto-Start mit Windows

### Plugin-Abhängigkeit

```toml
# Cargo.toml
tauri-plugin-autostart = "2"
```

```bash
npm install @tauri-apps/plugin-autostart
```

### Rust-Integration

```rust
use tauri_plugin_autostart::ManagerExt;

app.setup(|app| {
    // Auto-Start aktivieren
    let autostart = app.autostart();
    let _ = autostart.enable();
    Ok(())
});
```

### Capability

```json
"autostart:allow-enable",
"autostart:allow-disable",
"autostart:allow-is-enabled"
```

---

## 7. Icons für Tray

Tray-Icons unter `src-tauri/icons/`:
- `tray-icon.png` (32×32, für helle Taskleiste)
- `tray-icon-dark.png` (32×32, für dunkle Taskleiste)

Tauri sucht standardmäßig nach `icon.png` / `icon.ico` — diese auch als Tray-Icon nutzen.

---

## 8. Wichtige Fallstricke

| Problem | Lösung |
|---------|--------|
| **Hotkey funktioniert nicht** | Kein anderer Prozess belegt den Hotkey. Prüfen mit `global-shortcut:allow-is-registered` |
| **Tray-Icon unsichtbar** | PNG muss 32×32 oder 64×64 sein; `icon` in `TrayIconBuilder` korrekt setzen |
| **Auto-Start benötigt Admin** | Auf Windows: Registry-Eintrag unter `HKCU\...\Run` — kein Admin nötig |
| **Fenster schließt statt zu minimieren** | `CloseRequested` mit `api.prevent_close()` abfangen |
| **Hotkey nach System-Neustart weg** | `autostart` Plugin + Hotkey bei `setup` erneut registrieren |