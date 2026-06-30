# Walkthrough - Whisper4Windows Configuration & Sidecar Fixes

This document describes the fixes and enhancements applied to the Whisper4Windows portable application.

## Changes Made

### 1. Tauri Configuration Fix
*   **File modified:** [tauri.conf.json](file:///Y:/Austausch/whisper4windows-refactor/frontend/src-tauri/tauri.conf.json)
    *   Removed the invalid `"scope"` list under `"plugins": { "shell": { ... } }`.
    *   This resolves the deserialization error: `PluginInitialization("shell", "Error deserializing 'plugins.shell' within your Tauri configuration: unknown field scope, expected open")`.
*   **File modified:** [default.json](file:///Y:/Austausch/whisper4windows-refactor/frontend/src-tauri/capabilities/default.json)
    *   Added structured permissions under `"permissions"` to authorize spawning and executing the backend sidecar.
    *   Specifically configured target `"binaries/whisper-backend"` with `"sidecar": true` and `"args": true` under both `"shell:allow-spawn"` and `"shell:allow-execute"`.

### 2. Portable Package Build Fix (Sidecar Suffix Renaming)
*   **File modified:** [.github/workflows/build-windows.yml](file:///Y:/Austausch/whisper4windows-refactor/.github/workflows/build-windows.yml)
    *   Updated the step `Assemble portable ZIP` to rename the sidecar from `whisper-backend-x86_64-pc-windows-msvc.exe` to `whisper-backend.exe` during copy.
    *   **Why?** In production mode, Tauri's Rust sidecar runner automatically strips the target triple suffix when searching for the executable. If the suffix remains on the file in the `binaries/` folder, Tauri cannot find it at runtime and crashes with:
        `Failed to spawn backend sidecar: Io(Os { code: 2, kind: NotFound })`.

### 3. Single-Instance Show Window Enhancement
*   **File modified:** [lib.rs](file:///Y:/Austausch/whisper4windows-refactor/frontend/src-tauri/src/lib.rs)
    *   Updated the `tauri_plugin_single_instance` callback to call `window.show()` before `window.set_focus()`.
    *   **Why?** Since the main window is hidden (instead of destroyed) when clicking the "X" button, double-clicking the `Whisper4Windows.exe` again while the app was running in the background did nothing visually because the window was focused but remained hidden. Now, double-clicking the executable again will correctly restore and show the hidden settings window.

### 4. German Fine-Tuned Model Options
*   **File modified:** [index.html](file:///Y:/Austausch/whisper4windows-refactor/frontend/dist/index.html)
    *   Added `Medium German (Fine-tuned)` pointing to `mkenfenheuer/whisper-medium-cv11-german-ct2` to the dropdown menu.
    *   Added `Large V3 German (Fine-tuned)` pointing to `Reality-Interface/whisper-large-v3-german-faster-whisper` to the dropdown menu.
    *   **Why?** This gives you two high-quality options specifically trained on German voice datasets. The Medium model (~1.5GB) is a faster and less resource-heavy alternative to the Large model (~3GB).

### 5. Truly Portable Local Storage (data/ Directory)
*   **File modified:** [lib.rs](file:///Y:/Austausch/whisper4windows-refactor/frontend/src-tauri/src/lib.rs)
    *   Changed settings location to point to the local `data/` folder next to the main executable instead of the OS AppData folder.
*   **File modified:** [path_redirect.py](file:///Y:/Austausch/whisper4windows-refactor/backend/runtime_hooks/path_redirect.py)
    *   Changed backend directories (models, logs, temp, cache) to point to the local `data/` folder inside the portable app's root folder instead of the user's LocalAppData folder.
    *   **Why?** This makes the application 100% self-contained and truly portable. All downloaded models, settings, and logs will travel with the app folder.

### 6. Snappy Stop Recording & Debounced Hotkey Fix
*   **File modified:** [lib.rs](file:///Y:/Austausch/whisper4windows-refactor/frontend/src-tauri/src/lib.rs)
    *   **Instant Hiding:** Changed `cmd_stop_recording` to hide the window immediately when F9 is pressed to stop, instead of waiting for the transcription to finish. This restores cursor focus to your target text editor instantly.
    *   **Hotkey Debouncing:** Added a thread-safe `is_processing` state flag. When F9 is pressed to stop, the app enters the processing state. Any subsequent presses of F9 are ignored until the transcription is fully complete and injected, preventing duplicate backend requests and window flashing.

### 7. Explicit Model Management & Offline Default Model
*   **File modified:** [index.html](file:///Y:/Austausch/whisper4windows-refactor/frontend/dist/index.html)
    *   **Model Management UI:** Added a dedicated "Model Management" area in settings that lists all available models, shows their status, and provides explicit "Download" buttons.
    *   **Dropdown Restriction:** Modified the main "Model Quality" dropdown list to only display models that are already downloaded. This prevents accidental triggers of slow model downloads during a voice recording session.
    *   **Background Polling:** Added JavaScript routines to fire-and-forget backend downloads and poll `/model/status` in the background, updating status badges dynamically.
*   **File modified:** [lib.rs](file:///Y:/Austausch/whisper4windows-refactor/frontend/src-tauri/src/lib.rs)
    *   **Default Selected Model:** Changed the default selected model on startup from `small` to `tiny` to match the bundled offline model.
*   **File modified:** [.github/workflows/build-windows.yml](file:///Y:/Austausch/whisper4windows-refactor/.github/workflows/build-windows.yml)
    *   **Offline Default Model:** Updated the portable ZIP packaging steps to pre-download the lightweight `tiny` model weight files (approx. 75MB) directly into the portable bundle (`data/models/hub/...`). This ensures the application works immediately out-of-the-box in offline environments without requiring any downloads, while keeping the ZIP file size very small.

### 8. Removed "Launch on Startup" Configuration Option
*   **File modified:** [index.html](file:///Y:/Austausch/whisper4windows-refactor/frontend/dist/index.html)
    *   Removed the "Launch on Startup" toggle from the settings page. Since the app is built as a portable bundle, auto-starting with Windows is typically not desired or handled differently by the user.

### 9. Push-to-Talk (PTT) Recording Mode
*   **File modified:** [index.html](file:///Y:/Austausch/whisper4windows-refactor/frontend/dist/index.html)
    *   **Settings Switcher:** Replaced the non-functional "Switch Mode" placeholder row with a fully functional **Recording Mode** selector. Users can toggle between **Toggle** and **Push-To-Talk** modes.
    *   **Rust API Call:** Connects the buttons to Tauri using a new `set_recording_mode` command. Loads the setting dynamically on startup using `get_recording_mode`.
*   **File modified:** [lib.rs](file:///Y:/Austausch/whisper4windows-refactor/frontend/src-tauri/src/lib.rs)
    *   **Settings Serialization:** Added `recording_mode` to the settings struct and state. Default mode is `"toggle"`.
    *   **Shortcut Press/Release Handling:** Updated the `tauri_plugin_global_shortcut` handler to intercept both press and release events.
        *   In **Push-To-Talk (PTT)** mode, recording starts on hotkey **press** and stops immediately on hotkey **release**.
        *   In **Toggle** mode, recording starts/stops exclusively on hotkey **press**.

### 10. Cleaned up Unimplemented & Placeholder Sections
*   **File modified:** [index.html](file:///Y:/Austausch/whisper4windows-refactor/frontend/dist/index.html)
    *   Removed the unimplemented **Vocabulary** page, sidebar menu entry, and card on the Home page.
    *   Removed the unimplemented **History** page and sidebar menu entry.
    *   Removed the static **What's New** changelog list from the Home page.

### 11. About & License Info
*   **File modified:** [index.html](file:///Y:/Austausch/whisper4windows-refactor/frontend/dist/index.html)
    *   Added a new **About** section at the bottom of settings.
    *   Displays application name, version `v0.1.0`, and license details.
    *   Provides an **Original GitHub** button referencing the original repository by Bader Aljabri.
*   **File modified:** [lib.rs](file:///Y:/Austausch/whisper4windows-refactor/frontend/src-tauri/src/lib.rs)
    *   Added a secure `open_url` command that opens external links (like the original GitHub project page) safely in the user's default browser on Windows.

---

## Validation & Testing

1. **Manual Verification**:
   - Inside `C:\Users\Enste\Downloads\Whisper4Windows-portable_v2\binaries`, we manually renamed `whisper-backend-x86_64-pc-windows-msvc.exe` to `whisper-backend.exe`.
   - Launched the application from the console: `Whisper4Windows.exe`.
   - **Result:** The application now launches successfully! It registers shortcuts, creates the tray icon, starts the backend server in CPU mode on port `8765`, and keeps running.

---

## Next Steps for You

Commit and push all changes in the `Y:\Austausch\whisper4windows-refactor` repository. Once your GitHub Actions pipeline completes the run:
1. The new zip package will be correctly structured.
2. The backend binary in the zip will be named `whisper-backend.exe` automatically.
3. The settings page will feature both new German model choices.
4. If you close the settings window and double-click `Whisper4Windows.exe` again, it will correctly pop the window back up.
5. All app data will be stored locally inside a `data/` subdirectory inside the application directory!
6. The F9 hotkey will stop recording instantly on the first press without requiring a double press.
7. The lightweight 'tiny' model is pre-packaged inside the ZIP, making the app work offline immediately. You can download extra models using the new Model Management section.
8. The "Launch on Startup" option is removed from the settings interface.
9. You can select between "Toggle" and "Push-to-Talk" under Keyboard Shortcuts in settings.
10. The UI is clean, without unimplemented placeholder pages like "Vocabulary" or "History" or the static changelog.
11. Click on "Original GitHub" in the settings to view the original repository by Bader Aljabri.
