# Walkthrough - Whisper4Windows Configuration Fix

This document describes the Tauri configuration fixes applied to resolve the startup crash.

## Changes Made

### 1. Removed Invalid Scope Configuration
In [tauri.conf.json](file:///Y:/Austausch/whisper4windows-refactor/frontend/src-tauri/tauri.conf.json):
* Removed the `"scope"` list under `"plugins": { "shell": { ... } }`.
* This resolves the crash: `PluginInitialization("shell", "Error deserializing 'plugins.shell' within your Tauri configuration: unknown field scope, expected open")`.

### 2. Configured Sidecar Permissions & Scope
In [default.json](file:///Y:/Austausch/whisper4windows-refactor/frontend/src-tauri/capabilities/default.json):
* Added structured permissions under `"permissions"` to authorize spawning and executing the backend sidecar.
* Configured target `"binaries/whisper-backend"` with `"sidecar": true` and `"args": true` under both `"shell:allow-spawn"` and `"shell:allow-execute"`.

---

## Validation & Testing

1. **Backend Verification**:
   - The backend binary `whisper-backend-x86_64-pc-windows-msvc.exe` was executed locally via cmd.exe and it successfully launched the FastAPI server in CPU mode on port `8765`, confirming the backend bundle is clean.
2. **Frontend Build Verification**:
   - The modifications alignment with Tauri v2 capabilities has been checked against standard Tauri v2 documentation.

---

## Next Steps for You

Since the build tools are not locally installed in the current environment, you can trigger a build through your repository's automated CI/CD pipeline:

1. **Push Changes**: Commit the changes in the `frontend/src-tauri/` folder and push them to your GitHub repository.
2. **Download Portable App**: Once the `Build Windows Portable` GitHub Action finishes, download the newly generated portable zip.
3. **Launch**: Extract and launch `Whisper4Windows.exe` to verify that the app now starts successfully and tray icon functions correctly!
