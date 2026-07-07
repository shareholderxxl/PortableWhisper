# AGENTS.md

Compact guidance for OpenCode sessions working in this repo. Read `KONZEPT.md`
first for the 3-phase roadmap before any non-trivial change.

## What this is

PortableWhisper — local, offline speech-to-text for Windows. Stack:
- **Backend**: Python 3.11 + FastAPI + `faster-whisper`/`ctranslate2` (no torch).
  Flat module layout in `backend/` (`main.py`, `whisper_engine.py`,
  `audio_capture.py`, `gpu_manager.py`).
- **Frontend**: Tauri 2 (Rust in `frontend/src-tauri/`) + **static HTML**
  in `frontend/dist/` (`index.html`, `recording.html`). No JS framework,
  no bundler, no Node build step.

## Build platform — critical

Builds run **only on Windows** (`WINDOWS-ABHAENGIGKEITEN.md`). This Linux
server is for code preparation and analysis only. Do not attempt to compile
Tauri/PyInstaller artifacts here. Builds go through GitHub Actions on
`windows-latest` via `.github/workflows/build-windows.yml` (portable ZIP).

## Development workflow (Windows)

Two terminals:
```powershell
# Terminal 1 — backend (port 8765 after Phase 1; currently 8000)
cd backend
venv\Scripts\activate
python main.py

# Terminal 2 — frontend with hot reload
cd frontend\src-tauri
cargo tauri dev
```
- Frontend HTML/CSS/JS in `frontend/dist/` hot-reloads on save.
- Rust changes auto-rebuild.
- **Python has no hot reload** — restart `main.py` manually after backend edits.
- Production build: `cargo tauri build --no-bundle` (in `frontend/src-tauri/`).

## No Node/npm — common mistake

There is **no `package.json`** at root or in `frontend/`. Tauri's
`beforeBuildCommand` is intentionally empty — it bundles `../dist` directly.
Any CI/skill template that runs `npm ci` / `npm run build` is wrong for this
repo and must be adapted.

## Gitignore gotchas

| Path | Ignored? | Note |
|---|---|---|
| `backend/*.spec` | **yes** | PyInstaller spec files are not tracked. Add a `!backend/whisper-backend.spec` exception if you need it in version control. |
| `frontend/src-tauri/binaries/` | yes | Sidecar `.exe` is a build artifact; dir does not exist yet. |
| `backend/models/` | yes | Whisper models download on first run (~500MB+). |
| `backend/dist/`, `backend/build/` | yes | PyInstaller output. |
| `frontend/src-tauri/target/` | yes | Rust build (MSI artifact is un-ignored). |

## Sidecar wiring — already done

`frontend/src-tauri/src/lib.rs` already spawns the `whisper-backend` sidecar
(see `lib.rs:975` and the restart logic at `lib.rs:1123`). Registered in
`tauri.conf.json` via `bundle.externalBin: ["binaries/whisper-backend"]`.
Don't reinvent — extend in place. Sidecar permission is granted in
`capabilities/default.json` as structured `shell:allow-spawn`/`allow-execute`
entries (NOT via `plugins.shell.scope` in tauri.conf.json — that field is
invalid in Tauri 2 and crashes startup with "unknown field scope"). A global
`RunEvent::Exit` handler in `lib.rs` kills the backend on any exit path.

## No test / lint / format tooling

No eslint, prettier, ruff, black, rustfmt, pyproject.toml, or Makefile.
`backend/test_gpu_capabilities.py` is a **manual** diagnostic script
(`python test_gpu_capabilities.py`), not part of a pytest suite. Verify
Python changes by running the backend and hitting `http://127.0.0.1:<port>/health`.

## Python version

Use **Python 3.11** for anything PyInstaller-related (compat). The
`requirements.txt` header claiming "Compatible with Python 3.13" refers to
runtime dev use, not the freeze step.

## GPU libraries

CUDA libs are **not bundled** — `gpu_manager.py` downloads them on demand at
first run if an NVIDIA GPU is detected. Keeps the binary ~200MB instead of
~1.3GB. Don't re-bundle them in PyInstaller.

## Project skills (read before major work)

`.opencode/skills/` contains 7 phase-mapped skills:
`tauri-2-app-setup`, `tauri-audio-recording`, `tauri-portable-sidecar`,
`windows-hotkey-tray`, `onnx-directml-npu` (Phase 2), `local-llm-postprocessing`
(Phase 3), `github-actions-build`. Each has a `SKILL.md` with concrete config
and code templates — consult the matching one before touching that area.

## Current roadmap state

Phase 1 (Zero-Dependency & Portability) is **in implementation** on branch
`phase-1-2`. See `PHASE1_PLAN.md` for the full step list. Implemented decisions:
- Default model is `default` (manually placed CTranslate2 model in
  `data/models/default/`); `large-v3-turbo` is the quality option.
- Backend port = **8765** (constants `BACKEND_HOST`/`BACKEND_PORT` in `main.py`).
- PyInstaller **without torch** (ctranslate2 only); spec = `backend/whisper-backend.spec`.
- Build artifact: **portable ZIP only** (no MSI/NSIS, no UAC) via GitHub Actions.
- File paths route to `%LOCALAPPDATA%/PortableWhisper/{models,temp,logs}` via
  `backend/runtime_hooks/path_redirect.py` (imported first in `main.py`).
- Sidecar lifecycle: global `RunEvent::Exit` handler kills the backend on any
  exit path (`lib.rs`); sidecar permission via structured entries in
  `capabilities/default.json` (`shell:allow-spawn`/`allow-execute`). The
  `plugins.shell.scope` field in `tauri.conf.json` is INVALID in Tauri 2.
- Smart Pre-Load: Model loads async at app startup (`/load_model_async` endpoint).
- Default hotkey: **F9**; default language: **de** (German).

Phase 2 (Parakeet TDT v3 + onnx-asr) is **planned** — full migration plan in
`PHASE2_PLAN.md`. Key decisions:
- Engine: `onnx-asr` (pure Python, MIT license) replaces `faster-whisper`.
- Model: `nvidia/parakeet-tdt-0.6b-v3` (ONNX, 25 EU languages, 600M params).
- Expected speedup: **36× RTFx** on CPU vs ~0,3-1× with Whisper.
- DirectML/NPU (Phase 2B) is optional — CPU already fast enough.
- XDNA 2 NPU support via DirectML is experimental (AMD drivers still maturing).
- Streaming (Phase 2C) via sherpa-onnx only if live text is needed.

## Key docs

- `KONZEPT.md` — master 3-phase prompt (read first)
- `PHASE1_PLAN.md` — Phase 1 detailed steps
- `PHASE2_PLAN.md` — Phase 2 migration plan (faster-whisper → Parakeet TDT v3)
- `WINDOWS-ABHAENGIGKEITEN.md` — Windows build toolchain
- `INSTALLATION.md`, `BUILD.md`, `TECHNICAL.md`, `DEVELOPMENT_MODE.md`