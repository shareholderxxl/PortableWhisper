# AGENTS.md

Compact guidance for OpenCode sessions working in this repo. Read `KONZEPT.md`
first for the 3-phase roadmap before any non-trivial change.

## What this is

Whisper4Windows — local, offline speech-to-text for Windows. Stack:
- **Backend**: Python 3.11 + FastAPI + `faster-whisper`/`ctranslate2` (no torch).
  Flat module layout in `backend/` (`main.py`, `whisper_engine.py`,
  `audio_capture.py`, `gpu_manager.py`).
- **Frontend**: Tauri 2 (Rust in `frontend/src-tauri/`) + **static HTML**
  in `frontend/dist/` (`index.html`, `recording.html`). No JS framework,
  no bundler, no Node build step.

## Build platform — critical

Builds run **only on Windows** (`WINDOWS-ABHAENGIGKEITEN.md`). This Linux
server is for code preparation and analysis only. Do not attempt to compile
Tauri/PyInstaller artifacts here. Final builds go through GitHub Actions on
`windows-latest` (workflow not yet present — see `github-actions-build` skill).

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
Don't reinvent — extend in place. Open Phase-1 gaps: clean `child.kill()` on
app close and the `plugins.shell.scope` sidecar entry in
`capabilities/default.json`.

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

Phase 1 (Zero-Dependency & Portability) is **in planning**, not started.
Confirmed decisions an agent must respect:
- Default model stays `small`; `primeline/whisper-large-v3-german` is an option.
- Backend port → **8765** (currently 8000).
- PyInstaller **without torch** (ctranslate2 only).
- Build artifact: **portables ZIP only** (no MSI/NSIS, no UAC).
- Work happens on branch **`phase-1`**.
- File paths route to `%LOCALAPPDATA%/Whisper4Windows/{models,temp,logs}`.

See `PHASE1_PLAN.md` (to be created) for the full step list.

## Key docs

- `KONZEPT.md` — master 3-phase prompt (read first)
- `WINDOWS-ABHAENGIGKEITEN.md` — Windows build toolchain
- `INSTALLATION.md`, `BUILD.md`, `TECHNICAL.md`, `DEVELOPMENT_MODE.md`