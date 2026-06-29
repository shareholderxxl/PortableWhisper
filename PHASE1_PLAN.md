# Phase 1 — Implementierungsplan: Zero-Dependency & Portabilität

> Status: **geplant** (noch nicht gestartet)
> Stand: 29.06.2026
> Quelle: `KONZEPT.md` (Phase 1), Skills `tauri-portable-sidecar`, `github-actions-build`

---

## Rahmenbedingungen (vom Nutzer bestätigt)

- **Arbeitsort:** Nur Linux-Server (Vorbereitung). Build später via GitHub Actions auf `windows-latest`.
- **Default-Modell:** `small` bleibt Standard (schneller Erststart). `primeline/whisper-large-v3-german` wird als zusätzliche Option hinterlegt.
- **Build-Ziel:** Nur **portables ZIP** — kein MSI/NSIS, kein UAC.
- **PyInstaller:** **ohne torch**, nur `ctranslate2` via `faster-whisper` (deutlich kleineres Binary).
- **Backend-Port:** **8765** (vorher 8000).
- **Git-Branch:** **`phase-1`** (von `main`).

---

## Status Quo (vor Beginn)

| Bereich | Vorhanden | Lücke für Phase 1 |
|---|---|---|
| Backend (FastAPI) | `main.py`, `whisper_engine.py`, `gpu_manager.py`, `audio_capture.py` lauffähig | Default-Modell `small`/Systran → `primeline/whisper-large-v3-german` als Option |
| Pfad-Logik | drei Stellen nutzen `%APPDATA%` (Roaming) | Konzept will `%LOCALAPPDATA%/Whisper4Windows/{models,temp,logs}` + HF-Cache-Umleitung |
| `build_backend.py` | PyInstaller one-file, ohne Modell | `.spec` mit UPX, excludes, vollständigen hiddenimports ohne torch |
| Sidecar-Wiring (Rust) | `lib.rs:975` und `lib.rs:1123` spawnen `whisper-backend` | Shutdown-Handling + Restart-Logik härten |
| `tauri.conf.json` | `externalBin: ["binaries/whisper-backend"]` gesetzt | `targets: "all"` → portables ZIP; `plugins.shell.scope` für Sidecar fehlt teils |
| `src-tauri/binaries/` | **existiert nicht** | Muss erstellt und mit `.gitkeep` befüllt werden |
| `.github/workflows/` | **existiert nicht** | Workflow aus Skill `github-actions-build` anlegen (Phase-1-Variante) |
| Build-Doku | `WINDOWS-ABHAENGIGKEITEN.md` vorhanden | referenziert `requirements-portable.txt` (nicht vorhanden) |
| Git | sauber, letzter Commit „Initialer Fork: Whisper4Windows + OpenCode-Skills" | Phase-1-Arbeit beginnt bei 0 |

---

## Auszuführende Schritte

| # | Aktion | Dateien |
|---|---|---|
| 0 | Branch `phase-1` von `main` anlegen | git |
| 1 | Runtime-Hook für Pfad-Redirect anlegen | `backend/runtime_hooks/path_redirect.py` (neu) |
| 2 | `APPDATA`→`%LOCALAPPDATA%/Whisper4Windows` umstellen + HF-Cache setzen | `gpu_manager.py:34`, `whisper_engine.py:47,148`, `main.py` (früh im Import) |
| 3 | `primeline/whisper-large-v3-german` als wählbare Option (Default bleibt `small`) | `whisper_engine.py`, `main.py:41`, Frontend-Settings-UI (`frontend/dist/`) |
| 4 | Port 8000 → 8765 durchgängig | `main.py` (Log + uvicorn.run), Frontend-Konstante, ggf. `.env` |
| 5 | `whisper-backend.spec` anlegen (UPX, excludes, hiddenimports ohne torch) | `backend/whisper-backend.spec` (neu) |
| 6 | `requirements-portable.txt` anlegen (ohne torch, mit `huggingface_hub`, `pyinstaller`) | `backend/requirements-portable.txt` (neu) |
| 7 | Sidecar-Lifecycle härten (sauberes `child.kill()` beim App-Close) | `frontend/src-tauri/src/lib.rs:975,1123` |
| 8 | `tauri.conf.json` auf ZIP umstellen + `plugins.shell.scope` für Sidecar ergänzen | `frontend/src-tauri/tauri.conf.json`, `capabilities/default.json` |
| 9 | `frontend/src-tauri/binaries/.gitkeep` anlegen (Verz. existiert nicht) | neu |
| 10 | GitHub-Actions-Workflow anlegen (Phase-1-Variante: faster-whisper statt onnxruntime) | `.github/workflows/build-windows.yml` (neu) |
| 11 | Doku-Updates: Phase-1-Fortschritt, Modell-Optionen, Port-Wechsel | `KONZEPT.md`, `WINDOWS-ABHAENGIGKEITEN.md`, `INSTALLATION.md` |
| 12 | Commit pro logischem Schritt auf `phase-1` | git |

---

## Verifikation (unter Windows, nach GitHub-Actions-Build)

1. ZIP entpacken → `Whisper4Windows.exe` ohne Admin starten → **kein UAC-Prompt**.
2. Sidecar wird unsichtbar mitgestartet, API unter `http://127.0.0.1:8765` erreichbar.
3. Beim ersten Start (oder Modellwechsel auf „large-v3-german") lädt das Modell nach `%LOCALAPPDATA%/Whisper4Windows/models/hub/`.
4. App schließen → Sidecar-Prozess sauber beendet (Task-Manager prüfen, kein Zombie).
5. Kein installiertes Python/Node.js nötig — alle Dependencies im ZIP gebündelt.

---

## Offene Punkte (bei der Umsetzung selbst zu lösen)

- Wie der Frontend-Modell-Dialog genau erweitert wird (statische HTML-Dateien in `frontend/dist/`: `index.html`, `recording.html`) — Detail bei Schritt 3.
- Ob Tauri-Updater-Signing für das ZIP aktiviert wird (Default: nein, kein Secret voraussetzen).
- Anpassung des GitHub-Actions-Workflows aus Skill `github-actions-build`: dort ist `--collect-all onnxruntime` vorgesehen, das ist **Phase 2** — für Phase 1 weglassen und stattdessen `--collect-all ctranslate2` bzw. `faster_whisper` verwenden.

---

## Verweise

- `KONZEPT.md` — Master-Prompt mit 3-Phasen-Plan (Phase 1 ist aktuell)
- `.opencode/skills/tauri-portable-sidecar/SKILL.md` — Detail-Vorgaben für Sidecar + ZIP
- `.opencode/skills/github-actions-build/SKILL.md` — Workflow-Template (an Phase 1 anpassen)
- `WINDOWS-ABHAENGIGKEITEN.md` — Build-Toolchain-Dokumentation