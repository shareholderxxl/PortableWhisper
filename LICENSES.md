# Whisper4Windows - Lizenzinformationen

## Whisper4Windows

**Lizenz:** MIT License

Copyright (c) 2024 ShareholderXXL

Basiert auf dem Originalprojekt von Bader Aljabri.

Vollständiger Lizenztext siehe `LICENSE` Datei im App-Ordner.

## Whisper 3 Large Turbo Modell

**Source:** Systran/faster-whisper-large-v3-turbo  
**Lizenz:** MIT License  
**URL:** https://huggingface.co/Systran/faster-whisper-large-v3-turbo

Das Modell steht unter der MIT License zur Verfügung.

## Abhängigkeiten

### Backend (Python)

- **faster-whisper:** MIT License - https://github.com/SYSTRAN/faster-whisper
- **CTranslate2:** BSD-3-Clause - https://github.com/OpenNMT/CTranslate2
- **fastapi:** MIT License - https://github.com/tiangolo/fastapi
- **uvicorn:** BSD-3-Clause - https://github.com/encode/uvicorn
- **numpy:** BSD-3-Clause - https://numpy.org
- **soundfile:** BSD-3-Clause - https://github.com/bastibe/python-soundfile
- **scipy:** BSD-3-Clause - https://scipy.org
- **huggingface-hub:** Apache-2.0 - https://github.com/huggingface/huggingface_hub

### Frontend (Tauri)

- **Tauri 2:** MIT/Apache-2.0 - https://tauri.app
- **tokio:** MIT - https://github.com/tokio-rs/tokio
- **reqwest:** MIT/Apache-2.0 - https://github.com/seanmonstar/reqwest

### Build-Tools (nur im Build-Prozess)

- **PyTorch** (nur Build-Prozess): BSD-3-Clause - https://pytorch.org
- **Rust/Cargo:** Apache-2.0/MIT - https://www.rust-lang.org

## Danksagung

Danke an:
- OpenAI für das Whisper-Modell und die ursprüngliche Idee
- Systran für faster-whisper und CTranslate2
- Bader Aljabri für das ursprüngliche Projekt (whisper4windows)
- Der Whisper-Community für Beiträge und Verbesserungen
- Allen Mitwirkenden an den verwendeten Open-Source-Projekten

## Hinweis zur Modell-Nutzung

Die verwendeten Whisper-Modelle stehen unter der MIT License und können frei verwendet werden. Bitte beachten Sie jedoch:
- Die Modelle wurden von OpenAI trainiert
- Verwenden Sie die Modelle ethisch und verantwortungsbewusst
