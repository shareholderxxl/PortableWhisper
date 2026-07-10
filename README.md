# 🎙️ PortableWhisper

**Your voice, transcribed instantly — 100% private, no internet required.**

PortableWhisper is a local, offline speech-to-text tool for Windows. Press a hotkey,
speak, and your words appear wherever your cursor is — emails, notes, chats, code.
Everything runs on your machine; nothing is ever sent to the cloud.

> 🇩🇪 **Optimized for German.** The default recognition language is German (auto-detect
> available for 25 European languages).

---

## ✨ Features

- 🔒 **Fully private** — no cloud, no API keys, no subscriptions. Runs 100% offline.
- ⚡ **Fast** — Parakeet TDT 0.6B v3 on CPU delivers real-time transcription (~36× real-time).
- 🎯 **Global hotkey** — press **F9** from any app (Word, Chrome, VS Code, Slack…) and speak.
- 🌍 **Multilingual** — 25 European languages via NVIDIA Parakeet TDT v3, auto-detect or fixed.
- 📦 **Portable** — single ZIP, no installer, no UAC. Unpack and run.
- 🎨 **Clean UI** — light / dark / system theme, customizable shortcuts, sound effects.

---

## 🚀 Download & Run

1. Download the latest **PortableWhisper** ZIP from the [Releases](../../releases) page.
2. Unpack it to any folder (e.g. `C:\PortableWhisper`).
3. Place the model files (see below) into the `model/` folder.
4. Double-click **`PortableWhisper.exe`**.

That's it — no installation, no setup wizard.

---

## 🧠 Model Setup

PortableWhisper ships **without a bundled model** to keep the download small (~200 MB).
You place the ONNX model files yourself, once.

### Required files in `model/`

```
PortableWhisper/
└── model/
    ├── encoder-model.int4.onnx      (~373 MB, int4 encoder)
    ├── decoder_joint-model.int8.onnx (~18 MB, int8 decoder+joint)
    └── vocab.txt                     (~92 KB, SentencePiece vocabulary)
```

### Recommended model: Parakeet TDT 0.6B v3 (ONNX int4)

**Option A — HuggingFace CLI (easiest)**

```bash
pip install huggingface-hub

hf download efederici/parakeet-tdt-0.6b-v3-onnx-int4 --local-dir model/
```

**Option B — Manual download**

1. Open: https://huggingface.co/efederici/parakeet-tdt-0.6b-v3-onnx-int4
2. Download `encoder-model.int4.onnx`, `decoder_joint-model.int8.onnx`, and `vocab.txt`.
3. Copy them into the `model/` folder next to `PortableWhisper.exe`.

**Alternative (int8):** `istupakov/parakeet-tdt-0.6b-v3-onnx` — same procedure,
different quantization. Both work; int4 is smaller and faster on CPU.

> ⚠️ Only **one model** is supported (Parakeet TDT 0.6B v3). There are no size tiers
> like tiny/base/large — Parakeet is a single fixed architecture.

---

## ⌨️ Usage

1. **Press F9** — a minimal recording window appears at the top of your screen.
2. **Speak naturally** — watch the live audio visualizer respond.
3. **Press F9 again** — the transcribed text is inserted at your cursor.

Settings (F9 default, language, microphone, theme, GPU) are available from the
**Settings** page inside the app.

---

## 🖥️ System Requirements

- **OS:** Windows 10 / 11 (64-bit)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Microphone:** built-in or external
- **Disk:** ~600 MB (app + model)

### GPU acceleration (optional)

NVIDIA GPU users can enable CUDA acceleration. The CUDA libraries are **not bundled**
— install them once from **Settings → Install GPU Libraries** (~600 MB download, on demand).
The app automatically falls back to fast CPU mode if no GPU is present.

---

## ❓ FAQ

**Q: Is this really free?**
A: Yes. MIT-licensed, open source, no subscriptions or API keys.

**Q: Do I need internet?**
A: Only to download the model and (optionally) GPU libraries — once. Transcription itself
is fully offline.

**Q: Which languages are supported?**
A: 25 European languages via Parakeet TDT v3, with automatic detection or a fixed language.
German is the default.

**Q: Why is there no MSI installer?**
A: PortableWhisper is distributed as a portable ZIP — unpack and run, no admin rights needed.

**Q: How accurate is it?**
A: Parakeet TDT 0.6B v3 is a state-of-the-art streaming ASR model with excellent accuracy
for European languages, especially German.

---

## 🛠️ Troubleshooting

**"Model not found" dialog at startup?**
- Make sure `model/` contains all three files (`encoder-model.int4.onnx`,
  `decoder_joint-model.int8.onnx`, `vocab.txt`).
- Check `logs/whisper-backend.log` for details.

**Backend won't start?**
- Ensure no other app is using port `8765`.
- Run `CHECK_GPU_STATUS.bat` from the app folder to diagnose the environment.

**Hotkey not working?**
- The app must be running (visible in the system tray).
- Some applications block global hotkeys — try restarting PortableWhisper.

---

## 🧰 For Developers

PortableWhisper is built with:

- **Backend:** Python 3.11 + FastAPI, speech recognition via [`onnx-asr`](https://github.com/istupakov/onnx-asr)
  (pure Python, MIT) on ONNX Runtime.
- **Frontend:** [Tauri 2](https://tauri.app) (Rust) with static HTML — no Node build step.
- **Build:** portable ZIP via GitHub Actions (`windows-latest`), no local toolchain required.

See `KONZEPT.md`, `PHASE1_PLAN.md`, and `PHASE2_PLAN.md` for the architecture and roadmap.

---

## 📝 License

MIT License — Copyright (c) 2026 ShareholderXXL.
Based on the original project [Whisper4Windows](https://github.com/BaderJabri/Whisper4Windows) by Bader Aljabri.

### Third-party components

| Component | License |
|---|---|
| NVIDIA Parakeet TDT 0.6B v3 (ONNX) | CC-BY-4.0 |
| onnx-asr | MIT |
| ONNX Runtime | MIT |
| Tauri 2 | MIT / Apache-2.0 |

---

**Ready to dictate? Download PortableWhisper, drop in the model, and start speaking. 🎤**
