---
name: pyinstaller-onnx-bundle
description: Dictates rules for bundling onnxruntime shared libraries (.dll) via PyInstaller without corrupting runtime bindings.
license: MIT
compatibility: opencode-cli-v2
---

# Instructions

1. When freezing the FastAPI app, append `--collect-binaries onnxruntime` to the PyInstaller CLI command.
2. Ensure the `onnxruntime_providers_shared.dll` and `DirectML.dll` are explicitly bundled using the `--add-binary` flag if they are not picked up automatically.
3. Validate that runtime model instantiation utilizes absolute strings derived from `os.environ.get("LOCALAPPDATA")` rather than relative script paths (`__file__`).