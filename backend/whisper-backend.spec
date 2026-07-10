# -*- mode: python ; coding: utf-8 -*-
# whisper-backend.spec
# PyInstaller-Spezifikation für das PortableWhisper-Backend (Sidecar).
#
# Phase-2-Vorgaben (siehe PHASE2_PLAN.md):
#   - Modell wird NICHT eingebettet (datas=[]); User kopiert ONNX-Dateien
#     nach data/models/default/.
#   - onnx-asr + onnxruntime statt faster-whisper + ctranslate2.
#   - console=False: kein Konsolenfenster beim Sidecar-Start (unsichtbar).
#     stdout/stderr-None-Guard liegt in runtime_hooks/path_redirect.py.
#   - UPX-Kompression an (kleineres Binary); ggf. bei AV-Fehlalarmen abschalten.

import os
from pathlib import Path

backend_dir = Path(SPECPATH).resolve()

a = Analysis(
    ['main.py'],
    pathex=[str(backend_dir)],
    binaries=[],
    datas=[],  # ONNX-Modell nicht einbetten — liegt in data/models/default/
    hiddenimports=[
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'pydantic',
        'pydantic_settings',
        'huggingface_hub',
        'sounddevice',
        'soundfile',
        'numpy',
        'scipy',
        'runtime_hooks',
        'runtime_hooks.path_redirect',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(backend_dir / 'runtime_hooks' / 'path_redirect.py')],
    excludes=[
        'tkinter',
        'matplotlib',
        'notebook',
        'jupyter',
        'jupyterlab',
        'ipython',
        'test',
        'tests',
        'pytest',
        'torch',          # Phase 2: weiterhin OHNE torch
        'torchvision',
        'torchaudio',
        'faster_whisper', # Phase 1 Engine — nicht mehr benötigt
        'ctranslate2',    # Phase 1 Backend — nicht mehr benötigt
    ],
    noarchive=False,
    optimize=2,
)

# Phase 2: onnx-asr + onnxruntime collect_all für korrektes Bundling
from PyInstaller.utils.hooks import collect_all
onnx_asr_binaries, onnx_asr_datas = collect_all('onnx_asr')
onnx_runtime_binaries, onnx_runtime_datas = collect_all('onnxruntime')
a.binaries += onnx_asr_binaries + onnx_runtime_binaries
a.datas += onnx_asr_datas + onnx_runtime_datas

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='whisper-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,            # unsichtbarer Sidecar (Guard via path_redirect)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
