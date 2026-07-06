# -*- mode: python ; coding: utf-8 -*-
# whisper-backend.spec
# PyInstaller-Spezifikation für das PortableWhisper-Backend (Sidecar).
#
# Phase-1-Vorgaben (siehe PHASE1_PLAN.md):
#   - Modell wird NICHT eingebettet (datas=[]); Download beim ersten Start
#     via huggingface_hub nach %LOCALAPPDATA%/PortableWhisper/models/.
#   - OHNE torch — faster-whisper läuft nativ über ctranslate2.
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
    datas=[],  # Modell-Daten NICHT einbetten!
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
        'faster_whisper',
        'ctranslate2',
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
        'torch',          # Phase 1: bewusst OHNE torch
        'torchvision',
        'torchaudio',
    ],
    noarchive=False,
    optimize=2,
)

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
