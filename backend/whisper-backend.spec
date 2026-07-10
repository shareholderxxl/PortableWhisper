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
from PyInstaller.utils.hooks import collect_all

backend_dir = Path(SPECPATH).resolve()

# Phase 2: onnx-asr + onnxruntime korrekt bundeln. collect_all liefert
# (datas, binaries, hiddenimports[, ...]); nur die ersten 3 nutzen, damit es
# egal ist, ob PyInstaller 6.x (3 Werte) oder 7.x (4 Werte) verwendet wird.
def _gather(pkg):
    ret = collect_all(pkg)
    return ret[0], ret[1], ret[2]   # datas, binaries, hiddenimports

asr_datas, asr_bins, asr_hidden = _gather('onnx_asr')
ort_datas, ort_bins, ort_hidden = _gather('onnxruntime')

a = Analysis(
    ['main.py'],
    pathex=[str(backend_dir)],
    binaries=asr_bins + ort_bins,
    datas=asr_datas + ort_datas,  # ONNX-Modell selbst nicht einbetten — liegt in data/models/default/
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
        'onnx_asr',
        'onnxruntime',
    ] + asr_hidden + ort_hidden,
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
    ],
    noarchive=False,
    optimize=2,
)

# (collect_all-Ergebnisse werden oben direkt in die Analysis-Argumente
#  binaries/datas/hiddenimports eingespeist — kein post-Analysis a.pure += nötig)

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
