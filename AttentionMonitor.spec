# -*- mode: python ; coding: utf-8 -*-
import os

# MediaPipe 运行时需要 mediapipe 目录中的资源文件，直接把整个包目录打进去
_mp_path = os.path.join(os.getcwd(), 'venv', 'lib', 'python3.11', 'site-packages', 'mediapipe')
_mp_datas = [(_mp_path, 'mediapipe')] if os.path.isdir(_mp_path) else []

a = Analysis(
    ['attention_monitor.py'],
    pathex=[],
    binaries=[],
    datas=[('efficientdet_lite0.tflite', '.')] + _mp_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AttentionMonitor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AttentionMonitor',
)
app = BUNDLE(
    coll,
    name='AttentionMonitor.app',
    icon=None,
    bundle_identifier=None,
)
