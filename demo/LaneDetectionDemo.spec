# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Lane Detection demo.

Produces a single-file executable (LaneDetectionDemo.exe on Windows). The
project's src/ modules are bundled as data AND declared as hidden imports,
because they're imported by bare name at runtime via sys.path injection in
core._add_src_to_path().

Build with:  pyinstaller demo/LaneDetectionDemo.spec --noconfirm
"""

import os
from PyInstaller.utils.hooks import collect_submodules

# Spec files don't define __file__; SPECPATH is provided by PyInstaller.
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")

# Bundle every .py in src/ so the frozen app can import them by name.
src_datas = [(SRC, "src")]

hiddenimports = [
    "pipeline", "perspective", "thresholding",
    "sliding_window", "polyfit", "temporal",
]
hiddenimports += collect_submodules("matplotlib.backends")

a = Analysis(
    [os.path.join(ROOT, "demo", "app.py")],
    pathex=[SRC, os.path.join(ROOT, "demo")],
    binaries=[],
    datas=src_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.Qt3D*",
              "PySide6.QtMultimedia", "torch", "torchvision", "jupyter",
              "notebook", "IPython", "pandas"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="LaneDetectionDemo",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,         # GUI app — no console window on Windows
    disable_windowed_traceback=False,
    onefile=True,
)
