# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path.cwd()

datas = [
    (str(project_root / "config"), "config"),
    (str(project_root / "ml" / "models"), "ml/models"),
]

hiddenimports = [
    "customtkinter",
    "desktop_app.app",
    "desktop_app.dashboard_home",
    "desktop_app.log_formatter",
    "desktop_app.paths",
    "desktop_app.repo_status",
    "desktop_app.runner",
    "tkinter",
    "pandas",
]


a = Analysis(
    ["packaged_app_entry.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    name="AlpacaTradingBot",
    icon=str(project_root / "assets" / "alpaca_trading_bot.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AlpacaTradingBot",
)
