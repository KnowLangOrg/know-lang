# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

# Get the project root directory
project_root = Path.cwd()


def collect_knowlang_data():
    """Collect all data files from the knowlang package"""
    data_files = []
    
    # Add any configuration files or data files your package needs
    config_files = [
        ('settings', 'settings'),  # If you have a settings directory
    ]
    
    for src, dst in config_files:
        if (project_root / src).exists():
            data_files.append((str(project_root / src), dst))
    
    data_files += collect_data_files('pydantic_ai')
    data_files += copy_metadata('pydantic_ai_slim')
    
    return data_files

hidden_imports = [
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=collect_knowlang_data(),
    hiddenimports=hidden_imports,
    hookspath=['hooks'],
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
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='main',
)
