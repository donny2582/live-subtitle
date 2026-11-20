# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 設定檔：由 scripts\build_exe.ps1 呼叫
import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 - SPECPATH 由 PyInstaller 注入

datas = []
binaries = []
hiddenimports = ["webrtcvad", "soxr", "pyaudiowpatch"]

# 雲端套件：anthropic SDK 內部有延遲匯入的模組
hiddenimports += collect_submodules("anthropic")
datas += collect_data_files("opencc")

# 本機語音辨識（選用）：faster-whisper 內含 Silero VAD 的 ONNX 檔
for pkg in ("faster_whisper",):
    try:
        datas += collect_data_files(pkg)
    except Exception:  # 未安裝 requirements-local-asr.txt
        pass
for pkg in ("ctranslate2", "onnxruntime"):
    try:
        binaries += collect_dynamic_libs(pkg)
        hiddenimports += collect_submodules(pkg) if pkg == "ctranslate2" else []
    except Exception:
        pass

# 若有安裝 requirements-gpu.txt，一併打包 CUDA DLL（放在 _internal/nvidia/*/bin）
for sub in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
    try:
        binaries += collect_dynamic_libs(sub)
    except Exception:  # 未安裝 GPU 套件
        pass

a = Analysis(  # noqa: F821
    [os.path.join(ROOT, "src", "livesub", "__main__.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LiveSub",
    console=os.environ.get("LIVESUB_CONSOLE") == "1",
    upx=False,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="LiveSub",
)
