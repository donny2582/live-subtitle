# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 設定檔：由 tests\build_exe\build_exe.ps1 呼叫
#
# PyInstaller 會分析程式用到哪些模組，把 Python 本身與所有套件打包成 dist\LiveSub\LiveSub.exe，
# 讓沒有安裝 Python 的電腦也能執行。有些套件是「執行時才載入」或附帶資料檔／DLL，
# PyInstaller 自動分析找不到，所以要在下面手動列出（hiddenimports、datas、binaries）。
import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# 這個檔案在 <專案>\tests\build_exe\，往上兩層是專案根目錄
ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))  # noqa: F821 - SPECPATH 由 PyInstaller 注入
# exe 圖示：build_exe.ps1 會先產生這個檔案（藍底「字」，和系統匣圖示相同）
ICON = os.path.join(ROOT, "build", "livesub.ico")

datas = []  # 資料檔（例如 OpenCC 的轉換字典）
binaries = []  # DLL 等二進位檔
hiddenimports = ["webrtcvad", "soxr", "pyaudiowpatch"]  # 程式裡延遲 import 的套件

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

# 以下是 PyInstaller 的固定流程：Analysis（分析相依）→ PYZ（壓縮 Python 程式碼）
# → EXE（產生執行檔）→ COLLECT（把 exe 與所有檔案收集到 dist\LiveSub 資料夾）
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
    icon=ICON if os.path.exists(ICON) else None,
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
