"""Windows 上讓 CTranslate2 找到 pip 安裝的 CUDA DLL（cuBLAS、cuDNN）。

`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` 會把 DLL 放在
site-packages/nvidia/*/bin，但 Windows 預設不會搜尋該路徑。

DLL 是 Windows 的共用程式庫。CTranslate2（faster-whisper 底層）需要 NVIDIA 的 DLL 才能用 GPU，
找不到時會出現 `cublas64_12.dll` 之類的錯誤。這裡把那些資料夾加進搜尋路徑，
使用者就不需要另外安裝整套 CUDA Toolkit。
"""

from __future__ import annotations

import logging
import os
import site
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_dll_handles: list[object] = []  # 必須保留參考，否則目錄會被移除


def _candidate_roots() -> list[Path]:
    """可能放 nvidia 套件的位置：打包後的 exe 資料夾，或 Python 的 site-packages。"""
    roots: list[Path] = []
    if getattr(sys, "frozen", False):  # PyInstaller
        roots.append(Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)))
    for p in [*site.getsitepackages(), site.getusersitepackages()]:
        roots.append(Path(p))
    return roots


def register_cuda_dlls() -> list[Path]:
    """將 nvidia/*/bin 加入 DLL 搜尋路徑。非 Windows 時不做任何事。

    Returns:
        已加入的資料夾清單。
    """
    if os.name != "nt":
        return []
    added: list[Path] = []
    for root in _candidate_roots():
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in sorted(nvidia.glob("*/bin")):
            try:
                # 兩種方式都加：新版 Python 用 add_dll_directory，部分套件仍只看 PATH
                _dll_handles.append(os.add_dll_directory(str(bin_dir)))  # type: ignore[attr-defined]
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                added.append(bin_dir)
            except OSError as exc:  # pragma: no cover - 僅 Windows
                log.debug("無法加入 DLL 路徑 %s：%s", bin_dir, exc)
    if added:
        log.info("已加入 CUDA DLL 路徑：%s", ", ".join(str(p) for p in added))
    return added
