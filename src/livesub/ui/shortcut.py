"""在桌面建立「即時字幕」捷徑（設定視窗的「建立桌面捷徑」按鈕使用）。

開發版和打包的 exe 指向不同的程式：
* exe：直接指向 LiveSub.exe（圖示已內嵌在 exe 裡）。
* 開發版：指向 .venv 裡的 pythonw.exe -m livesub（pythonw 不會跳出黑色視窗）。

捷徑由 Windows 內建的 WScript.Shell 建立（透過 PowerShell 呼叫），不需要額外套件。
路徑等資料用環境變數傳給 PowerShell，避免路徑裡的空白或特殊字元造成問題。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..config import app_base_dir

SHORTCUT_NAME = "即時字幕"

_PS_SCRIPT = (
    "[Console]::OutputEncoding = [Text.Encoding]::UTF8;"
    "$desktop = [Environment]::GetFolderPath('Desktop');"  # 桌面被移到 OneDrive 也找得到
    "$path = Join-Path $desktop ($env:LS_NAME + '.lnk');"
    "$link = (New-Object -ComObject WScript.Shell).CreateShortcut($path);"
    "$link.TargetPath = $env:LS_TARGET;"
    "$link.Arguments = $env:LS_ARGS;"
    "$link.WorkingDirectory = $env:LS_DIR;"
    "$link.IconLocation = $env:LS_ICON;"
    "$link.Description = 'LiveSub live subtitles';"
    "$link.Save();"
    "Write-Output $path"
)


def shortcut_spec() -> dict[str, str]:
    """捷徑要指向的程式、參數、工作資料夾與圖示。"""
    if getattr(sys, "frozen", False):  # 打包的 exe
        exe = Path(sys.executable)
        return {"target": str(exe), "args": "", "dir": str(exe.parent), "icon": f"{exe},0"}
    from .icon import write_ico

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    icon = write_ico(Path(sys.prefix) / "livesub.ico")  # 開發版：圖示存在 .venv 裡
    return {"target": str(pythonw if pythonw.exists() else sys.executable), "args": "-m livesub",
            "dir": str(app_base_dir()), "icon": f"{icon},0"}


def create_desktop_shortcut(name: str = SHORTCUT_NAME) -> Path:
    """建立（或覆蓋）桌面捷徑，回傳捷徑檔的路徑。

    Raises:
        RuntimeError: 不是 Windows，或 PowerShell 建立失敗。
    """
    if sys.platform != "win32":
        raise RuntimeError("只有 Windows 支援建立桌面捷徑")
    spec = shortcut_spec()
    env = {**os.environ, "LS_NAME": name, "LS_TARGET": spec["target"], "LS_ARGS": spec["args"],
           "LS_DIR": spec["dir"], "LS_ICON": spec["icon"]}
    result = subprocess.run(  # noqa: S603 - 指令是固定的，資料只透過環境變數傳入
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",  # noqa: S607
         "-Command", _PS_SCRIPT],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # 不要閃出黑色視窗
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(result.stderr.strip() or "PowerShell 沒有回傳捷徑位置")
    return Path(result.stdout.strip().splitlines()[-1])
