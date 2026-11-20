"""讀寫 Windows「使用者環境變數」（等同 setx），讓設定視窗能直接儲存 API 金鑰。

金鑰存在 HKEY_CURRENT_USER\\Environment，不會寫進 config.toml。
"""

from __future__ import annotations

import os
import sys

_ENV_KEY = "Environment"


def read_user_env(name: str) -> str:
    """先看目前程式的環境變數，再看登錄檔（剛用 setx 設定、尚未重開終端機的情況）。"""
    value = os.environ.get(name, "").strip()
    if value or sys.platform != "win32":
        return value
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _ENV_KEY) as key:
            stored, _ = winreg.QueryValueEx(key, name)
            return str(stored).strip()
    except OSError:
        return ""


def set_user_env(name: str, value: str) -> None:
    """寫入（value 為空字串時刪除）使用者環境變數，並同步到目前程式。

    Raises:
        OSError: 寫入登錄檔失敗，或不是 Windows。
    """
    value = value.strip()
    if sys.platform != "win32":
        raise OSError("只有 Windows 支援從設定視窗儲存金鑰，請改用環境變數")
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _ENV_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if value:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
        else:
            try:
                winreg.DeleteValue(key, name)
            except FileNotFoundError:
                pass
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)
    _broadcast_change()


def _broadcast_change() -> None:
    """通知其他程式（檔案總管等）環境變數已變更，新開的程式才讀得到。"""
    try:
        import ctypes

        hwnd_broadcast, wm_settingchange, smto_abortifhung = 0xFFFF, 0x001A, 0x0002
        result = ctypes.c_ulong()
        ctypes.windll.user32.SendMessageTimeoutW(  # type: ignore[attr-defined]
            hwnd_broadcast, wm_settingchange, 0, "Environment", smto_abortifhung, 1000,
            ctypes.byref(result),
        )
    except Exception:  # noqa: BLE001, S110 - 通知失敗不影響已寫入的值
        pass


def mask_secret(value: str) -> str:
    """sk-abcdef...wxyz → sk-a…wxyz，只顯示頭尾方便確認是哪一把。"""
    if not value:
        return ""
    if len(value) <= 10:
        return "•" * len(value)
    return f"{value[:4]}…{value[-4:]}"
