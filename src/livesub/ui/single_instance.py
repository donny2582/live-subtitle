"""單一執行個體：同一個使用者同時只能開一個字幕程式。

做法：
* QLockFile（鎖定檔）：第一個啟動的程式拿到鎖；之後啟動的程式拿不到，就知道已經有人在執行。
  程式當掉時鎖會被自動判定為失效，不會永遠鎖住。
* QLocalServer／QLocalSocket（本機通訊管道）：第二個程式拿不到鎖時，
  透過管道通知第一個程式「使用者又點了一次」，第一個程式就把視窗叫到最前面，
  第二個程式隨即結束。
"""

from __future__ import annotations

import getpass
import hashlib
import logging
import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QLockFile, QObject, pyqtSignal
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

log = logging.getLogger(__name__)

_MESSAGE = b"activate"


def _instance_key() -> str:
    """每個 Windows 使用者一個名稱（通訊管道是全系統共用的，要避免不同使用者互相干擾）。"""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - 取不到使用者名稱時仍可運作
        user = "default"
    return "livesub-" + hashlib.sha1(user.encode("utf-8")).hexdigest()[:10]  # noqa: S324 - 只是取名稱


class SingleInstance(QObject):
    """拿到鎖的是「主程式」；activated 訊號代表另一個程式要求把視窗叫出來。"""

    activated = pyqtSignal()

    def __init__(self, key: str | None = None) -> None:
        super().__init__()
        self._key = key or _instance_key()
        self._lock = QLockFile(str(Path(tempfile.gettempdir()) / f"{self._key}.lock"))
        self._lock.setStaleLockTime(0)  # 只用「建立鎖的程式是否還活著」判斷失效，不看時間
        self._server: QLocalServer | None = None

    def acquire(self, wait_ms: int = 0) -> bool:
        """嘗試成為主程式。wait_ms：最多等幾毫秒（重新啟動時要等舊程式結束）。"""
        if not self._lock.tryLock(wait_ms):
            return False
        # 拿到鎖之後才開通訊管道；先清掉上次當掉時可能殘留的同名管道
        QLocalServer.removeServer(self._key)
        self._server = QLocalServer(self)
        if not self._server.listen(self._key):
            log.warning("無法建立單一執行個體的通訊管道：%s", self._server.errorString())
        self._server.newConnection.connect(self._on_connection)
        return True

    def notify_running(self, timeout_ms: int = 1500) -> bool:
        """通知已在執行的主程式把視窗叫出來。成功送出回傳 True。"""
        _allow_foreground()
        sock = QLocalSocket()
        sock.connectToServer(self._key)
        if not sock.waitForConnected(timeout_ms):
            return False
        sock.write(_MESSAGE)
        sock.flush()
        ok = sock.waitForBytesWritten(timeout_ms)
        sock.disconnectFromServer()
        return ok

    def release(self) -> None:
        """放開鎖與通訊管道（程式結束或重新啟動前呼叫）。可重複呼叫。"""
        if self._server is not None:
            self._server.close()
            self._server = None
        if self._lock.isLocked():
            self._lock.unlock()

    # ------------------------------------------------------------------ 內部
    def _on_connection(self) -> None:
        assert self._server is not None
        sock = self._server.nextPendingConnection()
        if sock is None:
            return

        def read() -> None:
            if bytes(sock.readAll()).startswith(_MESSAGE):
                self.activated.emit()
            sock.disconnectFromServer()
            sock.deleteLater()

        if sock.bytesAvailable():
            read()
        else:
            sock.readyRead.connect(read)


def _allow_foreground() -> None:
    """Windows 預設不讓背景程式搶到最前面；由使用者剛點開的這個程式授權給主程式。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.AllowSetForegroundWindow(-1)  # type: ignore[attr-defined]  # ASFW_ANY
    except Exception:  # noqa: BLE001, S110 - 授權失敗只是視窗不會跳到最前面
        pass


def bring_to_front(widget: object) -> None:
    """把視窗叫到最前面（最小化的話先還原）。"""
    from PyQt6.QtWidgets import QWidget

    if not isinstance(widget, QWidget):
        return
    widget.showNormal()
    widget.raise_()
    widget.activateWindow()
