"""程式圖示（藍底白色「字」）：系統匣與桌面捷徑共用。

圖示用程式畫出來，不需要另外附圖檔。
桌面捷徑需要 .ico 檔，由 write_ico() 產生（scripts\\create_shortcut.ps1 會呼叫）。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPainter, QPixmap


def make_pixmap(size: int = 64) -> QPixmap:
    """畫出 size × size 的圖示（以 64 × 64 的座標設計，再依大小縮放）。"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.scale(size / 64, size / 64)
    p.setBrush(QColor("#1f6feb"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 12, 12)  # 藍色圓角方塊
    p.setPen(QColor("white"))
    font = QFont("Microsoft JhengHei UI")
    font.setPixelSize(38)
    font.setWeight(QFont.Weight.Bold)
    p.setFont(font)
    p.drawText(0, 0, 64, 64, Qt.AlignmentFlag.AlignCenter, "字")
    p.end()
    return pix


def make_icon() -> QIcon:
    """系統匣用的圖示。"""
    return QIcon(make_pixmap(64))


def write_ico(path: str | Path, size: int = 256) -> Path:
    """把圖示存成 Windows .ico 檔（給桌面捷徑使用）。

    .ico 格式可以直接內嵌一張 PNG（Windows Vista 之後支援），
    所以只要在 PNG 前面加上 22 位元組的檔頭即可，不需要額外套件。
    """
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # 畫文字需要 Qt 應用程式物件
    # data 要用變數留住：QBuffer 不會保留它，若是臨時物件會被 Python 回收，Qt 就會寫到無效的記憶體
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    make_pixmap(size).save(buf, "PNG")
    buf.close()
    png = bytes(data)
    dim = 0 if size >= 256 else size  # .ico 裡 0 代表 256
    header = struct.pack("<HHH", 0, 1, 1)  # 保留欄位、類型 1 = 圖示、共 1 張圖
    # 圖片目錄：寬、高、色盤數、保留、色彩平面、位元深度、圖片大小、圖片從第幾個位元組開始
    entry = struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), 6 + 16)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(header + entry + png)
    del app
    return out
