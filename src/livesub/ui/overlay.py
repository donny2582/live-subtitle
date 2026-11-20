"""PyQt6 字幕浮層與系統匣選單。

* 無邊框、永遠置頂、半透明背景，不會搶走會議軟體的焦點。
* 左鍵拖曳移動位置；Ctrl + 滾輪調整字級；右鍵開啟選單。
* 開啟「點擊穿透」後滑鼠會直接點到底下的視窗，要關閉請用系統匣選單。
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QObject, QPoint, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QContextMenuEvent,
    QDesktopServices,
    QFont,
    QGuiApplication,
    QIcon,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import QApplication, QLabel, QMenu, QSystemTrayIcon, QVBoxLayout, QWidget

from ..config import OverlayConfig
from ..core.models import Stage, SubtitleEvent

_STATUS_VISIBLE_S = 6.0


class UiBridge(QObject):
    """背景執行緒 → UI 執行緒的訊號橋接（Qt 會自動以 queued connection 傳遞）。"""

    event = pyqtSignal(object)
    status = pyqtSignal(str)


class _Row(QWidget):
    """一句字幕：上方原文（小字）、下方譯文（大字）。"""

    def __init__(self, seq: int, cfg: OverlayConfig, parent: QWidget) -> None:
        super().__init__(parent)
        self.seq = seq
        self.final = False
        self.language = ""
        self.updated_at = time.monotonic()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.source = QLabel(self)
        self.target = QLabel(self)
        for lbl in (self.source, self.target):
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            lbl.setTextFormat(Qt.TextFormat.PlainText)
            lay.addWidget(lbl)
        self.target.hide()
        self.apply_style(cfg)

    def apply_style(self, cfg: OverlayConfig) -> None:
        self.source.setFont(QFont(cfg.font_family, cfg.source_font_size))
        target_font = QFont(cfg.font_family, cfg.font_size)
        target_font.setWeight(QFont.Weight.DemiBold)
        self.target.setFont(target_font)
        self._update_source_visibility(cfg.show_source)
        self._paint_colors()

    def _paint_colors(self) -> None:
        src_color = "#c8d0d8" if self.final else "#9aa4ae"
        tgt_color = "#ffffff" if self.final else "#d7e3ea"
        self.source.setStyleSheet(f"color: {src_color};")
        self.target.setStyleSheet(f"color: {tgt_color};")

    def update_from(self, evt: SubtitleEvent, show_source: bool) -> None:
        self.updated_at = time.monotonic()
        self.final = evt.stage is Stage.FINAL
        if evt.source_text:
            self.source.setText(evt.source_text)
        target = evt.translated_text
        if self.final and evt.error:
            target = f"（{evt.error}）"
        if target:
            self.target.setText(target)
            self.target.show()
        self.language = evt.language
        self._update_source_visibility(show_source)
        self._paint_colors()

    def _update_source_visibility(self, show_source: bool) -> None:
        has_target = bool(self.target.text())
        if self.language == "zh" and has_target:
            self.source.hide()  # 中文語音：譯文就是原文（已轉台灣用語）
        else:
            self.source.setVisible(show_source or not has_target)  # 尚無譯文時一定顯示原文


class SubtitleOverlay(QWidget):
    """字幕視窗。"""

    def __init__(self, cfg: OverlayConfig) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        super().__init__(None, flags)
        self.cfg = cfg
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("即時字幕")
        self._rows: dict[int, _Row] = {}
        self._menu: QMenu | None = None
        self._drag_offset: QPoint | None = None
        self._status_at = 0.0

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 10, 18, 12)
        self._layout.setSpacing(8)
        self._status = QLabel(self)
        self._status.setFont(QFont(cfg.font_family, max(10, cfg.source_font_size - 2)))
        self._status.setStyleSheet("color: #ffd27f;")
        self._status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._status.setWordWrap(True)
        self._layout.addWidget(self._status)

        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.setFixedWidth(int(screen.width() * cfg.width_ratio))
        self._anchor_cx = screen.x() + screen.width() // 2
        self._anchor_bottom = screen.y() + screen.height() - cfg.bottom_margin

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._expire)
        self._timer.start(500)
        if cfg.click_through:
            self.set_click_through(True)
        self._relayout()

    # ------------------------------------------------------------------ 外部呼叫
    def set_menu(self, menu: QMenu) -> None:
        self._menu = menu

    def show_status(self, text: str) -> None:
        self._status.setText(text)
        self._status.show()
        self._status_at = time.monotonic()
        self._relayout()

    def on_event(self, evt: SubtitleEvent) -> None:
        row = self._rows.get(evt.seq)
        if evt.stage is Stage.FINAL and not evt.source_text:
            if row:
                self._remove(evt.seq)
            return
        if row is None:
            if evt.stage is not Stage.FINAL and self._rows and evt.seq < max(self._rows):
                return  # 過時的中間結果
            row = _Row(evt.seq, self.cfg, self)
            self._rows[evt.seq] = row
            index = sorted(self._rows).index(evt.seq) + 1  # +1：狀態列在最上面
            self._layout.insertWidget(index, row)
        row.update_from(evt, self.cfg.show_source)
        self._trim()
        self._relayout()

    def set_click_through(self, enabled: bool) -> None:
        self.cfg = replace(self.cfg, click_through=enabled)
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, enabled)
        self.show()  # 變更視窗旗標後需要重新顯示

    def set_show_source(self, enabled: bool) -> None:
        self.cfg = replace(self.cfg, show_source=enabled)
        for row in self._rows.values():
            row.apply_style(self.cfg)
        self._relayout()

    def apply_config(self, cfg: OverlayConfig) -> None:
        """設定視窗存檔後立即套用外觀（不需重新啟動）。"""
        old = self.cfg
        self.cfg = cfg
        self._status.setFont(QFont(cfg.font_family, max(10, cfg.source_font_size - 2)))
        screen = (self.screen() or QGuiApplication.primaryScreen()).availableGeometry()
        if cfg.width_ratio != old.width_ratio:
            self.setFixedWidth(int(screen.width() * cfg.width_ratio))
        if cfg.bottom_margin != old.bottom_margin:
            self._anchor_cx = screen.x() + screen.width() // 2
            self._anchor_bottom = screen.y() + screen.height() - cfg.bottom_margin
        if cfg.click_through != old.click_through:
            self.set_click_through(cfg.click_through)
        for row in self._rows.values():
            row.apply_style(cfg)
        self._trim()
        self._relayout()
        self.update()

    def change_font_size(self, delta: int) -> None:
        size = min(72, max(12, self.cfg.font_size + delta))
        src = min(48, max(9, self.cfg.source_font_size + (1 if delta > 0 else -1)))
        self.cfg = replace(self.cfg, font_size=size, source_font_size=src)
        for row in self._rows.values():
            row.apply_style(self.cfg)
        self._relayout()

    # ------------------------------------------------------------------ 版面
    def _trim(self) -> None:
        while len(self._rows) > self.cfg.max_lines:
            self._remove(min(self._rows))

    def _remove(self, seq: int) -> None:
        row = self._rows.pop(seq, None)
        if row:
            self._layout.removeWidget(row)
            row.deleteLater()
            self._relayout()

    def _expire(self) -> None:
        now = time.monotonic()
        for seq, row in list(self._rows.items()):
            ttl = self.cfg.clear_after_s * (1 if row.final else 2)
            if now - row.updated_at > ttl:
                self._remove(seq)
        if self._status.isVisible() and self._rows and now - self._status_at > _STATUS_VISIBLE_S:
            self._status.hide()
            self._relayout()

    def _relayout(self) -> None:
        self._layout.activate()
        self.adjustSize()
        h = max(self.sizeHint().height(), 40)
        self.resize(self.width(), h)
        self.move(self._anchor_cx - self.width() // 2, self._anchor_bottom - h)

    # ------------------------------------------------------------------ 繪製與滑鼠
    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt 命名
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, int(255 * self.cfg.bg_opacity)))
        painter.drawRoundedRect(self.rect(), 14, 14)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            top_left = event.globalPosition().toPoint() - self._drag_offset
            self.move(top_left)
            self._anchor_cx = top_left.x() + self.width() // 2
            self._anchor_bottom = top_left.y() + self.height()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:  # noqa: N802
        self._drag_offset = None

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.change_font_size(2 if event.angleDelta().y() > 0 else -2)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802
        if self._menu:
            self._menu.exec(event.globalPos())


def _make_icon() -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#1f6feb"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 12, 12)
    p.setPen(QColor("white"))
    font = QFont("Microsoft JhengHei UI", 30)
    font.setWeight(QFont.Weight.Bold)
    p.setFont(font)
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "字")
    p.end()
    return QIcon(pix)


def open_folder(path: Path) -> None:
    """以檔案總管開啟資料夾。"""
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class TrayController(QObject):
    """系統匣圖示與選單。"""

    def __init__(
        self,
        app: QApplication,
        overlay: SubtitleOverlay,
        session_dir: Path,
        on_pause: Callable[[bool], None],
        on_reconnect: Callable[[], None],
        usage_summary: Callable[[], str],
        on_quit: Callable[[], None],
        on_settings: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(app)
        self._overlay = overlay
        self._usage = usage_summary
        self.menu = QMenu()

        self._pause = QAction("暫停", self.menu, checkable=True)
        self._pause.toggled.connect(on_pause)
        self._click = QAction("點擊穿透（滑鼠可點到後方視窗）", self.menu, checkable=True)
        self._click.setChecked(overlay.cfg.click_through)
        self._click.toggled.connect(overlay.set_click_through)
        self._source = QAction("顯示原文", self.menu, checkable=True)
        self._source.setChecked(overlay.cfg.show_source)
        self._source.toggled.connect(overlay.set_show_source)
        bigger = QAction("字放大", self.menu)
        bigger.triggered.connect(lambda: overlay.change_font_size(2))
        smaller = QAction("字縮小", self.menu)
        smaller.triggered.connect(lambda: overlay.change_font_size(-2))
        reconnect = QAction("重新連接音訊裝置（換耳機後使用）", self.menu)
        reconnect.triggered.connect(on_reconnect)
        folder = QAction("開啟本次逐字稿資料夾", self.menu)
        folder.triggered.connect(lambda: open_folder(session_dir))
        self._usage_action = QAction("", self.menu)
        self._usage_action.setEnabled(False)
        settings = QAction("設定…", self.menu)
        settings.setEnabled(on_settings is not None)
        if on_settings is not None:
            settings.triggered.connect(on_settings)
        quit_action = QAction("結束", self.menu)
        quit_action.triggered.connect(on_quit)

        for item in (settings, None, self._pause, self._click, self._source, None, bigger, smaller, None,
                     reconnect, folder, None, self._usage_action, quit_action):
            if item is None:
                self.menu.addSeparator()
            else:
                self.menu.addAction(item)
        self.menu.aboutToShow.connect(self._refresh_usage)

        self.tray = QSystemTrayIcon(_make_icon(), app)
        self.tray.setToolTip("即時字幕翻譯")
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._on_activated)
        self.tray.show()
        overlay.set_menu(self.menu)

    def _refresh_usage(self) -> None:
        self._usage_action.setText(self._usage())
        # 設定視窗可能改過外觀，讓勾選狀態與字幕視窗一致
        for action, value in ((self._click, self._overlay.cfg.click_through),
                              (self._source, self._overlay.cfg.show_source)):
            action.blockSignals(True)
            action.setChecked(value)
            action.blockSignals(False)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:  # 左鍵：切換點擊穿透
            self._click.toggle()
