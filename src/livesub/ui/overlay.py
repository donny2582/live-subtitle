"""PyQt6 字幕浮層與系統匣選單。

* 無邊框、永遠置頂、半透明背景，不會搶走會議軟體的焦點。
* 左鍵拖曳移動位置；Ctrl + 滾輪調整字級；右鍵開啟選單。
* 開啟「點擊穿透」後滑鼠會直接點到底下的視窗，要關閉請用系統匣選單。

PyQt6 基本觀念：
* QWidget：畫面上的一個元件（視窗、文字標籤…都是）。
* 訊號（signal）與槽（slot）：A.訊號.connect(B 的函式)，A 發出訊號時就會呼叫 B。
* 只有主執行緒（UI 執行緒）可以修改畫面；背景執行緒要透過訊號把資料「寄」過來。
* xxxEvent() 方法（paintEvent、mousePressEvent…）是 Qt 在對應事件發生時自動呼叫的。

字幕顯示規則：
* 每句話是一個 _Row，用句子序號（seq）對應；同一句的新事件會更新同一個 _Row。
* 中間結果顏色較淡，定稿是白色。
* 超過 max_lines 句時移除最舊的；太久沒更新的句子會自動消失。
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

_STATUS_VISIBLE_S = 6.0  # 狀態文字（黃字）顯示幾秒後隱藏（有字幕時才會隱藏）


class UiBridge(QObject):
    """背景執行緒 → UI 執行緒的訊號橋接（Qt 會自動以 queued connection 傳遞）。"""

    event = pyqtSignal(object)  # 傳送 SubtitleEvent（字幕）
    status = pyqtSignal(str)  # 傳送狀態文字（例如「聆聽中：喇叭」）


class _Row(QWidget):
    """一句字幕：上方原文（小字）、下方譯文（大字）。"""

    def __init__(self, seq: int, cfg: OverlayConfig, parent: QWidget) -> None:
        super().__init__(parent)
        self.seq = seq  # 句子序號
        self.final = False  # 是否已定稿
        self.language = ""
        self.updated_at = time.monotonic()  # 最後更新時間（判斷何時自動消失）
        lay = QVBoxLayout(self)  # 垂直排列：原文在上、譯文在下
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.source = QLabel(self)
        self.target = QLabel(self)
        for lbl in (self.source, self.target):
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            # 純文字模式：避免辨識結果裡剛好有 <b> 之類的字被當成 HTML 解讀
            lbl.setTextFormat(Qt.TextFormat.PlainText)
            lay.addWidget(lbl)
        self.target.hide()  # 還沒有譯文前先隱藏
        self.apply_style(cfg)

    def apply_style(self, cfg: OverlayConfig) -> None:
        """套用字型、字級與顏色。"""
        self.source.setFont(QFont(cfg.font_family, cfg.source_font_size))
        target_font = QFont(cfg.font_family, cfg.font_size)
        target_font.setWeight(QFont.Weight.DemiBold)
        self.target.setFont(target_font)
        self._update_source_visibility(cfg.show_source)
        self._paint_colors()

    def _paint_colors(self) -> None:
        """定稿用亮色、中間結果用較暗的顏色，讓使用者看得出哪些字還可能變動。"""
        src_color = "#c8d0d8" if self.final else "#9aa4ae"
        tgt_color = "#ffffff" if self.final else "#d7e3ea"
        self.source.setStyleSheet(f"color: {src_color};")
        self.target.setStyleSheet(f"color: {tgt_color};")

    def update_from(self, evt: SubtitleEvent, show_source: bool) -> None:
        """用新的事件更新這句字幕（翻譯失敗時把錯誤訊息顯示在譯文位置）。"""
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
    """字幕視窗。

    版面：最上面是狀態文字（黃字），下面依序排列每句字幕。
    位置以「底部中央」為錨點：字幕變多變高時往上長，底部位置不變。
    """

    def __init__(self, cfg: OverlayConfig) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint  # 沒有標題列與邊框
            | Qt.WindowType.WindowStaysOnTopHint  # 永遠在最上層
            | Qt.WindowType.Tool  # 不出現在工作列
        )
        super().__init__(None, flags)
        self.cfg = cfg
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)  # 背景透明，自己畫半透明圓角框
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)  # 顯示時不搶走焦點
        self.setWindowTitle("即時字幕")
        self._rows: dict[int, _Row] = {}  # 句子序號 → 那一句的元件
        self._menu: QMenu | None = None  # 右鍵選單（由 TrayController 提供）
        self._drag_offset: QPoint | None = None  # 拖曳時，滑鼠相對視窗左上角的位置
        self._status_at = 0.0  # 狀態文字最後更新時間

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 10, 18, 12)
        self._layout.setSpacing(8)
        self._status = QLabel(self)
        self._status.setFont(QFont(cfg.font_family, max(10, cfg.source_font_size - 2)))
        self._status.setStyleSheet("color: #ffd27f;")
        self._status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._status.setWordWrap(True)
        self._layout.addWidget(self._status)

        # 主螢幕可用範圍（扣掉工作列），字幕放在底部中央
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.setFixedWidth(int(screen.width() * cfg.width_ratio))
        self._anchor_cx = screen.x() + screen.width() // 2  # 錨點：水平中央
        self._anchor_bottom = screen.y() + screen.height() - cfg.bottom_margin  # 錨點：底部

        # 每 0.5 秒檢查一次有沒有該消失的字幕
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._expire)
        self._timer.start(500)
        if cfg.click_through:
            self.set_click_through(True)
        self._relayout()

    # ------------------------------------------------------------------ 外部呼叫
    def set_menu(self, menu: QMenu) -> None:
        """設定右鍵選單（與系統匣共用同一個選單）。"""
        self._menu = menu

    def show_status(self, text: str) -> None:
        """在字幕上方顯示狀態文字（例如「聆聽中」「翻譯失敗」）。"""
        self._status.setText(text)
        self._status.show()
        self._status_at = time.monotonic()
        self._relayout()

    def on_event(self, evt: SubtitleEvent) -> None:
        """收到字幕事件（由 pipeline 經 UiBridge 送來）：新增或更新對應的那一句。"""
        row = self._rows.get(evt.seq)
        if evt.stage is Stage.FINAL and not evt.source_text:
            # 空白定稿 = 這句沒有內容（辨識失敗或被過濾），把殘留的中間結果移除
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
        """開關點擊穿透（WindowTransparentForInput：滑鼠事件直接穿過這個視窗）。"""
        self.cfg = replace(self.cfg, click_through=enabled)
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, enabled)
        self.show()  # 變更視窗旗標後需要重新顯示

    def set_show_source(self, enabled: bool) -> None:
        """開關原文小字。"""
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
        """調整字級（Ctrl+滾輪、選單的字放大／縮小）。中文 12~72、原文 9~48。"""
        size = min(72, max(12, self.cfg.font_size + delta))
        src = min(48, max(9, self.cfg.source_font_size + (1 if delta > 0 else -1)))
        self.cfg = replace(self.cfg, font_size=size, source_font_size=src)
        for row in self._rows.values():
            row.apply_style(self.cfg)
        self._relayout()

    # ------------------------------------------------------------------ 版面
    def _trim(self) -> None:
        """句數超過 max_lines 時，移除最舊（序號最小）的句子。"""
        while len(self._rows) > self.cfg.max_lines:
            self._remove(min(self._rows))

    def _remove(self, seq: int) -> None:
        row = self._rows.pop(seq, None)
        if row:
            self._layout.removeWidget(row)
            row.deleteLater()  # 交給 Qt 在安全的時機釋放元件
            self._relayout()

    def _expire(self) -> None:
        """計時器每 0.5 秒呼叫：太久沒更新的字幕移除；有字幕時狀態文字幾秒後隱藏。"""
        now = time.monotonic()
        for seq, row in list(self._rows.items()):
            # 中間結果多等一倍時間（可能只是辨識比較慢，定稿還在路上）
            ttl = self.cfg.clear_after_s * (1 if row.final else 2)
            if now - row.updated_at > ttl:
                self._remove(seq)
        if self._status.isVisible() and self._rows and now - self._status_at > _STATUS_VISIBLE_S:
            self._status.hide()
            self._relayout()

    def _relayout(self) -> None:
        """依內容重新計算視窗高度，並維持底部中央的位置。"""
        self._layout.activate()
        self.adjustSize()
        h = max(self.sizeHint().height(), 40)
        self.resize(self.width(), h)
        self.move(self._anchor_cx - self.width() // 2, self._anchor_bottom - h)

    # ------------------------------------------------------------------ 繪製與滑鼠
    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt 命名
        """畫出半透明黑色圓角背景（透明度由 bg_opacity 決定）。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, int(255 * self.cfg.bg_opacity)))
        painter.drawRoundedRect(self.rect(), 14, 14)

    # ---- 拖曳移動：按下時記住位置，移動時跟著走，放開時結束 ----
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            top_left = event.globalPosition().toPoint() - self._drag_offset
            self.move(top_left)
            # 更新錨點，之後字幕變高變矮時以新位置為準
            self._anchor_cx = top_left.x() + self.width() // 2
            self._anchor_bottom = top_left.y() + self.height()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:  # noqa: N802
        self._drag_offset = None

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Ctrl + 滾輪調整字級。"""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.change_font_size(2 if event.angleDelta().y() > 0 else -2)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802
        """右鍵：在滑鼠位置開啟選單。"""
        if self._menu:
            self._menu.exec(event.globalPos())


def _make_icon() -> QIcon:
    """用程式畫出系統匣圖示（藍底白色「字」），不需要另外附圖檔。"""
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
    """系統匣圖示與選單。

    選單裡每個項目是一個 QAction；checkable=True 的項目會有勾選框。
    各項目被點選時呼叫的函式（on_pause 等）由 app.py 傳入，這裡只負責畫面。
    """

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
        self._usage_action = QAction("", self.menu)  # 本月花費（每次開選單時更新）
        self._usage_action.setEnabled(False)  # 只顯示，不能點
        settings = QAction("設定…", self.menu)
        settings.setEnabled(on_settings is not None)
        if on_settings is not None:
            settings.triggered.connect(on_settings)
        quit_action = QAction("結束", self.menu)
        quit_action.triggered.connect(on_quit)

        # 依序加入選單；None 代表分隔線
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
        overlay.set_menu(self.menu)  # 右鍵字幕也開同一個選單

    def _refresh_usage(self) -> None:
        """選單打開前更新花費文字與勾選狀態。"""
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
