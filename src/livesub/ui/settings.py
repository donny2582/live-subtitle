"""設定視窗：用圖形介面修改 config.toml 與 API 金鑰。

* 只會改動畫面上有的項目，config.toml 其他內容與註解保持不變。
* 金鑰存在 Windows 使用者環境變數（等同 setx），不寫進設定檔。
* 更進階的參數按「開啟設定檔（進階）」直接編輯 config.toml。

新增欄位的方法：在對應分頁的 `_build_*` 方法加一個元件，並用 `self._bind(...)`
指定它對應的設定鍵（例如 "overlay.font_size"），存檔時會自動比對並寫入。

存檔流程：
    每個元件在 self._bindings 登記「設定鍵 → 讀取目前值的函式」
    → 按「儲存」時 collect_updates() 逐一比對，只收集有改變的項目
    → save_config_updates() 寫回 config.toml（保留註解）
    → 通知 app.py（on_saved），外觀立即套用、其他設定詢問是否重新啟動

四個分頁：服務與費用、API 金鑰、音訊與斷句、字幕外觀（各由一個 _build_* 方法建立）。
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFontComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig, ConfigError, load_config, save_config_updates
from ..config.user_env import mask_secret, read_user_env, set_user_env
from .presets import (
    API_KEYS,
    ASR_PRESETS,
    TRANSLATE_PRESETS,
    Preset,
    asr_key,
    current_index,
    get_value,
    translate_key,
)

log = logging.getLogger(__name__)

_LANGUAGES = (("en", "英文"), ("ja", "日文"), ("ko", "韓文"), ("zh", "中文"))


def _hint(text: str) -> QLabel:
    """灰色的說明小字。"""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: palette(placeholder-text);")
    return lbl


class SettingsDialog(QDialog):
    """設定視窗。存檔成功後呼叫 on_saved(新設定, 變更的鍵)。"""

    def __init__(
        self,
        config_path: Path,
        on_saved: Callable[[AppConfig, set[str]], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("即時字幕 － 設定")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(560)
        self._path = config_path
        self._on_saved = on_saved
        try:
            self._cfg = load_config(config_path) if config_path.exists() else AppConfig()
        except ConfigError as exc:
            QMessageBox.warning(self, "設定檔有誤", f"{exc}\n\n將以預設值顯示；存檔會修正有誤的項目。")
            self._cfg = AppConfig()
        # 設定鍵 → 讀取元件目前值的函式
        self._bindings: dict[str, Callable[[], Any]] = {}
        # 金鑰名稱 → (狀態文字, 輸入框)
        self._key_rows: dict[str, tuple[QLabel, QLineEdit]] = {}

        tabs = QTabWidget(self)
        tabs.addTab(self._build_services(), "服務與費用")
        tabs.addTab(self._build_keys(), "API 金鑰")
        tabs.addTab(self._build_audio(), "音訊與斷句")
        tabs.addTab(self._build_overlay(), "字幕外觀")

        advanced = QPushButton("開啟設定檔（進階）")
        advanced.setToolTip("用記事本開啟 config.toml，可調整視窗裡沒有的細節。改完需重新啟動。")
        advanced.clicked.connect(lambda: _open_file(self._path))
        glossary = QPushButton("編輯術語表")
        glossary.setToolTip("glossary.csv：專有名詞的固定譯法，例如 section control → 區間測速。\n"
                            "存檔後重新啟動生效。")
        glossary.clicked.connect(self._open_glossary)
        buttons = QDialogButtonBox()
        buttons.addButton("儲存", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        bottom = QHBoxLayout()
        bottom.addWidget(advanced)
        bottom.addWidget(glossary)
        bottom.addStretch(1)
        bottom.addWidget(buttons)
        root = QVBoxLayout(self)
        root.addWidget(tabs)
        root.addLayout(bottom)
        self._refresh_cost()

    # ================================================================== 分頁：服務
    def _build_services(self) -> QWidget:
        """「服務與費用」分頁：選辨識／翻譯服務、翻譯選項、每月預算。"""
        page = QWidget()
        lay = QVBoxLayout(page)

        svc = QGroupBox("要用哪個服務")
        form = QFormLayout(svc)
        self._asr = self._preset_combo(ASR_PRESETS)
        self._asr.setToolTip("把語音轉成文字的服務。雲端不需要顯示卡，但會把會議音訊送出。")
        self._tr = self._preset_combo(TRANSLATE_PRESETS)
        self._tr.setToolTip("把原文翻成繁體中文的服務。")
        self._asr_tip, self._tr_tip = _hint(""), _hint("")
        form.addRow("語音辨識：", self._asr)
        form.addRow("", self._asr_tip)
        form.addRow("翻譯：", self._tr)
        form.addRow("", self._tr_tip)
        self._cost = QLabel()
        self._cost.setWordWrap(True)
        form.addRow("2 小時會議約：", self._cost)
        lay.addWidget(svc)

        opts = QGroupBox("翻譯選項")
        form = QFormLayout(opts)
        partial = self._check(
            "translate.partial_translation", "講到一半就先翻譯",
            "開啟：字幕更即時，但翻譯費用約變 4 倍。\n關閉：每句講完才翻譯，最省錢。",
        )
        taiwan = self._check(
            "translate.taiwan_convert", "統一成台灣用語（软件→軟體、視頻→影片）",
            "用 OpenCC 把簡體字與大陸用語轉成台灣正體與用語。",
        )
        context = self._spin("translate.context_turns", 0, 10, suffix=" 句",
                             tip="翻譯時參考前面幾句，讓同一個詞前後翻譯一致。\n越多越準，但費用越高。")
        form.addRow(partial)
        form.addRow(taiwan)
        form.addRow("參考前文：", context)
        partial.toggled.connect(self._refresh_cost)
        lay.addWidget(opts)

        budget = QGroupBox("費用上限")
        form = QFormLayout(budget)
        limit = self._dspin("budget.monthly_limit_usd", 0, 1000, 1.0, decimals=2, prefix="US$ ",
                            tip="語音辨識＋翻譯每月合計上限。用到 70% 後停止中間結果，\n"
                                "用完就暫停所有雲端請求，下個月自動恢復。0 = 不限制。")
        form.addRow("每月最多：", limit)
        form.addRow("", _hint("目前用量在系統匣選單最下方可以看到。"))
        lay.addWidget(budget)
        lay.addStretch(1)
        return page

    def _preset_combo(self, presets: tuple[Preset, ...]) -> QComboBox:
        """服務下拉選單。目前設定不符合任何選項時，多加一個「自訂」並選它。"""
        combo = QComboBox()
        for p in presets:
            combo.addItem(p.label)
        idx = current_index(self._cfg, presets)
        if idx < 0:
            combo.addItem("自訂（保留設定檔目前的內容）")
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)
        combo.setProperty("initial", idx)  # 記住一開始的選項，存檔時判斷有沒有改
        combo.currentIndexChanged.connect(self._refresh_cost)
        return combo

    @staticmethod
    def _selected(combo: QComboBox, presets: tuple[Preset, ...]) -> Preset | None:
        """目前選的服務；選「自訂」時回傳 None。"""
        i = combo.currentIndex()
        return presets[i] if 0 <= i < len(presets) else None

    def _refresh_cost(self) -> None:
        """換服務時更新說明文字、2 小時費用估算，以及金鑰分頁的紅字提示。"""
        asr = self._selected(self._asr, ASR_PRESETS)
        tr = self._selected(self._tr, TRANSLATE_PRESETS)
        self._asr_tip.setText(asr.tip if asr else "使用 config.toml 裡 [asr] 的設定。")
        self._tr_tip.setText(tr.tip if tr else "使用 config.toml 裡 [translate] 的設定。")
        parts = [f"辨識 {asr.cost if asr else '依設定檔'}", f"翻譯 {tr.cost if tr else '依設定檔'}"]
        self._cost.setText("；".join(parts))
        self._refresh_key_hints()

    # ================================================================== 分頁：金鑰
    def _build_keys(self) -> QWidget:
        """「API 金鑰」分頁：每把金鑰一列（狀態、輸入框、儲存、清除）。"""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.addWidget(_hint(
            "金鑰會存到 Windows 的「使用者環境變數」（和 setx 指令一樣），不會寫進設定檔，"
            "其他程式（包含終端機）重新開啟後也讀得到。標示「需要」的是目前選擇的服務會用到的金鑰。"
        ))
        form = QFormLayout()
        for env, desc in API_KEYS:
            status = QLabel()
            edit = QLineEdit()
            edit.setEchoMode(QLineEdit.EchoMode.Password)  # 輸入時顯示成 ●●●
            edit.setPlaceholderText("貼上新的金鑰")
            save = QPushButton("儲存")
            # e=env：把「這一列的金鑰名稱」固定在 lambda 裡（否則迴圈結束後全部會變成最後一個）
            save.clicked.connect(lambda _=False, e=env: self._save_key(e))
            clear = QPushButton("清除")
            clear.clicked.connect(lambda _=False, e=env: self._save_key(e, clear=True))
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(save)
            row.addWidget(clear)
            box = QVBoxLayout()
            box.addWidget(QLabel(f"<b>{env}</b>　{desc}"))
            box.addWidget(status)
            box.addLayout(row)
            form.addRow(box)
            self._key_rows[env] = (status, edit)
        lay.addLayout(form)
        lay.addStretch(1)
        return page

    def _needed_keys(self) -> set[str]:
        """目前選的服務需要哪些金鑰（本機服務不需要，回傳空字串後被排除）。"""
        asr = self._selected(self._asr, ASR_PRESETS)
        tr = self._selected(self._tr, TRANSLATE_PRESETS)
        need = {
            asr.key_env if asr else asr_key(self._cfg),  # 「自訂」時依設定檔
            tr.key_env if tr else translate_key(self._cfg),
        }
        return need - {""}

    def _refresh_key_hints(self) -> None:
        """更新每把金鑰的狀態：綠＝已設定（只顯示頭尾）、紅＝需要但沒設定、灰＝用不到。"""
        if not self._key_rows:
            return
        need = self._needed_keys()
        for env, (status, _edit) in self._key_rows.items():
            value = read_user_env(env)
            if value:
                text, color = f"已設定（{mask_secret(value)}）", "#2e7d32"
            elif env in need:
                text, color = "尚未設定（目前選擇的服務需要這把金鑰）", "#c62828"
            else:
                text, color = "未設定（目前用不到）", "gray"
            status.setText(text)
            status.setStyleSheet(f"color: {color};")

    def _save_key(self, env: str, clear: bool = False) -> None:
        """把金鑰存進 Windows 使用者環境變數（clear=True 則刪除）。金鑰本身不會寫進 log。"""
        _status, edit = self._key_rows[env]
        value = "" if clear else edit.text().strip()
        if not clear and not value:
            QMessageBox.information(self, "金鑰", "請先貼上金鑰。")
            return
        try:
            set_user_env(env, value)
        except OSError as exc:
            QMessageBox.warning(self, "金鑰", f"儲存失敗：{exc}")
            return
        edit.clear()
        self._refresh_key_hints()
        log.info("已%s使用者環境變數 %s", "清除" if clear else "更新", env)

    # ================================================================== 分頁：音訊
    def _build_audio(self) -> QWidget:
        """「音訊與斷句」分頁：聲音來源與裝置、斷句參數、會出現的語言。"""
        page = QWidget()
        lay = QVBoxLayout(page)

        src = QGroupBox("聲音來源")
        form = QFormLayout(src)
        self._source = QComboBox()
        self._source.addItem("電腦播放的聲音（會議軟體、影片）", "loopback")
        self._source.addItem("麥克風", "mic")
        self._source.addItem("電腦播放的聲音＋麥克風（會議中雙方都翻譯）", "both")
        self._source.setToolTip("「電腦播放的聲音＋麥克風」會把對方和自己的聲音混在一起辨識。\n"
                                "請戴耳機，否則麥克風會再收到一次喇叭的聲音，同一句會出現兩次。")
        self._source.setCurrentIndex(max(0, self._source.findData(self._cfg.audio.source)))
        self._bindings["audio.source"] = self._source.currentData
        self._device = QComboBox()
        self._device.setToolTip("通常用「自動」即可。換耳機後可在系統匣選「重新連接音訊裝置」。")
        self._mic_device = QComboBox()
        self._mic_device.setToolTip("自己說話用的麥克風，通常用「自動」即可。")
        self._source.currentIndexChanged.connect(self._fill_devices)
        self._fill_devices()
        self._bindings["audio.device_index"] = self._device.currentData
        self._bindings["audio.mic_device_index"] = self._mic_device.currentData
        form.addRow("來源：", self._source)
        form.addRow("裝置：", self._device)
        form.addRow("麥克風裝置：", self._mic_device)
        self._source_form = form
        self._update_mic_row()
        self._source.currentIndexChanged.connect(self._update_mic_row)
        lay.addWidget(src)

        seg = QGroupBox("斷句（什麼時候算一句講完）")
        form = QFormLayout(seg)
        form.addRow("停頓多久算一句：", self._spin(
            "segmenter.end_silence_ms", 200, 2000, step=50, suffix=" 毫秒",
            tip="越短：字幕出得越快，但句子容易被切碎、翻譯較不通順。\n"
                "越長：句子完整，但要等比較久。講者語速慢可調到 800。",
        ))
        form.addRow("一句最長：", self._dspin(
            "segmenter.max_utterance_s", 2, 30, 1.0, decimals=1, suffix=" 秒",
            tip="講者一直不停頓時，最多這麼長就強制切開，避免字幕一直不出來。",
        ))
        form.addRow("雜音過濾強度：", self._spin(
            "segmenter.vad_aggressiveness", 0, 3,
            tip="0 = 最寬鬆（小聲也會辨識）～ 3 = 最嚴格（背景音樂、雜音多時使用）。",
        ))
        lay.addWidget(seg)

        langs = QGroupBox("會出現的語言（沒勾的語言會被略過）")
        row = QHBoxLayout(langs)
        self._langs: dict[str, QCheckBox] = {}
        for code, name in _LANGUAGES:
            cb = QCheckBox(name)
            cb.setChecked(code in self._cfg.asr.allowed_languages)
            self._langs[code] = cb
            row.addWidget(cb)
        row.addStretch(1)
        self._bindings["asr.allowed_languages"] = lambda: tuple(
            c for c, cb in self._langs.items() if cb.isChecked()
        )
        langs.setToolTip("只勾實際會出現的語言，可以減少誤判（例如把英文口音誤認成其他語言）。")
        lay.addWidget(langs)
        lay.addStretch(1)
        return page

    def _update_mic_row(self) -> None:
        """「麥克風裝置」欄位只在選「電腦播放的聲音＋麥克風」時顯示。"""
        self._source_form.setRowVisible(self._mic_device, self._source.currentData() == "both")

    def _fill_devices(self) -> None:
        """依來源填入裝置清單：系統聲音列 loopback 裝置，麥克風列一般輸入裝置。"""
        try:
            from ..audio import list_devices

            devices = list_devices()
        except Exception as exc:  # noqa: BLE001 - 沒有音效卡或非 Windows
            log.debug("無法列出音訊裝置：%s", exc)
            devices = []
        want_loopback = self._source.currentData() in ("loopback", "both")
        self._fill_combo(self._device, self._cfg.audio.device_index,
                         [d for d in devices if d["loopback"] == want_loopback])
        self._fill_combo(self._mic_device, self._cfg.audio.mic_device_index,
                         [d for d in devices if not d["loopback"]])

    @staticmethod
    def _fill_combo(combo: QComboBox, configured: int, devices: list[dict[str, Any]]) -> None:
        """重新填入裝置下拉選單，並盡量保留原本選的裝置。"""
        # 第一次填：用設定檔的值；之後重填：保留使用者剛剛選的
        current = configured if combo.count() == 0 else combo.currentData()
        combo.blockSignals(True)  # 填的過程中不要觸發「選項改變」事件
        combo.clear()
        combo.addItem("自動（Windows 預設裝置）", -1)
        for d in devices:
            combo.addItem(f"[{d['index']}] {d['name']}", d["index"])
        idx = combo.findData(current)
        # 設定的裝置現在找不到（例如耳機沒插）：仍列出來，避免存檔時被默默改掉
        if idx < 0 and current not in (-1, None):
            combo.addItem(f"[{current}]（目前設定，找不到此裝置）", current)
            idx = combo.count() - 1
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)

    # ================================================================== 分頁：外觀
    def _build_overlay(self) -> QWidget:
        """「字幕外觀」分頁：字型、字級、句數、寬度、透明度、位置。"""
        page = QWidget()
        form = QFormLayout(page)
        form.addRow(self._check("overlay.show_source", "顯示原文（譯文上方的小字）",
                                "關閉後只顯示中文；還沒翻譯出來前仍會先顯示原文。"))
        font = QFontComboBox()
        font.setCurrentFont(QFont(self._cfg.overlay.font_family))
        # 設定的字型沒安裝時下拉選單會顯示別的字型；使用者真的選了才寫入
        chosen = {"family": self._cfg.overlay.font_family}
        font.currentFontChanged.connect(lambda f: chosen.update(family=f.family()))
        self._bindings["overlay.font_family"] = lambda: chosen["family"]
        form.addRow("字型：", font)
        form.addRow("中文字級：", self._spin("overlay.font_size", 12, 72,
                                         tip="也可以在字幕上按住 Ctrl 滾動滑鼠滾輪調整（不會存檔）。"))
        form.addRow("原文字級：", self._spin("overlay.source_font_size", 9, 48))
        form.addRow("同時顯示幾句：", self._spin("overlay.max_lines", 1, 6))
        form.addRow("字幕寬度：", self._dspin("overlay.width_ratio", 0.2, 1.0, 0.05, decimals=2,
                                         tip="佔螢幕寬度的比例，1.0 = 整個螢幕寬。"))
        form.addRow("背景不透明度：", self._dspin("overlay.bg_opacity", 0.0, 1.0, 0.05, decimals=2,
                                          tip="0 = 完全透明，1 = 全黑。"))
        form.addRow("幾秒沒新字幕就清除：", self._dspin("overlay.clear_after_s", 1, 60, 1.0,
                                              decimals=1, suffix=" 秒"))
        form.addRow("離螢幕底部：", self._spin("overlay.bottom_margin", 0, 600, step=10, suffix=" px",
                                        tip="也可以直接用滑鼠左鍵拖曳字幕移動位置（不會存檔）。"))
        form.addRow(_hint("字幕外觀按「儲存」後立即套用，不需重新啟動。"))
        return page

    # ================================================================== 綁定小工具
    # 以下三個函式建立元件、填入目前設定值，並登記到 self._bindings 供存檔時讀取
    def _check(self, key: str, text: str, tip: str = "") -> QCheckBox:
        """勾選框（true / false 設定）。"""
        cb = QCheckBox(text)
        cb.setChecked(bool(get_value(self._cfg, key)))
        cb.setToolTip(tip)
        self._bindings[key] = cb.isChecked
        return cb

    def _spin(self, key: str, lo: int, hi: int, step: int = 1, suffix: str = "", tip: str = "") -> QSpinBox:
        """整數輸入框（有上下箭頭）。"""
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setSuffix(suffix)
        sp.setValue(int(get_value(self._cfg, key)))
        sp.setToolTip(tip)
        self._bindings[key] = sp.value
        return sp

    def _dspin(self, key: str, lo: float, hi: float, step: float, decimals: int = 1,
               prefix: str = "", suffix: str = "", tip: str = "") -> QDoubleSpinBox:
        """小數輸入框。"""
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(step)
        sp.setPrefix(prefix)
        sp.setSuffix(suffix)
        sp.setValue(float(get_value(self._cfg, key)))
        sp.setToolTip(tip)
        self._bindings[key] = lambda: round(sp.value(), decimals)
        return sp

    # ================================================================== 存檔
    def collect_updates(self) -> dict[str, Any]:
        """畫面上與目前設定不同的項目（區段.鍵 → 新值）。"""
        updates: dict[str, Any] = {}
        # 1. 服務選單有換：套用該選項的整組設定（引擎、網址、模型、金鑰名稱、單價）
        for combo, presets in ((self._asr, ASR_PRESETS), (self._tr, TRANSLATE_PRESETS)):
            preset = self._selected(combo, presets)
            if preset is not None and combo.currentIndex() != combo.property("initial"):
                updates.update(preset.updates)
        # 2. 其他欄位：值和目前設定不同才寫入
        for key, getter in self._bindings.items():
            value = getter()
            old = get_value(self._cfg, key)
            if isinstance(old, float):
                value = float(value)
            if value != old:
                updates[key] = list(value) if isinstance(value, tuple) else value
        # 最後再過濾一次（服務選項裡和目前設定相同的值不必寫）
        return {k: v for k, v in updates.items() if _differs(get_value(self._cfg, k), v)}

    def _save(self) -> None:
        """按下「儲存」：檢查 → 寫檔 → 重新讀取確認 → 通知 app.py → 關閉視窗。"""
        if not any(cb.isChecked() for cb in self._langs.values()):
            QMessageBox.warning(self, "設定", "「會出現的語言」至少要勾一個。")
            return
        updates = self.collect_updates()
        if not updates:
            self.accept()
            return
        try:
            save_config_updates(self._path, updates)
            new_cfg = load_config(self._path)
        except (ConfigError, OSError) as exc:
            QMessageBox.warning(self, "儲存失敗", str(exc))
            return
        log.info("設定已更新：%s", ", ".join(sorted(updates)))
        self._cfg = new_cfg
        if self._on_saved:
            self._on_saved(new_cfg, set(updates))
        self.accept()

    def _open_glossary(self) -> None:
        """用記事本開啟術語表；檔案不存在時先從範例複製一份。"""
        path = self._path.parent / self._cfg.translate.glossary_path
        if not path.exists():
            example = self._path.parent / "glossary.example.csv"
            if example.exists():
                path.write_bytes(example.read_bytes())
            else:
                path.write_text("source,target\n", encoding="utf-8-sig")
        _open_file(path)


def _differs(old: Any, new: Any) -> bool:
    """比較新舊值（tuple 與 list 內容相同視為相同）。"""
    if isinstance(old, tuple):
        old = list(old)
    return bool(old != new)


def _open_file(path: Path) -> None:
    """用記事本開啟文字檔（非 Windows 用系統預設程式）。"""
    if sys.platform == "win32":
        subprocess.Popen(["notepad.exe", str(path)])  # noqa: S603, S607
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
