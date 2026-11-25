"""程式進入點：解析參數、載入設定、組裝管線與介面。

執行 `scripts\\run.ps1` 或 `python -m livesub` 時，最先執行的就是這裡的 main()。

main() 的流程：
    1. 解析命令列參數（--settings、--list-devices、--engine…）
    2. 設定 log（寫到 logs\\livesub.log）
    3. 顯示設定視窗（啟動模式），按「開始字幕」後讀取 config.toml
       （加 --skip-settings 可跳過；第一次執行會從範例檔複製一份設定檔）
    4. 依設定建立各元件：術語表、費用帳本、翻譯器、辨識器、逐字稿寫入器
    5. 建立畫面（字幕浮層、系統匣選單）
    6. 把元件交給 Pipeline 串起來，啟動背景處理
    7. 進入 Qt 事件迴圈，直到使用者按「結束」
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    AppConfig,
    ConfigError,
    app_base_dir,
    load_config_with_notes,
    replace,
    resolve_path,
    user_data_dir,
)

log = logging.getLogger("livesub")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """定義並解析命令列參數。這裡指定的值會暫時蓋過 config.toml（見 apply_overrides）。"""
    p = argparse.ArgumentParser(prog="livesub", description="即時語音翻譯字幕（英／日／韓 → 繁中）")
    p.add_argument("--config", type=Path, default=None, help="設定檔路徑（預設為 config.toml）")
    p.add_argument("--list-devices", action="store_true", help="列出音訊裝置後結束")
    p.add_argument("--settings", action="store_true", help="只開啟設定視窗（不啟動字幕）")
    p.add_argument("--skip-settings", action="store_true",
                   help="直接開始字幕，不先顯示設定視窗")
    # 內部使用：由「重新啟動」產生，代表要等舊程式結束、放開單一執行個體的鎖
    p.add_argument("--restarted", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--device-index", type=int, default=None, help="指定音訊裝置編號")
    p.add_argument("--source", choices=["loopback", "mic", "both"], default=None,
                   help="音訊來源：loopback = 系統聲音、mic = 麥克風、both = 兩者混音")
    p.add_argument("--asr", choices=["cloud", "local"], default=None, help="語音辨識引擎")
    p.add_argument("--model", default=None, help="本機 Whisper 模型名稱，例如 large-v3-turbo、small")
    p.add_argument("--engine", choices=["openai", "claude", "deepl", "none"], default=None, help="翻譯引擎")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def configure_logging(level: str, log_dir: Path, console: bool = True) -> None:
    """設定 log：同時寫到檔案（logs\\livesub.log）與終端機。

    檔案超過 2 MB 會自動換新檔，最多保留 3 個舊檔，避免越長越大。
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    # 每行格式：時間 等級 [執行緒名稱] 模組: 訊息
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    file_handler = RotatingFileHandler(
        log_dir / "livesub.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    if console:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        root.addHandler(handler)
    # 第三方套件的 log 太多太雜，只顯示警告以上
    for noisy in ("urllib3", "faster_whisper", "httpx"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """命令列參數覆蓋設定檔。

    設定物件是不可修改的（frozen），所以用 replace() 產生「改了某些欄位的新副本」。
    """
    audio, asr, tr = cfg.audio, cfg.asr, cfg.translate
    if args.device_index is not None:
        audio = replace(audio, device_index=args.device_index)
    if args.source:
        audio = replace(audio, source=args.source)
    if args.asr:
        asr = replace(asr, engine=args.asr)
    if args.model:
        asr = replace(asr, local=replace(asr.local, model=args.model))
    if args.engine:
        tr = replace(tr, engine=args.engine)
    return replace(cfg, audio=audio, asr=asr, translate=tr)


# 這些參數會覆蓋設定檔；從設定視窗重新啟動時要拿掉，新設定才會生效
_OVERRIDE_FLAGS = ("--device-index", "--source", "--asr", "--model", "--engine")


def restart_command() -> list[str]:
    """重新啟動本程式的指令（去掉覆蓋設定的參數與 --settings）。

    依啟動方式不同，重組出一樣的指令：打包的 .exe、`python -m livesub`、或直接跑腳本。
    使用者剛在設定視窗存檔才重新啟動，所以加上 --skip-settings，不再顯示一次設定視窗。
    """
    if getattr(sys, "frozen", False):  # PyInstaller 打包的 .exe
        prog, args = [sys.executable], sys.argv[1:]
    elif sys.orig_argv[1:2] == ["-m"]:  # python -m livesub ...
        prog, args = [sys.executable, "-m", sys.orig_argv[2]], sys.orig_argv[3:]
    else:  # python path/to/script.py ...
        prog, args = [sys.executable, *sys.orig_argv[1:2]], sys.orig_argv[2:]
    kept: list[str] = []
    skip = False  # 上一個參數是 "--engine" 這種分開寫值的旗標時，下一個（值）也要跳過
    for a in args:
        if skip:
            skip = False
            continue
        if a == "--settings" or a.split("=")[0] in _OVERRIDE_FLAGS:
            # "--engine=claude" 一個就結束；"--engine claude" 要連下一個一起丟掉
            skip = "=" not in a and a != "--settings"
            continue
        kept.append(a)
    for flag in ("--skip-settings", "--restarted"):
        if flag not in kept:
            kept.append(flag)
    return prog + kept


def run_settings_only(cfg_path: Path) -> int:
    """`--settings`：只開設定視窗，關閉後結束。"""
    from PyQt6.QtWidgets import QApplication

    from .ui.settings import SettingsDialog

    app = QApplication(sys.argv)  # noqa: F841 - 對話框需要 QApplication
    SettingsDialog(cfg_path).exec()
    return 0


def ensure_user_files(base: Path, data: Path | None = None) -> None:
    """第一次執行時，從範例檔（在 base）複製出可編輯的設定檔與術語表（到 data）。

    範例檔（*.example.*）會跟著程式更新；使用者只改複製出來的那份，更新時就不會被覆蓋。
    """
    data = data or base
    data.mkdir(parents=True, exist_ok=True)
    for example, target in (("config.example.toml", "config.toml"), ("glossary.example.csv", "glossary.csv")):
        src, dst = base / example, data / target
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)
            log.info("已建立 %s，可依需求修改", dst)


def print_devices() -> int:
    """`--list-devices`：印出所有可用的音訊輸入與 loopback 裝置。"""
    try:
        from .audio import list_devices
    except ImportError:  # 非 Windows 或沒裝 PyAudioWPatch
        list_devices = None

    try:
        if list_devices is None:
            raise ImportError
        devices = list_devices()
    except ImportError:
        print("需要 Windows 與 PyAudioWPatch 才能列出裝置。")
        return 1
    print(f"{'編號':>4}  {'類型':<8} {'聲道':>4} {'取樣率':>7}  名稱")
    for d in devices:
        kind = "loopback" if d["loopback"] else d["host_api"][:8]
        print(f"{d['index']:>4}  {kind:<8} {d['channels']:>4} {d['rate']:>7}  {d['name']}")
    print("\n在 config.toml 的 [audio] device_index 填入編號即可指定裝置。")
    return 0


def main(argv: list[str] | None = None) -> int:
    """程式主流程，回傳值是結束代碼（0 = 正常）。"""
    # 無主控台的 .exe 中 stdout/stderr 為 None，下載模型的進度條會因此當掉
    headless = sys.stderr is None
    if headless:
        sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # ---- 1~2. 參數與 log ----
    args = parse_args(argv)
    base = app_base_dir()  # 程式所在資料夾（開發時是專案根目錄，打包後是 exe 旁邊）
    data = user_data_dir(base)  # 設定、log、逐字稿的位置（通常就是 base；不能寫入時改用 %APPDATA%）
    configure_logging(args.log_level, data / "logs", console=not headless)
    if data != base:
        log.info("程式資料夾無法寫入，設定與紀錄改存到 %s", data)
    if args.list_devices:
        return print_devices()

    # ---- 3. 設定檔 ----
    ensure_user_files(base, data)
    cfg_path = args.config or (data / "config.toml")
    if args.settings:
        return run_settings_only(cfg_path)

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QDialog

    from .ui.settings import SettingsDialog
    from .ui.single_instance import SingleInstance, bring_to_front

    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # 關掉設定視窗時不要連整個程式一起結束

    # 同時只能開一個：已經在執行的話，叫原本那個把視窗拿到最前面，自己結束
    # （重新啟動時舊程式還在收尾，所以最多等 10 秒讓它放開鎖）
    instance = SingleInstance()
    if not instance.acquire(wait_ms=10_000 if args.restarted else 0):
        if instance.notify_running():
            log.info("即時字幕已在執行中，已通知原本的程式顯示視窗")
        else:
            log.warning("即時字幕已在執行中，但無法通知它")
        return 0

    # 另一個程式要求顯示視窗時：有對話框開著就把它拿到最前面，否則開啟設定視窗
    # 字幕還沒開始前 open_settings 是 None，此時一定是啟動用的設定視窗開著
    ui: dict[str, Any] = {"open_settings": None}

    def on_activated() -> None:
        dialog = app.activeModalWidget()
        if dialog is not None:
            bring_to_front(dialog)
        elif ui["open_settings"] is not None:
            ui["open_settings"]()

    # 用 singleShot 延後到目前的事件處理完再執行，避免在通訊管道的處理過程中開視窗
    instance.activated.connect(lambda: QTimer.singleShot(0, on_activated))

    # 啟動模式：先顯示設定視窗，按「開始字幕」才繼續；按「結束」或關閉視窗就離開
    if not args.skip_settings:
        if SettingsDialog(cfg_path, launcher=True).exec() != QDialog.DialogCode.Accepted:
            return 0

    # 設定視窗可能剛存檔，所以在這之後才讀設定檔
    try:
        # legacy：舊格式設定的鍵名清單，用來提醒使用者更新
        cfg, legacy = load_config_with_notes(cfg_path) if cfg_path.exists() else (AppConfig(), [])
    except ConfigError as exc:
        log.error("設定檔錯誤：%s", exc)
        if headless:
            _fatal_dialog(f"設定檔錯誤：\n{exc}")
        return 2
    if not cfg_path.exists():
        log.warning("找不到 %s，使用預設設定", cfg_path)
    if legacy:
        log.warning("設定檔使用舊格式（%s），已自動對應；建議參考 config.example.toml 更新",
                    ", ".join(legacy))
    cfg = apply_overrides(cfg, args)
    cfg_base = cfg_path.resolve().parent  # 設定檔裡的相對路徑都以設定檔所在資料夾為準

    # 延遲載入大型相依套件，讓 --help / --list-devices 不必等待
    from .asr import build_transcriber
    from .audio import make_vad
    from .config.user_env import with_user_env
    from .core.budget import BudgetLedger
    from .core.glossary import build_asr_prompt, load_glossary_safe
    from .output import TranscriptWriter, create_session_dir
    from .pipeline import Pipeline
    from .translation import build_translator
    from .ui.overlay import SubtitleOverlay, TrayController, UiBridge

    # ---- 4. 建立各元件 ----
    # API 金鑰：先看目前終端機的環境變數，沒有的話再讀 Windows 使用者環境變數
    # （用設定視窗或 setx 存的金鑰，在先前開啟的終端機裡看不到）
    t = cfg.translate
    env = with_user_env(dict(os.environ), (
        cfg.asr.cloud.api_key_env, t.openai.api_key_env, t.claude.api_key_env, t.deepl.api_key_env,
    ))
    glossary = load_glossary_safe(resolve_path(cfg_base, cfg.translate.glossary_path))
    ledger = BudgetLedger(cfg.budget, resolve_path(cfg_base, cfg.budget.ledger_path))
    translator = build_translator(cfg.translate, glossary, ledger, env)
    transcriber = build_transcriber(
        cfg.asr, cfg_base, env, ledger,
        prompt=build_asr_prompt(glossary) if cfg.asr.glossary_prompt else None,
    )
    # 逐字稿資料夾：使用者選的資料夾不能用（例如隨身碟拔掉了）時，改存到預設的 transcripts
    transcript_root = resolve_path(cfg_base, cfg.output.transcript_dir) or (data / "transcripts")
    transcript_warning = ""
    try:
        session_dir = create_session_dir(transcript_root)
    except OSError as exc:
        log.error("逐字稿資料夾無法使用（%s）：%s", transcript_root, exc)
        session_dir = create_session_dir(data / "transcripts")
        transcript_warning = f"逐字稿資料夾無法使用，本次改存到：{session_dir}"
    writer = TranscriptWriter(session_dir, cfg.output)
    log.info("語音辨識：%s；翻譯：%s；逐字稿：%s", transcriber.name, translator.name, session_dir)

    # ---- 5. 畫面 ----
    # 背景執行緒不能直接操作畫面，要透過 Qt 訊號（UiBridge）轉交給主執行緒
    bridge = UiBridge()
    overlay = SubtitleOverlay(cfg.overlay)
    bridge.event.connect(overlay.on_event)
    bridge.status.connect(overlay.show_status)

    # ---- 6. 管線 ----
    pipeline = Pipeline(
        cfg=cfg,
        transcriber=transcriber,
        translator=translator,
        vad_factory=lambda: make_vad(cfg.segmenter),
        writer=writer,
        ledger=ledger,
        on_event=bridge.event.emit,
        on_status=bridge.status.emit,
    )
    def on_settings_saved(new_cfg: AppConfig, changed: set[str]) -> None:
        """設定視窗存檔後：外觀立即套用；其他設定詢問是否重新啟動。"""
        overlay.apply_config(new_cfg.overlay)
        if all(k.startswith("overlay.") for k in changed):
            overlay.show_status("字幕外觀已更新")
            return
        from PyQt6.QtWidgets import QMessageBox

        answer = QMessageBox.question(
            None, "即時字幕", "設定已儲存。要現在重新啟動套用新設定嗎？\n（選「否」會在下次啟動時生效）",
        )
        if answer == QMessageBox.StandardButton.Yes:
            from PyQt6.QtCore import QProcess

            cmd = restart_command()
            # 先啟動新的程式（它帶 --restarted，會等最多 10 秒讓我們放開鎖），成功了才停掉自己。
            # 若先停管線再啟動，啟動失敗時字幕就停了卻還開著，使用者只能手動重開。
            if not QProcess.startDetached(cmd[0], cmd[1:], str(base)):
                overlay.show_status("重新啟動失敗，請手動關閉後再開啟")
                return
            pipeline.stop()
            instance.release()  # 放開「只能開一個」的鎖，新程式才能接手
            app.quit()

    def open_settings() -> None:
        # 設定視窗已經開著（例如又從選單點了一次）就拿到最前面，不要再開第二個
        if app.activeModalWidget() is not None:
            bring_to_front(app.activeModalWidget())
            return
        overlay.show_status("即時字幕執行中")
        dialog = SettingsDialog(cfg_path, on_saved=on_settings_saved)
        dialog.exec()

    ui["open_settings"] = open_settings

    tray = TrayController(  # noqa: F841 - 需保留參考（變數被回收的話系統匣圖示會消失）
        app,
        overlay,
        session_dir,
        on_pause=pipeline.set_paused,
        on_reconnect=pipeline.capture.reconnect,
        usage_summary=ledger.summary,
        on_quit=app.quit,
        on_settings=open_settings,
    )
    app.aboutToQuit.connect(pipeline.stop)

    # 讓 Ctrl+C 能結束 Qt 事件迴圈
    # （Qt 執行時 Python 收不到 Ctrl+C，每 250 毫秒用空的計時器讓 Python 有機會處理）
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start(250)

    # ---- 7. 開始執行 ----
    overlay.show()
    overlay.show_status(transcript_warning or "啟動中…（右鍵或系統匣圖示可開啟選單）")
    pipeline.start()
    code = app.exec()  # 這行會一直執行，直到程式結束
    pipeline.stop()
    instance.release()
    return code


def _fatal_dialog(message: str) -> None:
    """在沒有主控台的 .exe 中也能看到致命錯誤。"""
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        _app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "即時字幕", message)
    except Exception:  # noqa: BLE001, S110
        pass
