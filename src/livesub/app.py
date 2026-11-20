"""程式進入點：解析參數、載入設定、組裝管線與介面。"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from . import __version__
from .config import AppConfig, ConfigError, app_base_dir, load_config_with_notes, replace, resolve_path

log = logging.getLogger("livesub")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="livesub", description="即時語音翻譯字幕（英／日／韓 → 繁中）")
    p.add_argument("--config", type=Path, default=None, help="設定檔路徑（預設為 config.toml）")
    p.add_argument("--list-devices", action="store_true", help="列出音訊裝置後結束")
    p.add_argument("--settings", action="store_true", help="只開啟設定視窗（不啟動字幕）")
    p.add_argument("--device-index", type=int, default=None, help="指定音訊裝置編號")
    p.add_argument("--source", choices=["loopback", "mic"], default=None, help="音訊來源")
    p.add_argument("--asr", choices=["cloud", "local"], default=None, help="語音辨識引擎")
    p.add_argument("--model", default=None, help="本機 Whisper 模型名稱，例如 large-v3-turbo、small")
    p.add_argument("--engine", choices=["openai", "claude", "deepl", "none"], default=None, help="翻譯引擎")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def configure_logging(level: str, log_dir: Path, console: bool = True) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
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
    for noisy in ("urllib3", "faster_whisper", "httpx"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def apply_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    """命令列參數覆蓋設定檔。"""
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
    """重新啟動本程式的指令（去掉覆蓋設定的參數與 --settings）。"""
    if getattr(sys, "frozen", False):
        prog, args = [sys.executable], sys.argv[1:]
    elif sys.orig_argv[1:2] == ["-m"]:  # python -m livesub ...
        prog, args = [sys.executable, "-m", sys.orig_argv[2]], sys.orig_argv[3:]
    else:  # python path/to/script.py ...
        prog, args = [sys.executable, *sys.orig_argv[1:2]], sys.orig_argv[2:]
    kept: list[str] = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "--settings" or a.split("=")[0] in _OVERRIDE_FLAGS:
            skip = "=" not in a and a != "--settings"
            continue
        kept.append(a)
    return prog + kept


def run_settings_only(cfg_path: Path) -> int:
    """`--settings`：只開設定視窗，關閉後結束。"""
    from PyQt6.QtWidgets import QApplication

    from .ui.settings import SettingsDialog

    app = QApplication(sys.argv)  # noqa: F841 - 對話框需要 QApplication
    SettingsDialog(cfg_path).exec()
    return 0


def ensure_user_files(base: Path) -> None:
    """第一次執行時，從範例檔複製出可編輯的設定檔與術語表。"""
    for example, target in (("config.example.toml", "config.toml"), ("glossary.example.csv", "glossary.csv")):
        src, dst = base / example, base / target
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)
            log.info("已建立 %s，可依需求修改", dst)


def print_devices() -> int:
    try:
        from .audio import list_devices
    except ImportError:
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
    # 無主控台的 .exe 中 stdout/stderr 為 None，下載模型的進度條會因此當掉
    headless = sys.stderr is None
    if headless:
        sys.stdout = sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    args = parse_args(argv)
    base = app_base_dir()
    configure_logging(args.log_level, base / "logs", console=not headless)
    if args.list_devices:
        return print_devices()

    ensure_user_files(base)
    cfg_path = args.config or (base / "config.toml")
    if args.settings:
        return run_settings_only(cfg_path)
    try:
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
    cfg_base = cfg_path.resolve().parent

    # 延遲載入大型相依套件，讓 --help / --list-devices 不必等待
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from .asr import build_transcriber
    from .audio import make_vad
    from .core.budget import BudgetLedger
    from .core.glossary import build_asr_prompt, load_glossary_safe
    from .output import TranscriptWriter, create_session_dir
    from .pipeline import Pipeline
    from .translation import build_translator
    from .ui.overlay import SubtitleOverlay, TrayController, UiBridge
    from .ui.settings import SettingsDialog

    env = dict(os.environ)
    glossary = load_glossary_safe(resolve_path(cfg_base, cfg.translate.glossary_path))
    ledger = BudgetLedger(cfg.budget, resolve_path(cfg_base, cfg.budget.ledger_path))
    translator = build_translator(cfg.translate, glossary, ledger, env)
    transcriber = build_transcriber(
        cfg.asr, cfg_base, env, ledger,
        prompt=build_asr_prompt(glossary) if cfg.asr.glossary_prompt else None,
    )
    session_dir = create_session_dir(resolve_path(cfg_base, cfg.output.transcript_dir) or base)
    writer = TranscriptWriter(session_dir, cfg.output)
    log.info("語音辨識：%s；翻譯：%s；逐字稿：%s", transcriber.name, translator.name, session_dir)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    bridge = UiBridge()
    overlay = SubtitleOverlay(cfg.overlay)
    bridge.event.connect(overlay.on_event)
    bridge.status.connect(overlay.show_status)

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
            pipeline.stop()
            if QProcess.startDetached(cmd[0], cmd[1:], str(base)):
                app.quit()
            else:
                overlay.show_status("重新啟動失敗，請手動關閉後再開啟")

    def open_settings() -> None:
        dialog = SettingsDialog(cfg_path, on_saved=on_settings_saved)
        dialog.exec()

    tray = TrayController(  # noqa: F841 - 需保留參考
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
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start(250)

    overlay.show()
    overlay.show_status("啟動中…（右鍵或系統匣圖示可開啟選單）")
    pipeline.start()
    code = app.exec()
    pipeline.stop()
    return code


def _fatal_dialog(message: str) -> None:
    """在沒有主控台的 .exe 中也能看到致命錯誤。"""
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        _app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "即時字幕", message)
    except Exception:  # noqa: BLE001, S110
        pass
