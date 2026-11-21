"""逐字稿輸出：SRT（字幕檔）、TXT（易讀版）、JSONL（後續分析用）。

每寫一句就 flush，程式意外關閉也不會遺失已寫入的內容。

三種檔案的樣子：

subtitles.srt（標準字幕檔，播放器可載入）：
    1
    00:00:03,120 --> 00:00:05,480
    歡迎參加今天的會議。
    Welcome to today's meeting.

transcript.txt（給人看）：
    [00:00:03] [英] Welcome to today's meeting.
                   歡迎參加今天的會議。

transcript.jsonl（給程式分析，一行一句 JSON）：
    {"seq": 1, "stage": "final", "language": "en", "source_text": "...", ...}
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from ..config import OutputConfig
from ..core.models import Stage, SubtitleEvent

log = logging.getLogger(__name__)

_LANG_LABEL = {"en": "英", "ja": "日", "ko": "韓", "zh": "中"}
_MIN_SRT_DURATION_S = 0.8  # 每句字幕至少顯示這麼久，太短的句子才看得到


def format_srt_time(seconds: float) -> str:
    """秒數轉 SRT 時間格式 HH:MM:SS,mmm。"""
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_clock(seconds: float) -> str:
    """秒數轉 HH:MM:SS。"""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def create_session_dir(root: Path, now: datetime | None = None) -> Path:
    """建立本次工作階段資料夾，例如 transcripts/2026-11-20_104800。"""
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    # 同一秒內啟動兩次時，資料夾名稱加上 _1、_2… 避免撞名
    for i in range(100):
        candidate = root / (stamp if i == 0 else f"{stamp}_{i}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"無法在 {root} 建立工作階段資料夾")


class TranscriptWriter:
    """寫出定稿字幕。只應由單一執行緒呼叫 write。"""

    def __init__(self, session_dir: Path, cfg: OutputConfig, started: datetime | None = None) -> None:
        self.session_dir = session_dir
        self._cfg = cfg
        self._srt_index = 0  # SRT 字幕的編號（從 1 開始）
        self._files: list[IO[str]] = []  # 所有開啟的檔案，close() 時一起關閉
        started = started or datetime.now()
        # utf-8-sig 會在檔頭加 BOM，記事本與舊版播放器才不會把中文顯示成亂碼
        self._srt = self._open("subtitles.srt", "utf-8-sig") if cfg.write_srt else None
        self._txt = self._open("transcript.txt", "utf-8-sig") if cfg.write_txt else None
        self._jsonl = self._open("transcript.jsonl", "utf-8") if cfg.write_jsonl else None
        if self._txt:
            self._txt.write(f"# 工作階段開始：{started:%Y-%m-%d %H:%M:%S}\n\n")
            self._txt.flush()

    def _open(self, name: str, encoding: str) -> IO[str]:
        fh = (self.session_dir / name).open("w", encoding=encoding, newline="\n")
        self._files.append(fh)
        return fh

    def write(self, evt: SubtitleEvent) -> None:
        """寫入一句定稿。非定稿或空白內容會被忽略。"""
        if evt.stage is not Stage.FINAL or not evt.source_text:
            return
        target = evt.translated_text or ""
        try:
            # ---- SRT：編號、時間軸、譯文（＋原文）----
            if self._srt:
                self._srt_index += 1
                end = max(evt.ended_at, evt.started_at + _MIN_SRT_DURATION_S)
                lines = [target or evt.source_text]
                if self._cfg.srt_include_source and target and evt.language != "zh":
                    lines.append(evt.source_text)
                self._srt.write(
                    f"{self._srt_index}\n{format_srt_time(evt.started_at)} --> "
                    f"{format_srt_time(end)}\n" + "\n".join(lines) + "\n\n"
                )
                self._srt.flush()
            # ---- TXT：[時間] [語言] 原文，下一行縮排放譯文 ----
            if self._txt:
                label = _LANG_LABEL.get(evt.language, evt.language)
                indent = " " * 15
                self._txt.write(f"[{format_clock(evt.started_at)}] [{label}] {evt.source_text}\n")
                if target and evt.language != "zh":
                    self._txt.write(f"{indent}{target}\n")
                if evt.error:
                    self._txt.write(f"{indent}（翻譯失敗：{evt.error}）\n")
                self._txt.write("\n")
                self._txt.flush()
            # ---- JSONL：整筆事件轉成 JSON，時間取到毫秒 ----
            if self._jsonl:
                record: dict[str, Any] = asdict(evt)
                record["stage"] = evt.stage.value  # Enum 不能直接轉 JSON，改存字串
                for k in ("started_at", "ended_at", "asr_latency_s", "translate_latency_s"):
                    record[k] = round(float(record[k]), 3)
                self._jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._jsonl.flush()
        except OSError as exc:
            log.error("寫入逐字稿失敗：%s", exc)

    def close(self) -> None:
        """關閉所有檔案（可重複呼叫）。"""
        for fh in self._files:
            try:
                fh.close()
            except OSError:
                pass
        self._files.clear()

    # 讓它可以用 `with TranscriptWriter(...) as w:` 的寫法，離開 with 區塊時自動關檔
    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
