"""雲端用量（語音辨識＋翻譯）與每月預算控管。

各雲端服務自行依單價算出花費後呼叫 `record()`；帳本只負責累計與判斷上限。
用量以月份（本地時間 YYYY-MM）為單位寫入 JSON，跨工作階段累計。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from ..config import BudgetConfig

log = logging.getLogger(__name__)


@dataclass(slots=True)
class MonthUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    chars: int = 0
    requests: int = 0
    cost_usd: float = 0.0  # 合計
    audio_seconds: float = 0.0
    asr_cost_usd: float = 0.0


class BudgetLedger:
    """執行緒安全的用量帳本。"""

    def __init__(
        self,
        cfg: BudgetConfig,
        path: Path | None,
        month_fn: Callable[[], str] | None = None,
        save_interval_s: float = 10.0,
    ) -> None:
        self._cfg = cfg
        self._path = path
        self._month_fn = month_fn or (lambda: datetime.now().strftime("%Y-%m"))
        self._save_interval = save_interval_s
        self._lock = threading.Lock()
        self._data: dict[str, MonthUsage] = {}
        self._dirty = False
        self._last_save = 0.0
        self._load()

    # ------------------------------------------------------------------ 持久化
    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._data = {k: MonthUsage(**v) for k, v in raw.items()}
        except (OSError, ValueError, TypeError) as exc:
            backup = self._path.with_suffix(".corrupt.json")
            log.error("用量檔損毀（%s），已備份到 %s 並重新計算", exc, backup)
            try:
                self._path.replace(backup)
            except OSError:
                pass
            self._data = {}

    def save(self, force: bool = True) -> None:
        """寫入磁碟（先寫暫存檔再取代，避免當機時檔案損毀）。"""
        with self._lock:
            if not self._path or not self._dirty:
                return
            if not force and time.monotonic() - self._last_save < self._save_interval:
                return
            payload = {k: asdict(v) for k, v in self._data.items()}
            self._dirty = False
            self._last_save = time.monotonic()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as exc:
            log.error("無法寫入用量檔 %s：%s", self._path, exc)

    # ------------------------------------------------------------------ 計量
    def _current(self) -> MonthUsage:
        return self._data.setdefault(self._month_fn(), MonthUsage())

    def record(
        self,
        cost_usd: float,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        chars: int = 0,
        audio_s: float = 0.0,
    ) -> float:
        """記錄一次雲端請求，回傳該次花費（美元）。audio_s > 0 視為語音辨識費用。"""
        with self._lock:
            u = self._current()
            u.input_tokens += input_tokens
            u.output_tokens += output_tokens
            u.chars += chars
            u.audio_seconds += audio_s
            u.requests += 1
            u.cost_usd += cost_usd
            if audio_s:
                u.asr_cost_usd += cost_usd
            self._dirty = True
        self.save(force=False)
        return cost_usd

    def month_usage(self) -> MonthUsage:
        with self._lock:
            u = self._current()
            return MonthUsage(**asdict(u))

    def allows(self, partial: bool) -> bool:
        """是否還能呼叫雲端 API。中間結果翻譯在較低門檻就停止。"""
        limit = self._cfg.monthly_limit_usd
        if limit <= 0:
            return True  # 0 代表不設上限
        with self._lock:
            spent = self._current().cost_usd
        threshold = limit * (self._cfg.partial_cutoff_ratio if partial else 1.0)
        return spent < threshold

    def summary(self) -> str:
        u = self.month_usage()
        limit = self._cfg.monthly_limit_usd
        cap = f" / ${limit:.2f}" if limit > 0 else ""
        return f"本月雲端花費 ${u.cost_usd:.3f}{cap}（辨識 ${u.asr_cost_usd:.3f}，{u.requests} 次請求）"
