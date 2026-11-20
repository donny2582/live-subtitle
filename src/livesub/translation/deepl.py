"""DeepL API（Free 金鑰以 :fx 結尾，自動使用 api-free 網域）。

註：DeepL 的術語表需事先在 DeepL 端建立，本程式的 glossary.csv 僅對 LLM 引擎生效。
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import requests

from ..config import DeepLConfig
from ..core.budget import BudgetLedger
from .base import ChineseNormalizer, TranslationError, TranslationResult

_DEEPL_SOURCE = {"en": "EN", "ja": "JA", "ko": "KO", "zh": "ZH"}


class DeepLTranslator:
    is_cloud = True
    name = "deepl"

    def __init__(
        self,
        cfg: DeepLConfig,
        api_key: str,
        ledger: BudgetLedger | None,
        normalizer: ChineseNormalizer,
        context_turns: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        host = "https://api-free.deepl.com" if api_key.endswith(":fx") else "https://api.deepl.com"
        self._url = f"{host}/v2/translate"
        self._cfg = cfg
        self._ledger = ledger
        self._norm = normalizer
        self._session = session or requests.Session()
        self._session.headers.update({"Authorization": f"DeepL-Auth-Key {api_key}"})
        self._history: deque[str] = deque(maxlen=max(1, context_turns))

    def translate(self, text: str, source_lang: str, partial: bool = False) -> TranslationResult:
        t0 = time.perf_counter()
        body: dict[str, Any] = {"text": [text], "target_lang": "ZH-HANT"}
        if source_lang in _DEEPL_SOURCE:
            body["source_lang"] = _DEEPL_SOURCE[source_lang]
        if self._history:
            body["context"] = " ".join(self._history)  # context 不計費
        try:
            resp = self._session.post(self._url, json=body, timeout=self._cfg.timeout_s)
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise TranslationError(f"DeepL 無回應：{exc}") from exc
        if resp.status_code == 456:
            raise TranslationError("DeepL 本月額度已用完")
        if resp.status_code in (401, 403):
            raise TranslationError("DeepL 金鑰無效")
        if resp.status_code >= 400:
            raise TranslationError(f"DeepL HTTP {resp.status_code}")
        try:
            out = resp.json()["translations"][0]["text"]
        except (ValueError, KeyError, IndexError) as exc:
            raise TranslationError("DeepL 回應格式非預期") from exc
        cost = 0.0
        if self._ledger is not None:
            cost = self._ledger.record(len(text) * self._cfg.usd_per_million_chars / 1_000_000,
                                       chars=len(text))
        if not partial:
            self._history.append(text)
        return TranslationResult(self._norm(out.strip()), time.perf_counter() - t0, cost)

    def warmup(self) -> None:
        return None
