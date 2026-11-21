"""OpenAI 相容 /chat/completions（OpenAI、Gemini、DeepSeek、本機 Ollama…）。

很多廠商都提供和 OpenAI 一樣格式的 API，所以只要改 base_url 與 model，
同一份程式就能接不同的服務。直接用 HTTP（requests）呼叫，不需要額外安裝 openai 套件。
"""

from __future__ import annotations

import time
from typing import Any

import requests

from ..config import OpenAIConfig
from ..core.budget import BudgetLedger
from ..core.glossary import GlossaryEntry
from ..core.http import bearer_session, post_json
from .base import (
    ChineseNormalizer,
    ContextHistory,
    TranslationError,
    TranslationResult,
    build_system_prompt,
    clean_translation,
    token_cost,
)


class OpenAICompatTranslator:
    """呼叫 OpenAI 相容 API 的翻譯器（實作 base.Translator 介面）。"""

    is_cloud = True

    def __init__(
        self,
        cfg: OpenAIConfig,
        api_key: str,
        glossary: list[GlossaryEntry],
        ledger: BudgetLedger | None,
        normalizer: ChineseNormalizer,
        context_turns: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.name = f"openai:{cfg.model}"
        self._cfg = cfg
        self._glossary = glossary
        self._ledger = ledger
        self._norm = normalizer
        self._url = cfg.base_url.rstrip("/") + "/chat/completions"
        self._session = bearer_session(api_key, session)
        self._history = ContextHistory(context_turns)

    def _messages(self, text: str, source_lang: str, partial: bool) -> list[dict[str, str]]:
        """對話格式：系統提示詞 → 前幾句（原文／譯文）→ 這一句原文。"""
        system = build_system_prompt(text, source_lang, partial, self._glossary)
        return [{"role": "system", "content": system}, *self._history.as_messages(),
                {"role": "user", "content": text}]

    def translate(self, text: str, source_lang: str, partial: bool = False) -> TranslationResult:
        t0 = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": self._messages(text, source_lang, partial),
            "temperature": self._cfg.temperature,
            "max_tokens": max(64, len(text) * 3),  # 譯文長度上限：約原文的 3 倍，至少 64
            "stream": False,  # 一次拿到完整結果（字幕不需要逐字顯示）
        }
        data = post_json(self._session, self._url, json=payload, timeout=self._cfg.timeout_s,
                         retries=self._cfg.max_retries, error_cls=TranslationError, label="翻譯服務")
        try:
            # 回應格式：{"choices": [{"message": {"content": "譯文"}}], "usage": {...}}
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise TranslationError(f"API 回應格式非預期：{str(data)[:200]}") from exc
        out = self._norm(clean_translation(content))
        cost = 0.0
        if self._ledger is not None:
            usage = data.get("usage") or {}
            # 有些相容服務不回報用量，就用字數粗估
            tin = int(usage.get("prompt_tokens") or len(str(payload["messages"])) // 2)
            tout = int(usage.get("completion_tokens") or len(content))
            cost = self._ledger.record(
                token_cost(tin, tout, self._cfg.input_usd_per_million_tokens,
                           self._cfg.output_usd_per_million_tokens),
                input_tokens=tin, output_tokens=tout,
            )
        if not partial and out:
            self._history.add(text, out)
        return TranslationResult(out, time.perf_counter() - t0, cost)

    def warmup(self) -> None:
        """不額外花錢：只檢查設定，真正連線錯誤會在第一句翻譯時回報。"""
        if not self._cfg.model:
            raise TranslationError("translate.openai.model 未設定")
