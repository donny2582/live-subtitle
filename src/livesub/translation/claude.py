"""Anthropic Claude 翻譯（官方 anthropic SDK）。

* 思考（thinking）在 Opus 5.5 無法關閉，以 effort = low 讓它最快、最省；
  Haiku 5.5 可設 thinking = "off" 完全關閉。
* 支援伺服器端備援（refusal fallback）的模型會自動帶上 `fallbacks: "default"`：
  主模型拒答時，同一次請求內改由備援模型接手。

Claude 不提供語音辨識，只負責翻譯；語音辨識要另外選 Groq、OpenAI 或本機 Whisper。
金鑰：ANTHROPIC_API_KEY（可在設定視窗的「API 金鑰」分頁填入）。
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anthropic

from ..config import ClaudeConfig
from ..core.budget import BudgetLedger
from ..core.glossary import GlossaryEntry
from .base import (
    ChineseNormalizer,
    ContextHistory,
    TranslationError,
    TranslationResult,
    build_system_prompt,
    clean_translation,
    token_cost,
)

log = logging.getLogger(__name__)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# 支援伺服器端 fallbacks 參數的模型（其他模型送出會被拒絕）
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5")


def supports_server_fallback(model: str) -> bool:
    """這個模型能不能使用伺服器端備援（例如 Haiku 不支援，送出會被拒絕）。"""
    return model.startswith(_FALLBACK_MODELS)


class ClaudeTranslator:
    """用 Anthropic 官方 SDK 呼叫 Claude 翻譯（實作 base.Translator 介面）。"""

    is_cloud = True

    def __init__(
        self,
        cfg: ClaudeConfig,
        glossary: list[GlossaryEntry],
        ledger: BudgetLedger | None,
        normalizer: ChineseNormalizer,
        context_turns: int = 3,
        client: Any = None,
        api_key: str = "",
    ) -> None:
        self.name = f"claude:{cfg.model}"
        self._cfg = cfg
        self._glossary = glossary
        self._ledger = ledger
        self._norm = normalizer
        self._history = ContextHistory(context_turns)
        # client 參數是給測試用的（傳入假的 client，不必真的連網）
        if client is None:
            kwargs: dict[str, Any] = {"timeout": cfg.timeout_s, "max_retries": cfg.max_retries}
            if api_key:
                kwargs["api_key"] = api_key  # 沒給就由 SDK 依環境／ant auth login 解析
            client = anthropic.Anthropic(**kwargs)
        self._client = client
        self._fallback = cfg.refusal_fallback and supports_server_fallback(cfg.model)

    # ------------------------------------------------------------------ 請求
    def _params(self, text: str, source_lang: str, partial: bool) -> dict[str, Any]:
        """組出 API 請求內容。"""
        params: dict[str, Any] = {
            "model": self._cfg.model,
            "max_tokens": self._cfg.max_tokens,  # 回覆長度上限（含思考）
            "system": build_system_prompt(text, source_lang, partial, self._glossary),
            # 前幾句的原文／譯文 ＋ 這一句原文
            "messages": [*self._history.as_messages(), {"role": "user", "content": text}],
            # effort：模型要多認真思考；字幕翻譯用 low 最快最省
            "output_config": {"effort": self._cfg.effort},
        }
        if self._cfg.thinking == "off":
            params["thinking"] = {"type": "disabled"}
        return params

    def _create(self, params: dict[str, Any]) -> Any:
        """送出請求，並把 SDK 的各種錯誤轉成使用者看得懂的 TranslationError。"""
        try:
            if self._fallback:
                return self._client.beta.messages.create(
                    betas=[_FALLBACK_BETA], fallbacks="default", **params
                )
            return self._client.messages.create(**params)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise TranslationError("Claude API 金鑰無效或沒有權限（請設定 ANTHROPIC_API_KEY）") from exc
        except anthropic.RateLimitError as exc:
            raise TranslationError("Claude 請求過於頻繁（429），請稍後再試") from exc
        except anthropic.BadRequestError as exc:
            raise TranslationError(f"Claude 請求參數錯誤：{exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise TranslationError(f"Claude HTTP {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:  # 含逾時
            raise TranslationError(f"Claude 無回應：{exc}") from exc

    def translate(self, text: str, source_lang: str, partial: bool = False) -> TranslationResult:
        t0 = time.perf_counter()
        resp = self._create(self._params(text, source_lang, partial))
        cost = self._record_usage(resp)  # 拒答也會計費，所以先記帳
        # stop_reason 說明模型為什麼停止輸出；refusal = 安全機制拒絕回答
        if resp.stop_reason == "refusal":
            raise TranslationError("Claude 拒絕翻譯這一句")
        # 回覆由多個區塊組成（思考區塊、文字區塊…），只取文字區塊
        content = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if not content and resp.stop_reason == "max_tokens":
            raise TranslationError("Claude 輸出被 max_tokens 截斷，請調高 translate.claude.max_tokens")
        out = self._norm(clean_translation(content))
        if not partial and out:  # 只有定稿才當作之後的上下文
            self._history.add(text, out)
        return TranslationResult(out, time.perf_counter() - t0, cost)

    def _record_usage(self, resp: Any) -> float:
        """依回應裡的 token 用量計算費用並記帳。"""
        if self._ledger is None:
            return 0.0
        usage = getattr(resp, "usage", None)
        tin = int(getattr(usage, "input_tokens", 0) or 0)
        tout = int(getattr(usage, "output_tokens", 0) or 0)  # 含思考 token
        cost = token_cost(tin, tout, self._cfg.input_usd_per_million_tokens,
                          self._cfg.output_usd_per_million_tokens)
        return self._ledger.record(cost, input_tokens=tin, output_tokens=tout)

    def warmup(self) -> None:
        """不額外花錢：只檢查設定，真正連線錯誤會在第一句翻譯時回報。"""
        if not self._cfg.model:
            raise TranslationError("translate.claude.model 未設定")
