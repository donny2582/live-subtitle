"""依 translate.engine 建立翻譯器。新增服務時只要在 _BUILDERS 加一行。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from ..config import TranslateConfig, get_secret, is_local_url
from ..core.budget import BudgetLedger
from ..core.glossary import GlossaryEntry
from .base import ChineseNormalizer, Translator
from .passthrough import PassthroughTranslator

log = logging.getLogger(__name__)

Builder = Callable[
    [TranslateConfig, list[GlossaryEntry], "BudgetLedger | None", Mapping[str, str], ChineseNormalizer],
    "Translator | None",
]


def _openai(cfg, glossary, ledger, env, norm):  # type: ignore[no-untyped-def]
    from .openai_compat import OpenAICompatTranslator

    sub = cfg.openai
    local = is_local_url(sub.base_url)
    key = get_secret(env, sub.api_key_env)
    if not key and not local:
        log.error("找不到環境變數 %s，改為只顯示原文", sub.api_key_env)
        return None
    tr = OpenAICompatTranslator(sub, key or "local", glossary, None if local else ledger, norm,
                                context_turns=cfg.context_turns)
    tr.is_cloud = not local
    return tr


def _claude(cfg, glossary, ledger, env, norm):  # type: ignore[no-untyped-def]
    from .claude import ClaudeTranslator

    key = get_secret(env, cfg.claude.api_key_env)
    if not key:
        log.info("未設定 %s，改由 anthropic SDK 自動尋找憑證（ant auth login 等）", cfg.claude.api_key_env)
    try:
        return ClaudeTranslator(cfg.claude, glossary, ledger, norm,
                                context_turns=cfg.context_turns, api_key=key)
    except Exception as exc:  # noqa: BLE001 - 憑證缺失等，退回只顯示原文
        log.error("無法建立 Claude 翻譯器（%s），改為只顯示原文", exc)
        return None


def _deepl(cfg, glossary, ledger, env, norm):  # type: ignore[no-untyped-def]
    from .deepl import DeepLTranslator

    key = get_secret(env, cfg.deepl.api_key_env)
    if not key:
        log.error("找不到環境變數 %s，改為只顯示原文", cfg.deepl.api_key_env)
        return None
    return DeepLTranslator(cfg.deepl, key, ledger, norm, context_turns=cfg.context_turns)


_BUILDERS: dict[str, Builder] = {
    "openai": _openai,
    "claude": _claude,
    "deepl": _deepl,
}


def build_translator(
    cfg: TranslateConfig,
    glossary: list[GlossaryEntry],
    ledger: BudgetLedger | None,
    env: Mapping[str, str],
) -> Translator:
    """依設定建立翻譯器。缺金鑰或建立失敗時退回不翻譯（只顯示原文）。"""
    norm = ChineseNormalizer(cfg.taiwan_convert)
    builder = _BUILDERS.get(cfg.engine)
    translator = builder(cfg, glossary, ledger, env, norm) if builder else None
    return translator or PassthroughTranslator(norm)
