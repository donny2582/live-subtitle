"""不翻譯（engine = none，或來源本身就是中文）。"""

from __future__ import annotations

from .base import ChineseNormalizer, TranslationResult


class PassthroughTranslator:
    name = "none"
    is_cloud = False

    def __init__(self, normalizer: ChineseNormalizer | None = None) -> None:
        self._norm = normalizer or ChineseNormalizer(False)

    def translate(self, text: str, source_lang: str, partial: bool = False) -> TranslationResult:
        return TranslationResult(self._norm(text) if source_lang == "zh" else "", 0.0)

    def warmup(self) -> None:
        return None
