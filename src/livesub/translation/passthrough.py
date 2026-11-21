"""不翻譯（engine = none，或來源本身就是中文）。

也是「備援」：選了某個翻譯服務但缺金鑰時，registry 會改用這個，字幕只顯示原文而不會當掉。
"""

from __future__ import annotations

from .base import ChineseNormalizer, TranslationResult


class PassthroughTranslator:
    """中文：只做繁體／台灣用語轉換；其他語言：回傳空字串（畫面只顯示原文）。"""

    name = "none"
    is_cloud = False

    def __init__(self, normalizer: ChineseNormalizer | None = None) -> None:
        self._norm = normalizer or ChineseNormalizer(False)

    def translate(self, text: str, source_lang: str, partial: bool = False) -> TranslationResult:
        return TranslationResult(self._norm(text) if source_lang == "zh" else "", 0.0)

    def warmup(self) -> None:
        return None
