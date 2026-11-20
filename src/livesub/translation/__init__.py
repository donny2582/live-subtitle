"""翻譯模組：base（介面與共用工具）＋各服務實作＋registry（依設定建立）。"""

from .base import (
    ChineseNormalizer,
    TranslationError,
    TranslationResult,
    Translator,
    clean_translation,
)
from .passthrough import PassthroughTranslator
from .registry import build_translator

__all__ = [
    "ChineseNormalizer",
    "PassthroughTranslator",
    "TranslationError",
    "TranslationResult",
    "Translator",
    "build_translator",
    "clean_translation",
]
