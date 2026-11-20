"""語音辨識模組：base（介面與語言邏輯）＋ local_whisper／cloud_openai ＋ registry。"""

from .base import AsrError, AsrUnavailableError, LanguageTracker, Transcriber, guess_language
from .registry import build_transcriber

__all__ = [
    "AsrError",
    "AsrUnavailableError",
    "LanguageTracker",
    "Transcriber",
    "build_transcriber",
    "guess_language",
]
