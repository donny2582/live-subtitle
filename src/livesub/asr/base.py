"""語音辨識共用：介面、錯誤、語言判斷與「黏著」邏輯。

新增語音辨識服務的步驟：
1. 在 config/schema.py 加一個子區段（例如 [asr.myprovider]）。
2. 在本資料夾新增 myprovider.py，實作 `Transcriber` 介面（可重用 LanguageTracker）。
3. 在 registry.py 註冊建立函式。

ASR = Automatic Speech Recognition（自動語音辨識），也就是把聲音聽寫成文字。

「黏著」（sticky）是什麼？
    很短的句子（例如 "OK"、"はい"）很難判斷是哪種語言，判斷錯了翻譯就會怪。
    所以信心不夠時，就沿用上一句的語言——會議中大家通常不會每句換一種語言。
"""

from __future__ import annotations

import re
from typing import Protocol

from ..core.http import CloudError
from ..core.models import Transcription, Utterance


class AsrUnavailableError(RuntimeError):
    """辨識引擎無法啟動（模型載入失敗、缺金鑰等）。"""


class AsrError(CloudError):
    """單句辨識失敗（網路逾時等），不影響後續句子。"""


class Transcriber(Protocol):
    """所有語音辨識器（雲端、本機）都要提供的介面，pipeline 只透過這些方法使用它們。"""

    name: str  # 顯示在 log 裡的名稱，例如 "cloud:whisper-large-v3-turbo"
    is_cloud: bool  # 是否會花錢（要不要經過預算檢查）

    @property
    def realtime_partials(self) -> bool:
        """是否快到能每 0.7 秒重新辨識一次（partial_results = auto 時參考）。"""
        ...

    def load(self) -> None:
        """啟動前的準備（載入模型、檢查金鑰）；失敗丟出 AsrUnavailableError。"""
        ...

    def transcribe(self, utt: Utterance) -> Transcription | None:
        """辨識一段音訊；沒有內容、不是允許的語言或是幻覺時回傳 None。"""
        ...


# ---------------------------------------------------------------------- 語言
# 用 Unicode 字元範圍辨認文字系統
_KANA = re.compile(r"[぀-ヿㇰ-ㇿ]")
_HANGUL = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")
_HAN = re.compile(r"[一-鿿㐀-䶿]")
_LATIN = re.compile(r"[A-Za-z]")

# Whisper verbose_json 回傳的是語言全名
WHISPER_NAMES = {
    "english": "en", "japanese": "ja", "korean": "ko", "chinese": "zh",
    "mandarin": "zh", "cantonese": "zh", "yue": "zh",
}


def normalize_language(value: str | None) -> str | None:
    """'English' / 'en' / 'zh-TW' → 'en' / 'en' / 'zh'。"""
    if not value:
        return None
    v = value.strip().lower()
    head = v.split("-")[0]
    # 未列出的語言（例如 'french'）原樣回傳，之後會因不在允許清單而略過
    return WHISPER_NAMES.get(v) or (head if len(head) == 2 else v)


def guess_language(text: str) -> str | None:
    """以文字系統判斷語言：假名→日、諺文→韓、漢字→中、拉丁字母→英。

    順序很重要：日文同時有假名和漢字，所以要先檢查假名，才不會被當成中文。
    """
    if _KANA.search(text):
        return "ja"
    if _HANGUL.search(text):
        return "ko"
    if _HAN.search(text):
        return "zh"
    if _LATIN.search(text):
        return "en"
    return None


class LanguageTracker:
    """限制在允許語言內，短句／低信心沿用上一句語言；同一句的中間結果固定語言。"""

    def __init__(self, allowed: tuple[str, ...], sticky_prob: float) -> None:
        self.allowed = allowed  # 使用者勾選的語言
        self.sticky_prob = sticky_prob  # 信心低於這個值就沿用上一句的語言
        self.last: str | None = None  # 上一句定稿的語言
        self._partial: dict[int, str] = {}  # 進行中句子（seq）第一次判斷出的語言

    def fixed_for(self, utt: Utterance) -> str | None:
        """中間結果沿用該句第一次偵測到的語言，避免字幕語言來回跳動。"""
        return None if utt.is_final else self._partial.get(utt.seq)

    def choose(
        self,
        lang: str | None,
        prob: float,
        probs: dict[str, float] | None = None,
        min_allowed_prob: float = 0.0,
    ) -> tuple[str | None, float]:
        """回傳 (語言, 信心)。語言為 None 代表不屬於任何允許語言，應略過。"""
        probs = probs or {}
        if self.allowed and lang not in self.allowed:
            # 偵測到的語言沒勾選：有各語言機率可參考時（本機 Whisper），改選允許語言中機率最高的
            if not probs:
                return None, prob
            best = max(self.allowed, key=lambda c: probs.get(c, 0.0))
            best_p = probs.get(best, 0.0)
            if best_p < min_allowed_prob:
                return None, best_p
            lang, prob = best, best_p
        # 黏著：信心不足時沿用上一句的語言
        if prob < self.sticky_prob and self.last and self.last != lang:
            return self.last, probs.get(self.last, prob)
        return lang, prob

    def commit(self, utt: Utterance, lang: str) -> None:
        """確認這段音訊用的語言：定稿記為「上一句語言」；中間結果記為該句的固定語言。"""
        if utt.is_final:
            self.last = lang
        else:
            self._partial.setdefault(utt.seq, lang)

    def done(self, utt: Utterance) -> None:
        """一句定稿後，清掉它的中間結果紀錄。"""
        if utt.is_final:
            self._partial.pop(utt.seq, None)
            if len(self._partial) > 32:  # 防止極端情況下無限成長
                self._partial.clear()
