"""管線中傳遞的資料結構。

一句話在程式裡會依序變成三種資料：

    Utterance（一段音訊） ──辨識──► Transcription（原文文字） ──翻譯──► SubtitleEvent（字幕）

全部使用 frozen dataclass（建立後不可修改），因為它們會在不同執行緒之間傳遞，
不可修改就不必擔心兩個執行緒同時改到同一筆資料。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

# 整個程式統一使用的音訊格式：16 kHz、單聲道、float32，數值範圍 -1.0 ~ 1.0
AudioArray = npt.NDArray[np.float32]


class Stage(StrEnum):
    """字幕事件的階段。"""

    PARTIAL_SOURCE = "partial_source"  # 句子還在講，原文中間結果
    PARTIAL_TRANSLATION = "partial_translation"  # 中間結果的暫時譯文
    FINAL = "final"  # 句子結束後的定稿（只有這個會寫入逐字稿）


@dataclass(frozen=True, slots=True)
class Utterance:
    """一段語音（一句話，或句子進行中的快照）。

    Attributes:
        seq: 句子序號，同一句的中間快照與定稿共用序號。
        audio: 16 kHz 單聲道 float32 音訊。
        started_at: 相對於工作階段開始的秒數。
        ended_at: 相對於工作階段開始的秒數。
        is_final: False 代表句子還在進行中的快照。
        forced_cut: 是否因超過最長句長而被強制切斷。
    """

    seq: int
    audio: AudioArray
    started_at: float
    ended_at: float
    is_final: bool
    forced_cut: bool = False

    @property
    def duration(self) -> float:
        """音訊長度（秒）＝ 樣本數 ÷ 每秒 16000 個樣本。"""
        return self.audio.size / 16_000


@dataclass(frozen=True, slots=True)
class Transcription:
    """ASR（語音辨識）結果：把一段 Utterance 聽寫成文字。

    Attributes:
        language: 偵測到的語言代碼（en / ja / ko / zh）。
        language_prob: 語言判斷的信心（0~1），太低時會沿用上一句的語言。
        asr_latency_s: 辨識花了幾秒，用來在 log 中觀察延遲。
    """

    seq: int
    is_final: bool
    started_at: float
    ended_at: float
    language: str
    language_prob: float
    text: str
    asr_latency_s: float


@dataclass(frozen=True, slots=True)
class SubtitleEvent:
    """送往畫面與逐字稿的字幕事件。

    同一句話可能先後產生多個事件（中間原文 → 中間譯文 → 定稿），
    畫面用 seq 判斷是哪一句，新的事件會覆蓋同一句舊的顯示。
    error 有值時代表翻譯失敗，畫面只顯示原文。
    """

    seq: int
    stage: Stage
    started_at: float
    ended_at: float
    language: str
    source_text: str
    translated_text: str = ""
    asr_latency_s: float = 0.0
    translate_latency_s: float = 0.0
    error: str | None = None
