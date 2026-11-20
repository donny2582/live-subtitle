"""管線中傳遞的資料結構。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

AudioArray = npt.NDArray[np.float32]


class Stage(str, Enum):
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
        return self.audio.size / 16_000


@dataclass(frozen=True, slots=True)
class Transcription:
    """ASR 結果。"""

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
    """送往畫面與逐字稿的字幕事件。"""

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
