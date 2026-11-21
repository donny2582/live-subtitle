"""以 VAD 將連續音訊切成一句一句，並在句子進行中定期產生快照（中間結果）。

設計重點：
* WASAPI loopback 在「沒有聲音播放」時完全不送資料，因此不能只靠樣本數計時，
  需要用到達時間偵測斷流並視為靜音。
* 句子過長時改用較短的靜音門檻，盡量在自然停頓處切開，而不是硬切在字中間。

運作方式（狀態機）：

    ┌──────────┐  偵測到人聲   ┌──────────┐  靜音超過 end_silence_ms  ┌────────┐
    │ 等待說話 │ ───────────► │ 句子進行中 │ ──────────────────────► │ 定稿輸出 │
    └──────────┘              └──────────┘   或超過 max_utterance_s  └────────┘
         ▲                         │ 每隔 partial_interval_s                 │
         │                         ▼                                        │
         │                    中間快照輸出                                    │
         └──────────────────────────────────────────────────────────────────┘

音訊以「音框」（frame，預設 30 毫秒）為單位處理，每個音框問 VAD 一次「是不是人聲」。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Protocol

import numpy as np

from ..config import SegmenterConfig
from ..core.models import AudioArray, Utterance

log = logging.getLogger(__name__)

_GAP_TOLERANCE_S = 0.35  # 音訊到達時間比預期晚超過這麼多，就視為「中間斷流」
_SOFT_CUT_RATIO = 0.7  # 句長超過上限的 70% 後，改用較短的靜音門檻
_TAIL_KEEP_MS = 180  # 句尾保留的靜音長度（太短會切到尾音，太長浪費辨識費用）


class VoiceDetector(Protocol):
    """判斷單一音框是否為人聲。

    Protocol 是「介面」：只要有 is_speech() 方法的物件都能當 VAD 使用，
    所以 WebRtcVad、EnergyVad，以及測試用的假 VAD 都可以互換。
    """

    def is_speech(self, frame: AudioArray) -> bool:  # pragma: no cover - 介面
        ...


class EnergyVad:
    """以 RMS 能量判斷的簡易 VAD（webrtcvad 不可用時的備援，也用於測試）。"""

    def __init__(self, threshold_dbfs: float = -45.0) -> None:
        # dBFS 轉成振幅：-45 dBFS ≈ 0.0056（最大音量是 1.0）
        self._threshold = 10 ** (threshold_dbfs / 20)

    def is_speech(self, frame: AudioArray) -> bool:
        # RMS（均方根）＝ 這段聲音的平均音量；夠大聲就當作有人在說話
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))) if frame.size else 0.0
        return rms >= self._threshold


class WebRtcVad:
    """webrtcvad 包裝。只接受 16 kHz、10/20/30 ms 的音框。"""

    def __init__(self, aggressiveness: int, sample_rate: int = 16_000) -> None:
        import webrtcvad  # 延遲載入，讓沒有安裝的環境仍可執行測試

        self._vad = webrtcvad.Vad(aggressiveness)
        self._sr = sample_rate

    def is_speech(self, frame: AudioArray) -> bool:
        # webrtcvad 只吃 16 位元整數 PCM 位元組，先把 -1~1 的浮點數轉回來
        pcm = (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return bool(self._vad.is_speech(pcm, self._sr))


def make_vad(cfg: SegmenterConfig) -> VoiceDetector:
    """建立 VAD；webrtcvad 不存在時退回能量偵測。"""
    try:
        return WebRtcVad(cfg.vad_aggressiveness)
    except ImportError:
        log.warning("找不到 webrtcvad，改用能量式 VAD（準確度較差）")
        return EnergyVad()


class UtteranceSegmenter:
    """將音訊串流切分為 Utterance。

    非執行緒安全：只應由單一執行緒呼叫。
    """

    def __init__(
        self,
        cfg: SegmenterConfig,
        vad: VoiceDetector,
        sample_rate: int = 16_000,
        emit_partials: bool = True,
    ) -> None:
        self._cfg = cfg
        self._vad = vad
        self._sr = sample_rate
        self._frame_len = sample_rate * cfg.frame_ms // 1000
        self._frame_s = cfg.frame_ms / 1000
        self._partial_interval = cfg.partial_interval_s if emit_partials else 0.0
        # 還沒開始說話時，持續保留最近幾個音框；一偵測到人聲就把它們接在句首（避免切掉第一個字）
        self._pre_roll: deque[AudioArray] = deque(maxlen=max(1, cfg.pre_roll_ms // cfg.frame_ms))
        self._pending = np.zeros(0, dtype=np.float32)  # 不滿一個音框的剩餘樣本，等下次湊滿
        self._frames: list[AudioArray] = []  # 目前這句已收集的音框
        self._in_speech = False  # 是否正在一句話當中
        self._speech_frames = 0  # 這句裡被判定為人聲的音框數
        self._silence_run = 0  # 目前連續幾個靜音音框（用來判斷句子結束）
        self._utt_start = 0.0  # 這句開始的時間（秒）
        self._last_partial_frames = 0  # 上次輸出中間快照時，這句有幾個音框
        self._cursor: float | None = None  # 下一個音框開始的時間（秒）
        self._last_audio_at: float | None = None  # 最後收到音訊的時間（偵測斷流用）
        self._seq = 0  # 已定稿的句子數（句子序號）

    # ------------------------------------------------------------------ 公開 API
    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def push(self, samples: AudioArray, end_time: float) -> list[Utterance]:
        """餵入一段音訊。

        Args:
            samples: 16 kHz 單聲道 float32。
            end_time: 這段音訊最後一個樣本的時間（相對工作階段開始，秒）。

        Returns:
            這次產生的 Utterance（可能包含中間快照與定稿）。
        """
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return []
        out: list[Utterance] = []
        # 檢查斷流：這段音訊的開始時間，是否明顯晚於「接續上一段」應有的時間？
        # loopback 在電腦沒播放聲音時不送資料，所以中間的空白要當作靜音處理
        chunk_start = end_time - samples.size / self._sr
        expected = None if self._cursor is None else self._cursor + self._pending.size / self._sr
        if expected is None or chunk_start - expected > _GAP_TOLERANCE_S:
            gap = 0.0 if expected is None else chunk_start - expected
            if self._in_speech and gap * 1000 >= self._cfg.end_silence_ms:
                out.extend(self._finalize(forced=False))
            self._pending = np.zeros(0, dtype=np.float32)
            self._cursor = chunk_start
        self._last_audio_at = end_time

        # 把上次剩下的樣本接上新資料，切成一個個完整音框處理；不滿一框的留到下次
        buf = np.concatenate((self._pending, samples)) if self._pending.size else samples
        n = buf.size // self._frame_len
        for i in range(n):
            frame = buf[i * self._frame_len : (i + 1) * self._frame_len]
            out.extend(self._process_frame(frame))
            self._cursor += self._frame_s  # type: ignore[operator]
        self._pending = buf[n * self._frame_len :].copy()
        return out

    def on_idle(self, now: float) -> list[Utterance]:
        """一段時間沒有音訊時呼叫（loopback 在無聲時不送資料）。"""
        if (
            self._in_speech
            and self._last_audio_at is not None
            and now - self._last_audio_at >= self._cfg.idle_flush_s
        ):
            return self._finalize(forced=False)
        return []

    def flush(self) -> list[Utterance]:
        """強制結束目前句子（例如程式結束、暫停時）。"""
        return self._finalize(forced=False) if self._in_speech else []

    def reset(self) -> None:
        """丟棄所有狀態（暫停時使用）。"""
        self._reset_utterance()
        self._pre_roll.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        self._cursor = None

    # ------------------------------------------------------------------ 內部
    def _process_frame(self, frame: AudioArray) -> list[Utterance]:
        """處理一個音框，必要時輸出中間快照或定稿。"""
        is_speech = self._vad.is_speech(frame)
        if not self._in_speech:
            # 狀態「等待說話」：聽到人聲就開始一句，否則只放進 pre-roll 暫存
            if is_speech:
                self._in_speech = True
                self._frames = list(self._pre_roll)
                self._utt_start = self._cursor - len(self._pre_roll) * self._frame_s  # type: ignore[operator]
                self._pre_roll.clear()
                self._frames.append(frame)
                self._speech_frames = 1
                self._silence_run = 0
                self._last_partial_frames = 0
            else:
                self._pre_roll.append(frame)
            return []

        # 狀態「句子進行中」：收集音框，並計算連續靜音多久
        self._frames.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        duration = len(self._frames) * self._frame_s
        end_silence = self._cfg.end_silence_ms
        # 句子快到上限了：降低靜音門檻，下一個小停頓就切，避免之後被硬切在字中間
        if duration >= self._cfg.max_utterance_s * _SOFT_CUT_RATIO:
            end_silence = min(end_silence, self._cfg.soft_end_silence_ms)
        if self._silence_run * self._cfg.frame_ms >= end_silence:
            return self._finalize(forced=False)  # 停頓夠久 → 正常結束
        if duration >= self._cfg.max_utterance_s:
            return self._finalize(forced=True)  # 一直沒停頓 → 強制切開
        return self._maybe_partial()

    def _maybe_partial(self) -> list[Utterance]:
        """距離上次快照夠久、而且確定有人在說話時，輸出目前整句的快照。"""
        if self._partial_interval <= 0 or self._silence_run > 0:
            return []
        if self._speech_frames * self._cfg.frame_ms < self._cfg.min_speech_ms:
            return []
        new_frames = len(self._frames) - self._last_partial_frames
        if new_frames * self._frame_s < self._partial_interval:
            return []
        self._last_partial_frames = len(self._frames)
        audio = np.concatenate(self._frames)
        return [
            Utterance(
                seq=self._seq + 1,  # 這句還沒定稿，所以是「下一個」序號
                audio=audio,
                started_at=self._utt_start,
                ended_at=self._utt_start + audio.size / self._sr,
                is_final=False,
            )
        ]

    def _finalize(self, forced: bool) -> list[Utterance]:
        """結束目前這句，輸出定稿。人聲太短（咳嗽、雜音）就直接丟掉。"""
        frames = self._frames
        # 去掉句尾多餘的靜音，只留一小段（_TAIL_KEEP_MS）避免切到尾音
        keep_tail = _TAIL_KEEP_MS // self._cfg.frame_ms
        trim = max(0, self._silence_run - keep_tail)
        if trim:
            frames = frames[: len(frames) - trim]
        speech_ms = self._speech_frames * self._cfg.frame_ms
        start = self._utt_start
        self._reset_utterance()
        if speech_ms < self._cfg.min_speech_ms or not frames:
            return []
        self._seq += 1
        audio = np.concatenate(frames)
        return [
            Utterance(
                seq=self._seq,
                audio=audio,
                started_at=start,
                ended_at=start + audio.size / self._sr,
                is_final=True,
                forced_cut=forced,
            )
        ]

    def _reset_utterance(self) -> None:
        """回到「等待說話」狀態（不影響句子序號與時間軸）。"""
        self._in_speech = False
        self._frames = []
        self._speech_frames = 0
        self._silence_run = 0
        self._last_partial_frames = 0
