"""本機 faster-whisper 語音辨識（需安裝 requirements-local-asr.txt）。

* 逐句自動偵測語言，並限制在允許清單內（預設 en/ja/ko/zh）。
* GPU 載入失敗時自動改用 CPU 與較小模型。

faster-whisper 是 OpenAI Whisper 模型的加速版，完全在自己的電腦上執行，
不需要網路與金鑰、不花錢，會議內容也不會外傳；缺點是沒有 NVIDIA 顯示卡時會比較慢。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AsrConfig
from ..core.models import AudioArray, Transcription, Utterance
from .base import AsrUnavailableError, LanguageTracker
from .filters import is_hallucination, join_segments

log = logging.getLogger(__name__)


class WhisperTranscriber:
    """包裝 faster-whisper 的 WhisperModel。非執行緒安全：請只在 ASR 執行緒中呼叫。"""

    is_cloud = False

    def __init__(self, cfg: AsrConfig, model_dir: Path | None, prompt: str | None = None) -> None:
        self._common = cfg
        self._cfg = cfg.local
        self._model_dir = model_dir
        self._prompt = prompt or None
        self._model: Any = None  # load() 之後才有值
        self.device: str = "cpu"  # 實際使用的裝置（cuda = NVIDIA 顯示卡）
        self.model_name: str = self._cfg.model  # 實際載入的模型（GPU 失敗時會換成小模型）
        self._lang = LanguageTracker(cfg.allowed_languages, cfg.sticky_language_prob)

    @property
    def name(self) -> str:
        return f"local:{self.model_name}（{self.device}）"

    @property
    def realtime_partials(self) -> bool:
        """只有 GPU 夠快，能每 0.7 秒重新辨識一次產生中間結果。"""
        return self.device == "cuda"

    # ------------------------------------------------------------------ 載入
    def _resolve_device(self) -> tuple[str, str]:
        """決定用 GPU 還是 CPU，以及運算精度（GPU 用 float16、CPU 用 int8 比較快）。"""
        ct = self._cfg.compute_type
        if self._cfg.device in {"auto", "cuda"}:
            try:
                import ctranslate2

                count = ctranslate2.get_cuda_device_count()
            except Exception:  # noqa: BLE001 - 任何錯誤都視為沒有 GPU
                count = 0
            if count > 0:
                return "cuda", ("float16" if ct == "auto" else ct)
            if self._cfg.device == "cuda":
                raise AsrUnavailableError("設定為 cuda，但找不到可用的 NVIDIA GPU")
        return "cpu", ("int8" if ct == "auto" else ct)

    def load(self) -> None:
        """載入模型（第一次執行會自動下載，約 0.5~3 GB）。"""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AsrUnavailableError(
                "未安裝本機辨識套件：請執行 scripts\\setup.ps1 -LocalAsr，或把 asr.engine 改成 cloud"
            ) from exc

        device, compute = self._resolve_device()
        # CPU 跑大模型會跟不上即時，所以沒有 GPU 時改用 cpu_model（預設 small）
        name = self._cfg.model if device == "cuda" else self._cfg.cpu_model
        root = str(self._model_dir) if self._model_dir else None
        try:
            self._model = WhisperModel(name, device=device, compute_type=compute, download_root=root)
            self._warmup()
        except Exception as exc:
            # GPU 載入失敗（常見原因：缺 CUDA DLL、顯示卡記憶體不足）→ 自動退回 CPU
            # 但如果使用者明確指定 cuda，就照實回報錯誤
            if device != "cuda" or self._cfg.device == "cuda":
                raise AsrUnavailableError(f"無法載入 Whisper 模型 {name}：{exc}") from exc
            log.warning("GPU 載入失敗（%s），改用 CPU 與 %s 模型", exc, self._cfg.cpu_model)
            device, compute, name = "cpu", "int8", self._cfg.cpu_model
            self._model = WhisperModel(name, device=device, compute_type=compute, download_root=root)
            self._warmup()
        self.device, self.model_name = device, name
        log.info("Whisper 模型就緒：%s（%s / %s）", name, device, compute)

    def _warmup(self) -> None:
        """先辨識一秒的微弱雜音：讓模型完成初始化，第一句真正的話才不會特別慢。"""
        noise = (np.random.default_rng(0).standard_normal(16_000) * 1e-3).astype(np.float32)
        segments, _ = self._model.transcribe(noise, language="en", beam_size=1, vad_filter=False)
        list(segments)

    # ------------------------------------------------------------------ 辨識
    def _run(self, audio: AudioArray, language: str | None) -> tuple[list[Any], Any]:
        """呼叫 faster-whisper。language=None 代表讓模型自己判斷語言。"""
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=self._cfg.beam_size,
            vad_filter=True,  # 模型內建再過濾一次靜音段
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,  # 不參考前文：避免一句幻覺連帶影響後面
            without_timestamps=True,  # 不需要逐字時間，關掉比較快
            initial_prompt=self._prompt,
            no_speech_threshold=self._cfg.no_speech_threshold,
            log_prob_threshold=self._cfg.log_prob_threshold,
            compression_ratio_threshold=self._cfg.compression_ratio_threshold,
        )
        # segments 是「用到才計算」的產生器，list() 讓它在這裡一次算完
        return list(segments), info

    def _keep(self, seg: Any) -> bool:
        """這個片段可信嗎？不像語音又信心低、或文字重複度太高（幻覺）的片段丟掉。"""
        cfg = self._cfg
        if seg.no_speech_prob > cfg.no_speech_threshold and seg.avg_logprob < cfg.log_prob_threshold:
            return False
        return seg.compression_ratio <= self._cfg.compression_ratio_threshold

    def transcribe(self, utt: Utterance) -> Transcription | None:
        """辨識一段語音。無有效內容（靜音、幻覺、非允許語言）時回傳 None。"""
        if self._model is None:
            raise AsrUnavailableError("模型尚未載入")
        try:
            return self._transcribe(utt)
        finally:
            self._lang.done(utt)

    def _transcribe(self, utt: Utterance) -> Transcription | None:
        t0 = time.perf_counter()
        fixed = self._lang.fixed_for(utt)
        segments, info = self._run(utt.audio, fixed)
        if fixed:
            lang, prob = fixed, 1.0
        else:
            probs = dict(getattr(info, "all_language_probs", None) or ())
            lang, prob = self._lang.choose(info.language, float(info.language_probability), probs,
                                           self._cfg.min_allowed_language_prob)
            if lang is None:
                log.debug("seq=%d 非允許語言，略過", utt.seq)
                return None
            if lang != info.language:
                # 最後採用的語言和模型判斷的不同（例如改用上一句的語言）→ 指定語言重新辨識
                segments, _ = self._run(utt.audio, lang)

        text = join_segments([s.text for s in segments if self._keep(s)], lang)
        if is_hallucination(text):
            return None
        self._lang.commit(utt, lang)
        return Transcription(utt.seq, utt.is_final, utt.started_at, utt.ended_at, lang, prob, text,
                             time.perf_counter() - t0)
