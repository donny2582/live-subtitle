"""雲端語音辨識：OpenAI 相容 /audio/transcriptions。

同一套程式可接：
* OpenAI      base_url = https://api.openai.com/v1
              model = gpt-4o-mini-transcribe / gpt-4o-transcribe / whisper-1
* Groq        base_url = https://api.groq.com/openai/v1，model = whisper-large-v3-turbo
* 自架伺服器  任何實作同一 API 的服務（例如 faster-whisper-server）

每句音訊以 16 kHz 單聲道 WAV 上傳；whisper 系列模型使用 verbose_json 取得語言與
no_speech_prob，其他模型（gpt-4o-*-transcribe）只回傳文字，語言由文字系統判斷。

一句話的處理流程：
    音訊 → 轉成 WAV 檔 → HTTP POST 上傳 → 收到 JSON → 記帳 → 判斷語言 → 過濾幻覺 → Transcription
"""

from __future__ import annotations

import io
import logging
import time
import wave
from typing import Any

import numpy as np
import requests

from ..config import AsrConfig, is_local_url
from ..core.budget import BudgetLedger
from ..core.http import bearer_session, post_json
from ..core.models import AudioArray, Transcription, Utterance
from .base import AsrError, AsrUnavailableError, LanguageTracker, guess_language, normalize_language
from .filters import is_hallucination, join_segments, normalize

log = logging.getLogger(__name__)

# 片段同時「很可能不是語音」又「信心很低」時丟棄（只有 verbose_json 有這些數值）
_NO_SPEECH_PROB = 0.6
_MIN_LOGPROB = -1.0
_SHORT_TEXT_CHARS = 3  # 這麼短的句子語言判斷不可靠，交給「沿用上一句語言」處理


def to_wav_bytes(audio: AudioArray, sample_rate: int = 16_000) -> bytes:
    """float32 [-1, 1] → 16-bit PCM WAV。

    API 只接受音訊「檔案」，所以在記憶體裡（BytesIO）組出一個 WAV 檔，不寫到硬碟。
    """
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)  # 單聲道
        w.setsampwidth(2)  # 每個樣本 2 位元組（16 位元）
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class CloudTranscriber:
    """呼叫雲端 API 的語音辨識器（實作 base.Transcriber 介面）。"""

    is_cloud = True
    realtime_partials = False  # 每次中間結果都要重送整段音訊，費用與延遲都不划算

    def __init__(
        self,
        cfg: AsrConfig,
        api_key: str,
        ledger: BudgetLedger | None,
        prompt: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._cfg = cfg.cloud
        self.name = f"cloud:{self._cfg.model}"
        self._api_key = api_key
        # 網址是 localhost 時代表自架伺服器：不需要金鑰、不花錢、不記帳
        self._local = is_local_url(self._cfg.base_url)
        self.is_cloud = not self._local
        self._ledger = None if self._local else ledger
        self._prompt = prompt or None
        self._url = self._cfg.base_url.rstrip("/") + "/audio/transcriptions"
        self._session = bearer_session(api_key, session)  # 每次請求自動帶上金鑰的連線
        self._lang = LanguageTracker(cfg.allowed_languages, cfg.sticky_language_prob)
        # verbose_json 會多回傳語言與每段的信心值，但只有 whisper 系列模型支援
        fmt = self._cfg.response_format
        if fmt == "auto":
            fmt = "verbose_json" if "whisper" in self._cfg.model.lower() else "json"
        self._format = fmt

    def load(self) -> None:
        if not self._api_key and not self._local:
            raise AsrUnavailableError(
                f"找不到環境變數 {self._cfg.api_key_env}（雲端語音辨識需要 API 金鑰）"
            )
        log.info("雲端語音辨識：%s（%s）", self._cfg.model, self._cfg.base_url)

    # ------------------------------------------------------------------ 請求
    def _request(self, audio: AudioArray, language: str | None) -> dict[str, Any]:
        """上傳音訊並取得 JSON 結果（以 multipart 表單送出，和網頁上傳檔案的格式一樣）。"""
        form: dict[str, str] = {
            "model": self._cfg.model,
            "response_format": self._format,
            "temperature": "0",  # 0 = 結果最穩定，不要有隨機性
        }
        if language:  # 指定語言（同一句的中間結果固定用同一語言）
            form["language"] = language
        if self._prompt:  # 術語提示詞，幫助專有名詞聽得更準
            form["prompt"] = self._prompt
        files = {"file": ("speech.wav", to_wav_bytes(audio), "audio/wav")}
        return post_json(self._session, self._url, data=form, files=files, timeout=self._cfg.timeout_s,
                         retries=self._cfg.max_retries, error_cls=AsrError, label="語音辨識服務")

    def _record_cost(self, duration_s: float) -> None:
        """記帳：費用 = 計費分鐘數 × 每分鐘單價。"""
        if self._ledger is None:
            return
        billed = max(duration_s, self._cfg.min_billed_s)  # Groq 每次最少算 10 秒
        self._ledger.record(billed / 60 * self._cfg.usd_per_minute, audio_s=billed)

    @staticmethod
    def _extract_text(data: dict[str, Any], lang: str) -> str:
        """取出文字。有分段資訊（verbose_json）時，先丟掉不可信的片段再接起來。"""
        segments = data.get("segments")
        if isinstance(segments, list) and segments:
            kept = [
                s.get("text", "") for s in segments
                if not (s.get("no_speech_prob", 0.0) > _NO_SPEECH_PROB
                        and s.get("avg_logprob", 0.0) < _MIN_LOGPROB)
            ]
            return join_segments(kept, lang)
        return str(data.get("text") or "").strip()

    # ------------------------------------------------------------------ 辨識
    def transcribe(self, utt: Utterance) -> Transcription | None:
        """辨識一句；不論成功失敗，最後都要清理語言追蹤的紀錄（finally）。"""
        try:
            return self._transcribe(utt)
        finally:
            self._lang.done(utt)

    def _transcribe(self, utt: Utterance) -> Transcription | None:
        t0 = time.perf_counter()  # 計時，用來記錄辨識延遲
        fixed = self._lang.fixed_for(utt)
        data = self._request(utt.audio, fixed)
        self._record_cost(utt.duration)  # 送出就要付錢，所以不論結果如何都先記帳

        # ---- 決定這句的語言 ----
        raw_text = str(data.get("text") or "")
        if fixed:
            lang, prob = fixed, 1.0
        else:
            guessed = guess_language(raw_text)
            detected = normalize_language(data.get("language")) or guessed
            # 服務回報的語言不在清單時，只有文字系統明確（假名／諺文／漢字）才改用文字判斷；
            # 拉丁字母無法區分英、法、德…，此時尊重服務的判斷並略過
            if detected not in self._lang.allowed and guessed in {"ja", "ko", "zh"}:
                detected = guessed
            # 雲端服務不提供信心值，用句子長短估一個：短句 0.3（容易沿用上一句）、長句 0.9
            short = len(normalize(raw_text)) <= _SHORT_TEXT_CHARS
            chosen, prob = self._lang.choose(detected, 0.3 if short else 0.9)
            if chosen is None:
                log.debug("seq=%d 非允許語言（%s），略過", utt.seq, detected)
                return None
            lang = chosen

        # ---- 取出文字並過濾幻覺 ----
        text = self._extract_text(data, lang)
        if not text or is_hallucination(text):
            return None
        self._lang.commit(utt, lang)
        return Transcription(utt.seq, utt.is_final, utt.started_at, utt.ended_at, lang, prob, text,
                             time.perf_counter() - t0)
