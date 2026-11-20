"""以假的 ASR／翻譯器跑完整管線（不需要 GPU、音效卡或網路）。"""

from __future__ import annotations

import threading
import time

import numpy as np
from conftest import chunks, silence, tone

from livesub.audio import EnergyVad
from livesub.config import AppConfig, TranslateConfig, replace
from livesub.core.models import Stage, SubtitleEvent, Transcription, Utterance
from livesub.output import TranscriptWriter
from livesub.pipeline import LatestOnly, PartialThrottle, Pipeline
from livesub.translation import TranslationError, TranslationResult


class FakeAsr:
    name = "fake-asr"
    is_cloud = False
    realtime_partials = True

    def __init__(self, language: str = "en") -> None:
        self.language = language
        self.calls: list[Utterance] = []

    def load(self) -> None:
        pass

    def transcribe(self, utt: Utterance) -> Transcription:
        self.calls.append(utt)
        text = f"sentence {utt.seq} ({utt.duration:.1f}s)"
        return Transcription(utt.seq, utt.is_final, utt.started_at, utt.ended_at,
                             self.language, 0.99, text, 0.01)


class FakeTranslator:
    name = "fake"
    is_cloud = False

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def warmup(self) -> None:
        pass

    def translate(self, text: str, source_lang: str, partial: bool = False) -> TranslationResult:
        if self.fail and not partial:
            raise TranslationError("模擬逾時")
        return TranslationResult(("[暫]" if partial else "") + f"譯:{text}", 0.01)


def run_pipeline(tmp_path, asr, translator, signal: np.ndarray) -> list[SubtitleEvent]:
    cfg = replace(AppConfig(), translate=replace(TranslateConfig(), partial_min_interval_s=0.0,
                                                 partial_min_new_chars=0))
    events: list[SubtitleEvent] = []
    lock = threading.Lock()

    def on_event(e: SubtitleEvent) -> None:
        with lock:
            events.append(e)

    writer = TranscriptWriter(tmp_path, cfg.output)
    p = Pipeline(cfg, asr, translator, EnergyVad, writer, None, on_event, lambda _m: None)
    p.capture.start = lambda: None  # type: ignore[method-assign]  # 不使用真的音效卡
    p.start()
    deadline = time.time() + 5
    while len(p._threads) < 3 and time.time() < deadline:  # noqa: SLF001
        time.sleep(0.01)
    for part, t in chunks(signal):
        p._audio_q.put((part, t))  # noqa: SLF001
        time.sleep(0.002)
    deadline = time.time() + 5
    while time.time() < deadline:
        with lock:
            if sum(e.stage is Stage.FINAL for e in events) >= 2:
                break
        time.sleep(0.02)
    p.stop()
    return events


def test_pipeline_end_to_end(tmp_path):
    signal = np.concatenate([silence(0.3), tone(2.0), silence(1.0), tone(1.0), silence(1.0)])
    events = run_pipeline(tmp_path, FakeAsr(), FakeTranslator(), signal)
    finals = [e for e in events if e.stage is Stage.FINAL]
    assert [e.seq for e in finals] == [1, 2]
    assert all(e.translated_text.startswith("譯:sentence") for e in finals)
    assert any(e.stage is Stage.PARTIAL_SOURCE for e in events)
    assert any(e.stage is Stage.PARTIAL_TRANSLATION and e.translated_text.startswith("[暫]") for e in events)
    srt = (tmp_path / "subtitles.srt").read_text(encoding="utf-8-sig")
    assert srt.count("-->") == 2


def test_pipeline_translation_failure_still_writes_source(tmp_path):
    signal = np.concatenate([tone(1.0), silence(1.0), tone(1.0), silence(1.0)])
    events = run_pipeline(tmp_path, FakeAsr(), FakeTranslator(fail=True), signal)
    finals = [e for e in events if e.stage is Stage.FINAL]
    assert finals and all(e.error == "模擬逾時" and e.translated_text == "" for e in finals)
    assert "翻譯失敗：模擬逾時" in (tmp_path / "transcript.txt").read_text(encoding="utf-8-sig")


def test_chinese_speech_is_not_sent_to_translator(tmp_path):
    signal = np.concatenate([tone(1.0), silence(1.0), tone(1.0), silence(1.0)])
    events = run_pipeline(tmp_path, FakeAsr("zh"), FakeTranslator(fail=True), signal)
    finals = [e for e in events if e.stage is Stage.FINAL]
    assert finals and all(e.error is None and e.translated_text == e.source_text for e in finals)


def test_latest_only_and_throttle():
    box: LatestOnly[int] = LatestOnly()
    box.put(1)
    box.put(2)
    assert box.take() == 2 and box.take() is None
    th = PartialThrottle(min_interval_s=1.0, min_new_chars=3)
    assert th.should_translate(1, "abc", now=0.0)
    assert not th.should_translate(1, "abcdef", now=0.5)  # 太快
    assert not th.should_translate(1, "abcd", now=2.0)  # 新增字數不足
    assert th.should_translate(1, "abcdefg", now=2.0)
    th.forget(1)
    assert th.should_translate(1, "a", now=2.1)
