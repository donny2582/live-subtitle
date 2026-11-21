"""雲端語音辨識與 Claude 翻譯（以假的 HTTP／SDK 物件測試，不需網路）。"""

from __future__ import annotations

import io
import json
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from conftest import tone

from livesub.asr.base import LanguageTracker, guess_language, normalize_language
from livesub.asr.cloud_openai import CloudTranscriber, to_wav_bytes
from livesub.config import AsrConfig, BudgetConfig, ClaudeConfig, CloudAsrConfig, replace
from livesub.core.budget import BudgetLedger
from livesub.core.glossary import GlossaryEntry
from livesub.core.models import Utterance
from livesub.translation.base import ChineseNormalizer, TranslationError
from livesub.translation.claude import ClaudeTranslator, supports_server_fallback


# ---------------------------------------------------------------- 假的 HTTP
class FakeResp:
    def __init__(self, status: int, payload: Any) -> None:
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResp]) -> None:
        self.headers: dict[str, str] = {}
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, timeout: float, **kw: Any) -> FakeResp:
        self.calls.append({"url": url, **kw})
        return self.responses.pop(0)


def utt(seq: int = 1, final: bool = True, seconds: float = 2.0) -> Utterance:
    return Utterance(seq, tone(seconds), 0.0, seconds, final)


def asr(responses: list[FakeResp], ledger: BudgetLedger | None = None, **cloud: Any) -> tuple[
    CloudTranscriber, FakeSession
]:
    session = FakeSession(responses)
    cfg = AsrConfig(cloud=CloudAsrConfig(**cloud))
    return CloudTranscriber(cfg, "sk-test", ledger, prompt="section control", session=session), session


# ---------------------------------------------------------------- 語言工具
@pytest.mark.parametrize(
    ("text", "lang"),
    [("Hello there", "en"), ("はい、そうです", "ja"), ("구간단속 결과", "ko"),
     ("我們開始吧", "zh"), ("...", None)],
)
def test_guess_language(text, lang):
    assert guess_language(text) == lang


def test_normalize_language():
    assert normalize_language("English") == "en"
    assert normalize_language("japanese") == "ja"
    assert normalize_language("zh-TW") == "zh"
    assert normalize_language(None) is None


def test_language_tracker_sticky_and_partials():
    tr = LanguageTracker(("en", "ja"), sticky_prob=0.6)
    assert tr.choose("fr", 0.9) == (None, 0.9)  # 不在允許清單
    tr.commit(utt(1), "ja")
    assert tr.choose("en", 0.3)[0] == "ja"  # 低信心沿用上一句
    assert tr.choose("en", 0.9)[0] == "en"
    p = utt(2, final=False)
    tr.commit(p, "en")
    assert tr.fixed_for(utt(2, final=False)) == "en"
    tr.done(utt(2, final=True))
    assert tr.fixed_for(utt(2, final=False)) is None


# ---------------------------------------------------------------- 雲端辨識
def test_wav_encoding_roundtrip():
    data = to_wav_bytes(np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32))
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16_000, 4)


def test_cloud_asr_gpt4o_json_and_cost(tmp_path):
    ledger = BudgetLedger(BudgetConfig(), tmp_path / "u.json", month_fn=lambda: "2026-11")
    t, s = asr([FakeResp(200, {"text": "  こんにちは、始めましょう "})], ledger)
    res = t.transcribe(utt(seconds=3.0))
    assert res is not None and res.language == "ja" and res.text == "こんにちは、始めましょう"
    call = s.calls[0]
    assert call["url"] == "https://api.openai.com/v1/audio/transcriptions"
    assert call["data"]["model"] == "gpt-4o-mini-transcribe"
    assert call["data"]["response_format"] == "json"
    assert call["data"]["prompt"] == "section control" and "language" not in call["data"]
    assert call["files"]["file"][2] == "audio/wav"
    assert s.headers["Authorization"] == "Bearer sk-test"
    usage = ledger.month_usage()
    assert usage.audio_seconds == pytest.approx(3.0)
    assert usage.asr_cost_usd == pytest.approx(3.0 / 60 * 0.003)


def test_cloud_asr_whisper_verbose_json_filters_silence(tmp_path):
    ledger = BudgetLedger(BudgetConfig(), tmp_path / "u.json", month_fn=lambda: "2026-11")
    payload = {
        "text": "We need section control. Thank you.",
        "language": "english",
        "segments": [
            {"text": " We need section control.", "no_speech_prob": 0.01, "avg_logprob": -0.2},
            {"text": " Thank you.", "no_speech_prob": 0.9, "avg_logprob": -1.5},  # 靜音幻覺
        ],
    }
    t, s = asr([FakeResp(200, payload)], ledger, base_url="https://api.groq.com/openai/v1",
               model="whisper-large-v3-turbo", usd_per_minute=0.04 / 60, min_billed_s=10.0)
    res = t.transcribe(utt(seconds=2.0))
    assert res is not None and res.language == "en" and res.text == "We need section control."
    assert s.calls[0]["data"]["response_format"] == "verbose_json"
    assert ledger.month_usage().audio_seconds == 10.0  # 最少計費 10 秒


def test_cloud_asr_partial_reuses_language_and_drops_disallowed():
    t, s = asr([FakeResp(200, {"text": "Bonjour à tous", "language": "french"}),
                FakeResp(200, {"text": "We discuss"}),
                FakeResp(200, {"text": "We discuss it"}),
                FakeResp(200, {"text": "Thank you for watching!"})])
    assert t.transcribe(utt(1)) is None  # 服務回報法文，不在允許清單
    first = t.transcribe(utt(2, final=False))
    assert first is not None and first.language == "en"
    t.transcribe(utt(2, final=False))
    assert s.calls[2]["data"]["language"] == "en"  # 中間結果固定語言
    assert t.transcribe(utt(2)) is None  # 幻覺過濾


def test_cloud_asr_errors_and_missing_key():
    from livesub.asr import AsrError, AsrUnavailableError

    t, _ = asr([FakeResp(401, {})])
    with pytest.raises(AsrError, match="金鑰"):
        t.transcribe(utt())
    no_key = CloudTranscriber(AsrConfig(), "", None)
    with pytest.raises(AsrUnavailableError, match="OPENAI_API_KEY"):
        no_key.load()
    local = CloudTranscriber(AsrConfig(cloud=CloudAsrConfig(base_url="http://localhost:8000/v1")), "", None)
    local.load()
    assert local.is_cloud is False


# ---------------------------------------------------------------- Claude
class FakeMessages:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.calls: list[dict[str, Any]] = []

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        return self.replies.pop(0)

    @contextmanager
    def stream(self, **kw: Any) -> Iterator[Any]:
        """模擬 SDK 的 messages.stream()：把最終文字拆成兩段送出，最後回傳完整回應。"""
        self.calls.append({**kw, "_stream": True})
        resp = self.replies.pop(0)
        text = "".join(b.text for b in resp.content if b.type == "text")
        half = len(text) // 2
        yield SimpleNamespace(text_stream=iter([text[:half], text[half:]]),
                              get_final_message=lambda: resp)


class FakeAnthropic:
    def __init__(self, replies: list[Any]) -> None:
        self.messages = FakeMessages(replies)
        self.beta = SimpleNamespace(messages=self.messages)


def reply(text: str, stop: str = "end_turn", tin: int = 500, tout: int = 40) -> Any:
    return SimpleNamespace(
        stop_reason=stop,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=tin, output_tokens=tout),
    )


def test_claude_translator_request_shape_cost_and_context(tmp_path):
    ledger = BudgetLedger(BudgetConfig(), tmp_path / "u.json", month_fn=lambda: "2026-11")
    client = FakeAnthropic([reply("「我們討論區間測速」"), reply("下一句")])
    tr = ClaudeTranslator(ClaudeConfig(), [GlossaryEntry("section control", "區間測速")], ledger,
                          ChineseNormalizer(False), client=client)
    res = tr.translate("We discuss section control", "en")
    assert res.text == "我們討論區間測速"
    assert res.cost_usd == pytest.approx((500 * 4 + 40 * 20) / 1e6)
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert call["output_config"] == {"effort": "low"}
    assert "thinking" not in call and "temperature" not in call
    assert "section control → 區間測速" in call["system"] and "英文" in call["system"]
    assert call["messages"][-1]["content"] == "<source>We discuss section control</source>"

    tr.translate("Next one", "en")
    msgs = client.messages.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]


def test_claude_haiku_thinking_off_without_fallback():
    cfg = replace(ClaudeConfig(), model="claude-haiku-5-5", thinking="off")
    client = FakeAnthropic([reply("好")])
    tr = ClaudeTranslator(cfg, [], None, ChineseNormalizer(False), client=client)
    assert tr.translate("ok", "en", partial=True).text == "好"
    call = client.messages.calls[0]
    assert call["thinking"] == {"type": "disabled"} and "fallbacks" not in call
    assert "還沒說完" in call["system"]
    assert not supports_server_fallback("claude-haiku-5-5") and supports_server_fallback("claude-opus-5-5")


def test_claude_refusal_and_truncation_raise():
    client = FakeAnthropic([reply("", stop="refusal"), reply("", stop="max_tokens")])
    tr = ClaudeTranslator(ClaudeConfig(), [], None, ChineseNormalizer(False), client=client)
    with pytest.raises(TranslationError, match="拒絕"):
        tr.translate("x", "en")
    with pytest.raises(TranslationError, match="max_tokens"):
        tr.translate("x", "en")


def test_claude_sdk_error_is_mapped():
    import anthropic

    class Boom:
        def create(self, **_: Any) -> Any:
            raise anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]

    client = SimpleNamespace(messages=Boom(), beta=SimpleNamespace(messages=Boom()))
    tr = ClaudeTranslator(ClaudeConfig(), [], None, ChineseNormalizer(False), client=client)
    with pytest.raises(TranslationError, match="無回應"):
        tr.translate("x", "en")


def test_claude_streaming_sends_growing_text_and_records_cost(tmp_path):
    ledger = BudgetLedger(BudgetConfig(), tmp_path / "u.json", month_fn=lambda: "2026-11")
    cfg = replace(ClaudeConfig(), model="claude-haiku-5-5", thinking="off")
    client = FakeAnthropic([reply("「我們討論區間測速」")])
    tr = ClaudeTranslator(cfg, [], ledger, ChineseNormalizer(False), client=client)
    assert tr.supports_streaming
    seen: list[str] = []
    res = tr.translate("We discuss section control", "en", on_delta=seen.append)
    assert client.messages.calls[0]["_stream"]
    assert len(seen) == 2 and seen[-1] == "我們討論區間測速"  # 第二段時已是完整譯文
    assert res.text == "我們討論區間測速" and res.cost_usd > 0  # 用量取自最終回應


def test_claude_streaming_error_is_mapped():
    import anthropic

    class BoomStream:
        @contextmanager
        def stream(self, **_: Any) -> Iterator[Any]:
            def pieces() -> Iterator[str]:
                yield "半"
                raise anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]

            yield SimpleNamespace(text_stream=pieces(), get_final_message=lambda: None)

    client = SimpleNamespace(messages=BoomStream(), beta=SimpleNamespace(messages=BoomStream()))
    tr = ClaudeTranslator(ClaudeConfig(), [], None, ChineseNormalizer(False), client=client)
    with pytest.raises(TranslationError, match="無回應"):
        tr.translate("x", "en", on_delta=lambda _t: None)
