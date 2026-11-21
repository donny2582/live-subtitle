from __future__ import annotations

import json
from typing import Any

import pytest

from livesub.config import BudgetConfig, DeepLConfig, OpenAIConfig, OutputConfig, TranslateConfig
from livesub.core.budget import BudgetLedger
from livesub.core.glossary import GlossaryEntry
from livesub.core.models import Stage, SubtitleEvent
from livesub.output.transcript import TranscriptWriter, format_clock, format_srt_time
from livesub.translation import (
    ChineseNormalizer,
    PassthroughTranslator,
    TranslationError,
    build_translator,
    clean_translation,
)
from livesub.translation.deepl import DeepLTranslator
from livesub.translation.openai_compat import OpenAICompatTranslator


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

    def post(self, url: str, json: Any, timeout: float) -> FakeResp:  # noqa: A002
        self.calls.append({"url": url, "json": json})
        return self.responses.pop(0)


def ok(content: str, pt: int = 100, ct: int = 20) -> FakeResp:
    return FakeResp(200, {"choices": [{"message": {"content": content}}],
                          "usage": {"prompt_tokens": pt, "completion_tokens": ct}})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("「我們需要區間測速」", "我們需要區間測速"),
        ("譯文：你好", "你好"),
        ("<think>hmm</think>\n你好", "你好"),
        ('"他說「好」"', '他說「好」'),
        ("「A」和「B」", "「A」和「B」"),
        ("<source>你好</source>", "你好"),  # 照抄標籤
        ("政府宣布新政策。\n註：該新聞內容需要查核", "政府宣布新政策。"),  # 整行附註
        ("政府宣布新政策（註：以上內容需查核）", "政府宣布新政策"),  # 句尾括號附註
        ("譯註：原文有誤\n他明天出發", "他明天出發"),
        ("注意：前方施工", "注意：前方施工"),  # 講者真的說「注意」不能刪
        ("請看第三頁（附錄）", "請看第三頁（附錄）"),  # 一般括號保留
    ],
)
def test_clean_translation(raw, expected):
    assert clean_translation(raw) == expected


def test_openai_translator_context_glossary_and_cost(tmp_path):
    ledger = BudgetLedger(BudgetConfig(), tmp_path / "u.json", month_fn=lambda: "2026-11")
    session = FakeSession([ok("「我們討論區間測速」", 1000, 100), ok("下一句")])
    tr = OpenAICompatTranslator(
        OpenAIConfig(), "sk-test", [GlossaryEntry("section control", "區間測速"),
                                       GlossaryEntry("speed camera", "測速照相")],
        ledger, ChineseNormalizer(False), session=session,  # type: ignore[arg-type]
    )
    res = tr.translate("We discuss section control", "en")
    assert res.text == "我們討論區間測速"
    assert res.cost_usd == pytest.approx((1000 * 0.4 + 100 * 1.6) / 1e6)
    system = session.calls[0]["json"]["messages"][0]["content"]
    assert "section control → 區間測速" in system and "speed camera" not in system
    assert "不要加任何評論" in system  # 禁止額外內容的規則
    user_msg = session.calls[0]["json"]["messages"][-1]["content"]
    assert user_msg == "<source>We discuss section control</source>"  # 原文用標籤包起來
    assert "英文" in system
    assert session.headers["Authorization"] == "Bearer sk-test"

    tr.translate("Next one", "en")
    msgs = session.calls[1]["json"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]  # 帶入上下文
    assert msgs[1]["content"] == "<source>We discuss section control</source>"  # 上下文的原文也加標籤
    ledger.save()
    saved = json.loads((tmp_path / "u.json").read_text(encoding="utf-8"))
    assert saved["2026-11"]["requests"] == 2


def test_partial_translation_not_added_to_history():
    session = FakeSession([ok("片段"), ok("完整")])
    tr = OpenAICompatTranslator(OpenAIConfig(), "k", [], None, ChineseNormalizer(False),
                                session=session)  # type: ignore[arg-type]
    tr.translate("We disc", "en", partial=True)
    assert "還沒說完" in session.calls[0]["json"]["messages"][0]["content"]
    tr.translate("We discuss", "en")
    assert len(session.calls[1]["json"]["messages"]) == 2  # system + user，沒有片段上下文


def test_openai_errors_and_retry():
    session = FakeSession([FakeResp(500, {}), ok("好")])
    tr = OpenAICompatTranslator(OpenAIConfig(max_retries=1), "k", [], None,
                                ChineseNormalizer(False), session=session)  # type: ignore[arg-type]
    assert tr.translate("ok", "en").text == "好"
    bad = OpenAICompatTranslator(OpenAIConfig(), "k", [], None, ChineseNormalizer(False),
                                 session=FakeSession([FakeResp(401, {})]))  # type: ignore[arg-type]
    with pytest.raises(TranslationError, match="金鑰"):
        bad.translate("hi", "en")


def test_deepl_free_host_and_quota():
    s = FakeSession([FakeResp(200, {"translations": [{"text": "你好"}]}), FakeResp(456, {})])
    tr = DeepLTranslator(DeepLConfig(), "abc:fx", None, ChineseNormalizer(False),
                         session=s)  # type: ignore[arg-type]
    assert tr.translate("Hello", "ja").text == "你好"
    assert s.calls[0]["url"].startswith("https://api-free.deepl.com")
    assert s.calls[0]["json"]["target_lang"] == "ZH-HANT" and s.calls[0]["json"]["source_lang"] == "JA"
    with pytest.raises(TranslationError, match="額度"):
        tr.translate("Again", "ja")


def test_build_translator_without_key_falls_back():
    tr = build_translator(TranslateConfig(), [], None, env={})
    assert isinstance(tr, PassthroughTranslator)
    local = build_translator(
        TranslateConfig(openai=OpenAIConfig(base_url="http://localhost:11434/v1")), [], None, env={}
    )
    assert local.is_cloud is False and local.name.startswith("openai:")


def test_budget_ledger_limits(tmp_path):
    cfg = BudgetConfig(monthly_limit_usd=1.0, partial_cutoff_ratio=0.5)
    month = {"m": "2026-11"}
    ledger = BudgetLedger(cfg, tmp_path / "u.json", month_fn=lambda: month["m"])
    assert ledger.allows(partial=True)
    ledger.record(0.0)
    assert ledger.allows(partial=True)
    ledger.record(0.6, input_tokens=10)
    assert not ledger.allows(partial=True) and ledger.allows(partial=False)
    ledger.record(0.6, audio_s=30.0)  # 語音辨識費用
    assert not ledger.allows(partial=False)
    usage = ledger.month_usage()
    assert usage.asr_cost_usd == pytest.approx(0.6) and usage.audio_seconds == 30.0
    month["m"] = "2026-12"  # 換月自動重置
    assert ledger.allows(partial=False)
    ledger.save()
    reloaded = BudgetLedger(cfg, tmp_path / "u.json", month_fn=lambda: "2026-11")
    assert reloaded.month_usage().cost_usd == pytest.approx(1.2)
    assert "辨識 $0.600" in reloaded.summary()


def test_budget_ledger_reads_old_file_format(tmp_path):
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"2026-10": {"input_tokens": 1, "output_tokens": 2, "chars": 0,
                                         "requests": 3, "cost_usd": 0.5}}), encoding="utf-8")
    ledger = BudgetLedger(BudgetConfig(), p, month_fn=lambda: "2026-10")
    assert ledger.month_usage().cost_usd == 0.5 and ledger.month_usage().audio_seconds == 0.0


def test_budget_ledger_recovers_from_corrupt_file(tmp_path):
    p = tmp_path / "u.json"
    p.write_text("{not json", encoding="utf-8")
    ledger = BudgetLedger(BudgetConfig(), p)
    assert ledger.month_usage().requests == 0
    assert (tmp_path / "u.corrupt.json").exists()


def test_time_format():
    assert format_srt_time(3723.4567) == "01:02:03,457"
    assert format_srt_time(-1) == "00:00:00,000"
    assert format_clock(3723.9) == "01:02:03"


def test_transcript_writer(tmp_path):
    with TranscriptWriter(tmp_path, OutputConfig()) as w:
        w.write(SubtitleEvent(1, Stage.PARTIAL_SOURCE, 0, 1, "en", "ignored"))
        w.write(SubtitleEvent(1, Stage.FINAL, 1.0, 1.2, "en", "Hello", "你好"))
        w.write(SubtitleEvent(2, Stage.FINAL, 2.0, 4.0, "ja", "こんにちは", "", error="逾時"))
        w.write(SubtitleEvent(3, Stage.FINAL, 5.0, 6.0, "zh", "中文", "中文"))
    srt = (tmp_path / "subtitles.srt").read_text(encoding="utf-8-sig")
    assert "1\n00:00:01,000 --> 00:00:01,800\n你好\nHello\n" in srt  # 最短顯示 0.8 秒
    assert "2\n00:00:02,000 --> 00:00:04,000\nこんにちは\n" in srt
    assert "3\n00:00:05,000 --> 00:00:06,000\n中文\n\n" in srt  # 中文不重複顯示
    txt = (tmp_path / "transcript.txt").read_text(encoding="utf-8-sig")
    assert "[00:00:01] [英] Hello" in txt and "翻譯失敗：逾時" in txt
    lines = (tmp_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["stage"] == "final"


def test_passthrough_only_for_chinese():
    p = PassthroughTranslator()
    assert p.translate("中文", "zh").text == "中文"
    assert p.translate("Hello", "en").text == ""
