from __future__ import annotations

from pathlib import Path

import pytest

from livesub.asr.filters import is_hallucination, join_segments, normalize
from livesub.config import AppConfig, ConfigError, load_config, load_config_with_notes
from livesub.core.glossary import (
    GlossaryEntry,
    build_asr_prompt,
    load_glossary,
    render_prompt_section,
    select_relevant,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "text",
    [
        "Thank you for watching!",
        "ご視聴ありがとうございました。",
        "시청해주셔서 감사합니다.",
        "字幕由Amara.org社区提供",
        "",
        "...",
        "はいはいはいはいはいはいはい",
    ],
)
def test_hallucinations_removed(text):
    assert is_hallucination(text)


@pytest.mark.parametrize(
    "text",
    ["Thank you.", "We need section control in long tunnels.",
     "구간단속 결과를 보겠습니다", "はい、そうです"],
)
def test_real_speech_kept(text):
    assert not is_hallucination(text)


def test_join_and_normalize():
    assert join_segments([" こんにちは", "世界 "], "ja") == "こんにちは世界"
    assert join_segments(["Hello", " world"], "en") == "Hello world"
    assert normalize("Hello,   World!") == "hello world"


def test_glossary_load_and_select(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text(
        "﻿source,target\n# 註解\nsection control,區間測速\nbad-row\nSection Control,重複\n구간단속,區間測速\n",
        encoding="utf-8",
    )
    entries = load_glossary(p)
    assert entries == [GlossaryEntry("section control", "區間測速"), GlossaryEntry("구간단속", "區間測速")]
    rel = select_relevant(entries, "The SECTION CONTROL pilot")
    assert [e.target for e in rel] == ["區間測速"]
    assert "section control → 區間測速" in render_prompt_section(rel)
    assert render_prompt_section([]) == ""
    assert build_asr_prompt(entries) == "section control, 구간단속"


def test_example_glossary_is_valid():
    entries = load_glossary(ROOT / "glossary.example.csv")
    assert len(entries) > 20


def test_example_config_loads():
    cfg = load_config(ROOT / "config.example.toml")
    assert cfg.translate.openai.model == "gpt-4.1-mini"
    assert cfg.translate.claude.model == "claude-opus-5-5"
    assert cfg.asr.engine == "cloud" and cfg.asr.cloud.model == "gpt-4o-mini-transcribe"
    assert cfg.asr.allowed_languages == ("en", "ja", "ko", "zh")
    assert cfg.budget.monthly_limit_usd == 5.0


def test_defaults_match_example():
    """範例檔與程式預設值應一致，避免文件與行為不同步。"""
    assert load_config(ROOT / "config.example.toml") == AppConfig()


def test_unknown_key_and_bad_value(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[asr]\nmodle = 'small'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="modle"):
        load_config(p)
    p.write_text("[translate]\nengine = 'google'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="engine"):
        load_config(p)
    p.write_text("[asr\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_unknown_key_in_provider_section(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[translate.claude]\nmodle = 'x'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"translate\.claude.*modle"):
        load_config(p)
    p.write_text("[translate.claude]\neffort = 'turbo'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="effort"):
        load_config(p)


def test_legacy_flat_config_is_migrated(tmp_path):
    p = tmp_path / "old.toml"
    p.write_text(
        "[asr]\nmodel = 'small'\ndevice = 'cpu'\n"
        "[translate]\nmodel = 'gpt-4o-mini'\ndeepl_api_key_env = 'MY_DEEPL'\n"
        "[budget]\ninput_usd_per_million_tokens = 0.15\nusd_per_million_chars = 20.0\n",
        encoding="utf-8",
    )
    cfg, moved = load_config_with_notes(p)
    assert cfg.asr.engine == "local" and cfg.asr.local.model == "small" and cfg.asr.local.device == "cpu"
    assert cfg.translate.openai.model == "gpt-4o-mini"
    assert cfg.translate.deepl.api_key_env == "MY_DEEPL"
    assert cfg.translate.openai.input_usd_per_million_tokens == 0.15
    assert cfg.translate.deepl.usd_per_million_chars == 20.0
    assert "translate.model" in moved and "budget.usd_per_million_chars" in moved


def test_plaintext_api_key_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[translate]\napi_key = 'sk-x'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="環境變數"):
        load_config(p)
