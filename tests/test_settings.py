"""設定視窗：TOML 就地更新、服務選項、對話框存檔（Qt 以 offscreen 執行）。"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from livesub.config import AppConfig, ConfigError, load_config, save_config_updates, update_toml_text
from livesub.ui.presets import ASR_PRESETS, TRANSLATE_PRESETS, current_index, get_value, translate_key

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- TOML 就地更新
def test_update_keeps_comments_and_alignment():
    text = (
        "# 標題註解\n"
        "[translate]\n"
        'engine = "openai"          # openai / claude\n'
        "context_turns = 3\n"
        "\n"
        "[translate.claude]\n"
        'model = "claude-opus-5-5"  # 模型 # 含井號\n'
    )
    out = update_toml_text(text, {
        "translate.engine": "claude",
        "translate.claude.model": "claude-haiku-5-5",
        "translate.claude.thinking": "off",  # 原本沒有 → 加在區段最後
        "budget.monthly_limit_usd": 3.0,  # 區段不存在 → 附加
    })
    lines = out.splitlines()
    assert lines[0] == "# 標題註解"
    assert lines[2] == 'engine = "claude"          # openai / claude'
    assert lines[6] == 'model = "claude-haiku-5-5" # 模型 # 含井號'
    assert lines[7] == 'thinking = "off"'
    assert lines[-2:] == ["[budget]", "monthly_limit_usd = 3.0"]


def test_update_handles_hash_inside_string_and_lists():
    text = '[overlay]\nfont_family = "A#B"  # 字型\n[asr]\nallowed_languages = ["en"]\n'
    out = update_toml_text(text, {"overlay.font_family": 'C "D"', "asr.allowed_languages": ["en", "ja"]})
    assert 'font_family = "C \\"D\\""' in out and "# 字型" in out
    assert 'allowed_languages = ["en", "ja"]' in out


def test_update_preserves_crlf():
    out = update_toml_text("[overlay]\r\nfont_size = 26\r\n", {"overlay.font_size": 30})
    assert out == "[overlay]\r\nfont_size = 30\r\n"


def test_save_validates_before_writing(tmp_path):
    p = tmp_path / "config.toml"
    shutil.copyfile(ROOT / "config.example.toml", p)
    before = p.read_text(encoding="utf-8")
    with pytest.raises(ConfigError):
        save_config_updates(p, {"overlay.bg_opacity": 5.0})
    assert p.read_text(encoding="utf-8") == before  # 不合法時原檔不變

    save_config_updates(p, {"translate.engine": "claude", "overlay.font_size": 30})
    cfg = load_config(p)
    assert cfg.translate.engine == "claude" and cfg.overlay.font_size == 30
    assert p.read_text(encoding="utf-8").count("#") == before.count("#")  # 註解都還在


# ---------------------------------------------------------------- 服務選項
def test_every_preset_produces_valid_config(tmp_path):
    p = tmp_path / "config.toml"
    for preset in (*ASR_PRESETS, *TRANSLATE_PRESETS):
        shutil.copyfile(ROOT / "config.example.toml", p)
        save_config_updates(p, preset.updates)
        cfg = load_config(p)
        presets = ASR_PRESETS if preset in ASR_PRESETS else TRANSLATE_PRESETS
        assert presets[current_index(cfg, presets)] == preset, preset.label
        for key, value in preset.updates.items():
            assert get_value(cfg, key) == value


def test_current_index_defaults_and_custom():
    cfg = AppConfig()
    assert ASR_PRESETS[current_index(cfg, ASR_PRESETS)].updates["asr.cloud.model"] == "gpt-4o-mini-transcribe"
    assert TRANSLATE_PRESETS[current_index(cfg, TRANSLATE_PRESETS)].label == "OpenAI gpt-4.1-mini"
    from livesub.config import OpenAIConfig, TranslateConfig, replace

    custom = replace(cfg, translate=TranslateConfig(openai=OpenAIConfig(model="qwen2.5:7b")))
    assert current_index(custom, TRANSLATE_PRESETS) == -1
    assert translate_key(custom) == "OPENAI_API_KEY"


# ---------------------------------------------------------------- 對話框
@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PyQt6.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication(sys.argv)


def test_dialog_saves_only_changes(tmp_path, qapp):
    from livesub.ui.settings import SettingsDialog

    p = tmp_path / "config.toml"
    shutil.copyfile(ROOT / "config.example.toml", p)
    saved: list[set[str]] = []
    dlg = SettingsDialog(p, on_saved=lambda _cfg, changed: saved.append(changed))
    assert dlg.collect_updates() == {}  # 沒改任何東西

    haiku = next(i for i, x in enumerate(TRANSLATE_PRESETS) if "Haiku" in x.label)
    dlg._tr.setCurrentIndex(haiku)
    dlg._langs["ko"].setChecked(False)
    updates = dlg.collect_updates()
    assert updates["translate.engine"] == "claude"
    assert updates["asr.allowed_languages"] == ["en", "ja", "zh"]
    assert "overlay.font_size" not in updates
    assert "ANTHROPIC_API_KEY" in dlg._needed_keys()

    dlg._save()
    cfg = load_config(p)
    assert cfg.translate.claude.model == "claude-haiku-5-5"
    assert cfg.asr.allowed_languages == ("en", "ja", "zh")
    assert saved and "translate.engine" in saved[0]
