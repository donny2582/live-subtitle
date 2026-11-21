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


# ---------------------------------------------------------------- 啟動模式
def _launcher(tmp_path, monkeypatch, keys: dict[str, str], answer=None):
    from PyQt6.QtWidgets import QMessageBox

    from livesub.ui import settings

    p = tmp_path / "config.toml"
    shutil.copyfile(ROOT / "config.example.toml", p)
    monkeypatch.setattr(settings, "read_user_env", lambda name: keys.get(name, ""))
    asked: list[str] = []

    def fake_question(_parent, _title, text, *_args):
        asked.append(text)
        return answer or QMessageBox.StandardButton.No

    monkeypatch.setattr(settings.QMessageBox, "question", fake_question)
    return settings.SettingsDialog(p, launcher=True), asked


def test_launcher_starts_when_keys_present(tmp_path, monkeypatch, qapp):
    from PyQt6.QtWidgets import QDialog

    dlg, asked = _launcher(tmp_path, monkeypatch, {"OPENAI_API_KEY": "sk-test"})
    assert dlg.windowTitle() == "即時字幕"
    assert dlg.missing_keys() == []
    dlg._start()
    assert dlg.result() == QDialog.DialogCode.Accepted
    assert asked == []


def test_launcher_blocks_on_missing_key_unless_confirmed(tmp_path, monkeypatch, qapp):
    from PyQt6.QtWidgets import QDialog, QMessageBox

    dlg, asked = _launcher(tmp_path, monkeypatch, {})
    assert dlg.missing_keys() == ["OPENAI_API_KEY"]
    dlg._start()  # 回答「否」：不啟動，切到金鑰分頁
    assert dlg.result() != QDialog.DialogCode.Accepted
    assert asked and "OPENAI_API_KEY" in asked[0]
    assert dlg._tabs.currentIndex() == dlg._keys_tab

    dlg2, _ = _launcher(tmp_path, monkeypatch, {}, answer=QMessageBox.StandardButton.Yes)
    dlg2._start()  # 回答「是」：仍然啟動
    assert dlg2.result() == QDialog.DialogCode.Accepted


def test_restart_skips_launcher(monkeypatch):
    from livesub import app

    monkeypatch.setattr(sys, "orig_argv", ["python", "-m", "livesub", "--settings", "--engine", "claude"])
    cmd = app.restart_command()
    assert cmd[-2:] == ["--skip-settings", "--restarted"]
    assert "--settings" not in cmd and "claude" not in cmd
    assert app.parse_args(["--skip-settings"]).skip_settings


def test_write_ico(tmp_path, qapp):
    from livesub.ui.icon import write_ico

    data = write_ico(tmp_path / "x.ico").read_bytes()
    assert data[:6] == b"\x00\x00\x01\x00\x01\x00"  # .ico 檔頭：類型 1、共 1 張
    assert data[22:30] == b"\x89PNG\r\n\x1a\n"  # 第 22 位元組開始是 PNG


# ---------------------------------------------------------------- 單一執行個體
def test_single_instance_blocks_second_and_activates_first(qapp):
    import subprocess
    import time
    import uuid

    from livesub.ui.single_instance import SingleInstance

    key = f"livesub-test-{uuid.uuid4().hex[:8]}"
    first = SingleInstance(key)
    activated: list[bool] = []
    first.activated.connect(lambda: activated.append(True))
    # 第二個程式：拿不到鎖 → 通知第一個 → 以結束代碼回報結果（0 = 照預期運作）
    second_code = "\n".join((
        "import sys",
        "from PyQt6.QtCore import QCoreApplication",
        "from livesub.ui.single_instance import SingleInstance",
        "app = QCoreApplication(sys.argv)",
        f"s = SingleInstance({key!r})",
        "sys.exit(0 if (not s.acquire() and s.notify_running()) else 1)",
    ))
    try:
        assert first.acquire()
        proc = subprocess.Popen([sys.executable, "-c", second_code])  # noqa: S603 - 測試自己產生的程式碼
        deadline = time.monotonic() + 15
        while (proc.poll() is None or not activated) and time.monotonic() < deadline:
            qapp.processEvents()  # 第一個程式要處理事件才收得到通知
            time.sleep(0.01)
        assert proc.wait(timeout=5) == 0
        assert activated  # 第一個收到「把視窗叫出來」

        first.release()
        second = SingleInstance(key)
        assert second.acquire()  # 第一個結束後，新的程式可以接手
        second.release()
    finally:
        first.release()


# ---------------------------------------------------------------- 打包版（exe）相關
def test_user_data_dir_falls_back_when_not_writable(tmp_path, monkeypatch):
    from livesub.config import loader

    assert loader.user_data_dir(tmp_path) == tmp_path  # 可以寫入：和程式放一起
    monkeypatch.setattr(loader, "is_writable_dir", lambda _p: False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    assert loader.user_data_dir(tmp_path / "Program Files") == tmp_path / "roaming" / "LiveSub"


def test_ensure_user_files_copies_examples_to_data_dir(tmp_path):
    from livesub.app import ensure_user_files

    base, data = tmp_path / "exe", tmp_path / "data"
    base.mkdir()
    shutil.copyfile(ROOT / "config.example.toml", base / "config.example.toml")
    ensure_user_files(base, data)
    assert (data / "config.toml").exists() and not (base / "config.toml").exists()


def test_shortcut_points_to_exe_when_frozen(tmp_path, monkeypatch):
    from livesub.ui import shortcut

    exe = tmp_path / "LiveSub" / "LiveSub.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    spec = shortcut.shortcut_spec()
    assert spec == {"target": str(exe), "args": "", "dir": str(exe.parent), "icon": f"{exe},0"}


def test_output_tab_saves_transcript_folder(tmp_path, qapp):
    from livesub.ui.settings import SettingsDialog

    p = tmp_path / "config.toml"
    shutil.copyfile(ROOT / "config.example.toml", p)
    dlg = SettingsDialog(p)
    assert dlg._output_path() == tmp_path / "transcripts"  # 預設：設定檔旁邊
    target = tmp_path / "我的文件" / "逐字稿"
    dlg._set_output_dir(str(target))
    assert "output.write_jsonl" in dlg._bindings  # 有綁定輸出格式
    assert dlg.collect_updates() == {"output.transcript_dir": str(target)}
    dlg._save()
    assert load_config(p).output.transcript_dir == str(target)
