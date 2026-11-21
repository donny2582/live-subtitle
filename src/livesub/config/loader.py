"""TOML → AppConfig，並提供路徑與金鑰的共用工具。

未知的鍵會直接報錯，避免拼錯設定名稱卻默默使用預設值。

讀取流程：
    config.toml ──tomllib──► dict ──_migrate_legacy──► 新格式 dict ──_build──► AppConfig
"""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from .schema import AppConfig, ConfigError

T = TypeVar("T")

# 舊版（單層 [translate]／[asr]）設定鍵 → 新的子區段，讓舊 config.toml 仍可使用
_LEGACY_KEYS: dict[str, dict[str, tuple[str, str]]] = {
    "translate": {
        "base_url": ("openai", "base_url"),
        "model": ("openai", "model"),
        "api_key_env": ("openai", "api_key_env"),
        "temperature": ("openai", "temperature"),
        "max_retries": ("openai", "max_retries"),
        "timeout_s": ("openai", "timeout_s"),
        "deepl_api_key_env": ("deepl", "api_key_env"),
    },
    "asr": {
        k: ("local", k)
        for k in ("model", "cpu_model", "device", "compute_type", "model_dir", "beam_size",
                  "min_allowed_language_prob", "no_speech_threshold", "log_prob_threshold",
                  "compression_ratio_threshold")
    },
}
_LEGACY_BUDGET_PRICES = {
    "input_usd_per_million_tokens": ("translate", "openai"),
    "output_usd_per_million_tokens": ("translate", "openai"),
    "usd_per_million_chars": ("translate", "deepl"),
}


def _migrate_legacy(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """把舊版平面設定搬到子區段。回傳 (新資料, 被搬移的鍵)。"""
    moved: list[str] = []
    for section, mapping in _LEGACY_KEYS.items():
        table = raw.get(section)
        if not isinstance(table, dict):
            continue
        for old, (sub, new) in mapping.items():
            # 例：[translate] model = "x"  →  [translate.openai] model = "x"
            # （新格式已經寫了的話以新格式為準，setdefault 不會覆蓋）
            if old in table and not isinstance(table[old], dict):
                table.setdefault(sub, {}).setdefault(new, table.pop(old))
                moved.append(f"{section}.{old}")
        if section == "asr" and moved and "engine" not in table:
            table["engine"] = "local"  # 舊設定檔一律是本機辨識
    budget = raw.get("budget")
    if isinstance(budget, dict):
        for key, (section, sub) in _LEGACY_BUDGET_PRICES.items():
            if key in budget:
                raw.setdefault(section, {}).setdefault(sub, {}).setdefault(key, budget.pop(key))
                moved.append(f"budget.{key}")
    tr = raw.get("translate")
    if isinstance(tr, dict) and "api_key" in tr:
        raise ConfigError("translate.api_key 已移除：請改用環境變數（setx OPENAI_API_KEY ...）")
    return raw, moved


def _build(cls: type[T], data: Any, path: str) -> T:
    """遞迴將 dict 轉成 dataclass，並拒絕未知鍵。

    遇到欄位型別本身也是 dataclass（例如 AppConfig.asr 是 AsrConfig），就往下一層再轉一次。
    path 是目前位置（例如 "asr.cloud"），只用在錯誤訊息裡告訴使用者是哪個區段出錯。
    """
    if not isinstance(data, dict):
        raise ConfigError(f"[{path}] 應為表格（table），實際為 {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"[{path or 'root'}] 未知的設定項目：{', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints[f.name]
        if isinstance(hint, type) and is_dataclass(hint):
            value = _build(hint, value, f"{path}.{f.name}" if path else f.name)
        elif isinstance(value, list):
            value = tuple(value)  # list 可以被修改，frozen 設定要用不可修改的 tuple
        kwargs[f.name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:  # 型別錯誤，例如字串填到數字欄位造成比較失敗
        raise ConfigError(f"[{path or 'root'}] 設定值型別錯誤：{exc}") from exc


def parse_config(raw: dict[str, Any]) -> tuple[AppConfig, list[str]]:
    """dict → AppConfig。回傳 (設定, 從舊格式搬移的鍵)。"""
    raw, moved = _migrate_legacy(raw)
    return _build(AppConfig, raw, ""), moved


def load_config(path: Path) -> AppConfig:
    """讀取 TOML 設定檔。

    Raises:
        ConfigError: 檔案格式錯誤或設定值不合法。
        FileNotFoundError: 檔案不存在。
    """
    return load_config_with_notes(path)[0]


def load_config_with_notes(path: Path) -> tuple[AppConfig, list[str]]:
    """同 load_config，但另外回傳「從舊格式自動搬移的鍵」，讓啟動時可以提醒使用者。"""
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} 不是合法的 TOML：{exc}") from exc
    return parse_config(raw)


def app_base_dir() -> Path:
    """程式的基準資料夾：打包成 .exe 時為 exe 所在資料夾；開發時為專案根目錄。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # 這個檔案在 <專案>/src/livesub/config/loader.py，往上 3 層就是專案根目錄
    return Path(__file__).resolve().parents[3]


def is_writable_dir(path: Path) -> bool:
    """實際建立一個暫存檔測試能不能寫入（只看權限設定在 Windows 上不準）。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".livesub_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def user_data_dir(base: Path) -> Path:
    """存放 config.toml、logs、用量、逐字稿的資料夾。

    * 一般情況：和程式放在一起（base），整個資料夾可以直接帶走（免安裝、可攜）。
    * 程式放在不能寫入的地方（例如 C:\\Program Files）：改用 %APPDATA%\\LiveSub。
    """
    if is_writable_dir(base):
        return base
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "LiveSub"


def resolve_path(base: Path, value: str) -> Path | None:
    """把設定中的相對路徑解析為絕對路徑；空字串回傳 None。"""
    if not value:
        return None
    # 支援 %USERPROFILE%、$HOME、~ 這類寫法
    p = Path(os.path.expandvars(value)).expanduser()
    return p if p.is_absolute() else (base / p)


def get_secret(env: Mapping[str, str], name: str) -> str:
    """從環境變數讀取金鑰（金鑰刻意不放在設定檔）。"""
    return env.get(name, "").strip() if name else ""


def is_local_url(url: str) -> bool:
    """本機服務（Ollama 等）不需要金鑰，也不計費。"""
    return any(h in url for h in ("localhost", "127.0.0.1", "[::1]"))
