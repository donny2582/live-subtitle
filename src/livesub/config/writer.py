"""在保留註解與排版的前提下，更新 config.toml 中的個別設定值。

設定視窗只改動使用者調整過的項目，其他內容（包含註解）原封不動。
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .loader import parse_config
from .schema import ConfigError

_HEADER = re.compile(r"^\s*\[([A-Za-z0-9_.\-]+)\]\s*(#.*)?$")
_KEY = re.compile(r"^(\s*)([A-Za-z0-9_\-]+)(\s*=\s*)(.*)$")


def format_value(value: Any) -> str:
    """Python 值 → TOML 字面值。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    raise TypeError(f"不支援的設定值型別：{type(value).__name__}")


def _split_value_comment(rest: str) -> tuple[str, str]:
    """把 `"abc"   # 註解` 拆成值與（含前置空白的）註解；字串內的 # 不算。"""
    quote: str | None = None
    i = 0
    while i < len(rest):
        ch = rest[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            j = i
            while j > 0 and rest[j - 1] in " \t":
                j -= 1
            return rest[:j], rest[j:]
        i += 1
    return rest.rstrip(), ""


def _replace_value(line: str, new_value: str) -> str:
    m = _KEY.match(line)
    assert m is not None
    indent, key, eq, rest = m.groups()
    old, comment = _split_value_comment(rest)
    if comment:
        # 盡量讓註解維持原本的對齊位置
        pad = max(1, len(old) + len(comment) - len(comment.lstrip()) - len(new_value))
        comment = " " * pad + comment.lstrip()
    return f"{indent}{key}{eq}{new_value}{comment}"


def update_toml_text(text: str, updates: Mapping[str, Any]) -> str:
    """依「區段.鍵」更新 TOML 文字，例如 {"translate.claude.model": "claude-haiku-5-5"}。

    已存在的鍵就地替換（保留註解）；不存在的鍵加到該區段最後一個設定之後；
    區段不存在則附加到檔案結尾。
    """
    newline = "\r\n" if "\r\n" in text else "\n"  # 沿用原檔的換行格式（Windows 是 \r\n）
    lines = text.splitlines()
    # pending：還沒寫進去的更新，{區段: {鍵: TOML 字面值}}；寫入後就從這裡移除
    pending: dict[str, dict[str, str]] = {}
    for dotted, value in updates.items():
        section, _, key = dotted.rpartition(".")
        if not section or not key:
            raise ValueError(f"設定鍵需包含區段：{dotted}")
        pending.setdefault(section, {})[key] = format_value(value)

    # 逐行掃描：記住目前在哪個 [區段]，遇到要更新的鍵就就地替換數值
    current: str | None = None
    last_key_line: dict[str, int] = {}  # 每個區段最後一個設定在第幾行（新鍵要插在它後面）
    for i, line in enumerate(lines):
        h = _HEADER.match(line)
        if h:
            current = h.group(1)
            last_key_line.setdefault(current, i)
            continue
        m = _KEY.match(line)
        if m and current is not None:
            last_key_line[current] = i
            key = m.group(2)
            if key in pending.get(current, {}):
                lines[i] = _replace_value(line, pending[current].pop(key))

    # 尚未寫入的鍵：插在區段最後一個設定之後（由後往前插，索引才不會位移）
    inserts: list[tuple[int, list[str]]] = []
    appended: list[str] = []
    for section, keys in pending.items():
        if not keys:
            continue
        new_lines = [f"{k} = {v}" for k, v in keys.items()]
        if section in last_key_line:
            inserts.append((last_key_line[section] + 1, new_lines))
        else:
            appended += ["", f"[{section}]", *new_lines]
    for index, new_lines in sorted(inserts, reverse=True):
        lines[index:index] = new_lines
    lines += appended
    return newline.join(lines) + newline


def save_config_updates(path: Path, updates: Mapping[str, Any]) -> None:
    """更新設定檔；寫入前先驗證新內容，失敗時原檔不受影響。

    Raises:
        ConfigError: 更新後的設定不合法。
    """
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    new_text = update_toml_text(text, updates)
    try:
        parse_config(tomllib.loads(new_text))  # 先驗證：不合法就丟出例外，不會寫檔
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"更新後不是合法的 TOML：{exc}") from exc
    # 先寫到暫存檔再整個替換：寫到一半當機也不會留下壞掉的設定檔
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8", newline="")
    os.replace(tmp, path)
