"""術語表：固定特定名詞的譯法（例如交通安全專有名詞）。

CSV 格式（UTF-8，可用 Excel 另存為「CSV UTF-8」）：

    source,target
    section control,區間測速
    구간단속,區間測速

以 # 開頭的行視為註解。

術語表有兩個用途：
1. 翻譯：這句出現的術語會附在 LLM 提示詞裡，要求一定要用指定譯法。
2. 辨識（選用，asr.glossary_prompt = true）：把術語原文提示給語音辨識，專有名詞比較不會聽錯。
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """一筆術語。"""

    source: str  # 原文，例如 "section control"
    target: str  # 固定譯法，例如 "區間測速"


def load_glossary(path: Path) -> list[GlossaryEntry]:
    """讀取術語表。

    Raises:
        FileNotFoundError: 檔案不存在。
    """
    entries: list[GlossaryEntry] = []
    seen: set[str] = set()  # 已出現過的原文（不分大小寫），用來找重複
    # utf-8-sig：容忍 Excel 存檔時加入的 BOM
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for lineno, row in enumerate(csv.reader(fh), start=1):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue  # 空行或註解
            if lineno == 1 and [c.strip().lower() for c in row[:2]] == ["source", "target"]:
                continue  # 標題列
            if len(row) < 2 or not row[1].strip():
                log.warning("術語表第 %d 行格式錯誤，已略過：%r", lineno, row)
                continue
            src, tgt = row[0].strip(), row[1].strip()
            key = src.casefold()
            if key in seen:
                log.warning("術語表第 %d 行重複的原文「%s」，以第一筆為準", lineno, src)
                continue
            seen.add(key)
            entries.append(GlossaryEntry(src, tgt))
    log.info("已載入術語 %d 筆：%s", len(entries), path)
    return entries


def select_relevant(entries: list[GlossaryEntry], text: str) -> list[GlossaryEntry]:
    """只挑出原文中出現的術語，讓提示詞保持精簡（省 token 也省延遲）。"""
    folded = text.casefold()  # casefold：不分大小寫比對（比 lower() 更完整）
    return [e for e in entries if e.source.casefold() in folded]


def render_prompt_section(entries: list[GlossaryEntry]) -> str:
    """產生放進 LLM system prompt 的術語段落。"""
    if not entries:
        return ""
    lines = "\n".join(f"- {e.source} → {e.target}" for e in entries)
    return f"\n術語表（出現時必須使用以下譯法）：\n{lines}"


def build_asr_prompt(entries: list[GlossaryEntry], max_chars: int = 200) -> str:
    """把術語原文串成 Whisper initial_prompt，提高專有名詞的辨識率。

    Whisper 的提示詞有長度限制，所以最多放 max_chars 個字，超過的術語就不放。
    """
    out: list[str] = []
    total = 0
    for e in entries:
        if total + len(e.source) + 2 > max_chars:
            break
        out.append(e.source)
        total += len(e.source) + 2
    return ", ".join(out)


def load_glossary_safe(path: Path | None) -> list[GlossaryEntry]:
    """檔案不存在或讀取失敗時回傳空清單（只記錄錯誤，不中止程式）。"""
    if not path or not path.exists():
        return []
    try:
        return load_glossary(path)
    except (OSError, UnicodeDecodeError) as exc:
        log.error("術語表讀取失敗：%s", exc)
        return []
