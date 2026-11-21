"""Whisper 幻覺（hallucination）過濾與文字整理。

Whisper 在靜音、音樂或雜訊段常會憑空產生固定句子，例如
"Thank you for watching"、「ご視聴ありがとうございました」。

原因是 Whisper 用大量 YouTube 影片訓練，影片結尾常出現這些字幕，
所以聽到「沒有內容」時，模型會猜這是影片結尾而產生這些句子。這裡把它們過濾掉。
"""

from __future__ import annotations

import re
import unicodedata

# 整句完全等於這些內容時丟棄（比對前會先正規化：小寫、去標點與空白）
_EXACT_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "thank you for watching and see you next time",
    "please subscribe",
    "please like and subscribe",
    "subtitles by",
    "ご視聴ありがとうございました",
    "ご視聴いただきありがとうございました",
    "チャンネル登録お願いします",
    "チャンネル登録よろしくお願いします",
    "시청해주셔서 감사합니다",
    "시청해 주셔서 감사합니다",
    "구독과 좋아요 부탁드립니다",
    "謝謝觀看",
    "谢谢观看",
    "感謝收看",
    "请不吝点赞订阅转发打赏支持明镜与点点栏目",
}

# 只要包含這些片段就丟棄（典型的字幕組署名幻覺）
_CONTAINS_PHRASES = (
    "amaraorg",
    "字幕由",
    "字幕製作",
    "字幕制作",
    "明镜与点点",
    "mbc 뉴스",
)

# 同一小段（1~12 字）連續重複 5 次以上，例如「哈哈哈哈哈」或 "you you you you you"
_REPEAT_RE = re.compile(r"(.{1,12}?)\1{4,}")
_NO_SPACE_LANGS = {"ja", "zh", "yue"}  # 字與字之間不加空格的語言


def normalize(text: str) -> str:
    """小寫並移除標點、符號與多餘空白（保留單一空格以區分英文單字）。"""
    cleaned = []
    for ch in text.lower():
        # Unicode 分類：P = 標點、S = 符號、Z = 空白類
        cat = unicodedata.category(ch)
        if cat[0] in {"P", "S"}:
            continue
        cleaned.append(" " if cat[0] == "Z" or ch.isspace() else ch)
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def _key(text: str) -> str:
    """比對用的鍵：正規化後再去掉所有空格。"""
    return normalize(text).replace(" ", "")


# 程式載入時就先把清單轉成比對鍵，之後每句只需要查表
_EXACT_KEYS = {_key(p) for p in _EXACT_PHRASES}
_CONTAINS_KEYS = tuple(_key(p) for p in _CONTAINS_PHRASES)


def is_hallucination(text: str) -> bool:
    """判斷 ASR 結果是否為幻覺或無意義內容。"""
    key = _key(text)
    if not key:
        return True
    if key in _EXACT_KEYS:
        return True
    if any(k and k in key for k in _CONTAINS_KEYS):
        return True
    # 重複的部分佔整句 60% 以上、且句子不太短，才當作幻覺（避免誤殺「哈哈」這種正常短句）
    match = _REPEAT_RE.search(key)
    return bool(match and len(match.group(0)) >= 0.6 * len(key) and len(key) >= 8)


def join_segments(texts: list[str], language: str) -> str:
    """依語言接合多個片段：日文與中文不加空格，其他語言以空格分隔。"""
    parts = [t.strip() for t in texts if t and t.strip()]
    sep = "" if language in _NO_SPACE_LANGS else " "
    return sep.join(parts).strip()
