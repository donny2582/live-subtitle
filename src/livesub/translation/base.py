"""翻譯模組共用：介面、結果、錯誤、提示詞、後處理、上下文。

新增翻譯服務的步驟：
1. 在 config/schema.py 加一個子區段（例如 [translate.myprovider]）。
2. 在本資料夾新增 myprovider.py，實作 `Translator` 介面。
3. 在 registry.py 註冊建立函式。

LLM（大型語言模型，例如 Claude、GPT）翻譯的做法：
    系統提示詞（翻譯規則＋術語）＋ 前幾句的原文／譯文（上下文）＋ 這一句原文 → 模型回傳譯文
回傳後再經過 clean_translation()（去掉多餘的引號、前綴）與 ChineseNormalizer（轉台灣用語）。
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from ..core.glossary import GlossaryEntry, render_prompt_section, select_relevant
from ..core.http import CloudError

log = logging.getLogger(__name__)

LANGUAGE_NAMES = {"en": "英文", "ja": "日文", "ko": "韓文", "zh": "中文"}

# 給 LLM 的翻譯規則。{lang} 會換成「英文」等；{glossary} 會換成這句用到的術語清單
SYSTEM_PROMPT = """你是專業的同步口譯員，負責把{lang}語音辨識結果翻譯成繁體中文（台灣用語），作為即時字幕。
規則：
1. 只輸出譯文本身，不要加引號、說明、拼音或原文。
2. 使用台灣慣用詞彙（例如：軟體、網路、影片、品質、資料、專案、行人、機車）。
3. 語音辨識可能有同音錯字或斷句不完整，請依上下文合理推斷，但不要補充原文沒有的內容。
4. 人名、公司名、產品名若無通用中文譯名，保留原文。
5. 口語、精簡，適合一眼讀完。{glossary}"""

# 翻譯中間結果時額外加上的規則（避免模型自己把沒說完的句子補完）
PARTIAL_HINT = "\n6. 這是講者還沒說完的句子片段，請只翻譯目前已出現的內容，不要猜測後文。"


class TranslationError(CloudError):
    """翻譯失敗（網路、金鑰、額度等）。"""


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """一次翻譯的結果。"""

    text: str  # 譯文
    latency_s: float  # 花了幾秒
    cost_usd: float = 0.0  # 這次花了多少錢


class Translator(Protocol):
    """所有翻譯器都要提供的介面，pipeline 只透過這些方法使用它們。"""

    name: str  # 例如 "claude:claude-haiku-5-5"；"none" 代表不翻譯
    is_cloud: bool  # 是否會花錢

    def translate(self, text: str, source_lang: str, partial: bool = False) -> TranslationResult:
        """翻譯一句。失敗丟出 TranslationError。partial=True 代表這是講到一半的句子。"""
        ...

    def warmup(self) -> None:
        """啟動前的檢查（不花錢）；設定有問題時丟出 TranslationError。"""
        ...


# ---------------------------------------------------------------------- 提示詞
def build_system_prompt(text: str, source_lang: str, partial: bool, glossary: list[GlossaryEntry]) -> str:
    """LLM 引擎共用的系統提示詞（只附上這句有出現的術語）。"""
    # 只放相關術語：提示詞越短越便宜，模型也比較不會被無關的術語干擾
    section = render_prompt_section(select_relevant(glossary, text))
    system = SYSTEM_PROMPT.format(lang=LANGUAGE_NAMES.get(source_lang, "外語"), glossary=section)
    return system + (PARTIAL_HINT if partial else "")


class ContextHistory:
    """最近幾句（原文, 譯文），讓 LLM 用詞前後一致。執行緒安全。

    deque(maxlen=N)：滿了之後再加入，最舊的會自動被擠掉，永遠只保留最近 N 句。
    """

    def __init__(self, turns: int) -> None:
        self._items: deque[tuple[str, str]] = deque(maxlen=turns)
        self._lock = threading.Lock()

    def add(self, source: str, target: str) -> None:
        """記下一句定稿的原文與譯文（turns = 0 時不記）。"""
        if self._items.maxlen:
            with self._lock:
                self._items.append((source, target))

    def as_messages(self) -> list[dict[str, str]]:
        """轉成對話格式：使用者說原文、助理回譯文，模型會照著前面的譯法繼續翻。"""
        with self._lock:
            out: list[dict[str, str]] = []
            for src, tgt in self._items:
                out.append({"role": "user", "content": src})
                out.append({"role": "assistant", "content": tgt})
            return out

    def sources(self) -> list[str]:
        """只取原文（給不支援對話格式的服務使用）。"""
        with self._lock:
            return [s for s, _ in self._items]


# ---------------------------------------------------------------------- 後處理
class ChineseNormalizer:
    """以 OpenCC s2twp 轉為台灣正體與用語；未安裝時不做轉換。

    s2twp = 簡體 → 台灣正體，並轉換慣用詞（例如「软件」→「軟體」、「视频」→「影片」）。
    就算提示詞要求台灣用語，模型偶爾還是會出現大陸用詞，這裡再保險一次。
    """

    def __init__(self, enabled: bool) -> None:
        self._cc: Any = None
        if not enabled:
            return
        try:
            from opencc import OpenCC

            self._cc = OpenCC("s2twp")
        except Exception as exc:  # noqa: BLE001 - 套件缺失或設定檔缺失
            log.warning("OpenCC 無法使用（%s），將不進行台灣用語轉換", exc)

    def __call__(self, text: str) -> str:
        return self._cc.convert(text) if self._cc and text else text


# 部分模型（例如 DeepSeek、Qwen）會把思考過程包在 <think>…</think> 裡一起輸出
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LABEL_RE = re.compile(r"^\s*(譯文|翻譯|中文翻譯|translation)\s*[:：]\s*", re.IGNORECASE)
_QUOTE_PAIRS = (("「", "」"), ("『", "』"), ('"', '"'), ("“", "”"), ("'", "'"))


def clean_translation(text: str) -> str:
    """移除 LLM 常見的多餘輸出：思考標籤、「譯文：」前綴、外層引號。"""
    text = _THINK_RE.sub("", text).strip()
    text = _LABEL_RE.sub("", text).strip()
    for left, right in _QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            inner = text[len(left) : -len(right)]
            # 裡面還有同樣的引號，代表引號是句子內容的一部分（例如「A」和「B」），不能拿掉
            if left not in inner and right not in inner:
                text = inner.strip()
            break
    return text


def token_cost(input_tokens: int, output_tokens: int, in_price: float, out_price: float) -> float:
    """依每百萬 token 單價計算美元花費。"""
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000
