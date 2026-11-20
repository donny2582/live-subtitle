"""設定視窗的服務選項（純資料，不依賴 Qt）。

要在設定視窗新增一個選項，只要在下面的清單加一筆：
* label     下拉選單顯示的文字
* updates   選這個選項時要寫進 config.toml 的設定（區段.鍵 → 值）
* cost      2 小時會議的大約費用說明（只是顯示用）
* key_env   需要的 API 金鑰環境變數（金鑰分頁會提示）

模型的單價一定要跟著模型一起寫進 updates，否則預算計算會不準。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..config import AppConfig, is_local_url


@dataclass(frozen=True)
class Preset:
    label: str
    updates: Mapping[str, Any]
    cost: str
    key_env: str = ""
    tip: str = ""


ASR_PRESETS: tuple[Preset, ...] = (
    Preset(
        "雲端 OpenAI gpt-4o-mini-transcribe（推薦）",
        {
            "asr.engine": "cloud",
            "asr.cloud.base_url": "https://api.openai.com/v1",
            "asr.cloud.model": "gpt-4o-mini-transcribe",
            "asr.cloud.api_key_env": "OPENAI_API_KEY",
            "asr.cloud.usd_per_minute": 0.003,
            "asr.cloud.min_billed_s": 0.0,
        },
        "約 0.25 美元",
        "OPENAI_API_KEY",
        "便宜、準確度好。不回報語言，程式以文字判斷（只有漢字的日文短句可能被當成中文）。",
    ),
    Preset(
        "雲端 OpenAI gpt-4o-transcribe（較準）",
        {
            "asr.engine": "cloud",
            "asr.cloud.base_url": "https://api.openai.com/v1",
            "asr.cloud.model": "gpt-4o-transcribe",
            "asr.cloud.api_key_env": "OPENAI_API_KEY",
            "asr.cloud.usd_per_minute": 0.006,
            "asr.cloud.min_billed_s": 0.0,
        },
        "約 0.5 美元",
        "OPENAI_API_KEY",
        "比 mini 稍準，價格兩倍。",
    ),
    Preset(
        "雲端 OpenAI whisper-1（會回報語言）",
        {
            "asr.engine": "cloud",
            "asr.cloud.base_url": "https://api.openai.com/v1",
            "asr.cloud.model": "whisper-1",
            "asr.cloud.api_key_env": "OPENAI_API_KEY",
            "asr.cloud.usd_per_minute": 0.006,
            "asr.cloud.min_billed_s": 0.0,
        },
        "約 0.5 美元",
        "OPENAI_API_KEY",
        "語言判斷最可靠（服務直接回報語言），適合中日混雜的場合。",
    ),
    Preset(
        "雲端 Groq whisper-large-v3-turbo（最便宜）",
        {
            "asr.engine": "cloud",
            "asr.cloud.base_url": "https://api.groq.com/openai/v1",
            "asr.cloud.model": "whisper-large-v3-turbo",
            "asr.cloud.api_key_env": "GROQ_API_KEY",
            "asr.cloud.usd_per_minute": 0.000667,
            "asr.cloud.min_billed_s": 10.0,
        },
        "約 0.1 美元以下",
        "GROQ_API_KEY",
        "需要另外申請 Groq 帳號與金鑰。速度很快，會回報語言。",
    ),
    Preset(
        "本機 faster-whisper（需 NVIDIA 顯示卡）",
        {"asr.engine": "local"},
        "0 元",
        "",
        "需先執行 scripts\\setup.ps1 -Gpu。沒有顯示卡會很慢。",
    ),
)

TRANSLATE_PRESETS: tuple[Preset, ...] = (
    Preset(
        "Claude Haiku 5.5（推薦，便宜）",
        {
            "translate.engine": "claude",
            "translate.claude.model": "claude-haiku-5-5",
            "translate.claude.thinking": "off",
            "translate.claude.input_usd_per_million_tokens": 0.10,
            "translate.claude.output_usd_per_million_tokens": 0.50,
        },
        "約 0.05 美元（開中間結果約 0.2）",
        "ANTHROPIC_API_KEY",
        "速度快、價格低，字幕翻譯品質已足夠。",
    ),
    Preset(
        "Claude Opus 5.5（品質最好，很貴）",
        {
            "translate.engine": "claude",
            "translate.claude.model": "claude-opus-5-5",
            "translate.claude.thinking": "auto",
            "translate.claude.input_usd_per_million_tokens": 4.00,
            "translate.claude.output_usd_per_million_tokens": 20.00,
        },
        "約 3 美元以上（開中間結果 12 以上）",
        "ANTHROPIC_API_KEY",
        "專業內容翻得最好，但每月 5 美元預算大概只夠 1～2 場會議。",
    ),
    Preset(
        "OpenAI gpt-4.1-mini",
        {
            "translate.engine": "openai",
            "translate.openai.base_url": "https://api.openai.com/v1",
            "translate.openai.model": "gpt-4.1-mini",
            "translate.openai.api_key_env": "OPENAI_API_KEY",
            "translate.openai.input_usd_per_million_tokens": 0.40,
            "translate.openai.output_usd_per_million_tokens": 1.60,
        },
        "約 0.2 美元（開中間結果約 0.8）",
        "OPENAI_API_KEY",
        "和語音辨識共用 OpenAI 金鑰。",
    ),
    Preset(
        "OpenAI gpt-4o-mini（較省）",
        {
            "translate.engine": "openai",
            "translate.openai.base_url": "https://api.openai.com/v1",
            "translate.openai.model": "gpt-4o-mini",
            "translate.openai.api_key_env": "OPENAI_API_KEY",
            "translate.openai.input_usd_per_million_tokens": 0.15,
            "translate.openai.output_usd_per_million_tokens": 0.60,
        },
        "約 0.07 美元（開中間結果約 0.3）",
        "OPENAI_API_KEY",
    ),
    Preset(
        "DeepL",
        {"translate.engine": "deepl"},
        "Free 方案 0 元（每月 50 萬字元）",
        "DEEPL_API_KEY",
        "不會參考上下文與術語表，專有名詞較容易翻錯。",
    ),
    Preset("不翻譯（只顯示原文）", {"translate.engine": "none"}, "0 元"),
)

# 金鑰分頁顯示的項目：(環境變數, 說明)
API_KEYS: tuple[tuple[str, str], ...] = (
    ("OPENAI_API_KEY", "OpenAI（sk-…）：雲端語音辨識、OpenAI 翻譯"),
    ("ANTHROPIC_API_KEY", "Anthropic（sk-ant-…）：Claude 翻譯"),
    ("GROQ_API_KEY", "Groq（gsk_…）：Groq 語音辨識"),
    ("DEEPL_API_KEY", "DeepL（…:fx）：DeepL 翻譯"),
)


def get_value(cfg: AppConfig, dotted: str) -> Any:
    """依「區段.鍵」取出目前設定值，例如 'translate.claude.model'。"""
    obj: Any = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def matches(cfg: AppConfig, preset: Preset) -> bool:
    """目前設定是否就是這個選項。只比對會影響行為的鍵（engine、model、base_url）。"""
    engine_key = next((k for k in preset.updates if k.endswith(".engine")), None)
    if engine_key is None or get_value(cfg, engine_key) != preset.updates[engine_key]:
        return False
    for key, value in preset.updates.items():
        if key.endswith((".model", ".base_url")) and get_value(cfg, key) != value:
            return False
    return True


def current_index(cfg: AppConfig, presets: tuple[Preset, ...]) -> int:
    """回傳符合目前設定的選項索引；都不符合（使用者手動改過設定檔）回傳 -1。"""
    return next((i for i, p in enumerate(presets) if matches(cfg, p)), -1)


def asr_key(cfg: AppConfig) -> str:
    """目前語音辨識設定需要的金鑰環境變數（不需要時回傳空字串）。"""
    if cfg.asr.engine == "cloud" and not is_local_url(cfg.asr.cloud.base_url):
        return cfg.asr.cloud.api_key_env
    return ""


def translate_key(cfg: AppConfig) -> str:
    """目前翻譯設定需要的金鑰環境變數（不需要時回傳空字串）。"""
    tr = cfg.translate
    if tr.engine == "openai" and not is_local_url(tr.openai.base_url):
        return tr.openai.api_key_env
    if tr.engine == "claude":
        return tr.claude.api_key_env
    if tr.engine == "deepl":
        return tr.deepl.api_key_env
    return ""
