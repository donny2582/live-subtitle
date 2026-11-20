"""設定結構（frozen dataclass）。

每個模組一個區段，每個雲端服務一個子區段，例如：

    [asr]            共用：引擎選擇、允許語言、中間結果
    [asr.local]      本機 faster-whisper
    [asr.cloud]      雲端語音辨識（OpenAI 相容 /audio/transcriptions）
    [translate]      共用：引擎選擇、上下文、中間結果翻譯
    [translate.openai] / [translate.claude] / [translate.deepl]

新增雲端服務時：在這裡加一個子區段 dataclass，並在對應的 registry 註冊即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "AppConfig",
    "AsrConfig",
    "AudioConfig",
    "BudgetConfig",
    "ClaudeConfig",
    "CloudAsrConfig",
    "ConfigError",
    "DeepLConfig",
    "LocalAsrConfig",
    "OpenAIConfig",
    "OutputConfig",
    "OverlayConfig",
    "SegmenterConfig",
    "TranslateConfig",
]


class ConfigError(ValueError):
    """設定內容不合法。"""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


# ====================================================================== 音訊
@dataclass(frozen=True, slots=True)
class AudioConfig:
    """音訊擷取設定。"""

    source: str = "loopback"  # loopback = 系統播放的聲音；mic = 麥克風
    device_index: int = -1  # -1 = 自動使用預設裝置
    target_sample_rate: int = 16_000
    max_queue_chunks: int = 400  # 約 20 秒的緩衝（每塊 50 ms）

    def __post_init__(self) -> None:
        _check(self.source in {"loopback", "mic"}, "audio.source 只能是 loopback 或 mic")
        _check(self.target_sample_rate == 16_000, "audio.target_sample_rate 目前僅支援 16000")
        _check(self.max_queue_chunks >= 20, "audio.max_queue_chunks 至少 20")


@dataclass(frozen=True, slots=True)
class SegmenterConfig:
    """VAD 切句設定。"""

    frame_ms: int = 30
    vad_aggressiveness: int = 2  # 0~3，越大越嚴格（背景音樂多時調高）
    min_speech_ms: int = 300
    end_silence_ms: int = 600  # 靜音多久視為一句結束
    soft_end_silence_ms: int = 250  # 句子偏長時改用較短靜音門檻，避免硬切在字中間
    max_utterance_s: float = 10.0
    pre_roll_ms: int = 240
    idle_flush_s: float = 0.8  # loopback 沒有任何資料進來多久後強制結束當前句
    partial_interval_s: float = 0.7  # 0 = 關閉中間結果

    def __post_init__(self) -> None:
        _check(self.frame_ms in {10, 20, 30}, "segmenter.frame_ms 只能是 10、20、30")
        _check(0 <= self.vad_aggressiveness <= 3, "segmenter.vad_aggressiveness 須介於 0~3")
        _check(self.max_utterance_s >= 2.0, "segmenter.max_utterance_s 至少 2 秒")
        _check(self.end_silence_ms >= self.frame_ms, "segmenter.end_silence_ms 太小")
        _check(self.partial_interval_s >= 0, "segmenter.partial_interval_s 不可為負")


# ====================================================================== 語音辨識
@dataclass(frozen=True, slots=True)
class LocalAsrConfig:
    """本機 faster-whisper（需安裝 requirements-local-asr.txt）。"""

    model: str = "large-v3-turbo"
    cpu_model: str = "small"  # 沒有 GPU 時改用的模型
    device: str = "auto"  # auto / cuda / cpu
    compute_type: str = "auto"  # auto = GPU 用 float16，CPU 用 int8
    model_dir: str = "models"
    beam_size: int = 1
    min_allowed_language_prob: float = 0.05
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4

    def __post_init__(self) -> None:
        _check(self.device in {"auto", "cuda", "cpu"}, "asr.local.device 只能是 auto、cuda、cpu")
        _check(self.beam_size >= 1, "asr.local.beam_size 至少 1")


@dataclass(frozen=True, slots=True)
class CloudAsrConfig:
    """雲端語音辨識：任何 OpenAI 相容的 /audio/transcriptions（OpenAI、Groq…）。"""

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini-transcribe"
    api_key_env: str = "OPENAI_API_KEY"
    # auto = whisper 系列模型用 verbose_json（會回傳語言），其他用 json（語言由文字判斷）
    response_format: str = "auto"
    timeout_s: float = 10.0
    max_retries: int = 1
    usd_per_minute: float = 0.003  # 依實際模型價格填寫，供預算計算
    min_billed_s: float = 0.0  # 每次請求最少計費秒數（Groq 為 10 秒）

    def __post_init__(self) -> None:
        _check(self.response_format in {"auto", "json", "verbose_json"},
               "asr.cloud.response_format 只能是 auto、json、verbose_json")
        _check(self.timeout_s > 0, "asr.cloud.timeout_s 必須大於 0")
        _check(self.max_retries >= 0, "asr.cloud.max_retries 不可為負")
        _check(self.usd_per_minute >= 0, "asr.cloud.usd_per_minute 不可為負")


@dataclass(frozen=True, slots=True)
class AsrConfig:
    """語音辨識共用設定。"""

    engine: str = "cloud"  # cloud / local
    allowed_languages: tuple[str, ...] = ("en", "ja", "ko", "zh")
    sticky_language_prob: float = 0.6  # 偵測信心低於此值時沿用上一句的語言
    glossary_prompt: bool = False  # 把術語原文當作辨識提示詞
    # auto = 只有本機 GPU 才開啟（雲端每次中間結果都要重送整段音訊，費用約變 4 倍）
    partial_results: str = "auto"
    max_backlog: int = 4
    local: LocalAsrConfig = field(default_factory=LocalAsrConfig)
    cloud: CloudAsrConfig = field(default_factory=CloudAsrConfig)

    def __post_init__(self) -> None:
        _check(self.engine in {"cloud", "local"}, "asr.engine 只能是 cloud 或 local")
        _check(self.partial_results in {"auto", "on", "off"}, "asr.partial_results 只能是 auto/on/off")
        _check(self.max_backlog >= 1, "asr.max_backlog 至少 1")


# ====================================================================== 翻譯
@dataclass(frozen=True, slots=True)
class OpenAIConfig:
    """OpenAI 相容 /chat/completions（OpenAI、Gemini、DeepSeek、本機 Ollama…）。"""

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 8.0
    max_retries: int = 1
    temperature: float = 0.2
    input_usd_per_million_tokens: float = 0.40
    output_usd_per_million_tokens: float = 1.60

    def __post_init__(self) -> None:
        _check(self.timeout_s > 0, "translate.openai.timeout_s 必須大於 0")
        _check(self.max_retries >= 0, "translate.openai.max_retries 不可為負")


@dataclass(frozen=True, slots=True)
class ClaudeConfig:
    """Anthropic Claude（官方 anthropic SDK）。"""

    model: str = "claude-opus-5-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    effort: str = "low"  # low / medium / high / xhigh / max；字幕翻譯用 low 最快最省
    thinking: str = "auto"  # auto = 依模型預設；off = 關閉思考（僅 Haiku 5.5 等支援的模型）
    refusal_fallback: bool = True  # 模型拒答時由伺服器自動改用備援模型（僅支援的模型會送出）
    max_tokens: int = 2048  # 含思考 token
    timeout_s: float = 15.0
    max_retries: int = 1
    input_usd_per_million_tokens: float = 4.00
    output_usd_per_million_tokens: float = 20.00

    def __post_init__(self) -> None:
        _check(self.effort in {"low", "medium", "high", "xhigh", "max"},
               "translate.claude.effort 只能是 low、medium、high、xhigh、max")
        _check(self.thinking in {"auto", "off"}, "translate.claude.thinking 只能是 auto 或 off")
        _check(self.max_tokens >= 256, "translate.claude.max_tokens 至少 256")
        _check(self.timeout_s > 0, "translate.claude.timeout_s 必須大於 0")


@dataclass(frozen=True, slots=True)
class DeepLConfig:
    """DeepL API（Free 金鑰以 :fx 結尾）。"""

    api_key_env: str = "DEEPL_API_KEY"
    timeout_s: float = 8.0
    usd_per_million_chars: float = 0.0  # DeepL Free 填 0；Pro 依方案填入


@dataclass(frozen=True, slots=True)
class TranslateConfig:
    """翻譯共用設定。"""

    engine: str = "openai"  # openai / claude / deepl / none
    context_turns: int = 3
    partial_translation: bool = True
    partial_min_interval_s: float = 1.2
    partial_min_new_chars: int = 6
    taiwan_convert: bool = True  # 使用 OpenCC s2twp 統一成台灣用語
    glossary_path: str = "glossary.csv"
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    deepl: DeepLConfig = field(default_factory=DeepLConfig)

    def __post_init__(self) -> None:
        _check(self.engine in {"openai", "claude", "deepl", "none"},
               "translate.engine 只能是 openai、claude、deepl、none")
        _check(0 <= self.context_turns <= 10, "translate.context_turns 須介於 0~10")


# ====================================================================== 其他
@dataclass(frozen=True, slots=True)
class BudgetConfig:
    """雲端費用控管（語音辨識＋翻譯合計）。各服務單價寫在各自的區段。"""

    monthly_limit_usd: float = 5.0
    partial_cutoff_ratio: float = 0.7  # 用到預算 70% 後停止中間結果的雲端請求
    ledger_path: str = "usage_ledger.json"

    def __post_init__(self) -> None:
        _check(self.monthly_limit_usd >= 0, "budget.monthly_limit_usd 不可為負")
        _check(0 < self.partial_cutoff_ratio <= 1, "budget.partial_cutoff_ratio 須介於 0~1")


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """字幕浮層外觀。"""

    show_source: bool = True
    font_family: str = "Microsoft JhengHei UI"
    font_size: int = 26
    source_font_size: int = 15
    max_lines: int = 2
    width_ratio: float = 0.7
    bottom_margin: int = 90
    bg_opacity: float = 0.6
    click_through: bool = False
    clear_after_s: float = 8.0

    def __post_init__(self) -> None:
        _check(0.2 <= self.width_ratio <= 1.0, "overlay.width_ratio 須介於 0.2~1.0")
        _check(0.0 <= self.bg_opacity <= 1.0, "overlay.bg_opacity 須介於 0~1")
        _check(1 <= self.max_lines <= 6, "overlay.max_lines 須介於 1~6")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """逐字稿輸出設定。"""

    transcript_dir: str = "transcripts"
    write_srt: bool = True
    write_txt: bool = True
    write_jsonl: bool = True
    srt_include_source: bool = True


@dataclass(frozen=True, slots=True)
class AppConfig:
    """整體設定。"""

    audio: AudioConfig = field(default_factory=AudioConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
