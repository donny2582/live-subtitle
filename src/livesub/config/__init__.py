"""設定：schema（資料結構）＋ loader（讀檔、舊格式相容、路徑與金鑰工具）＋ writer（設定視窗存檔）。"""

from dataclasses import replace

from .loader import (
    app_base_dir,
    get_secret,
    is_local_url,
    load_config,
    load_config_with_notes,
    parse_config,
    resolve_path,
)
from .schema import (
    AppConfig,
    AsrConfig,
    AudioConfig,
    BudgetConfig,
    ClaudeConfig,
    CloudAsrConfig,
    ConfigError,
    DeepLConfig,
    LocalAsrConfig,
    OpenAIConfig,
    OutputConfig,
    OverlayConfig,
    SegmenterConfig,
    TranslateConfig,
)
from .writer import save_config_updates, update_toml_text

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
    "app_base_dir",
    "get_secret",
    "is_local_url",
    "load_config",
    "load_config_with_notes",
    "parse_config",
    "replace",
    "resolve_path",
    "save_config_updates",
    "update_toml_text",
]
