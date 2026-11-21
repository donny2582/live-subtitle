"""依 asr.engine 建立語音辨識器。新增服務時只要在 _BUILDERS 加一行。

這是「工廠」模式：app.py 不需要知道有哪些辨識器，只要呼叫 build_transcriber()，
這裡依設定（cloud / local）挑出對應的建立函式。
各建立函式在函式內才 import，沒用到的服務（例如本機 Whisper）就不必安裝它的套件。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from ..config import AsrConfig, get_secret, resolve_path
from ..core.budget import BudgetLedger
from .base import Transcriber


def _local(cfg: AsrConfig, base: Path, env: Mapping[str, str], ledger: BudgetLedger | None,
           prompt: str | None) -> Transcriber:
    """本機 faster-whisper。"""
    from .gpu import register_cuda_dlls
    from .local_whisper import WhisperTranscriber

    register_cuda_dlls()  # 必須在載入 faster-whisper 之前做
    return WhisperTranscriber(cfg, resolve_path(base, cfg.local.model_dir), prompt=prompt)


def _cloud(cfg: AsrConfig, base: Path, env: Mapping[str, str], ledger: BudgetLedger | None,
           prompt: str | None) -> Transcriber:
    """雲端（OpenAI、Groq 等 OpenAI 相容服務）。"""
    from .cloud_openai import CloudTranscriber

    return CloudTranscriber(cfg, get_secret(env, cfg.cloud.api_key_env), ledger, prompt=prompt)


# asr.engine 的值 → 建立函式
_BUILDERS: dict[str, Callable[..., Transcriber]] = {
    "local": _local,
    "cloud": _cloud,
}


def build_transcriber(
    cfg: AsrConfig,
    base: Path,
    env: Mapping[str, str],
    ledger: BudgetLedger | None,
    prompt: str | None = None,
) -> Transcriber:
    """依設定建立辨識器（模型載入／金鑰檢查在 load() 才做，不會卡住 UI 啟動）。"""
    return _BUILDERS[cfg.engine](cfg, base, env, ledger, prompt)
