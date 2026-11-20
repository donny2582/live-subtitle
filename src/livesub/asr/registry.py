"""依 asr.engine 建立語音辨識器。新增服務時只要在 _BUILDERS 加一行。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from ..config import AsrConfig, get_secret, resolve_path
from ..core.budget import BudgetLedger
from .base import Transcriber


def _local(cfg: AsrConfig, base: Path, env: Mapping[str, str], ledger: BudgetLedger | None,
           prompt: str | None) -> Transcriber:
    from .gpu import register_cuda_dlls
    from .local_whisper import WhisperTranscriber

    register_cuda_dlls()
    return WhisperTranscriber(cfg, resolve_path(base, cfg.local.model_dir), prompt=prompt)


def _cloud(cfg: AsrConfig, base: Path, env: Mapping[str, str], ledger: BudgetLedger | None,
           prompt: str | None) -> Transcriber:
    from .cloud_openai import CloudTranscriber

    return CloudTranscriber(cfg, get_secret(env, cfg.cloud.api_key_env), ledger, prompt=prompt)


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
