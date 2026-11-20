from __future__ import annotations

import numpy as np
import pytest

SR = 16_000


def tone(seconds: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def chunks(signal: np.ndarray, start: float = 0.0, chunk_s: float = 0.05):
    """以 (樣本, 結束時間) 形式切塊，模擬音訊 callback。"""
    n = int(chunk_s * SR)
    for i in range(0, signal.size, n):
        part = signal[i : i + n]
        yield part, start + (i + part.size) / SR


@pytest.fixture
def sr() -> int:
    return SR
