from __future__ import annotations

import queue

import numpy as np
import pytest

from livesub.audio.capture import AudioCapture, Mixer
from livesub.config import AudioConfig, ConfigError


def test_mixer_adds_secondary_and_pads_with_silence():
    m = Mixer(sample_rate=100, max_lag_s=1.0)
    m.push_secondary(np.full(3, 0.25, dtype=np.float32))
    out = m.mix_primary(np.full(5, 0.5, dtype=np.float32))
    np.testing.assert_allclose(out, [0.75, 0.75, 0.75, 0.5, 0.5])
    # 系統聲音已用完：只剩麥克風
    np.testing.assert_allclose(m.mix_primary(np.full(2, 0.1, dtype=np.float32)), [0.1, 0.1])


def test_mixer_clips_and_bounds_lag():
    m = Mixer(sample_rate=10, max_lag_s=0.5)  # 最多保留 5 個樣本
    m.push_secondary(np.arange(1, 9, dtype=np.float32) / 10)  # 0.1 ~ 0.8，只留最後 5 個
    out = m.mix_primary(np.full(6, 0.5, dtype=np.float32))
    np.testing.assert_allclose(out, [0.9, 1.0, 1.0, 1.0, 1.0, 0.5])


class _FakePa:
    def get_default_wasapi_loopback(self):
        return {"index": 10, "name": "喇叭 [Loopback]"}

    def get_default_input_device_info(self):
        return {"index": 2, "name": "麥克風"}

    def get_device_info_by_index(self, i):
        return {"index": i, "name": f"dev{i}"}


def _devices(**kw):
    cap = AudioCapture(AudioConfig(**kw), queue.Queue(), clock=lambda: 0.0)
    return [d["index"] for d in cap._resolve_devices(_FakePa())]


def test_resolve_devices_per_source():
    assert _devices(source="loopback") == [10]
    assert _devices(source="mic") == [2]
    assert _devices(source="both") == [2, 10]  # 麥克風在前（主時鐘）
    assert _devices(source="both", device_index=7, mic_device_index=3) == [3, 7]
    assert _devices(source="loopback", device_index=7, mic_device_index=3) == [7]


def test_bad_source_rejected():
    with pytest.raises(ConfigError):
        AudioConfig(source="stereo")
