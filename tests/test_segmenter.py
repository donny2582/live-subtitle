from __future__ import annotations

import numpy as np
from conftest import chunks, silence, tone

from livesub.audio import EnergyVad, UtteranceSegmenter
from livesub.config import SegmenterConfig


def run(seg: UtteranceSegmenter, signal: np.ndarray, start: float = 0.0):
    out = []
    for part, t in chunks(signal, start):
        out.extend(seg.push(part, t))
    return out


def make(**kw) -> UtteranceSegmenter:
    return UtteranceSegmenter(SegmenterConfig(**kw), EnergyVad(), emit_partials=True)


def test_single_utterance_with_partials():
    seg = make()
    out = run(seg, np.concatenate([silence(0.5), tone(1.5), silence(1.0)]))
    finals = [u for u in out if u.is_final]
    partials = [u for u in out if not u.is_final]
    assert len(finals) == 1
    f = finals[0]
    assert f.seq == 1
    assert abs(f.started_at - (0.5 - 0.24)) < 0.05  # 含 pre-roll
    assert 1.5 < f.duration < 2.1  # 語音 + pre-roll + 保留尾音
    assert partials and all(p.seq == 1 for p in partials)
    assert all(p.duration < f.duration for p in partials)


def test_short_noise_is_ignored():
    seg = make()
    assert run(seg, np.concatenate([tone(0.12), silence(1.0)])) == []


def test_forced_cut_on_long_speech():
    seg = make(max_utterance_s=4.0)
    out = [u for u in run(seg, tone(9.0)) if u.is_final]
    assert len(out) == 2
    assert all(u.forced_cut for u in out)
    assert [u.seq for u in out] == [1, 2]


def test_idle_flush_when_loopback_stops_sending():
    seg = make()
    out = run(seg, tone(1.0))
    assert not [u for u in out if u.is_final]
    assert seg.on_idle(1.3) == []  # 還沒到門檻
    finals = seg.on_idle(1.9)
    assert len(finals) == 1 and finals[0].is_final


def test_gap_between_chunks_finalizes_previous():
    seg = make()
    run(seg, tone(1.0))
    out = run(seg, tone(1.0), start=3.0)  # 中間 2 秒沒有資料
    finals = [u for u in out if u.is_final]
    assert len(finals) == 1 and finals[0].ended_at <= 1.05


def test_partials_disabled():
    seg = UtteranceSegmenter(SegmenterConfig(), EnergyVad(), emit_partials=False)
    out = run(seg, np.concatenate([tone(2.0), silence(1.0)]))
    assert all(u.is_final for u in out) and len(out) == 1


def test_reset_discards_state():
    seg = make()
    run(seg, tone(1.0))
    seg.reset()
    assert seg.flush() == []
