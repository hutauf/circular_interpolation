from __future__ import annotations

import numpy as np

from raw_stream_wrapper import interpolate_raw_circular_stream


def _integrate(t: np.ndarray, v: np.ndarray, initial: float = 11.0) -> np.ndarray:
    angle = np.empty_like(v)
    angle[0] = initial
    angle[1:] = initial + np.cumsum(0.5 * (v[:-1] + v[1:]) * np.diff(t))
    return np.mod(angle, 360.0)


def test_boundary_and_middle_standstill_are_preserved():
    fs = 1000.0
    t = np.arange(0.0, 2.001, 1.0 / fs)
    v = np.zeros_like(t)
    # Smooth acceleration/deceleration: a real motor cannot jump from full speed
    # to zero in one sample.
    m = (t >= 0.3) & (t < 0.5)
    v[m] = 6000.0 * (t[m] - 0.3) / 0.2
    v[(t >= 0.5) & (t < 0.7)] = 6000.0
    m = (t >= 0.7) & (t < 0.9)
    v[m] = 6000.0 * (1.0 - (t[m] - 0.7) / 0.2)
    m = (t >= 1.1) & (t < 1.3)
    v[m] = 6000.0 * (t[m] - 1.1) / 0.2
    v[(t >= 1.3) & (t < 1.5)] = 6000.0
    m = (t >= 1.5) & (t < 1.7)
    v[m] = 6000.0 * (1.0 - (t[m] - 1.5) / 0.2)
    raw = _integrate(t, v)

    result = interpolate_raw_circular_stream(t, raw)

    assert not np.any(result.repaired_mask)
    assert np.any(result.preserved_standstill_mask)


def test_single_held_sample_at_high_speed_is_repaired():
    fs = 1000.0
    t = np.arange(0.0, 1.001, 1.0 / fs)
    raw = np.mod(19.0 + 60_000.0 * t, 360.0)
    raw[500] = raw[499]

    result = interpolate_raw_circular_stream(t, raw)

    assert result.repaired_mask[500]
    assert any(d.kind == "held_dropout" for d in result.decisions)


def test_slow_quantized_motion_is_not_repaired():
    fs = 1000.0
    t = np.arange(0.0, 1.001, 1.0 / fs)
    true = 17.0 + 30.0 * t  # 5 rpm
    raw = np.mod(np.round(true / 1.0) * 1.0, 360.0)

    result = interpolate_raw_circular_stream(t, raw)

    assert not np.any(result.repaired_mask)
    assert np.any(result.preserved_standstill_mask)


def test_reversal_plateau_is_left_ambiguous():
    fs = 1000.0
    t = np.arange(0.0, 1.501, 1.0 / fs)
    v = np.where(t < 0.75, 12_000.0, -12_000.0)
    raw = _integrate(t, v)
    raw[700:800] = raw[699]

    result = interpolate_raw_circular_stream(t, raw)

    assert not np.any(result.repaired_mask[700:800])
    assert np.all(result.ambiguous_mask[700:800])
