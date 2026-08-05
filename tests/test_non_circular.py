from __future__ import annotations

import numpy as np
import pytest

from circular_interpolation import (
    interpolate_circular_auto,
    interpolate_raw_circular_stream,
)


def test_period_none_skips_wrap_and_unwrap_across_360():
    time_s = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([340.0, 350.0, np.nan, 10.0, 20.0])
    invalid = np.isnan(values)

    circular = interpolate_circular_auto(time_s, values, invalid)
    linear = interpolate_circular_auto(time_s, values, invalid, period=None)

    assert np.isclose(circular.angle_deg[2], 0.0)
    assert np.isclose(linear.angle_deg[2], 180.0)
    assert np.array_equal(linear.angle_deg, linear.unwrapped_deg)
    assert linear.chosen_model_by_gap[0][2] == "linear"


def test_period_none_repairs_resamples_and_extrapolates_without_wrapping():
    time_s = np.arange(0.0, 0.101, 0.001)
    truth = 350.0 + 720.0 * time_s
    values = truth.copy()
    invalid = (time_s >= 0.040) & (time_s <= 0.060)
    values[invalid] = np.nan
    target = np.unique(
        np.r_[
            np.linspace(-0.020, 0.120, 431),
            np.array([0.0405, 0.0473, 0.0599]),
        ]
    )

    result = interpolate_circular_auto(
        time_s,
        values,
        invalid,
        period=None,
        target_time_s=target,
    )

    expected = 350.0 + 720.0 * target
    assert np.array_equal(result.time_s, target)
    assert np.allclose(result.angle_deg, expected, atol=1e-6)
    assert np.array_equal(result.angle_deg, result.unwrapped_deg)
    assert np.max(result.angle_deg) > 360.0


def test_raw_stream_period_none_repairs_stale_values_and_preserves_anchor():
    sample_rate_hz = 1000.0
    time_s = np.arange(0.0, 1.001, 1.0 / sample_rate_hz)
    truth = -1000.0 + 60_000.0 * time_s
    raw = truth.copy()
    anchor = 499
    repeat_start = 500
    repeat_end = 504
    raw[repeat_start:repeat_end] = raw[anchor]

    result = interpolate_raw_circular_stream(time_s, raw, period=None)

    assert not result.repaired_mask[anchor]
    assert np.all(result.repaired_mask[repeat_start:repeat_end])
    assert np.allclose(
        result.angle_deg[repeat_start:repeat_end],
        truth[repeat_start:repeat_end],
        atol=1e-5,
    )
    assert np.array_equal(result.angle_deg, result.unwrapped_deg)
    assert np.max(result.angle_deg) > 360.0


def test_raw_stream_period_none_without_repairs_is_identity():
    time_s = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([-500.0, 200.0, 800.0, 1400.0])

    result = interpolate_raw_circular_stream(time_s, values, period=None)

    assert not np.any(result.repaired_mask)
    assert np.array_equal(result.angle_deg, values)
    assert np.array_equal(result.unwrapped_deg, values)


@pytest.mark.parametrize("period", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_invalid_periods_are_rejected(period):
    time_s = np.array([0.0, 1.0, 2.0])
    values = np.array([1.0, 2.0, 3.0])
    invalid = np.zeros(3, dtype=bool)

    with pytest.raises(ValueError, match="finite positive number or None"):
        interpolate_circular_auto(time_s, values, invalid, period=period)

    with pytest.raises(ValueError, match="finite positive number or None"):
        interpolate_raw_circular_stream(time_s, values, period=period)
