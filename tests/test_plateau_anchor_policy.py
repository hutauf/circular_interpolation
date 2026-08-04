from __future__ import annotations

import numpy as np
import pytest

from circular_interpolation import interpolate_raw_circular_stream


def _held_value_plateau() -> tuple[np.ndarray, np.ndarray, int, int, int]:
    fs = 1000.0
    time_s = np.arange(0.0, 1.001, 1.0 / fs)
    angle_deg = np.mod(19.0 + 60_000.0 * time_s, 360.0)

    anchor = 499
    repeat_start = 500
    repeat_end = 504
    angle_deg[repeat_start:repeat_end] = angle_deg[anchor]
    return time_s, angle_deg, anchor, repeat_start, repeat_end


def test_default_preserves_first_measured_plateau_value():
    time_s, angle_deg, anchor, repeat_start, repeat_end = _held_value_plateau()

    result = interpolate_raw_circular_stream(time_s, angle_deg)

    assert not result.repaired_mask[anchor]
    assert np.all(result.repaired_mask[repeat_start:repeat_end])
    assert not result.repaired_mask[repeat_end]
    assert result.angle_deg[anchor] == angle_deg[anchor]

    decision = next(d for d in result.decisions if d.kind == "held_dropout")
    assert decision.repeat_start == repeat_start
    assert decision.repeat_end == repeat_end


def test_anchor_can_be_included_explicitly():
    time_s, angle_deg, anchor, repeat_start, repeat_end = _held_value_plateau()

    result = interpolate_raw_circular_stream(
        time_s,
        angle_deg,
        plateau_anchor_policy="repair",
    )

    assert not result.repaired_mask[anchor - 1]
    assert np.all(result.repaired_mask[anchor:repeat_end])
    assert not result.repaired_mask[repeat_end]


def test_plateau_anchor_policy_is_validated():
    time_s, angle_deg, *_ = _held_value_plateau()

    with pytest.raises(ValueError, match="plateau_anchor_policy"):
        interpolate_raw_circular_stream(
            time_s,
            angle_deg,
            plateau_anchor_policy="invalid",
        )
