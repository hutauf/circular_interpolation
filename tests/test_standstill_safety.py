from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent / "benchmarks"))
from standstill_safety import (  # noqa: E402
    StandstillSafetyCase,
    make_stop_go_profile,
    run_benchmark,
)

from circular_interpolation import interpolate_raw_circular_stream


def test_boundary_standstill_is_never_repaired_by_ambiguous_policy():
    case = StandstillSafetyCase(1000.0, 6000.0, 0.200, 0.200)
    time_s, _, observed, _, trailing_mask = make_stop_go_profile(case)

    result = interpolate_raw_circular_stream(
        time_s,
        observed,
        ambiguous_policy="repair",
    )

    assert not np.any(result.repaired_mask[trailing_mask])
    assert np.array_equal(result.angle_deg[trailing_mask], observed[trailing_mask])
    assert any(decision.kind == "boundary_plateau" for decision in result.decisions)


def test_duration_cap_preserves_long_internal_standstill():
    case = StandstillSafetyCase(1000.0, 6000.0, 0.200, 0.200)
    time_s, _, observed, middle_mask, _ = make_stop_go_profile(case)

    uncapped = interpolate_raw_circular_stream(
        time_s,
        observed,
        ambiguous_policy="repair",
    )
    capped = interpolate_raw_circular_stream(
        time_s,
        observed,
        ambiguous_policy="repair",
        max_plateau_repair_duration_s=0.050,
    )

    # This profile is deliberately difficult: the edge windows still contain
    # the end of the deceleration and the beginning of the next acceleration.
    assert np.any(uncapped.repaired_mask[middle_mask])
    assert not np.any(capped.repaired_mask[middle_mask])
    assert np.array_equal(capped.angle_deg[middle_mask], observed[middle_mask])
    assert any(
        "max_plateau_repair_duration_s" in decision.reason
        for decision in capped.decisions
    )


def test_duration_cap_still_repairs_short_held_dropout():
    sample_rate_hz = 1000.0
    time_s = np.arange(0.0, 1.001, 1.0 / sample_rate_hz)
    observed = np.mod(19.0 + 60_000.0 * time_s, 360.0)
    repeat_start = 500
    repeat_end = 504
    observed[repeat_start:repeat_end] = observed[repeat_start - 1]

    result = interpolate_raw_circular_stream(
        time_s,
        observed,
        ambiguous_policy="repair",
        max_plateau_repair_duration_s=0.010,
    )

    assert np.all(result.repaired_mask[repeat_start:repeat_end])
    assert not result.repaired_mask[repeat_start - 1]


def test_duration_cap_can_block_long_detected_dropout():
    sample_rate_hz = 1000.0
    time_s = np.arange(0.0, 1.001, 1.0 / sample_rate_hz)
    observed = np.mod(19.0 + 6000.0 * time_s, 360.0)
    repeat_start = 400
    repeat_end = 500
    observed[repeat_start:repeat_end] = observed[repeat_start - 1]

    result = interpolate_raw_circular_stream(
        time_s,
        observed,
        ambiguous_policy="repair",
        max_plateau_repair_duration_s=0.020,
    )

    assert not np.any(result.repaired_mask[repeat_start:repeat_end])
    assert np.array_equal(
        result.angle_deg[repeat_start:repeat_end],
        observed[repeat_start:repeat_end],
    )
    assert any(
        "max_plateau_repair_duration_s" in decision.reason
        for decision in result.decisions
    )


def test_explicit_missing_samples_ignore_plateau_duration_cap():
    time_s = np.arange(0.0, 1.001, 0.001)
    observed = np.mod(25.0 + 720.0 * time_s, 360.0)
    missing = slice(300, 500)
    observed[missing] = np.nan

    result = interpolate_raw_circular_stream(
        time_s,
        observed,
        max_plateau_repair_duration_s=0.010,
    )

    assert np.all(result.repaired_mask[missing])
    assert np.all(np.isfinite(result.angle_deg[missing]))


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_plateau_duration_cap_is_validated(value):
    time_s = np.array([0.0, 0.1, 0.2])
    observed = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="max_plateau_repair_duration_s"):
        interpolate_raw_circular_stream(
            time_s,
            observed,
            max_plateau_repair_duration_s=value,
        )


def test_stop_go_benchmark_exposes_risk_and_bounds_long_standstills():
    cap = 0.020
    results = run_benchmark(max_plateau_repair_duration_s=cap)

    assert len(results) == 72
    assert any(result.uncapped_repaired_samples > 0 for result in results)
    assert all(result.uncapped_tail_repairs == 0 for result in results)
    assert all(result.capped_tail_repairs == 0 for result in results)

    long_standstills = [
        result
        for result in results
        if result.standstill_duration_s > cap + 1e-12
    ]
    assert long_standstills
    assert all(result.capped_repaired_samples == 0 for result in long_standstills)
    assert all(result.capped_max_error_deg < 1e-9 for result in long_standstills)
