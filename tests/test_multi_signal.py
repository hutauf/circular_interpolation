from __future__ import annotations

import numpy as np
import pytest

from circular_interpolation import (
    JointPlateauDecision,
    MultiSignalInterpolationResult,
    interpolate_raw_circular_stream,
)


def _moving_signals() -> tuple[np.ndarray, np.ndarray]:
    time_s = np.arange(0.0, 1.001, 0.001)
    data = np.vstack(
        [
            np.mod(350.0 + 720.0 * time_s, 360.0),
            10.0 + 200.0 * time_s,
            -3.0 + 50.0 * time_s,
        ]
    )
    return time_s, data


def test_local_single_signal_plateau_is_not_a_recorder_freeze():
    time_s, data = _moving_signals()
    data[0, 400:420] = data[0, 399]

    result = interpolate_raw_circular_stream(
        time_s,
        data,
        period=[360.0, None, None],
        ambiguous_policy="repair",
    )

    assert isinstance(result, MultiSignalInterpolationResult)
    assert not np.any(result.joint_candidate_mask)
    assert not np.any(result.joint_repair_mask)
    assert not np.any(result.repaired_mask)
    assert np.array_equal(result.angle_deg[0], data[0])


def test_joint_freeze_is_repaired_for_every_signal_and_anchor_is_preserved():
    time_s, data = _moving_signals()
    start, end = 400, 420
    data[:, start:end] = data[:, start - 1, None]

    result = interpolate_raw_circular_stream(
        time_s,
        data,
        period=[360.0, None, None],
        ambiguous_policy="repair",
    )

    assert not result.joint_candidate_mask[start - 1]
    assert np.all(result.joint_candidate_mask[start:end])
    assert not result.joint_repair_mask[start - 1]
    assert np.all(result.joint_repair_mask[start:end])
    assert np.all(result.repaired_mask[:, start:end])
    assert result.period_by_signal == (360.0, None, None)
    assert result.angle_deg.shape == data.shape
    assert isinstance(result.decisions[0], JointPlateauDecision)


def test_all_signals_are_evaluated_on_the_same_target_grid():
    time_s, data = _moving_signals()
    data[:, 400:420] = data[:, 399, None]
    target_time_s = np.linspace(-0.010, 1.010, 1500)

    result = interpolate_raw_circular_stream(
        time_s,
        data,
        period=[360.0, None, None],
        ambiguous_policy="repair",
        target_time_s=target_time_s,
    )

    assert result.angle_deg.shape == (3, target_time_s.size)
    assert result.unwrapped_deg.shape == (3, target_time_s.size)
    assert np.array_equal(result.time_s, target_time_s)
    assert result.repaired_mask.shape == data.shape
    assert np.all((result.angle_deg[0] >= 0.0) & (result.angle_deg[0] < 360.0))
    assert np.max(result.angle_deg[1]) > 200.0


def test_trailing_joint_plateau_is_preserved_even_with_aggressive_policy():
    time_s, data = _moving_signals()
    data[:, -100:] = data[:, -101, None]

    result = interpolate_raw_circular_stream(
        time_s,
        data,
        period=[360.0, None, None],
        ambiguous_policy="repair",
    )

    assert np.all(result.joint_candidate_mask[-100:])
    assert not np.any(result.joint_repair_mask[-100:])
    assert np.all(result.preserved_standstill_mask[-100:])
    assert np.array_equal(result.angle_deg[:, -100:], data[:, -100:])
    assert result.decisions[-1].kind == "boundary_plateau"


def test_duration_guard_preserves_a_long_joint_internal_plateau():
    time_s, data = _moving_signals()
    start, end = 400, 501
    data[:, start:end] = data[:, start - 1, None]

    result = interpolate_raw_circular_stream(
        time_s,
        data,
        period=[360.0, None, None],
        ambiguous_policy="repair",
        max_plateau_repair_duration_s=0.020,
    )

    assert np.all(result.joint_candidate_mask[start:end])
    assert not np.any(result.joint_repair_mask[start:end])
    assert not np.any(result.repaired_mask[:, start:end])
    blocked = [
        decision
        for decision in result.decisions
        if decision.repeat_start == start
    ]
    assert blocked
    assert "max_plateau_repair_duration_s" in blocked[0].reason


def test_short_joint_locally_stationary_run_obeys_ambiguous_policy():
    time_s, data = _moving_signals()
    start, end = 400, 408
    data[:, start:end] = data[:, start - 1, None]
    thresholds = [1e9, 1e9, 1e9]

    preserved = interpolate_raw_circular_stream(
        time_s,
        data,
        period=[360.0, None, None],
        standstill_speed_threshold_deg_s=thresholds,
        ambiguous_policy="preserve",
    )
    repaired = interpolate_raw_circular_stream(
        time_s,
        data,
        period=[360.0, None, None],
        standstill_speed_threshold_deg_s=thresholds,
        ambiguous_policy="repair",
        max_plateau_repair_duration_s=0.020,
    )

    assert preserved.decisions[0].kind == "ambiguous"
    assert not np.any(preserved.joint_repair_mask[start:end])
    assert repaired.decisions[0].kind == "ambiguous"
    assert np.all(repaired.joint_repair_mask[start:end])


def test_row_specific_nonfinite_samples_are_not_promoted_to_shared_freeze():
    time_s, data = _moving_signals()
    data[1, 500:505] = np.nan

    result = interpolate_raw_circular_stream(
        time_s,
        data,
        period=[360.0, None, None],
        ambiguous_policy="repair",
    )

    assert np.all(result.repaired_mask[1, 500:505])
    assert not np.any(result.repaired_mask[[0, 2], 500:505])
    assert not np.any(result.joint_candidate_mask[500:505])
    assert not np.any(result.joint_repair_mask[500:505])


def test_period_scalar_broadcasts_and_sequence_length_is_validated():
    time_s, data = _moving_signals()
    circular_data = np.mod(data, 360.0)

    broadcast = interpolate_raw_circular_stream(
        time_s,
        circular_data,
        period=360.0,
    )
    assert broadcast.period_by_signal == (360.0, 360.0, 360.0)

    linear = interpolate_raw_circular_stream(time_s, data, period=None)
    assert linear.period_by_signal == (None, None, None)

    with pytest.raises(ValueError, match=r"data\.shape\[0\]"):
        interpolate_raw_circular_stream(
            time_s,
            data,
            period=[360.0, None],
        )


@pytest.mark.parametrize(
    "period",
    [
        [360.0, None, np.inf],
        [360.0, None, 0.0],
        [360.0, None, -5.0],
    ],
)
def test_each_period_entry_is_validated(period):
    time_s, data = _moving_signals()
    with pytest.raises(ValueError, match="period"):
        interpolate_raw_circular_stream(time_s, data, period=period)


def test_value_domain_thresholds_may_be_configured_per_signal():
    time_s, data = _moving_signals()
    start, end = 400, 410
    data[:, start:end] = data[:, start - 1, None]
    data[2, start:end] += 1e-5

    result = interpolate_raw_circular_stream(
        time_s,
        data,
        period=[360.0, None, None],
        repeat_tolerance_deg=[0.0, 0.0, 1e-4],
        standstill_speed_threshold_deg_s=[100.0, 100.0, 1.0],
        min_motion_evidence_deg=[0.15, 0.15, 1e-6],
        max_abs_acceleration_deg_s2=[1e6, 1e6, 1e6],
        ambiguous_policy="repair",
    )

    assert np.all(result.joint_candidate_mask[start:end])
    assert np.all(result.joint_repair_mask[start:end])


def test_randomized_local_plateaus_are_rejected_and_shared_ones_are_repaired():
    rng = np.random.default_rng(20260805)
    time_s = np.arange(0.0, 0.501, 0.001)
    slopes = np.array([700.0, 31.0, -120.0, 9.0, 450.0])
    offsets = rng.normal(size=5)
    data = offsets[:, None] + slopes[:, None] * time_s[None, :]
    data[[0, 4]] = np.mod(data[[0, 4]], 360.0)

    local_windows = [(40, 48), (90, 101), (145, 153), (210, 220), (275, 284)]
    for row, (start, end) in enumerate(local_windows):
        data[row, start:end] = data[row, start - 1]

    shared_windows = [(330, 338), (410, 420)]
    for start, end in shared_windows:
        data[:, start:end] = data[:, start - 1, None]

    result = interpolate_raw_circular_stream(
        time_s,
        data,
        period=[360.0, None, None, None, 360.0],
        ambiguous_policy="repair",
        max_plateau_repair_duration_s=0.020,
    )

    for start, end in local_windows:
        assert not np.any(result.joint_candidate_mask[start:end])
        assert not np.any(result.repaired_mask[:, start:end])

    for start, end in shared_windows:
        assert np.all(result.joint_candidate_mask[start:end])
        assert np.all(result.joint_repair_mask[start:end])
        assert np.all(result.repaired_mask[:, start:end])


def test_one_dimensional_input_remains_supported_with_target_grid():
    time_s = np.arange(0.0, 0.101, 0.001)
    values = 10.0 + 20.0 * time_s
    target_time_s = np.linspace(-0.010, 0.110, 301)

    result = interpolate_raw_circular_stream(
        time_s,
        values,
        period=None,
        target_time_s=target_time_s,
    )

    assert result.angle_deg.shape == target_time_s.shape
    assert result.unwrapped_deg.shape == target_time_s.shape
    assert np.array_equal(result.time_s, target_time_s)


def test_input_orientation_is_rows_by_samples():
    time_s = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match=r"data\.shape\[1\]"):
        interpolate_raw_circular_stream(
            time_s,
            np.zeros((5, 2)),
            period=None,
        )


def test_single_row_2d_input_keeps_single_signal_classification_semantics():
    time_s = np.arange(0.0, 0.101, 0.001)
    data = np.full((1, time_s.size), 12.0)

    result = interpolate_raw_circular_stream(
        time_s,
        data,
        period=None,
        ambiguous_policy="repair",
    )

    assert result.angle_deg.shape == data.shape
    assert result.decisions[0].kind == "boundary_plateau"
    assert not np.any(result.joint_repair_mask)
