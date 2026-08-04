from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from ._auto_common import _GapTrajectory, _contiguous_runs
from ._auto_curves import (
    _boundary_kinematics,
    _evaluate_gap_trajectory,
    _limited_kinematic_extrapolation,
)
from ._auto_kalman import interpolate_circular_constant_acceleration
from ._periodic import Period


def _evaluate_constant_acceleration_targets(
    t: np.ndarray,
    y: np.ndarray,
    invalid: np.ndarray,
    target: np.ndarray,
    gap_trajectories: list[_GapTrajectory],
    output: np.ndarray,
    *,
    period: Period,
) -> None:
    """Evaluate RTS trajectories at inserted prediction-only timestamps.

    Splitting a continuous white-jerk state transition at extra timestamps does
    not change the state at the original timestamps. Therefore the Kalman/RTS
    model can be run once on the union of source timestamps and requested target
    timestamps inside CA-selected gaps. This evaluates the actual selected model
    directly instead of approximating it with a second interpolation polynomial.
    """
    ca_trajectories = [
        trajectory
        for trajectory in gap_trajectories
        if trajectory.model == "constant_acceleration_rts"
    ]
    if not ca_trajectories:
        return

    requested = np.zeros(target.shape, dtype=bool)
    for trajectory in ca_trajectories:
        requested |= (
            (target > trajectory.left_time_s)
            & (target < trajectory.right_time_s)
        )
    if not np.any(requested):
        return

    combined_time = np.unique(np.concatenate((t, target[requested])))
    source_positions = np.searchsorted(combined_time, t)
    combined_angle = np.full(combined_time.shape, np.nan, dtype=float)
    combined_invalid = np.ones(combined_time.shape, dtype=bool)
    combined_angle[source_positions] = y
    combined_invalid[source_positions] = invalid

    dense_ca = interpolate_circular_constant_acceleration(
        combined_time,
        combined_angle,
        combined_invalid,
        period=period,
    )

    for trajectory in ca_trajectories:
        mask = (
            (target > trajectory.left_time_s)
            & (target < trajectory.right_time_s)
        )
        if not np.any(mask):
            continue
        combined_positions = np.searchsorted(combined_time, target[mask])
        output[mask] = (
            dense_ca.unwrapped_deg[combined_positions]
            + trajectory.candidate_offset_deg
        )


def _evaluate_on_target_grid(
    t: np.ndarray,
    y: np.ndarray,
    source_unwrapped: np.ndarray,
    invalid: np.ndarray,
    target: np.ndarray,
    gap_trajectories: list[_GapTrajectory],
    *,
    period: Period,
) -> np.ndarray:
    valid = ~invalid
    output = np.full(target.shape, np.nan, dtype=float)

    for start, end in _contiguous_runs(valid):
        run_t = t[start:end]
        run_y = source_unwrapped[start:end]
        if run_t.size >= 2:
            mask = (
                (target >= run_t[0])
                & (target <= run_t[-1])
                & ~np.isfinite(output)
            )
            if np.any(mask):
                output[mask] = PchipInterpolator(
                    run_t,
                    run_y,
                    extrapolate=False,
                )(target[mask])
        else:
            mask = (target == run_t[0]) & ~np.isfinite(output)
            output[mask] = run_y[0]

    for trajectory in gap_trajectories:
        if trajectory.model == "constant_acceleration_rts":
            continue
        mask = (
            (target > trajectory.left_time_s)
            & (target < trajectory.right_time_s)
        )
        if np.any(mask):
            output[mask] = _evaluate_gap_trajectory(target[mask], trajectory)

    _evaluate_constant_acceleration_targets(
        t,
        y,
        invalid,
        target,
        gap_trajectories,
        output,
        period=period,
    )

    first_valid = int(np.flatnonzero(valid)[0])
    last_valid = int(np.flatnonzero(valid)[-1])

    left_mask = target < t[first_valid]
    if np.any(left_mask):
        bt, bp, bv, ba, horizon = _boundary_kinematics(
            t,
            source_unwrapped,
            valid,
            side="left",
        )
        output[left_mask] = _limited_kinematic_extrapolation(
            target[left_mask],
            boundary_time_s=bt,
            boundary_position_deg=bp,
            boundary_velocity_deg_s=bv,
            boundary_acceleration_deg_s2=ba,
            acceleration_horizon_s=horizon,
        )

    right_mask = target > t[last_valid]
    if np.any(right_mask):
        bt, bp, bv, ba, horizon = _boundary_kinematics(
            t,
            source_unwrapped,
            valid,
            side="right",
        )
        output[right_mask] = _limited_kinematic_extrapolation(
            target[right_mask],
            boundary_time_s=bt,
            boundary_position_deg=bp,
            boundary_velocity_deg_s=bv,
            boundary_acceleration_deg_s2=ba,
            acceleration_horizon_s=horizon,
        )

    missing = ~np.isfinite(output)
    if np.any(missing):
        # Degenerate one-sample valid runs can leave isolated points uncovered.
        # The final fallback stays on the already selected continuous branch.
        output[missing] = np.interp(target[missing], t, source_unwrapped)

    # Preserve the historical source-grid result exactly whenever the requested
    # grid explicitly contains a source timestamp.
    target_positions = np.searchsorted(target, t)
    inside_supported_span = (
        (np.arange(t.size) >= first_valid)
        & (np.arange(t.size) <= last_valid)
    )
    candidates = (target_positions < target.size) & inside_supported_span
    source_indices = np.flatnonzero(candidates)
    exact = target[target_positions[candidates]] == t[candidates]
    source_indices = source_indices[exact]
    output[target_positions[source_indices]] = source_unwrapped[source_indices]

    return output
