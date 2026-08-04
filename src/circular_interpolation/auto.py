from __future__ import annotations

import numpy as np

from ._auto_common import (
    AutomaticInterpolationResult,
    KinematicInterpolationResult,
    _GapTrajectory,
    _contiguous_runs,
    _fit_boundary_kinematics,
    _validate,
    _validate_target_time_s,
)
from ._auto_curves import _run_bounds
from ._auto_grid import _evaluate_on_target_grid
from ._auto_hermite import interpolate_circular_boundary_hermite
from ._auto_kalman import interpolate_circular_constant_acceleration
from ._periodic import (
    Period,
    branch_offset,
    interpolate_linear,
    validate_period,
    wrap_values,
)


def interpolate_circular_auto(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    invalid_mask: np.ndarray,
    *,
    period: Period = 360.0,
    target_time_s: np.ndarray | None = None,
) -> AutomaticInterpolationResult:
    """Select a robust gap model and optionally evaluate a new time grid.

    A positive finite ``period`` enables circular branch selection and wrapping.
    Set ``period=None`` for an ordinary scalar signal; no values are wrapped or
    unwrapped and ``angle_deg`` equals ``unwrapped_deg`` in the returned result.

    ``target_time_s`` is the only resampling control. When it is omitted, the
    function returns the source-grid result. When supplied, selected continuous
    trajectories are evaluated directly at those times. The grid may be
    irregular and may extend before or after the measurements.

    Internal gaps use automatic model selection. Valid source runs are evaluated
    with shape-preserving interpolation on the continuous branch. Outside the
    measured support, local position, velocity, and acceleration are estimated at
    the nearest boundary; acceleration is used only across the local fit horizon
    and extrapolation then continues at constant velocity to avoid quadratic
    blow-up far from the data.
    """
    period = validate_period(period)
    t, y, invalid = _validate(time_s, angle_deg, invalid_mask)
    target = _validate_target_time_s(target_time_s, t)

    _, linear_u = interpolate_linear(t, y, invalid, period=period)
    ca = interpolate_circular_constant_acceleration(t, y, invalid, period=period)
    hermite = interpolate_circular_boundary_hermite(t, y, invalid, period=period)

    output_u = hermite.unwrapped_deg.copy()
    choices: list[tuple[int, int, str]] = []
    confidences: list[tuple[int, int, str]] = []
    trajectories: list[_GapTrajectory] = []
    valid = ~invalid

    for start, end in _contiguous_runs(invalid):
        left = start - 1
        right = end
        if left < 0 or right >= t.size:
            model = "boundary_velocity_hermite"
            confidence = "low"
            candidate = hermite.unwrapped_deg
        else:
            duration = t[right] - t[left]
            v0 = float(hermite.angular_velocity_deg_s[left])
            v1 = float(hermite.angular_velocity_deg_s[right])
            a0 = float(hermite.angular_acceleration_deg_s2[left])
            a1 = float(hermite.angular_acceleration_deg_s2[right])
            max_speed = max(abs(v0), abs(v1))
            near_standstill = max_speed < 100.0 and max(abs(a0), abs(a1)) < 5_000.0

            if near_standstill:
                model = "linear_shortest_arc" if period is not None else "linear"
                candidate = linear_u
                confidence = "high"
            elif duration <= 0.006:
                model = "constant_acceleration_rts"
                candidate = ca.unwrapped_deg
                confidence = "high"
            else:
                model = "boundary_velocity_hermite"
                candidate = hermite.unwrapped_deg
                same_direction = v0 * v1 >= 0.0
                relative_change = abs(v1 - v0) / max(max_speed, 1.0)
                if duration <= 0.050 or (same_direction and relative_change < 0.10):
                    confidence = "high"
                elif duration <= 0.100 and same_direction:
                    confidence = "medium"
                else:
                    confidence = "low"

        anchor = max(0, left)
        offset = branch_offset(
            float(output_u[anchor]),
            float(candidate[anchor]),
            period,
        )
        output_u[start:end] = candidate[start:end] + offset
        choices.append((start, end, model))
        confidences.append((start, end, confidence))

        if left >= 0 and right < t.size:
            left_velocity = 0.0
            right_velocity = 0.0
            if model == "boundary_velocity_hermite":
                left_start, left_end = _run_bounds(valid, left)
                right_start, right_end = _run_bounds(valid, right)
                left_indices = np.arange(max(left_start, left_end - 20), left_end)
                right_indices = np.arange(right_start, min(right_end, right_start + 20))
                left_velocity, _ = _fit_boundary_kinematics(
                    t,
                    output_u,
                    left_indices,
                    t[left],
                )
                right_velocity, _ = _fit_boundary_kinematics(
                    t,
                    output_u,
                    right_indices,
                    t[right],
                )

            trajectories.append(
                _GapTrajectory(
                    start=start,
                    end=end,
                    model=model,
                    confidence=confidence,
                    left_time_s=float(t[left]),
                    right_time_s=float(t[right]),
                    left_position_deg=float(output_u[left]),
                    right_position_deg=float(output_u[right]),
                    left_velocity_deg_s=left_velocity,
                    right_velocity_deg_s=right_velocity,
                    candidate_offset_deg=float(offset),
                )
            )

    source_output = wrap_values(output_u, period)
    source_output[~invalid] = wrap_values(y[~invalid], period)

    if target.shape == t.shape and np.array_equal(target, t):
        return AutomaticInterpolationResult(
            angle_deg=source_output,
            unwrapped_deg=output_u,
            chosen_model_by_gap=choices,
            confidence_by_gap=confidences,
            time_s=target,
        )

    target_continuous = _evaluate_on_target_grid(
        t,
        y,
        output_u,
        invalid,
        target,
        trajectories,
        period=period,
    )
    return AutomaticInterpolationResult(
        angle_deg=wrap_values(target_continuous, period),
        unwrapped_deg=target_continuous,
        chosen_model_by_gap=choices,
        confidence_by_gap=confidences,
        time_s=target,
    )
