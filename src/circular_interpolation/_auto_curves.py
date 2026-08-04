from __future__ import annotations

import numpy as np

from ._auto_common import (
    _GapTrajectory,
    _contiguous_runs,
    _fit_boundary_kinematics,
)


def _run_bounds(valid: np.ndarray, index: int) -> tuple[int, int]:
    start = index
    while start > 0 and valid[start - 1]:
        start -= 1
    end = index + 1
    while end < valid.size and valid[end]:
        end += 1
    return start, end


def _cubic_hermite(
    query_time_s: np.ndarray,
    trajectory: _GapTrajectory,
) -> np.ndarray:
    duration = trajectory.right_time_s - trajectory.left_time_s
    u = (query_time_s - trajectory.left_time_s) / duration
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    return (
        h00 * trajectory.left_position_deg
        + h10 * duration * trajectory.left_velocity_deg_s
        + h01 * trajectory.right_position_deg
        + h11 * duration * trajectory.right_velocity_deg_s
    )


def _evaluate_gap_trajectory(
    query_time_s: np.ndarray,
    trajectory: _GapTrajectory,
) -> np.ndarray:
    if trajectory.model in {"linear_shortest_arc", "linear"}:
        u = (
            (query_time_s - trajectory.left_time_s)
            / (trajectory.right_time_s - trajectory.left_time_s)
        )
        return trajectory.left_position_deg + u * (
            trajectory.right_position_deg - trajectory.left_position_deg
        )
    if trajectory.model == "constant_acceleration_rts":
        raise ValueError(
            "constant_acceleration_rts must be evaluated with the state-space "
            "model on the combined source/target grid."
        )
    return _cubic_hermite(query_time_s, trajectory)


def _boundary_kinematics(
    t: np.ndarray,
    unwrapped: np.ndarray,
    valid: np.ndarray,
    *,
    side: str,
    max_points: int = 20,
) -> tuple[float, float, float, float, float]:
    runs = _contiguous_runs(valid)
    if side == "left":
        start, end = runs[0]
        boundary_index = start
        indices = np.arange(start, min(end, start + max_points))
    else:
        start, end = runs[-1]
        boundary_index = end - 1
        indices = np.arange(max(start, end - max_points), end)

    velocity, acceleration = _fit_boundary_kinematics(
        t, unwrapped, indices, t[boundary_index]
    )

    if indices.size >= 2:
        horizon = float(t[indices[-1]] - t[indices[0]])
    else:
        horizon = float(np.median(np.diff(t)))
    horizon = max(horizon, float(np.median(np.diff(t))))

    if indices.size >= 3:
        slopes = np.diff(unwrapped[indices]) / np.diff(t[indices])
        slope_times = 0.5 * (t[indices][1:] + t[indices][:-1])
        finite_slopes = np.abs(slopes[np.isfinite(slopes)])
        speed_scale = max(
            abs(velocity),
            float(np.percentile(finite_slopes, 90.0)) if finite_slopes.size else 0.0,
            1.0,
        )
        acceleration_limit = 4.0 * speed_scale / horizon
        if slopes.size >= 2:
            local_acceleration = np.diff(slopes) / np.diff(slope_times)
            finite_acceleration = np.abs(local_acceleration[np.isfinite(local_acceleration)])
            if finite_acceleration.size:
                acceleration_limit = min(
                    acceleration_limit,
                    max(
                        4.0 * float(np.percentile(finite_acceleration, 90.0)),
                        1e-9,
                    ),
                )
        acceleration = float(
            np.clip(acceleration, -acceleration_limit, acceleration_limit)
        )

    return (
        float(t[boundary_index]),
        float(unwrapped[boundary_index]),
        velocity,
        acceleration,
        horizon,
    )


def _limited_kinematic_extrapolation(
    query_time_s: np.ndarray,
    *,
    boundary_time_s: float,
    boundary_position_deg: float,
    boundary_velocity_deg_s: float,
    boundary_acceleration_deg_s2: float,
    acceleration_horizon_s: float,
) -> np.ndarray:
    dt = query_time_s - boundary_time_s
    accelerated_dt = np.clip(
        dt,
        -acceleration_horizon_s,
        acceleration_horizon_s,
    )
    accelerated_position = (
        boundary_position_deg
        + boundary_velocity_deg_s * accelerated_dt
        + 0.5 * boundary_acceleration_deg_s2 * accelerated_dt**2
    )
    terminal_velocity = (
        boundary_velocity_deg_s
        + boundary_acceleration_deg_s2 * accelerated_dt
    )
    return accelerated_position + terminal_velocity * (dt - accelerated_dt)
