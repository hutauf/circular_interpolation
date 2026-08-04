from __future__ import annotations

import numpy as np

from ._auto_common import (
    KinematicInterpolationResult,
    _contiguous_runs,
    _fit_boundary_kinematics,
    _validate,
)
from .inertia import wrap_deg

def interpolate_circular_boundary_hermite(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    invalid_mask: np.ndarray,
    *,
    period: float = 360.0,
    fit_window_samples: int = 20,
) -> KinematicInterpolationResult:
    """Offline gap interpolation using samples on both sides of each gap."""
    t, y, invalid = _validate(time_s, angle_deg, invalid_mask)
    valid = ~invalid
    n = t.size
    unwrapped = np.full(n, np.nan)
    velocity = np.full(n, np.nan)
    acceleration = np.full(n, np.nan)

    valid_runs = _contiguous_runs(valid)
    if not valid_runs:
        raise ValueError("No valid samples.")

    s0, e0 = valid_runs[0]
    unwrapped[s0:e0] = np.unwrap(y[s0:e0], period=period)

    for run_index in range(1, len(valid_runs)):
        prev_start, prev_end = valid_runs[run_index - 1]
        right_start, right_end = valid_runs[run_index]
        left = prev_end - 1
        right = right_start

        right_local = np.unwrap(y[right_start:right_end], period=period)
        left_indices = np.arange(max(prev_start, prev_end - fit_window_samples), prev_end)
        temp_right = np.full(n, np.nan)
        temp_right[right_start:right_end] = right_local
        right_indices = np.arange(right_start, min(right_end, right_start + fit_window_samples))

        v0, _ = _fit_boundary_kinematics(t, unwrapped, left_indices, t[left])
        v1, _ = _fit_boundary_kinematics(t, temp_right, right_indices, t[right])

        duration = t[right] - t[left]
        expected_displacement = 0.5 * (v0 + v1) * duration
        target_right = unwrapped[left] + expected_displacement
        branch_offset = round((target_right - right_local[0]) / period) * period
        unwrapped[right_start:right_end] = right_local + branch_offset

        y0 = unwrapped[left]
        y1 = unwrapped[right]
        gap_indices = np.arange(left + 1, right)
        if gap_indices.size:
            u = (t[gap_indices] - t[left]) / duration
            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            h01 = -2 * u**3 + 3 * u**2
            h11 = u**3 - u**2
            unwrapped[gap_indices] = (
                h00 * y0
                + h10 * duration * v0
                + h01 * y1
                + h11 * duration * v1
            )

    first_valid = int(np.flatnonzero(valid)[0])
    last_valid = int(np.flatnonzero(valid)[-1])
    if first_valid > 0:
        inds = np.arange(first_valid, min(n, first_valid + fit_window_samples))
        v, a = _fit_boundary_kinematics(t, unwrapped, inds, t[first_valid])
        dt = t[:first_valid] - t[first_valid]
        unwrapped[:first_valid] = unwrapped[first_valid] + v * dt + 0.5 * a * dt**2
    if last_valid < n - 1:
        inds = np.arange(max(0, last_valid - fit_window_samples + 1), last_valid + 1)
        v, a = _fit_boundary_kinematics(t, unwrapped, inds, t[last_valid])
        dt = t[last_valid + 1 :] - t[last_valid]
        unwrapped[last_valid + 1 :] = unwrapped[last_valid] + v * dt + 0.5 * a * dt**2

    velocity[:] = np.gradient(unwrapped, t)
    acceleration[:] = np.gradient(velocity, t)
    output = wrap_deg(unwrapped, period)
    output[valid] = wrap_deg(y[valid], period)

    return KinematicInterpolationResult(
        angle_deg=output,
        unwrapped_deg=unwrapped,
        angular_velocity_deg_s=velocity,
        angular_acceleration_deg_s2=acceleration,
        rejected_outliers=np.zeros(n, dtype=bool),
    )


