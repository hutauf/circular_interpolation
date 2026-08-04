from __future__ import annotations

import numpy as np

from ._auto_common import KinematicInterpolationResult, _validate
from ._periodic import (
    Period,
    unwrap_values,
    validate_period,
    value_difference,
    wrap_values,
)


def _initial_kinematics(
    t: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    first: int,
    period: Period,
    max_points: int = 40,
) -> tuple[float, float]:
    indices = np.flatnonzero(valid & (np.arange(t.size) >= first))[:max_points]
    if indices.size < 3:
        return 0.0, 0.0
    local_t = t[indices] - t[indices[0]]
    local_y = unwrap_values(y[indices], period)
    degree = min(2, indices.size - 1)
    coeff = np.polyfit(local_t, local_y, degree)
    if degree == 1:
        return float(coeff[0]), 0.0
    return float(coeff[1]), float(2.0 * coeff[0])


def interpolate_circular_constant_acceleration(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    invalid_mask: np.ndarray,
    *,
    period: Period = 360.0,
    acceleration_random_walk_std: float = 2_000_000.0,
    measurement_std_deg: float = 0.03,
    gate_sigma: float = 8.0,
    preserve_valid_samples: bool = True,
) -> KinematicInterpolationResult:
    """Constant-acceleration Kalman filter plus RTS smoother.

    ``period=None`` treats measurements as ordinary scalar values. A positive
    finite period keeps the circular nearest-branch measurement update.
    """
    period = validate_period(period)
    t, y, invalid = _validate(time_s, angle_deg, invalid_mask)
    valid = ~invalid
    n = t.size
    first = int(np.flatnonzero(valid)[0])

    xf = np.full((n, 3), np.nan)
    pf = np.full((n, 3, 3), np.nan)
    xp = np.full((n, 3), np.nan)
    pp = np.full((n, 3, 3), np.nan)
    transitions = np.full((n, 3, 3), np.nan)
    rejected = np.zeros(n, dtype=bool)

    velocity0, acceleration0 = _initial_kinematics(t, y, valid, first, period)
    state = np.array([y[first], velocity0, acceleration0], dtype=float)
    covariance = np.diag(
        [
            max(measurement_std_deg**2 * 4.0, 1e-8),
            (20_000.0) ** 2,
            (300_000.0) ** 2,
        ]
    )

    xf[first] = state
    pf[first] = covariance
    xp[first] = state
    pp[first] = covariance
    transitions[first] = np.eye(3)

    r = measurement_std_deg**2
    q = acceleration_random_walk_std**2
    identity = np.eye(3)

    for k in range(first + 1, n):
        dt = t[k] - t[k - 1]
        f = np.array(
            [
                [1.0, dt, 0.5 * dt * dt],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ]
        )
        qk = q * np.array(
            [
                [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
                [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
                [dt**3 / 6.0, dt**2 / 2.0, dt],
            ]
        )
        state_pred = f @ state
        covariance_pred = f @ covariance @ f.T + qk

        transitions[k] = f
        xp[k] = state_pred
        pp[k] = covariance_pred

        if valid[k]:
            innovation = float(
                value_difference(
                    y[k],
                    wrap_values(state_pred[0], period),
                    period,
                )
            )
            innovation_variance = covariance_pred[0, 0] + r
            if innovation**2 <= gate_sigma**2 * innovation_variance:
                gain = covariance_pred[:, 0] / innovation_variance
                state = state_pred + gain * innovation
                kh = np.zeros((3, 3))
                kh[:, 0] = gain
                covariance = (
                    (identity - kh)
                    @ covariance_pred
                    @ (identity - kh).T
                    + np.outer(gain, gain) * r
                )
            else:
                state = state_pred
                covariance = covariance_pred
                rejected[k] = True
        else:
            state = state_pred
            covariance = covariance_pred

        xf[k] = state
        pf[k] = covariance

    xs = xf.copy()
    ps = pf.copy()
    for k in range(n - 2, first - 1, -1):
        f = transitions[k + 1]
        smoother_gain = np.linalg.solve(pp[k + 1].T, (pf[k] @ f.T).T).T
        xs[k] = xf[k] + smoother_gain @ (xs[k + 1] - xp[k + 1])
        ps[k] = pf[k] + smoother_gain @ (ps[k + 1] - pp[k + 1]) @ smoother_gain.T

    for k in range(first - 1, -1, -1):
        dt = t[first] - t[k]
        xs[k, 0] = xs[first, 0] - xs[first, 1] * dt + 0.5 * xs[first, 2] * dt**2
        xs[k, 1] = xs[first, 1] - xs[first, 2] * dt
        xs[k, 2] = xs[first, 2]

    output = wrap_values(xs[:, 0], period)
    if preserve_valid_samples:
        keep = valid & ~rejected
        output[keep] = wrap_values(y[keep], period)

    return KinematicInterpolationResult(
        angle_deg=output,
        unwrapped_deg=xs[:, 0],
        angular_velocity_deg_s=xs[:, 1],
        angular_acceleration_deg_s2=xs[:, 2],
        rejected_outliers=rejected,
    )
