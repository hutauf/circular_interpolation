from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .inertia import (
    interpolate_circular_linear,
    circular_difference_deg,
    wrap_deg,
)


class KinematicInterpolationResult:
    def __init__(self, angle_deg, unwrapped_deg, angular_velocity_deg_s, angular_acceleration_deg_s2, rejected_outliers):
        self.angle_deg = angle_deg
        self.unwrapped_deg = unwrapped_deg
        self.angular_velocity_deg_s = angular_velocity_deg_s
        self.angular_acceleration_deg_s2 = angular_acceleration_deg_s2
        self.rejected_outliers = rejected_outliers


@dataclass
class AutomaticInterpolationResult:
    angle_deg: np.ndarray
    unwrapped_deg: np.ndarray
    chosen_model_by_gap: list[tuple[int, int, str]]
    confidence_by_gap: list[tuple[int, int, str]]


def _validate(time_s: np.ndarray, angle_deg: np.ndarray, invalid_mask: np.ndarray):
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(angle_deg, dtype=float)
    invalid = np.asarray(invalid_mask, dtype=bool) | ~np.isfinite(y)
    if t.ndim != 1 or y.ndim != 1 or invalid.ndim != 1:
        raise ValueError("All inputs must be one-dimensional.")
    if not (t.shape == y.shape == invalid.shape):
        raise ValueError("All inputs must have equal shape.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be strictly increasing.")
    if np.count_nonzero(~invalid) < 3:
        raise ValueError("At least three valid samples are required.")
    return t, y, invalid


def _initial_kinematics(
    t: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    first: int,
    period: float,
    max_points: int = 40,
) -> tuple[float, float]:
    indices = np.flatnonzero(valid & (np.arange(t.size) >= first))[:max_points]
    if indices.size < 3:
        return 0.0, 0.0
    local_t = t[indices] - t[indices[0]]
    local_y = np.unwrap(y[indices], period=period)
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
    period: float = 360.0,
    acceleration_random_walk_std: float = 2_000_000.0,
    measurement_std_deg: float = 0.03,
    gate_sigma: float = 8.0,
    preserve_valid_samples: bool = True,
) -> KinematicInterpolationResult:
    """
    Circular constant-acceleration Kalman filter plus RTS smoother.

    State: [angle, angular velocity, angular acceleration]. Acceleration follows
    a random walk driven by white jerk. The wrapped measurement is assigned to
    the revolution nearest the predicted unwrapped angle.
    """
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
        # Continuous white jerk model.
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
                circular_difference_deg(
                    y[k], wrap_deg(state_pred[0], period), period
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

    output = wrap_deg(xs[:, 0], period)
    if preserve_valid_samples:
        keep = valid & ~rejected
        output[keep] = wrap_deg(y[keep], period)

    return KinematicInterpolationResult(
        angle_deg=output,
        unwrapped_deg=xs[:, 0],
        angular_velocity_deg_s=xs[:, 1],
        angular_acceleration_deg_s2=xs[:, 2],
        rejected_outliers=rejected,
    )


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def _fit_boundary_kinematics(
    t: np.ndarray,
    unwrapped: np.ndarray,
    indices: np.ndarray,
    at_time: float,
) -> tuple[float, float]:
    if indices.size < 2:
        return 0.0, 0.0
    x = t[indices] - at_time
    degree = min(2, indices.size - 1)
    coeff = np.polyfit(x, unwrapped[indices], degree)
    if degree == 1:
        return float(coeff[0]), 0.0
    return float(coeff[1]), float(2.0 * coeff[0])


def interpolate_circular_boundary_hermite(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    invalid_mask: np.ndarray,
    *,
    period: float = 360.0,
    fit_window_samples: int = 20,
) -> KinematicInterpolationResult:
    """
    Offline gap interpolation using samples on both sides of each gap.

    Each valid segment is unwrapped locally. The unknown revolution offset of
    the segment after a gap is selected from the endpoint velocity estimate.
    A cubic Hermite curve then connects position and velocity at both ends.
    """
    t, y, invalid = _validate(time_s, angle_deg, invalid_mask)
    valid = ~invalid
    n = t.size
    unwrapped = np.full(n, np.nan)
    velocity = np.full(n, np.nan)
    acceleration = np.full(n, np.nan)

    valid_runs = _contiguous_runs(valid)
    if not valid_runs:
        raise ValueError("No valid samples.")

    # First valid segment establishes the initial branch.
    s0, e0 = valid_runs[0]
    unwrapped[s0:e0] = np.unwrap(y[s0:e0], period=period)

    for run_index in range(1, len(valid_runs)):
        prev_start, prev_end = valid_runs[run_index - 1]
        right_start, right_end = valid_runs[run_index]
        left = prev_end - 1
        right = right_start

        right_local = np.unwrap(y[right_start:right_end], period=period)

        left_indices = np.arange(max(prev_start, prev_end - fit_window_samples), prev_end)
        # Temporarily insert the right local segment so its local velocity can be estimated.
        temp_right = np.full(n, np.nan)
        temp_right[right_start:right_end] = right_local
        right_indices = np.arange(right_start, min(right_end, right_start + fit_window_samples))

        v0, a0 = _fit_boundary_kinematics(t, unwrapped, left_indices, t[left])
        v1, a1 = _fit_boundary_kinematics(t, temp_right, right_indices, t[right])

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

    # Extrapolate leading/trailing invalid samples if present.
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


def interpolate_circular_auto(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    invalid_mask: np.ndarray,
    *,
    period: float = 360.0,
) -> AutomaticInterpolationResult:
    """Empirical offline selector derived from the motor-profile test suite.

    Rule:
    - Near standstill: shortest-arc linear interpolation.
    - Moving and endpoint-to-endpoint gap <= 6 ms: constant-acceleration RTS.
    - Otherwise: boundary-velocity Hermite interpolation.

    Long dynamic gaps are still reconstructed, but marked low confidence when
    the boundary velocities disagree strongly or change direction.
    """
    t, y, invalid = _validate(time_s, angle_deg, invalid_mask)
    linear_wrapped, linear_u = interpolate_circular_linear(t, y, invalid, period=period)
    ca = interpolate_circular_constant_acceleration(t, y, invalid, period=period)
    hermite = interpolate_circular_boundary_hermite(t, y, invalid, period=period)

    output_u = hermite.unwrapped_deg.copy()
    choices: list[tuple[int, int, str]] = []
    confidences: list[tuple[int, int, str]] = []

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
                model = "linear_shortest_arc"
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

        # Align the selected candidate to the Hermite branch at the left edge.
        anchor = max(0, left)
        offset = round((output_u[anchor] - candidate[anchor]) / period) * period
        output_u[start:end] = candidate[start:end] + offset
        choices.append((start, end, model))
        confidences.append((start, end, confidence))

    output_angle = wrap_deg(output_u, period)
    output_angle[~invalid] = wrap_deg(y[~invalid], period)
    return AutomaticInterpolationResult(
        angle_deg=output_angle,
        unwrapped_deg=output_u,
        chosen_model_by_gap=choices,
        confidence_by_gap=confidences,
    )
