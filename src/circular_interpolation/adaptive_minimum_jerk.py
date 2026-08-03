from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GapDecision:
    """Diagnostic information for one interpolated gap."""

    start: int
    end: int
    endpoint_span_s: float
    selected_revolution_offset: int
    used_linear_fallback: bool
    kinematic_evidence_sigma: float
    left_velocity_deg_s: float
    right_velocity_deg_s: float
    left_acceleration_deg_s2: float
    right_acceleration_deg_s2: float
    left_fit_residual_deg: float
    right_fit_residual_deg: float


@dataclass
class AdaptiveMinimumJerkResult:
    angle_deg: np.ndarray
    unwrapped_deg: np.ndarray
    angular_velocity_deg_s: np.ndarray
    angular_acceleration_deg_s2: np.ndarray
    decisions: list[GapDecision]


@dataclass(frozen=True)
class _BoundaryFit:
    velocity: float
    acceleration: float
    velocity_se: float
    acceleration_se: float
    residual_std: float


def _wrap_deg(values: np.ndarray, period: float) -> np.ndarray:
    return np.mod(values, period)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def _fit_boundary_kinematics(
    time_s: np.ndarray,
    unwrapped_deg: np.ndarray,
    boundary_time_s: float,
    *,
    degree: int,
    noise_floor_deg: float,
) -> _BoundaryFit:
    """Weighted local polynomial fit and uncertainty of its first derivatives."""
    x = np.asarray(time_s, dtype=float) - boundary_time_s
    y = np.asarray(unwrapped_deg, dtype=float)
    if x.size < 3:
        raise ValueError("At least three samples are required for a boundary fit.")

    degree = min(degree, x.size - 1)
    scale = max(float(np.max(np.abs(x))), 1e-12)
    z = x / scale
    weights = 0.25 + 0.75 * np.exp(-2.0 * np.abs(z))
    design = np.column_stack([z**order for order in range(degree + 1)])
    sqrt_w = np.sqrt(weights)
    weighted_design = design * sqrt_w[:, None]
    weighted_y = y * sqrt_w
    coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)

    residual = y - design @ coefficients
    degrees_of_freedom = max(1, x.size - degree - 1)
    residual_variance = float(
        np.sum(weights * residual**2) / degrees_of_freedom
    )
    residual_variance = max(residual_variance, noise_floor_deg**2)
    covariance = residual_variance * np.linalg.pinv(
        weighted_design.T @ weighted_design
    )

    velocity = float(coefficients[1] / scale) if degree >= 1 else 0.0
    acceleration = (
        float(2.0 * coefficients[2] / scale**2) if degree >= 2 else 0.0
    )
    velocity_se = (
        float(np.sqrt(max(covariance[1, 1], 0.0)) / scale)
        if degree >= 1
        else np.inf
    )
    acceleration_se = (
        float(2.0 * np.sqrt(max(covariance[2, 2], 0.0)) / scale**2)
        if degree >= 2
        else np.inf
    )

    return _BoundaryFit(
        velocity=velocity,
        acceleration=acceleration,
        velocity_se=velocity_se,
        acceleration_se=acceleration_se,
        residual_std=float(np.sqrt(residual_variance)),
    )


def _basis_derivatives(order: int, points: np.ndarray) -> np.ndarray:
    """Derivatives of phi_j(u)=u^(j+1)-u^(j+2), for j=0..3."""
    points = np.asarray(points, dtype=float)
    output = np.zeros((points.size, 4), dtype=float)
    for column in range(4):
        coefficients = np.zeros(column + 3, dtype=float)
        coefficients[column + 1] = 1.0
        coefficients[column + 2] = -1.0
        for _ in range(order):
            if coefficients.size <= 1:
                coefficients = np.zeros(1, dtype=float)
                break
            coefficients = np.array(
                [index * coefficients[index] for index in range(1, coefficients.size)],
                dtype=float,
            )
        output[:, column] = np.polynomial.polynomial.polyval(
            points, coefficients
        )
    return output


def _integral_gram(order: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, 401)
    basis = _basis_derivatives(order, u)
    return np.trapezoid(
        basis[:, :, None] * basis[:, None, :], u, axis=0
    )


_VELOCITY_BASIS = np.vstack(
    [
        _basis_derivatives(1, np.array([0.0]))[0],
        _basis_derivatives(1, np.array([1.0]))[0],
    ]
)
_ACCELERATION_BASIS = np.vstack(
    [
        _basis_derivatives(2, np.array([0.0]))[0],
        _basis_derivatives(2, np.array([1.0]))[0],
    ]
)
_ACCELERATION_GRAM = _integral_gram(2)
_JERK_GRAM = _integral_gram(3)


def interpolate_circular_adaptive_minimum_jerk(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    invalid_mask: np.ndarray,
    *,
    period: float = 360.0,
    fit_window_samples: int = 10,
    fit_degree: int = 2,
    acceleration_penalty: float = 0.001,
    jerk_penalty: float = 0.01,
    kinematic_significance_threshold: float = 8.0,
    measurement_noise_floor_deg: float = 0.02,
    derivative_uncertainty_floor_deg: float = 0.03,
    branch_search_turns: int = 8,
    preserve_valid_samples: bool = True,
) -> AdaptiveMinimumJerkResult:
    """Offline interpolation of wrapped angles with an exact linear fallback.

    For every internal gap, the method:

    1. Estimates velocity and acceleration from valid samples on both sides.
    2. Tests plausible integer revolution offsets for the right-hand segment.
    3. Starts with the straight line between the selected endpoint positions.
    4. Adds a quintic correction whose endpoint positions are exactly zero.
    5. Fits that correction softly to the estimated endpoint velocity and
       acceleration while penalizing acceleration and jerk inside the gap.
    6. Sets the correction exactly to zero when the evidence for curvature is
       below ``kinematic_significance_threshold``.

    Therefore, simple motion does not merely approximate linear interpolation:
    it becomes exactly linear after the physically plausible winding number has
    been selected.
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(angle_deg, dtype=float)
    invalid = np.asarray(invalid_mask, dtype=bool) | ~np.isfinite(y)

    if t.ndim != 1 or y.ndim != 1 or invalid.ndim != 1:
        raise ValueError("All inputs must be one-dimensional.")
    if not (t.shape == y.shape == invalid.shape):
        raise ValueError("All inputs must have equal shape.")
    if t.size < 3 or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be finite, increasing, and have >=3 samples.")
    if fit_window_samples < 3:
        raise ValueError("fit_window_samples must be at least 3.")
    if period <= 0:
        raise ValueError("period must be positive.")

    valid = ~invalid
    if np.count_nonzero(valid) < 3:
        raise ValueError("At least three valid samples are required.")

    output_unwrapped = np.full(t.size, np.nan, dtype=float)
    valid_runs = _contiguous_runs(valid)
    first_start, first_end = valid_runs[0]
    output_unwrapped[first_start:first_end] = np.unwrap(
        y[first_start:first_end], period=period
    )
    decisions: list[GapDecision] = []

    for run_number in range(1, len(valid_runs)):
        previous_start, previous_end = valid_runs[run_number - 1]
        right_start, right_end = valid_runs[run_number]
        left_index = previous_end - 1
        right_index = right_start
        duration = float(t[right_index] - t[left_index])

        right_local = np.unwrap(y[right_start:right_end], period=period)
        left_fit_indices = np.arange(
            max(previous_start, previous_end - fit_window_samples),
            previous_end,
        )
        right_fit_indices = np.arange(
            right_start,
            min(right_end, right_start + fit_window_samples),
        )

        left_fit = _fit_boundary_kinematics(
            t[left_fit_indices],
            output_unwrapped[left_fit_indices],
            t[left_index],
            degree=fit_degree,
            noise_floor_deg=measurement_noise_floor_deg,
        )
        right_fit = _fit_boundary_kinematics(
            t[right_fit_indices],
            right_local[: right_fit_indices.size],
            t[right_index],
            degree=fit_degree,
            noise_floor_deg=measurement_noise_floor_deg,
        )

        left_position = float(output_unwrapped[left_index])
        expected_right_position = left_position + 0.5 * (
            left_fit.velocity + right_fit.velocity
        ) * duration
        central_turn = int(
            round((expected_right_position - right_local[0]) / period)
        )

        best_solution: tuple[
            float, int, float, np.ndarray, float
        ] | None = None

        for turn_offset in range(
            central_turn - branch_search_turns,
            central_turn + branch_search_turns + 1,
        ):
            right_position = float(right_local[0] + turn_offset * period)
            displacement = right_position - left_position

            target_velocity_correction = np.array(
                [
                    duration * left_fit.velocity - displacement,
                    duration * right_fit.velocity - displacement,
                ],
                dtype=float,
            )
            target_acceleration = np.array(
                [
                    duration**2 * left_fit.acceleration,
                    duration**2 * right_fit.acceleration,
                ],
                dtype=float,
            )

            velocity_uncertainty = np.array(
                [
                    max(
                        duration * left_fit.velocity_se,
                        derivative_uncertainty_floor_deg,
                    ),
                    max(
                        duration * right_fit.velocity_se,
                        derivative_uncertainty_floor_deg,
                    ),
                ],
                dtype=float,
            )
            acceleration_uncertainty = np.array(
                [
                    max(
                        duration**2 * left_fit.acceleration_se,
                        derivative_uncertainty_floor_deg,
                    ),
                    max(
                        duration**2 * right_fit.acceleration_se,
                        derivative_uncertainty_floor_deg,
                    ),
                ],
                dtype=float,
            )

            velocity_design = _VELOCITY_BASIS / velocity_uncertainty[:, None]
            acceleration_design = (
                _ACCELERATION_BASIS / acceleration_uncertainty[:, None]
            )
            velocity_target = (
                target_velocity_correction / velocity_uncertainty
            )
            acceleration_target = (
                target_acceleration / acceleration_uncertainty
            )

            normal_matrix = (
                velocity_design.T @ velocity_design
                + acceleration_design.T @ acceleration_design
                + acceleration_penalty * _ACCELERATION_GRAM
                + jerk_penalty * _JERK_GRAM
            )
            right_hand_side = (
                velocity_design.T @ velocity_target
                + acceleration_design.T @ acceleration_target
            )
            correction_coefficients = np.linalg.solve(
                normal_matrix + 1e-12 * np.eye(4), right_hand_side
            )

            velocity_residual = (
                velocity_design @ correction_coefficients - velocity_target
            )
            acceleration_residual = (
                acceleration_design @ correction_coefficients
                - acceleration_target
            )
            smoothness_cost = (
                acceleration_penalty
                * float(
                    correction_coefficients
                    @ _ACCELERATION_GRAM
                    @ correction_coefficients
                )
                + jerk_penalty
                * float(
                    correction_coefficients
                    @ _JERK_GRAM
                    @ correction_coefficients
                )
            )
            total_cost = float(
                velocity_residual @ velocity_residual
                + acceleration_residual @ acceleration_residual
                + smoothness_cost
            )
            evidence = float(
                np.sqrt(
                    velocity_target @ velocity_target
                    + acceleration_target @ acceleration_target
                )
            )

            if best_solution is None or total_cost < best_solution[0]:
                best_solution = (
                    total_cost,
                    turn_offset,
                    right_position,
                    correction_coefficients,
                    evidence,
                )

        assert best_solution is not None
        _, turn_offset, right_position, coefficients, evidence = best_solution
        use_linear_fallback = evidence < kinematic_significance_threshold
        if use_linear_fallback:
            coefficients = np.zeros_like(coefficients)

        output_unwrapped[right_start:right_end] = (
            right_local + turn_offset * period
        )

        gap_indices = np.arange(left_index + 1, right_index)
        if gap_indices.size:
            normalized_time = (
                t[gap_indices] - t[left_index]
            ) / duration
            straight_line = left_position + (
                right_position - left_position
            ) * normalized_time
            correction_basis = _basis_derivatives(0, normalized_time)
            output_unwrapped[gap_indices] = (
                straight_line + correction_basis @ coefficients
            )

        decisions.append(
            GapDecision(
                start=left_index + 1,
                end=right_index,
                endpoint_span_s=duration,
                selected_revolution_offset=turn_offset,
                used_linear_fallback=use_linear_fallback,
                kinematic_evidence_sigma=evidence,
                left_velocity_deg_s=left_fit.velocity,
                right_velocity_deg_s=right_fit.velocity,
                left_acceleration_deg_s2=left_fit.acceleration,
                right_acceleration_deg_s2=right_fit.acceleration,
                left_fit_residual_deg=left_fit.residual_std,
                right_fit_residual_deg=right_fit.residual_std,
            )
        )

    first_valid = int(np.flatnonzero(valid)[0])
    last_valid = int(np.flatnonzero(valid)[-1])
    if first_valid > 0:
        output_unwrapped[:first_valid] = output_unwrapped[first_valid]
    if last_valid < t.size - 1:
        output_unwrapped[last_valid + 1 :] = output_unwrapped[last_valid]

    velocity = np.gradient(output_unwrapped, t)
    acceleration = np.gradient(velocity, t)
    wrapped = _wrap_deg(output_unwrapped, period)
    if preserve_valid_samples:
        wrapped[valid] = _wrap_deg(y[valid], period)

    return AdaptiveMinimumJerkResult(
        angle_deg=wrapped,
        unwrapped_deg=output_unwrapped,
        angular_velocity_deg_s=velocity,
        angular_acceleration_deg_s2=acceleration,
        decisions=decisions,
    )
