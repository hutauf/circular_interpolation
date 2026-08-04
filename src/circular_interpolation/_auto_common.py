from __future__ import annotations

from dataclasses import dataclass

import numpy as np

class KinematicInterpolationResult:
    def __init__(
        self,
        angle_deg,
        unwrapped_deg,
        angular_velocity_deg_s,
        angular_acceleration_deg_s2,
        rejected_outliers,
    ):
        self.angle_deg = angle_deg
        self.unwrapped_deg = unwrapped_deg
        self.angular_velocity_deg_s = angular_velocity_deg_s
        self.angular_acceleration_deg_s2 = angular_acceleration_deg_s2
        self.rejected_outliers = rejected_outliers


@dataclass
class AutomaticInterpolationResult:
    """Automatic interpolation evaluated on ``time_s``."""

    angle_deg: np.ndarray
    unwrapped_deg: np.ndarray
    chosen_model_by_gap: list[tuple[int, int, str]]
    confidence_by_gap: list[tuple[int, int, str]]
    time_s: np.ndarray | None = None


@dataclass(frozen=True)
class _GapTrajectory:
    start: int
    end: int
    model: str
    confidence: str
    left_time_s: float
    right_time_s: float
    left_position_deg: float
    right_position_deg: float
    left_velocity_deg_s: float = 0.0
    right_velocity_deg_s: float = 0.0
    candidate_offset_deg: float = 0.0


def _validate(time_s: np.ndarray, angle_deg: np.ndarray, invalid_mask: np.ndarray):
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(angle_deg, dtype=float)
    invalid = np.asarray(invalid_mask, dtype=bool) | ~np.isfinite(y)
    if t.ndim != 1 or y.ndim != 1 or invalid.ndim != 1:
        raise ValueError("All inputs must be one-dimensional.")
    if not (t.shape == y.shape == invalid.shape):
        raise ValueError("All inputs must have equal shape.")
    if np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be finite and strictly increasing.")
    if np.count_nonzero(~invalid) < 3:
        raise ValueError("At least three valid samples are required.")
    return t, y, invalid


def _validate_target_time_s(
    target_time_s: np.ndarray | None,
    source_time_s: np.ndarray,
) -> np.ndarray:
    if target_time_s is None:
        return source_time_s.copy()

    target = np.array(target_time_s, dtype=float, copy=True)
    if target.ndim != 1:
        raise ValueError("target_time_s must be one-dimensional.")
    if target.size == 0:
        raise ValueError("target_time_s must contain at least one sample.")
    if np.any(~np.isfinite(target)) or np.any(np.diff(target) <= 0.0):
        raise ValueError("target_time_s must be finite and strictly increasing.")
    return target


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


