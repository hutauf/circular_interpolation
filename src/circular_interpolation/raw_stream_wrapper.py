from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .inertia import (
    circular_difference_deg,
    wrap_deg,
)
from .auto import interpolate_circular_auto


PlateauKind = Literal[
    "held_dropout",
    "true_standstill",
    "quantization_plateau",
    "boundary_plateau",
    "ambiguous",
]


@dataclass(frozen=True)
class PlateauDecision:
    """Classification of one run of repeated angular samples.

    ``repeat_start`` is the first sample that repeats its predecessor.
    ``repeat_end`` is exclusive. The unchanged plateau therefore also includes
    the anchor sample ``repeat_start - 1``.
    """

    repeat_start: int
    repeat_end: int
    kind: PlateauKind
    confidence: Literal["high", "medium", "low"]
    left_velocity_deg_s: float
    right_velocity_deg_s: float
    endpoint_span_ms: float
    predicted_motion_deg: float
    required_stop_acceleration_deg_s2: float
    reason: str


@dataclass
class RawStreamInterpolationResult:
    """Output of :func:`interpolate_raw_circular_stream`."""

    angle_deg: np.ndarray
    unwrapped_deg: np.ndarray
    repaired_mask: np.ndarray
    preserved_standstill_mask: np.ndarray
    ambiguous_mask: np.ndarray
    decisions: list[PlateauDecision]
    chosen_model_by_gap: list[tuple[int, int, str]]
    interpolation_confidence_by_gap: list[tuple[int, int, str]]


def _validate(time_s: np.ndarray, angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(angle_deg, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.shape != y.shape:
        raise ValueError("time_s and angle_deg must be one-dimensional and equally sized.")
    if t.size < 3:
        raise ValueError("At least three samples are required.")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        raise ValueError("time_s must be finite and strictly increasing.")
    return t, y


def _repeat_runs(
    y: np.ndarray,
    *,
    period: float,
    repeat_tolerance_deg: float,
    min_repeated_samples: int,
) -> list[tuple[int, int]]:
    """Return runs [start, end) of samples repeating their predecessor."""
    repeated = np.zeros(y.size, dtype=bool)
    finite_pairs = np.isfinite(y[1:]) & np.isfinite(y[:-1])
    repeated[1:] = finite_pairs & (
        np.abs(circular_difference_deg(y[1:], y[:-1], period))
        <= repeat_tolerance_deg
    )

    padded = np.r_[False, repeated, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs: list[tuple[int, int]] = []
    for start, end in edges.reshape(-1, 2):
        if end - start >= min_repeated_samples:
            runs.append((int(start), int(end)))
    return runs


def _robust_edge_velocity(
    t: np.ndarray,
    y: np.ndarray,
    *,
    start: int,
    end: int,
    side: Literal["left", "right"],
    window_samples: int,
    period: float,
) -> float:
    """Estimate local velocity near a plateau edge using median sample slopes."""
    if side == "left":
        lo = max(0, start - window_samples + 1)
        hi = start + 1
    else:
        lo = end
        hi = min(y.size, end + window_samples)

    indices = np.arange(lo, hi)
    finite = np.isfinite(y[indices])
    indices = indices[finite]
    if indices.size < 2:
        return 0.0

    local_t = t[indices]
    local_y = np.unwrap(y[indices], period=period)
    dt = np.diff(local_t)
    slopes = np.diff(local_y) / dt
    if slopes.size == 0:
        return 0.0

    # The median is intentionally conservative around quantization plateaus and
    # rejects isolated noisy encoder steps better than an endpoint derivative.
    return float(np.median(slopes))


def _hermite_motion_evidence(
    *,
    y0: float,
    y1_wrapped: float,
    v0: float,
    v1: float,
    duration_s: float,
    sample_times_s: np.ndarray,
    period: float,
) -> tuple[float, float]:
    """Return maximum predicted motion and selected endpoint displacement."""
    expected_displacement = 0.5 * (v0 + v1) * duration_s
    y1 = y1_wrapped + round((y0 + expected_displacement - y1_wrapped) / period) * period

    if duration_s <= 0.0 or sample_times_s.size == 0:
        return 0.0, float(y1 - y0)

    u = np.clip(sample_times_s / duration_s, 0.0, 1.0)
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    path = h00 * y0 + h10 * duration_s * v0 + h01 * y1 + h11 * duration_s * v1
    return float(np.max(np.abs(path - y0))), float(y1 - y0)


def classify_repeated_plateaus(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    *,
    period: float = 360.0,
    repeat_tolerance_deg: float = 0.0,
    min_repeated_samples: int = 1,
    fit_window_samples: int = 12,
    standstill_speed_threshold_deg_s: float = 100.0,
    min_motion_evidence_deg: float = 0.15,
    long_standstill_duration_s: float = 0.050,
    max_abs_acceleration_deg_s2: float = 1_000_000.0,
) -> list[PlateauDecision]:
    """Classify unchanged runs without requiring an externally supplied mask.

    The detector is deliberately conservative:

    * leading and trailing plateaus are preserved;
    * plateaus with near-zero edge velocity are preserved as true standstill or
      encoder quantization;
    * only an internal plateau with clear two-sided motion evidence is marked as
      a held-value dropout;
    * uncertain cases are returned as ``ambiguous`` and are not repaired by the
      default wrapper policy.

    Angle data alone cannot distinguish every true stop from every frozen sensor.
    Physical acceleration limits and two-sided look-ahead make the distinction
    much safer, but not mathematically identifiable in all cases.
    """
    t, y = _validate(time_s, angle_deg)
    if repeat_tolerance_deg < 0.0:
        raise ValueError("repeat_tolerance_deg must be non-negative.")
    if min_repeated_samples < 1:
        raise ValueError("min_repeated_samples must be at least one.")
    if fit_window_samples < 2:
        raise ValueError("fit_window_samples must be at least two.")

    dt_typical = float(np.median(np.diff(t)))
    decisions: list[PlateauDecision] = []

    for repeat_start, repeat_end in _repeat_runs(
        y,
        period=period,
        repeat_tolerance_deg=repeat_tolerance_deg,
        min_repeated_samples=min_repeated_samples,
    ):
        anchor = repeat_start - 1
        right = repeat_end
        touches_start = anchor == 0
        touches_end = right >= y.size
        endpoint_span_s = (
            t[right] - t[anchor] if not touches_end else t[-1] - t[anchor]
        )
        repeated_duration_s = t[repeat_end - 1] - t[anchor]

        if touches_start or touches_end:
            decisions.append(
                PlateauDecision(
                    repeat_start=repeat_start,
                    repeat_end=repeat_end,
                    kind="boundary_plateau",
                    confidence="high",
                    left_velocity_deg_s=0.0,
                    right_velocity_deg_s=0.0,
                    endpoint_span_ms=1000.0 * endpoint_span_s,
                    predicted_motion_deg=0.0,
                    required_stop_acceleration_deg_s2=0.0,
                    reason=(
                        "Plateau touches the beginning or end of the measurement. "
                        "Without two-sided evidence it is preserved as a normal boundary standstill."
                    ),
                )
            )
            continue

        v_left = _robust_edge_velocity(
            t,
            y,
            start=anchor,
            end=right,
            side="left",
            window_samples=fit_window_samples,
            period=period,
        )
        v_right = _robust_edge_velocity(
            t,
            y,
            start=anchor,
            end=right,
            side="right",
            window_samples=fit_window_samples,
            period=period,
        )

        relative_times = t[repeat_start:repeat_end] - t[anchor]
        motion_evidence, endpoint_displacement = _hermite_motion_evidence(
            y0=float(y[anchor]),
            y1_wrapped=float(y[right]),
            v0=v_left,
            v1=v_right,
            duration_s=endpoint_span_s,
            sample_times_s=relative_times,
            period=period,
        )

        left_speed = abs(v_left)
        right_speed = abs(v_right)
        maximum_speed = max(left_speed, right_speed)
        minimum_speed = min(left_speed, right_speed)
        same_direction = (
            v_left * v_right > 0.0
            or minimum_speed <= standstill_speed_threshold_deg_s
        )
        velocity_change_ratio = abs(v_right - v_left) / max(maximum_speed, 1.0)

        # A genuine stop beginning immediately at the plateau edge would need at
        # least this acceleration over one sample interval. This is not a claim
        # about the exact motor trajectory; it is a useful plausibility bound.
        required_stop_accel = maximum_speed / max(dt_typical, 1e-12)
        stop_is_physically_hard = required_stop_accel > max_abs_acceleration_deg_s2

        if maximum_speed <= standstill_speed_threshold_deg_s:
            if repeated_duration_s >= long_standstill_duration_s:
                kind: PlateauKind = "true_standstill"
                reason = (
                    "Both edge-velocity estimates are near zero and the plateau is long; "
                    "it is preserved as a genuine standstill."
                )
            else:
                kind = "quantization_plateau"
                reason = (
                    "The plateau is short and edge motion is below the standstill threshold; "
                    "it is treated as normal encoder quantization or micro-motion."
                )
            confidence: Literal["high", "medium", "low"] = "high"

        elif (
            same_direction
            and minimum_speed > standstill_speed_threshold_deg_s
            and motion_evidence >= min_motion_evidence_deg
            and stop_is_physically_hard
            and velocity_change_ratio <= 1.0
        ):
            kind = "held_dropout"
            confidence = "high" if velocity_change_ratio <= 0.35 else "medium"
            reason = (
                "The motor is clearly moving on both sides in a compatible direction, "
                "the look-ahead trajectory moves away from the held value, and an "
                "instantaneous true stop would exceed the configured acceleration limit."
            )

        elif (
            same_direction
            and maximum_speed > 2.0 * standstill_speed_threshold_deg_s
            and motion_evidence >= 2.0 * min_motion_evidence_deg
            and stop_is_physically_hard
            and velocity_change_ratio <= 1.5
        ):
            kind = "held_dropout"
            confidence = "low"
            reason = (
                "There is substantial two-sided motion evidence, but one edge estimate is "
                "weak or the velocity changes strongly. The plateau is probably a held value."
            )

        else:
            kind = "ambiguous"
            confidence = "low"
            if v_left * v_right < 0.0:
                reason = (
                    "The estimated direction changes across the plateau. This may be a true "
                    "stop/reversal or a dropout spanning a reversal, so it is preserved."
                )
            elif abs(endpoint_displacement) < min_motion_evidence_deg:
                reason = (
                    "The two-sided trajectory predicts too little angular motion to justify "
                    "overwriting the repeated samples."
                )
            else:
                reason = (
                    "The evidence is insufficient to distinguish a true stop from a frozen "
                    "sensor value safely."
                )

        decisions.append(
            PlateauDecision(
                repeat_start=repeat_start,
                repeat_end=repeat_end,
                kind=kind,
                confidence=confidence,
                left_velocity_deg_s=v_left,
                right_velocity_deg_s=v_right,
                endpoint_span_ms=1000.0 * endpoint_span_s,
                predicted_motion_deg=motion_evidence,
                required_stop_acceleration_deg_s2=required_stop_accel,
                reason=reason,
            )
        )

    return decisions


def interpolate_raw_circular_stream(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    *,
    period: float = 360.0,
    repeat_tolerance_deg: float = 0.0,
    min_repeated_samples: int = 1,
    fit_window_samples: int = 12,
    standstill_speed_threshold_deg_s: float = 100.0,
    min_motion_evidence_deg: float = 0.15,
    long_standstill_duration_s: float = 0.050,
    max_abs_acceleration_deg_s2: float = 1_000_000.0,
    ambiguous_policy: Literal["preserve", "repair"] = "preserve",
) -> RawStreamInterpolationResult:
    """Detect held-value plateaus and interpolate them from a raw angle stream.

    No invalid-data mask is required. True standstill, boundary plateaus and
    quantization plateaus are retained. By default, ambiguous plateaus are also
    retained. Set ``ambiguous_policy='repair'`` only if false negatives are more
    costly than false positives for the application.
    """
    t, y = _validate(time_s, angle_deg)
    decisions = classify_repeated_plateaus(
        t,
        y,
        period=period,
        repeat_tolerance_deg=repeat_tolerance_deg,
        min_repeated_samples=min_repeated_samples,
        fit_window_samples=fit_window_samples,
        standstill_speed_threshold_deg_s=standstill_speed_threshold_deg_s,
        min_motion_evidence_deg=min_motion_evidence_deg,
        long_standstill_duration_s=long_standstill_duration_s,
        max_abs_acceleration_deg_s2=max_abs_acceleration_deg_s2,
    )

    repair_mask = ~np.isfinite(y)
    standstill_mask = np.zeros(y.size, dtype=bool)
    ambiguous_mask = np.zeros(y.size, dtype=bool)

    for decision in decisions:
        sl = slice(decision.repeat_start, decision.repeat_end)
        if decision.kind == "held_dropout":
            repair_mask[sl] = True
        elif decision.kind in {
            "true_standstill",
            "quantization_plateau",
            "boundary_plateau",
        }:
            standstill_mask[sl] = True
        else:
            ambiguous_mask[sl] = True
            if ambiguous_policy == "repair":
                repair_mask[sl] = True

    if np.any(repair_mask):
        auto = interpolate_circular_auto(t, y, repair_mask, period=period)
        output_angle = auto.angle_deg
        output_unwrapped = auto.unwrapped_deg
        chosen_models = auto.chosen_model_by_gap
        interpolation_confidence = auto.confidence_by_gap
    else:
        output_unwrapped = np.unwrap(y, period=period)
        output_angle = wrap_deg(output_unwrapped, period)
        chosen_models = []
        interpolation_confidence = []

    # Preserve every sample that was not explicitly selected for repair.
    output_angle[~repair_mask & np.isfinite(y)] = wrap_deg(
        y[~repair_mask & np.isfinite(y)], period
    )

    return RawStreamInterpolationResult(
        angle_deg=output_angle,
        unwrapped_deg=output_unwrapped,
        repaired_mask=repair_mask,
        preserved_standstill_mask=standstill_mask,
        ambiguous_mask=ambiguous_mask,
        decisions=decisions,
        chosen_model_by_gap=chosen_models,
        interpolation_confidence_by_gap=interpolation_confidence,
    )
