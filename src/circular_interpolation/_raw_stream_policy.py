from __future__ import annotations

from typing import Literal

import numpy as np

from .auto import interpolate_circular_auto
from .inertia import wrap_deg
from .raw_stream_wrapper import (
    RawStreamInterpolationResult,
    interpolate_raw_circular_stream as _interpolate_raw_circular_stream,
)


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
    plateau_anchor_policy: Literal["preserve", "repair"] = "preserve",
) -> RawStreamInterpolationResult:
    """Detect and repair held-value plateaus in a raw circular signal.

    The first occurrence of a repeated value is the *plateau anchor*. It was
    measured before the sensor began returning the same stale value, so the
    default ``plateau_anchor_policy='preserve'`` keeps it. For example,
    ``[1, 2, 3, 3, 3, 3, 7]`` repairs only the final three ``3`` samples.

    Set ``plateau_anchor_policy='repair'`` only when the sensor protocol says
    that the first occurrence may already be stale. Classification itself is
    unchanged; the option controls whether the anchor joins a plateau that was
    selected for reconstruction.
    """
    if ambiguous_policy not in {"preserve", "repair"}:
        raise ValueError("ambiguous_policy must be 'preserve' or 'repair'.")
    if plateau_anchor_policy not in {"preserve", "repair"}:
        raise ValueError("plateau_anchor_policy must be 'preserve' or 'repair'.")

    result = _interpolate_raw_circular_stream(
        time_s=time_s,
        angle_deg=angle_deg,
        period=period,
        repeat_tolerance_deg=repeat_tolerance_deg,
        min_repeated_samples=min_repeated_samples,
        fit_window_samples=fit_window_samples,
        standstill_speed_threshold_deg_s=standstill_speed_threshold_deg_s,
        min_motion_evidence_deg=min_motion_evidence_deg,
        long_standstill_duration_s=long_standstill_duration_s,
        max_abs_acceleration_deg_s2=max_abs_acceleration_deg_s2,
        ambiguous_policy=ambiguous_policy,
    )

    if plateau_anchor_policy == "preserve":
        return result

    repair_mask = result.repaired_mask.copy()
    for decision in result.decisions:
        selected_for_repair = decision.kind == "held_dropout" or (
            decision.kind == "ambiguous" and ambiguous_policy == "repair"
        )
        if selected_for_repair:
            repair_mask[max(0, decision.repeat_start - 1) : decision.repeat_end] = True

    if np.array_equal(repair_mask, result.repaired_mask):
        return result

    t = np.asarray(time_s, dtype=float)
    y = np.asarray(angle_deg, dtype=float)
    auto = interpolate_circular_auto(t, y, repair_mask, period=period)
    output_angle = auto.angle_deg.copy()

    # Samples outside the explicitly selected repair mask remain exact sensor
    # measurements, matching the established raw-stream result semantics.
    keep = ~repair_mask & np.isfinite(y)
    output_angle[keep] = wrap_deg(y[keep], period)

    return RawStreamInterpolationResult(
        angle_deg=output_angle,
        unwrapped_deg=auto.unwrapped_deg,
        repaired_mask=repair_mask,
        preserved_standstill_mask=result.preserved_standstill_mask,
        ambiguous_mask=result.ambiguous_mask,
        decisions=result.decisions,
        chosen_model_by_gap=auto.chosen_model_by_gap,
        interpolation_confidence_by_gap=auto.confidence_by_gap,
    )
