from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from ._periodic import Period, unwrap_values, wrap_values
from .auto import interpolate_circular_auto
from .raw_stream_wrapper import (
    PlateauDecision,
    RawStreamInterpolationResult,
    interpolate_raw_circular_stream as _interpolate_raw_circular_stream,
)


def _plateau_duration_s(
    time_s: np.ndarray,
    decision: PlateauDecision,
) -> float:
    """Return the full repeated-value duration, including its measured anchor."""
    anchor = max(0, decision.repeat_start - 1)
    last_repeated = decision.repeat_end - 1
    return float(time_s[last_repeated] - time_s[anchor])


def interpolate_raw_circular_stream(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    *,
    period: Period = 360.0,
    repeat_tolerance_deg: float = 0.0,
    min_repeated_samples: int = 1,
    fit_window_samples: int = 12,
    standstill_speed_threshold_deg_s: float = 100.0,
    min_motion_evidence_deg: float = 0.15,
    long_standstill_duration_s: float = 0.050,
    max_abs_acceleration_deg_s2: float = 1_000_000.0,
    ambiguous_policy: Literal["preserve", "repair"] = "preserve",
    plateau_anchor_policy: Literal["preserve", "repair"] = "preserve",
    max_plateau_repair_duration_s: float | None = None,
) -> RawStreamInterpolationResult:
    """Detect and repair held-value plateaus in a raw signal.

    A positive finite ``period`` enables circular branch handling. Set
    ``period=None`` for an ordinary scalar signal with no wrap or unwrap step.

    The first occurrence of a repeated value is the *plateau anchor*. It was
    measured before the sensor began returning the same stale value, so the
    default ``plateau_anchor_policy='preserve'`` keeps it. For example,
    ``[1, 2, 3, 3, 3, 3, 7]`` repairs only the final three ``3`` samples.

    Set ``plateau_anchor_policy='repair'`` only when the sensor protocol says
    that the first occurrence may already be stale. Classification itself is
    unchanged; the option controls whether the anchor joins a plateau that was
    selected for reconstruction.

    ``ambiguous_policy='repair'`` is intentionally aggressive. A genuine
    internal standstill and a frozen sensor can be indistinguishable from values
    alone. Set ``max_plateau_repair_duration_s`` to the longest plausible
    automatically detected dropout when false repairs are costly. Longer
    detected plateaus are preserved even if they were classified as ambiguous
    or as a held dropout. Explicit non-finite samples are still repaired, and
    leading or trailing boundary plateaus remain preserved as before.
    """
    if ambiguous_policy not in {"preserve", "repair"}:
        raise ValueError("ambiguous_policy must be 'preserve' or 'repair'.")
    if plateau_anchor_policy not in {"preserve", "repair"}:
        raise ValueError("plateau_anchor_policy must be 'preserve' or 'repair'.")
    if max_plateau_repair_duration_s is not None:
        max_plateau_repair_duration_s = float(max_plateau_repair_duration_s)
        if (
            not np.isfinite(max_plateau_repair_duration_s)
            or max_plateau_repair_duration_s <= 0.0
        ):
            raise ValueError(
                "max_plateau_repair_duration_s must be positive and finite, "
                "or None."
            )

    # Ask the lower-level detector to keep ambiguous plateaus. This wrapper then
    # applies the complete public repair policy once, including the anchor and
    # duration safeguards, so repaired_mask always describes the actual work.
    detected = _interpolate_raw_circular_stream(
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
        ambiguous_policy="preserve",
    )

    t = np.asarray(time_s, dtype=float)
    y = np.asarray(angle_deg, dtype=float)
    repair_mask = ~np.isfinite(y)
    decisions = list(detected.decisions)

    for index, decision in enumerate(decisions):
        selected_for_repair = decision.kind == "held_dropout" or (
            decision.kind == "ambiguous" and ambiguous_policy == "repair"
        )
        if not selected_for_repair:
            continue

        plateau_duration_s = _plateau_duration_s(t, decision)
        if (
            max_plateau_repair_duration_s is not None
            and plateau_duration_s > max_plateau_repair_duration_s
        ):
            decisions[index] = replace(
                decision,
                reason=(
                    f"{decision.reason} The samples were preserved because the "
                    f"plateau duration ({plateau_duration_s:.6g} s) exceeds "
                    "max_plateau_repair_duration_s="
                    f"{max_plateau_repair_duration_s:.6g} s."
                ),
            )
            continue

        start = decision.repeat_start
        if plateau_anchor_policy == "repair":
            start = max(0, start - 1)
        repair_mask[start : decision.repeat_end] = True

    if (
        np.array_equal(repair_mask, detected.repaired_mask)
        and decisions == detected.decisions
    ):
        return detected

    if np.any(repair_mask):
        auto = interpolate_circular_auto(t, y, repair_mask, period=period)
        output_angle = auto.angle_deg.copy()
        output_unwrapped = auto.unwrapped_deg
        chosen_models = auto.chosen_model_by_gap
        interpolation_confidence = auto.confidence_by_gap
    else:
        output_unwrapped = unwrap_values(y, period)
        output_angle = wrap_values(output_unwrapped, period)
        chosen_models = []
        interpolation_confidence = []

    # Every finite sample outside the final repair mask remains the exact sensor
    # value. This includes boundary plateaus and duration-limited internal runs.
    keep = ~repair_mask & np.isfinite(y)
    output_angle[keep] = wrap_values(y[keep], period)

    return RawStreamInterpolationResult(
        angle_deg=output_angle,
        unwrapped_deg=output_unwrapped,
        repaired_mask=repair_mask,
        preserved_standstill_mask=detected.preserved_standstill_mask,
        ambiguous_mask=detected.ambiguous_mask,
        decisions=decisions,
        chosen_model_by_gap=chosen_models,
        interpolation_confidence_by_gap=interpolation_confidence,
    )
