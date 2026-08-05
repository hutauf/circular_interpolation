from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from ._multi_signal import interpolate_multi_signal_stream
from ._multi_signal_common import (
    JointPlateauDecision,
    MultiSignalInterpolationResult,
    PeriodInput,
    SignalScalarInput,
    normalize_periods,
    normalize_signal_scalars,
    validate_target_time_s,
)
from ._periodic import Period, unwrap_values, wrap_values
from .auto import interpolate_circular_auto
from .raw_stream_wrapper import (
    RawStreamInterpolationResult,
    interpolate_raw_circular_stream as _interpolate_raw_circular_stream,
)


def _plateau_duration_s(time_s: np.ndarray, decision: object) -> float:
    repeat_start = int(getattr(decision, "repeat_start"))
    repeat_end = int(getattr(decision, "repeat_end"))
    anchor = max(0, repeat_start - 1)
    return float(time_s[repeat_end - 1] - time_s[anchor])


def _attach_single_signal_metadata(
    result: RawStreamInterpolationResult,
    *,
    time_s: np.ndarray,
    period: Period,
    inferred_repair_mask: np.ndarray,
) -> RawStreamInterpolationResult:
    # RawStreamInterpolationResult predates target-grid and multi-signal support
    # and has no slots. Additive attributes preserve its constructor and all
    # existing field semantics.
    result.time_s = np.asarray(time_s, dtype=float)
    result.period_by_signal = (period,)
    result.inferred_freeze_mask = np.asarray(inferred_repair_mask, dtype=bool)
    return result


def _interpolate_single_signal(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    *,
    period: Period,
    target_time_s: np.ndarray | None,
    repeat_tolerance_deg: float,
    min_repeated_samples: int,
    fit_window_samples: int,
    standstill_speed_threshold_deg_s: float,
    min_motion_evidence_deg: float,
    long_standstill_duration_s: float,
    max_abs_acceleration_deg_s2: float,
    ambiguous_policy: Literal["preserve", "repair"],
    plateau_anchor_policy: Literal["preserve", "repair"],
    max_plateau_repair_duration_s: float | None,
) -> RawStreamInterpolationResult:
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
    target = validate_target_time_s(target_time_s, t)
    values = np.asarray(angle_deg, dtype=float)
    repair_mask = ~np.isfinite(values)
    inferred_repair_mask = np.zeros(t.size, dtype=bool)
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
        inferred_repair_mask[start : decision.repeat_end] = True
        repair_mask[start : decision.repeat_end] = True

    if (
        target_time_s is None
        and np.array_equal(repair_mask, detected.repaired_mask)
        and decisions == detected.decisions
    ):
        return _attach_single_signal_metadata(
            detected,
            time_s=t.copy(),
            period=period,
            inferred_repair_mask=inferred_repair_mask,
        )

    if np.any(repair_mask) or target_time_s is not None:
        auto = interpolate_circular_auto(
            t,
            values,
            repair_mask,
            period=period,
            target_time_s=target,
        )
        output_angle = np.asarray(auto.angle_deg, dtype=float).copy()
        output_unwrapped = np.asarray(auto.unwrapped_deg, dtype=float)
        output_time = np.asarray(auto.time_s, dtype=float)
        chosen_models = auto.chosen_model_by_gap
        interpolation_confidence = auto.confidence_by_gap
    else:
        output_unwrapped = unwrap_values(values, period)
        output_angle = wrap_values(output_unwrapped, period)
        output_time = t.copy()
        chosen_models = []
        interpolation_confidence = []

    if output_time.shape == t.shape and np.array_equal(output_time, t):
        keep = ~repair_mask & np.isfinite(values)
        output_angle[keep] = wrap_values(values[keep], period)

    result = RawStreamInterpolationResult(
        angle_deg=output_angle,
        unwrapped_deg=output_unwrapped,
        repaired_mask=repair_mask,
        preserved_standstill_mask=detected.preserved_standstill_mask,
        ambiguous_mask=detected.ambiguous_mask,
        decisions=decisions,
        chosen_model_by_gap=chosen_models,
        interpolation_confidence_by_gap=interpolation_confidence,
    )
    return _attach_single_signal_metadata(
        result,
        time_s=output_time,
        period=period,
        inferred_repair_mask=inferred_repair_mask,
    )


def interpolate_raw_circular_stream(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    *,
    period: PeriodInput = 360.0,
    target_time_s: np.ndarray | None = None,
    repeat_tolerance_deg: SignalScalarInput = 0.0,
    min_repeated_samples: int = 1,
    fit_window_samples: int = 12,
    standstill_speed_threshold_deg_s: SignalScalarInput = 100.0,
    min_motion_evidence_deg: SignalScalarInput = 0.15,
    long_standstill_duration_s: float = 0.050,
    max_abs_acceleration_deg_s2: SignalScalarInput = 1_000_000.0,
    ambiguous_policy: Literal["preserve", "repair"] = "preserve",
    plateau_anchor_policy: Literal["preserve", "repair"] = "preserve",
    max_plateau_repair_duration_s: float | None = None,
) -> RawStreamInterpolationResult | MultiSignalInterpolationResult:
    """Detect and repair frozen values in one or more synchronized signals.

    A one-dimensional input preserves the historical API. A two-dimensional
    input must have shape ``(signals, samples)``. For 2-D data, an inferred
    recorder-freeze candidate exists only where every row repeats its own
    preceding value at the same source timestamps.

    ``period`` may be one scalar/``None`` broadcast to all rows or a sequence
    with one entry per row, for example ``[360.0, None, None, 360.0]``. The
    value-domain thresholds may likewise be scalars or per-signal sequences.
    ``target_time_s`` evaluates every signal on the same output grid.
    """
    if ambiguous_policy not in {"preserve", "repair"}:
        raise ValueError("ambiguous_policy must be 'preserve' or 'repair'.")
    if plateau_anchor_policy not in {"preserve", "repair"}:
        raise ValueError("plateau_anchor_policy must be 'preserve' or 'repair'.")
    if min_repeated_samples < 1:
        raise ValueError("min_repeated_samples must be at least one.")
    if fit_window_samples < 2:
        raise ValueError("fit_window_samples must be at least two.")
    if (
        not np.isfinite(long_standstill_duration_s)
        or long_standstill_duration_s < 0.0
    ):
        raise ValueError(
            "long_standstill_duration_s must be finite and non-negative."
        )
    if max_plateau_repair_duration_s is not None:
        max_plateau_repair_duration_s = float(max_plateau_repair_duration_s)
        if (
            not np.isfinite(max_plateau_repair_duration_s)
            or max_plateau_repair_duration_s <= 0.0
        ):
            raise ValueError(
                "max_plateau_repair_duration_s must be positive and finite, or None."
            )

    values = np.asarray(angle_deg, dtype=float)
    if values.ndim == 2:
        return interpolate_multi_signal_stream(
            time_s,
            values,
            period=period,
            target_time_s=target_time_s,
            repeat_tolerance_deg=repeat_tolerance_deg,
            min_repeated_samples=min_repeated_samples,
            fit_window_samples=fit_window_samples,
            standstill_speed_threshold_deg_s=standstill_speed_threshold_deg_s,
            min_motion_evidence_deg=min_motion_evidence_deg,
            long_standstill_duration_s=long_standstill_duration_s,
            max_abs_acceleration_deg_s2=max_abs_acceleration_deg_s2,
            ambiguous_policy=ambiguous_policy,
            plateau_anchor_policy=plateau_anchor_policy,
            max_plateau_repair_duration_s=max_plateau_repair_duration_s,
        )
    if values.ndim != 1:
        raise ValueError(
            "angle_deg must be one-dimensional or have shape (signals, samples)."
        )

    single_period = normalize_periods(period, 1)[0]
    repeat_tolerance = normalize_signal_scalars(
        repeat_tolerance_deg,
        1,
        name="repeat_tolerance_deg",
        allow_zero=True,
    )[0]
    standstill_threshold = normalize_signal_scalars(
        standstill_speed_threshold_deg_s,
        1,
        name="standstill_speed_threshold_deg_s",
        allow_zero=True,
    )[0]
    motion_threshold = normalize_signal_scalars(
        min_motion_evidence_deg,
        1,
        name="min_motion_evidence_deg",
        allow_zero=True,
    )[0]
    acceleration_limit = normalize_signal_scalars(
        max_abs_acceleration_deg_s2,
        1,
        name="max_abs_acceleration_deg_s2",
        allow_zero=False,
    )[0]

    return _interpolate_single_signal(
        time_s,
        values,
        period=single_period,
        target_time_s=target_time_s,
        repeat_tolerance_deg=repeat_tolerance,
        min_repeated_samples=min_repeated_samples,
        fit_window_samples=fit_window_samples,
        standstill_speed_threshold_deg_s=standstill_threshold,
        min_motion_evidence_deg=motion_threshold,
        long_standstill_duration_s=long_standstill_duration_s,
        max_abs_acceleration_deg_s2=acceleration_limit,
        ambiguous_policy=ambiguous_policy,
        plateau_anchor_policy=plateau_anchor_policy,
        max_plateau_repair_duration_s=max_plateau_repair_duration_s,
    )


__all__ = [
    "JointPlateauDecision",
    "MultiSignalInterpolationResult",
    "interpolate_raw_circular_stream",
]
