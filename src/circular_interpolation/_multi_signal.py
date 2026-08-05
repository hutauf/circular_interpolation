from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from ._multi_signal_common import (
    MultiSignalInterpolationResult,
    PeriodInput,
    SignalScalarInput,
    contiguous_runs,
    joint_repeat_mask,
    normalize_periods,
    normalize_signal_scalars,
    validate_multi_input,
    validate_target_time_s,
)
from ._multi_signal_detection import classify_joint_run
from ._periodic import unwrap_values, wrap_values
from .auto import interpolate_circular_auto


def _plateau_duration_s(t: np.ndarray, repeat_start: int, repeat_end: int) -> float:
    anchor = max(0, repeat_start - 1)
    return float(t[repeat_end - 1] - t[anchor])


def interpolate_multi_signal_stream(
    time_s: np.ndarray,
    data: np.ndarray,
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
) -> MultiSignalInterpolationResult:
    """Repair synchronized recorder freezes in ``(signals, samples)`` data."""
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

    t, values = validate_multi_input(time_s, data)
    target = validate_target_time_s(target_time_s, t)
    signal_count = values.shape[0]
    periods = normalize_periods(period, signal_count)
    repeat_tolerances = normalize_signal_scalars(
        repeat_tolerance_deg,
        signal_count,
        name="repeat_tolerance_deg",
        allow_zero=True,
    )
    standstill_thresholds = normalize_signal_scalars(
        standstill_speed_threshold_deg_s,
        signal_count,
        name="standstill_speed_threshold_deg_s",
        allow_zero=True,
    )
    motion_thresholds = normalize_signal_scalars(
        min_motion_evidence_deg,
        signal_count,
        name="min_motion_evidence_deg",
        allow_zero=True,
    )
    acceleration_limits = normalize_signal_scalars(
        max_abs_acceleration_deg_s2,
        signal_count,
        name="max_abs_acceleration_deg_s2",
        allow_zero=False,
    )

    raw_candidate_mask = joint_repeat_mask(
        values,
        periods=periods,
        repeat_tolerances=repeat_tolerances,
    )
    joint_candidate_mask = np.zeros(t.size, dtype=bool)
    joint_repair_mask = np.zeros(t.size, dtype=bool)
    preserved_standstill_mask = np.zeros(t.size, dtype=bool)
    ambiguous_mask = np.zeros(t.size, dtype=bool)
    decisions = []

    for repeat_start, repeat_end in contiguous_runs(raw_candidate_mask):
        if repeat_end - repeat_start < min_repeated_samples:
            continue
        joint_candidate_mask[repeat_start:repeat_end] = True
        decision = classify_joint_run(
            t,
            values,
            repeat_start=repeat_start,
            repeat_end=repeat_end,
            periods=periods,
            fit_window_samples=fit_window_samples,
            standstill_thresholds=standstill_thresholds,
            min_motion_evidence=motion_thresholds,
            long_standstill_duration_s=long_standstill_duration_s,
            max_abs_accelerations=acceleration_limits,
        )

        plateau_slice = slice(decision.repeat_start, decision.repeat_end)
        if decision.kind in {
            "true_standstill",
            "quantization_plateau",
            "boundary_plateau",
        }:
            preserved_standstill_mask[plateau_slice] = True
        if decision.kind == "ambiguous":
            ambiguous_mask[plateau_slice] = True

        selected_for_repair = decision.kind == "held_dropout" or (
            decision.kind == "ambiguous" and ambiguous_policy == "repair"
        )
        if selected_for_repair and max_plateau_repair_duration_s is not None:
            duration_s = _plateau_duration_s(
                t,
                decision.repeat_start,
                decision.repeat_end,
            )
            if duration_s > max_plateau_repair_duration_s:
                decision = replace(
                    decision,
                    reason=(
                        f"{decision.reason} The samples were preserved because the "
                        f"joint plateau duration ({duration_s:.6g} s) exceeds "
                        "max_plateau_repair_duration_s="
                        f"{max_plateau_repair_duration_s:.6g} s."
                    ),
                )
                selected_for_repair = False

        if selected_for_repair:
            start = decision.repeat_start
            if plateau_anchor_policy == "repair":
                start = max(0, start - 1)
            joint_repair_mask[start : decision.repeat_end] = True
        decisions.append(decision)

    output_wrapped: list[np.ndarray] = []
    output_continuous: list[np.ndarray] = []
    repaired_masks: list[np.ndarray] = []
    chosen_models: list[list[tuple[int, int, str]]] = []
    confidences: list[list[tuple[int, int, str]]] = []
    output_time: np.ndarray | None = None

    for signal_index, signal_period in enumerate(periods):
        row = values[signal_index]
        repair_mask = joint_repair_mask | ~np.isfinite(row)
        repaired_masks.append(repair_mask.copy())

        if np.any(repair_mask) or target_time_s is not None:
            try:
                result = interpolate_circular_auto(
                    t,
                    row,
                    repair_mask,
                    period=signal_period,
                    target_time_s=target,
                )
            except ValueError as exc:
                raise ValueError(f"signal {signal_index}: {exc}") from exc
            wrapped = np.asarray(result.angle_deg, dtype=float)
            continuous = np.asarray(result.unwrapped_deg, dtype=float)
            signal_time = np.asarray(result.time_s, dtype=float)
            chosen_models.append(result.chosen_model_by_gap)
            confidences.append(result.confidence_by_gap)
        else:
            continuous = unwrap_values(row, signal_period)
            wrapped = wrap_values(continuous, signal_period)
            signal_time = t.copy()
            chosen_models.append([])
            confidences.append([])

        if output_time is None:
            output_time = signal_time
        elif not np.array_equal(output_time, signal_time):
            raise RuntimeError("All signals must be evaluated on the same time grid.")
        output_wrapped.append(wrapped)
        output_continuous.append(continuous)

    assert output_time is not None
    return MultiSignalInterpolationResult(
        angle_deg=np.stack(output_wrapped, axis=0),
        unwrapped_deg=np.stack(output_continuous, axis=0),
        time_s=output_time,
        repaired_mask=np.stack(repaired_masks, axis=0),
        joint_candidate_mask=joint_candidate_mask,
        joint_repair_mask=joint_repair_mask,
        preserved_standstill_mask=preserved_standstill_mask,
        ambiguous_mask=ambiguous_mask,
        decisions=decisions,
        chosen_model_by_gap=tuple(chosen_models),
        interpolation_confidence_by_gap=tuple(confidences),
        period_by_signal=periods,
    )
