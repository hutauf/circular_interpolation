from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._multi_signal_common import Confidence, JointPlateauDecision
from ._periodic import Period
from .raw_stream_wrapper import (
    PlateauDecision,
    PlateauKind,
    _hermite_motion_evidence,
    _robust_edge_velocity,
)


def _boundary_decision(
    repeat_start: int,
    repeat_end: int,
    endpoint_span_ms: float,
) -> PlateauDecision:
    return PlateauDecision(
        repeat_start=repeat_start,
        repeat_end=repeat_end,
        kind="boundary_plateau",
        confidence="high",
        left_velocity_deg_s=0.0,
        right_velocity_deg_s=0.0,
        endpoint_span_ms=endpoint_span_ms,
        predicted_motion_deg=0.0,
        required_stop_acceleration_deg_s2=0.0,
        reason=(
            "The joint plateau touches the beginning or end of the measurement. "
            "Without two-sided evidence it is preserved."
        ),
    )


def _classify_signal_run(
    t: np.ndarray,
    row: np.ndarray,
    *,
    repeat_start: int,
    repeat_end: int,
    period: Period,
    fit_window_samples: int,
    standstill_speed_threshold: float,
    min_motion_evidence: float,
    long_standstill_duration_s: float,
    max_abs_acceleration: float,
) -> PlateauDecision:
    anchor = repeat_start - 1
    right = repeat_end
    endpoint_span_s = float(t[right] - t[anchor])
    repeated_duration_s = float(t[repeat_end - 1] - t[anchor])

    velocity_left = _robust_edge_velocity(
        t,
        row,
        start=anchor,
        end=right,
        side="left",
        window_samples=fit_window_samples,
        period=period,
    )
    velocity_right = _robust_edge_velocity(
        t,
        row,
        start=anchor,
        end=right,
        side="right",
        window_samples=fit_window_samples,
        period=period,
    )

    relative_times = t[repeat_start:repeat_end] - t[anchor]
    motion_evidence, endpoint_displacement = _hermite_motion_evidence(
        y0=float(row[anchor]),
        y1_observed=float(row[right]),
        v0=velocity_left,
        v1=velocity_right,
        duration_s=endpoint_span_s,
        sample_times_s=relative_times,
        period=period,
    )

    left_speed = abs(velocity_left)
    right_speed = abs(velocity_right)
    maximum_speed = max(left_speed, right_speed)
    minimum_speed = min(left_speed, right_speed)
    same_direction = (
        velocity_left * velocity_right > 0.0
        or minimum_speed <= standstill_speed_threshold
    )
    velocity_change_ratio = (
        abs(velocity_right - velocity_left) / max(maximum_speed, 1.0)
    )
    typical_dt_s = float(np.median(np.diff(t)))
    required_stop_acceleration = maximum_speed / max(typical_dt_s, 1e-12)
    stop_is_physically_hard = required_stop_acceleration > max_abs_acceleration

    if maximum_speed <= standstill_speed_threshold:
        if repeated_duration_s >= long_standstill_duration_s:
            kind: PlateauKind = "true_standstill"
            reason = (
                "Both edge-velocity estimates are near zero and the plateau is "
                "long enough to be consistent with a genuine standstill."
            )
        else:
            kind = "quantization_plateau"
            reason = (
                "Edge motion is below the standstill threshold; this channel is "
                "consistent with quantization or micro-motion."
            )
        confidence: Confidence = "high"
    elif (
        same_direction
        and minimum_speed > standstill_speed_threshold
        and motion_evidence >= min_motion_evidence
        and stop_is_physically_hard
        and velocity_change_ratio <= 1.0
    ):
        kind = "held_dropout"
        confidence = "high" if velocity_change_ratio <= 0.35 else "medium"
        reason = (
            "The signal moves on both sides in a compatible direction and an "
            "instantaneous true stop would exceed the acceleration limit."
        )
    elif (
        same_direction
        and maximum_speed > 2.0 * standstill_speed_threshold
        and motion_evidence >= 2.0 * min_motion_evidence
        and stop_is_physically_hard
        and velocity_change_ratio <= 1.5
    ):
        kind = "held_dropout"
        confidence = "low"
        reason = (
            "There is substantial two-sided motion evidence, but one edge is weak "
            "or the velocity changes strongly."
        )
    else:
        kind = "ambiguous"
        confidence = "low"
        if velocity_left * velocity_right < 0.0:
            reason = (
                "The estimated direction changes across the plateau, so a true "
                "stop or reversal cannot be separated safely from a freeze."
            )
        elif abs(endpoint_displacement) < min_motion_evidence:
            reason = (
                "The two-sided trajectory predicts too little motion to justify "
                "overwriting the repeated samples confidently."
            )
        else:
            reason = (
                "The evidence is insufficient to distinguish a true stop from a "
                "frozen recorder value safely."
            )

    return PlateauDecision(
        repeat_start=repeat_start,
        repeat_end=repeat_end,
        kind=kind,
        confidence=confidence,
        left_velocity_deg_s=velocity_left,
        right_velocity_deg_s=velocity_right,
        endpoint_span_ms=1000.0 * endpoint_span_s,
        predicted_motion_deg=motion_evidence,
        required_stop_acceleration_deg_s2=required_stop_acceleration,
        reason=reason,
    )


def _best_confidence(decisions: Sequence[PlateauDecision]) -> Confidence:
    order: dict[Confidence, int] = {"low": 0, "medium": 1, "high": 2}
    return max(
        (decision.confidence for decision in decisions),
        key=lambda confidence: order[confidence],
    )


def classify_joint_run(
    t: np.ndarray,
    data: np.ndarray,
    *,
    repeat_start: int,
    repeat_end: int,
    periods: tuple[Period, ...],
    fit_window_samples: int,
    standstill_thresholds: tuple[float, ...],
    min_motion_evidence: tuple[float, ...],
    long_standstill_duration_s: float,
    max_abs_accelerations: tuple[float, ...],
) -> JointPlateauDecision:
    anchor = repeat_start - 1
    right = repeat_end
    touches_start = anchor == 0
    touches_end = right >= t.size
    endpoint_span_s = (
        float(t[right] - t[anchor])
        if not touches_end
        else float(t[-1] - t[anchor])
    )
    endpoint_span_ms = 1000.0 * endpoint_span_s

    if touches_start or touches_end:
        signal_decisions = tuple(
            _boundary_decision(repeat_start, repeat_end, endpoint_span_ms)
            for _ in range(data.shape[0])
        )
        return JointPlateauDecision(
            repeat_start=repeat_start,
            repeat_end=repeat_end,
            kind="boundary_plateau",
            confidence="high",
            endpoint_span_ms=endpoint_span_ms,
            signal_decisions=signal_decisions,
            reason=(
                "Every signal repeats, but the run touches a measurement boundary "
                "and is therefore preserved."
            ),
        )

    signal_decisions = tuple(
        _classify_signal_run(
            t,
            data[signal_index],
            repeat_start=repeat_start,
            repeat_end=repeat_end,
            period=periods[signal_index],
            fit_window_samples=fit_window_samples,
            standstill_speed_threshold=standstill_thresholds[signal_index],
            min_motion_evidence=min_motion_evidence[signal_index],
            long_standstill_duration_s=long_standstill_duration_s,
            max_abs_acceleration=max_abs_accelerations[signal_index],
        )
        for signal_index in range(data.shape[0])
    )

    if len(signal_decisions) == 1:
        only = signal_decisions[0]
        return JointPlateauDecision(
            repeat_start=repeat_start,
            repeat_end=repeat_end,
            kind=only.kind,
            confidence=only.confidence,
            endpoint_span_ms=endpoint_span_ms,
            signal_decisions=signal_decisions,
            reason=(
                "A single-row 2-D input uses the original per-signal "
                f"classification: {only.reason}"
            ),
        )

    held = [
        decision
        for decision in signal_decisions
        if decision.kind == "held_dropout"
    ]
    ambiguous = [
        decision for decision in signal_decisions if decision.kind == "ambiguous"
    ]
    stationary_kinds = {"true_standstill", "quantization_plateau"}

    if held:
        kind: PlateauKind = "held_dropout"
        confidence = _best_confidence(held)
        moving_indices = [
            index
            for index, decision in enumerate(signal_decisions)
            if decision.kind == "held_dropout"
        ]
        reason = (
            "All signals repeat simultaneously and motion evidence in signal(s) "
            f"{moving_indices} supports a shared recorder freeze."
        )
    elif ambiguous:
        kind = "ambiguous"
        confidence = "low"
        ambiguous_indices = [
            index
            for index, decision in enumerate(signal_decisions)
            if decision.kind == "ambiguous"
        ]
        reason = (
            "All signals repeat simultaneously, but signal(s) "
            f"{ambiguous_indices} do not distinguish a recorder freeze from a "
            "genuine system standstill."
        )
    elif all(
        decision.kind in stationary_kinds for decision in signal_decisions
    ):
        # Simultaneous repetition is strong recorder-level evidence even if each
        # channel looks locally slow in its own units. Keep this joint case
        # ambiguous so the caller's policy and duration guard decide it.
        kind = "ambiguous"
        confidence = "medium"
        reason = (
            "Every signal repeats simultaneously and every channel looks locally "
            "stationary. This may be a shared recorder freeze or a genuine "
            "whole-system standstill."
        )
    else:
        kind = "ambiguous"
        confidence = "low"
        reason = (
            "Every signal repeats simultaneously, but the combined channel-level "
            "evidence is insufficient for a safe classification."
        )

    return JointPlateauDecision(
        repeat_start=repeat_start,
        repeat_end=repeat_end,
        kind=kind,
        confidence=confidence,
        endpoint_span_ms=endpoint_span_ms,
        signal_decisions=signal_decisions,
        reason=reason,
    )
