from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._periodic import Period, validate_period, value_difference
from .raw_stream_wrapper import PlateauDecision, PlateauKind


Confidence = Literal["high", "medium", "low"]
PeriodInput = Period | Sequence[Period]
SignalScalarInput = float | Sequence[float]


@dataclass(frozen=True)
class JointPlateauDecision:
    """Classification of a repeated-value run shared by every input signal."""

    repeat_start: int
    repeat_end: int
    kind: PlateauKind
    confidence: Confidence
    endpoint_span_ms: float
    signal_decisions: tuple[PlateauDecision, ...]
    reason: str


@dataclass
class MultiSignalInterpolationResult:
    """Result for an input array shaped ``(signals, samples)``.

    Signal arrays are evaluated on ``time_s``. Source masks retain the original
    source grid. ``joint_candidate_mask`` marks simultaneous repeated-value runs,
    while ``joint_repair_mask`` is the subset selected after policy and duration
    safeguards. ``repaired_mask`` additionally contains row-specific explicit
    non-finite samples.
    """

    angle_deg: np.ndarray
    unwrapped_deg: np.ndarray
    time_s: np.ndarray
    repaired_mask: np.ndarray
    joint_candidate_mask: np.ndarray
    joint_repair_mask: np.ndarray
    preserved_standstill_mask: np.ndarray
    ambiguous_mask: np.ndarray
    decisions: list[JointPlateauDecision]
    chosen_model_by_gap: tuple[list[tuple[int, int, str]], ...]
    interpolation_confidence_by_gap: tuple[list[tuple[int, int, str]], ...]
    period_by_signal: tuple[Period, ...]

    @property
    def shared_freeze_mask(self) -> np.ndarray:
        """Backward-friendly alias for the final inferred joint repair mask."""
        return self.joint_repair_mask

    @property
    def periods(self) -> tuple[Period, ...]:
        """Concise alias for ``period_by_signal``."""
        return self.period_by_signal


def normalize_periods(period: PeriodInput, signal_count: int) -> tuple[Period, ...]:
    """Broadcast one period or validate one period per signal."""
    if signal_count < 1:
        raise ValueError("At least one signal is required.")

    if period is None or np.isscalar(period):
        normalized = validate_period(period)
        return (normalized,) * signal_count

    if isinstance(period, np.ndarray) and period.ndim == 0:
        normalized = validate_period(period.item())
        return (normalized,) * signal_count

    if isinstance(period, (str, bytes)):
        raise ValueError(
            "period must be a finite positive number, None, or a sequence "
            "with one entry per signal."
        )

    try:
        values = list(period)
    except TypeError as exc:
        raise ValueError(
            "period must be a finite positive number, None, or a sequence "
            "with one entry per signal."
        ) from exc

    if len(values) != signal_count:
        raise ValueError(
            "A period sequence must have length data.shape[0] "
            f"({signal_count}); got {len(values)}."
        )
    return tuple(validate_period(value) for value in values)


def normalize_signal_scalars(
    value: SignalScalarInput,
    signal_count: int,
    *,
    name: str,
    allow_zero: bool,
) -> tuple[float, ...]:
    """Broadcast or validate one finite scalar per signal."""
    if np.isscalar(value):
        values = [float(value)] * signal_count
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        values = [float(value.item())] * signal_count
    else:
        try:
            values = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a finite scalar or a sequence with one entry "
                "per signal."
            ) from exc
        if len(values) != signal_count:
            raise ValueError(
                f"A {name} sequence must have length data.shape[0] "
                f"({signal_count}); got {len(values)}."
            )

    lower_ok = (
        (lambda item: item >= 0.0)
        if allow_zero
        else (lambda item: item > 0.0)
    )
    if any(not np.isfinite(item) or not lower_ok(item) for item in values):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be finite and {qualifier}.")
    return tuple(values)


def validate_multi_input(
    time_s: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_s, dtype=float)
    values = np.asarray(data, dtype=float)
    if t.ndim != 1:
        raise ValueError("time_s must be one-dimensional.")
    if values.ndim != 2:
        raise ValueError("Multi-signal data must have shape (signals, samples).")
    if values.shape[0] < 1:
        raise ValueError("At least one signal is required.")
    if values.shape[1] != t.size:
        raise ValueError("data.shape[1] must equal time_s.size.")
    if t.size < 3:
        raise ValueError("At least three samples are required.")
    if np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        raise ValueError("time_s must be finite and strictly increasing.")
    return t, values


def validate_target_time_s(
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


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in edges.reshape(-1, 2)]


def joint_repeat_mask(
    data: np.ndarray,
    *,
    periods: tuple[Period, ...],
    repeat_tolerances: tuple[float, ...],
) -> np.ndarray:
    """Mark timestamps where every row repeats its own preceding sample."""
    signal_count, sample_count = data.shape
    repeated = np.zeros((signal_count, sample_count), dtype=bool)
    for signal_index in range(signal_count):
        row = data[signal_index]
        finite_pairs = np.isfinite(row[1:]) & np.isfinite(row[:-1])
        repeated[signal_index, 1:] = finite_pairs & (
            np.abs(
                value_difference(
                    row[1:],
                    row[:-1],
                    periods[signal_index],
                )
            )
            <= repeat_tolerances[signal_index]
        )
    return np.all(repeated, axis=0)
