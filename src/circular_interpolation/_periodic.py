from __future__ import annotations

import numpy as np

Period = float | None


def validate_period(period: Period) -> Period:
    """Return a normalized period or ``None`` for an ordinary linear signal."""
    if period is None:
        return None

    value = float(period)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("period must be a finite positive number or None.")
    return value


def wrap_values(values: np.ndarray | float, period: Period) -> np.ndarray:
    """Wrap values to ``[0, period)`` or return an unchanged copy for ``None``."""
    array = np.asarray(values, dtype=float)
    if period is None:
        return array.copy()
    return np.mod(array, period)


def unwrap_values(values: np.ndarray, period: Period) -> np.ndarray:
    """Unwrap a periodic sequence or return an unchanged copy for ``None``."""
    array = np.asarray(values, dtype=float)
    if period is None:
        return array.copy()
    return np.unwrap(array, period=period)


def value_difference(
    a: np.ndarray | float,
    b: np.ndarray | float,
    period: Period,
) -> np.ndarray:
    """Return ``a - b`` using shortest-period distance when a period exists."""
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if period is None:
        return difference
    return (difference + period / 2.0) % period - period / 2.0


def branch_offset(reference: float, candidate: float, period: Period) -> float:
    """Align a periodic candidate to the branch nearest ``reference``."""
    if period is None:
        return 0.0
    return float(round((reference - candidate) / period) * period)


def interpolate_linear(
    time_s: np.ndarray,
    values: np.ndarray,
    invalid_mask: np.ndarray,
    *,
    period: Period,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation on either a periodic or ordinary scalar signal."""
    valid = ~invalid_mask
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid samples are required.")

    continuous_valid = unwrap_values(values[valid], period)
    continuous = np.interp(time_s, time_s[valid], continuous_valid)
    return wrap_values(continuous, period), continuous
