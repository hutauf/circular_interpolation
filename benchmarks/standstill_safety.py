from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from circular_interpolation import interpolate_raw_circular_stream


@dataclass(frozen=True)
class StandstillSafetyCase:
    sample_rate_hz: float
    peak_speed_deg_s: float
    ramp_duration_s: float
    standstill_duration_s: float

    @property
    def case_id(self) -> str:
        return (
            f"fs={self.sample_rate_hz:g},speed={self.peak_speed_deg_s:g},"
            f"ramp={self.ramp_duration_s:g},stop={self.standstill_duration_s:g}"
        )


@dataclass(frozen=True)
class StandstillSafetyResult:
    case_id: str
    standstill_duration_s: float
    uncapped_repaired_samples: int
    capped_repaired_samples: int
    uncapped_max_error_deg: float
    capped_max_error_deg: float
    uncapped_tail_repairs: int
    capped_tail_repairs: int


def _integrate_velocity(
    time_s: np.ndarray,
    velocity_deg_s: np.ndarray,
    *,
    initial_angle_deg: float = 11.0,
) -> np.ndarray:
    angle = np.empty_like(velocity_deg_s)
    angle[0] = initial_angle_deg
    angle[1:] = initial_angle_deg + np.cumsum(
        0.5
        * (velocity_deg_s[:-1] + velocity_deg_s[1:])
        * np.diff(time_s)
    )
    return angle


def make_stop_go_profile(
    case: StandstillSafetyCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create ramp-up, ramp-down, standstill, and a second motion cycle."""
    dt = 1.0 / case.sample_rate_hz
    initial_standstill_s = 0.050
    constant_speed_s = 0.050
    trailing_standstill_s = 0.100

    first_ramp_up = initial_standstill_s
    first_constant = first_ramp_up + case.ramp_duration_s
    first_ramp_down = first_constant + constant_speed_s
    middle_standstill = first_ramp_down + case.ramp_duration_s
    second_ramp_up = middle_standstill + case.standstill_duration_s
    second_constant = second_ramp_up + case.ramp_duration_s
    second_ramp_down = second_constant + constant_speed_s
    trailing_standstill = second_ramp_down + case.ramp_duration_s
    end_time = trailing_standstill + trailing_standstill_s

    time_s = np.arange(0.0, end_time + 0.5 * dt, dt)
    velocity = np.zeros_like(time_s)

    mask = (time_s >= first_ramp_up) & (time_s < first_constant)
    velocity[mask] = (
        case.peak_speed_deg_s
        * (time_s[mask] - first_ramp_up)
        / case.ramp_duration_s
    )
    velocity[(time_s >= first_constant) & (time_s < first_ramp_down)] = (
        case.peak_speed_deg_s
    )
    mask = (time_s >= first_ramp_down) & (time_s < middle_standstill)
    velocity[mask] = case.peak_speed_deg_s * (
        1.0
        - (time_s[mask] - first_ramp_down) / case.ramp_duration_s
    )

    mask = (time_s >= second_ramp_up) & (time_s < second_constant)
    velocity[mask] = (
        case.peak_speed_deg_s
        * (time_s[mask] - second_ramp_up)
        / case.ramp_duration_s
    )
    velocity[(time_s >= second_constant) & (time_s < second_ramp_down)] = (
        case.peak_speed_deg_s
    )
    mask = (time_s >= second_ramp_down) & (time_s < trailing_standstill)
    velocity[mask] = case.peak_speed_deg_s * (
        1.0
        - (time_s[mask] - second_ramp_down) / case.ramp_duration_s
    )

    truth_unwrapped = _integrate_velocity(time_s, velocity)
    observed = np.mod(truth_unwrapped, 360.0)
    middle_mask = (time_s >= middle_standstill) & (time_s < second_ramp_up)
    trailing_mask = time_s >= trailing_standstill
    return time_s, truth_unwrapped, observed, middle_mask, trailing_mask


def _aligned_error(
    estimated_unwrapped: np.ndarray,
    truth_unwrapped: np.ndarray,
) -> np.ndarray:
    shift = round(
        (truth_unwrapped[0] - estimated_unwrapped[0]) / 360.0
    ) * 360.0
    return estimated_unwrapped + shift - truth_unwrapped


def evaluate_standstill_case(
    case: StandstillSafetyCase,
    *,
    max_plateau_repair_duration_s: float,
) -> StandstillSafetyResult:
    time_s, truth, observed, middle_mask, trailing_mask = make_stop_go_profile(
        case
    )

    uncapped = interpolate_raw_circular_stream(
        time_s,
        observed,
        ambiguous_policy="repair",
    )
    capped = interpolate_raw_circular_stream(
        time_s,
        observed,
        ambiguous_policy="repair",
        max_plateau_repair_duration_s=max_plateau_repair_duration_s,
    )

    uncapped_error = _aligned_error(uncapped.unwrapped_deg, truth)
    capped_error = _aligned_error(capped.unwrapped_deg, truth)

    return StandstillSafetyResult(
        case_id=case.case_id,
        standstill_duration_s=case.standstill_duration_s,
        uncapped_repaired_samples=int(
            np.count_nonzero(uncapped.repaired_mask & middle_mask)
        ),
        capped_repaired_samples=int(
            np.count_nonzero(capped.repaired_mask & middle_mask)
        ),
        uncapped_max_error_deg=float(
            np.max(np.abs(uncapped_error[middle_mask]), initial=0.0)
        ),
        capped_max_error_deg=float(
            np.max(np.abs(capped_error[middle_mask]), initial=0.0)
        ),
        uncapped_tail_repairs=int(
            np.count_nonzero(uncapped.repaired_mask & trailing_mask)
        ),
        capped_tail_repairs=int(
            np.count_nonzero(capped.repaired_mask & trailing_mask)
        ),
    )


def make_benchmark_cases() -> list[StandstillSafetyCase]:
    return [
        StandstillSafetyCase(1000.0, speed, ramp, standstill)
        for speed in (300.0, 1000.0, 6000.0)
        for ramp in (0.020, 0.050, 0.100, 0.200)
        for standstill in (0.010, 0.020, 0.050, 0.100, 0.200, 0.500)
    ]


def run_benchmark(
    *,
    max_plateau_repair_duration_s: float = 0.020,
) -> list[StandstillSafetyResult]:
    return [
        evaluate_standstill_case(
            case,
            max_plateau_repair_duration_s=max_plateau_repair_duration_s,
        )
        for case in make_benchmark_cases()
    ]


def _summarize(results: list[StandstillSafetyResult]) -> None:
    uncapped_errors = np.array(
        [result.uncapped_max_error_deg for result in results], dtype=float
    )
    capped_errors = np.array(
        [result.capped_max_error_deg for result in results], dtype=float
    )

    print(f"cases: {len(results)}")
    print(
        "internal standstills modified: "
        f"uncapped={sum(r.uncapped_repaired_samples > 0 for r in results)}, "
        f"capped={sum(r.capped_repaired_samples > 0 for r in results)}"
    )
    print(
        "maximum standstill error [deg]: "
        f"uncapped median={np.median(uncapped_errors):.6g}, "
        f"p95={np.percentile(uncapped_errors, 95):.6g}, "
        f"worst={np.max(uncapped_errors):.6g}; "
        f"capped median={np.median(capped_errors):.6g}, "
        f"p95={np.percentile(capped_errors, 95):.6g}, "
        f"worst={np.max(capped_errors):.6g}"
    )
    print(
        "boundary-tail repairs: "
        f"uncapped={sum(r.uncapped_tail_repairs for r in results)}, "
        f"capped={sum(r.capped_tail_repairs for r in results)}"
    )

    worst = sorted(
        results,
        key=lambda result: result.uncapped_max_error_deg,
        reverse=True,
    )[:10]
    print("\nworst uncapped cases:")
    for result in worst:
        print(
            f"  {result.case_id}: "
            f"uncapped={result.uncapped_max_error_deg:.6g} deg, "
            f"capped={result.capped_max_error_deg:.6g} deg"
        )


if __name__ == "__main__":
    _summarize(run_benchmark())
