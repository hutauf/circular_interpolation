from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from circular_interpolation.inertia import (
    CircularInterpolationResult,
    add_single_held_dropout,
    circular_difference_deg,
    interpolate_circular_inertial,
    interpolate_circular_linear,
    wrap_deg,
)
from circular_interpolation.auto import (
    AutomaticInterpolationResult,
    KinematicInterpolationResult,
    interpolate_circular_auto,
    interpolate_circular_boundary_hermite,
    interpolate_circular_constant_acceleration,
)


RPM_TO_DEG_S = 6.0


@dataclass
class MotorProfile:
    name: str
    description: str
    time_s: np.ndarray
    angle_unwrapped_deg: np.ndarray
    velocity_deg_s: np.ndarray
    acceleration_deg_s2: np.ndarray
    measured_angle_deg: np.ndarray
    phase_centers_s: dict[str, float]


def _integrate_velocity(time_s: np.ndarray, velocity_deg_s: np.ndarray, initial_angle=15.0):
    angle = np.empty_like(velocity_deg_s)
    angle[0] = initial_angle
    dt = np.diff(time_s)
    angle[1:] = initial_angle + np.cumsum(
        0.5 * (velocity_deg_s[:-1] + velocity_deg_s[1:]) * dt
    )
    return angle


def _linear_transition(t, start, end, v0, v1):
    u = np.clip((t - start) / (end - start), 0.0, 1.0)
    return v0 + (v1 - v0) * u


def _smoothstep5(u):
    u = np.clip(u, 0.0, 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def _smooth_transition(t, start, end, v0, v1):
    u = (t - start) / (end - start)
    return v0 + (v1 - v0) * _smoothstep5(u)


def _make_measured(angle, noise_std, quantization_deg, rng):
    measured = angle + rng.normal(0.0, noise_std, angle.size)
    if quantization_deg > 0:
        measured = np.round(measured / quantization_deg) * quantization_deg
    return wrap_deg(measured)


def make_motor_profiles(
    *,
    sample_rate_hz: float = 1000.0,
    noise_std_deg: float = 0.03,
    quantization_deg: float = 0.01,
    seed: int = 0,
) -> list[MotorProfile]:
    rng = np.random.default_rng(seed)
    vmax_10k = 10_000.0 * RPM_TO_DEG_S
    vmax_6k = 6_000.0 * RPM_TO_DEG_S
    profiles: list[MotorProfile] = []

    def build(name, description, duration, velocity_fn, phases):
        n = int(round(duration * sample_rate_hz)) + 1
        t = np.arange(n) / sample_rate_hz
        v = velocity_fn(t)
        a = np.gradient(v, t)
        angle = _integrate_velocity(t, v)
        measured = _make_measured(angle, noise_std_deg, quantization_deg, rng)
        profiles.append(
            MotorProfile(name, description, t, angle, v, a, measured, phases)
        )

    build(
        "constant_10000_rpm",
        "Konstantfahrt mit 10.000 min^-1",
        1.4,
        lambda t: np.full_like(t, vmax_10k),
        {"high_speed_cruise": 0.70},
    )

    def aggressive_trapezoid(t):
        v = np.zeros_like(t)
        m = (t >= 0.20) & (t < 0.30)
        v[m] = _linear_transition(t[m], 0.20, 0.30, 0.0, vmax_10k)
        v[(t >= 0.30) & (t < 0.65)] = vmax_10k
        m = (t >= 0.65) & (t < 0.75)
        v[m] = _linear_transition(t[m], 0.65, 0.75, vmax_10k, 0.0)
        m = (t >= 0.95) & (t < 1.05)
        v[m] = _linear_transition(t[m], 0.95, 1.05, 0.0, -vmax_10k)
        v[(t >= 1.05) & (t < 1.40)] = -vmax_10k
        m = (t >= 1.40) & (t < 1.50)
        v[m] = _linear_transition(t[m], 1.40, 1.50, -vmax_10k, 0.0)
        return v

    build(
        "aggressive_trapezoid_10k",
        "0 -> +10.000 rpm in 100 ms, Stopp, danach -10.000 rpm",
        1.8,
        aggressive_trapezoid,
        {
            "positive_acceleration": 0.25,
            "positive_cruise": 0.48,
            "positive_braking": 0.70,
            "standstill": 0.85,
            "negative_acceleration": 1.00,
            "negative_cruise": 1.22,
            "negative_braking": 1.45,
        },
    )

    def loaded_servo(t):
        v = np.zeros_like(t)
        m = (t >= 0.20) & (t < 0.60)
        v[m] = _linear_transition(t[m], 0.20, 0.60, 0.0, vmax_6k)
        v[(t >= 0.60) & (t < 1.00)] = vmax_6k
        m = (t >= 1.00) & (t < 1.40)
        v[m] = _linear_transition(t[m], 1.00, 1.40, vmax_6k, 0.0)
        m = (t >= 1.65) & (t < 2.05)
        v[m] = _linear_transition(t[m], 1.65, 2.05, 0.0, -vmax_6k)
        v[(t >= 2.05) & (t < 2.45)] = -vmax_6k
        m = (t >= 2.45) & (t < 2.85)
        v[m] = _linear_transition(t[m], 2.45, 2.85, -vmax_6k, 0.0)
        return v

    build(
        "loaded_servo_6000_rpm",
        "Belasteter Industrie-Servo, 400-ms-Rampen bis 6.000 rpm",
        3.1,
        loaded_servo,
        {
            "positive_acceleration": 0.40,
            "positive_cruise": 0.80,
            "positive_braking": 1.20,
            "standstill": 1.52,
            "negative_acceleration": 1.85,
            "negative_cruise": 2.25,
            "negative_braking": 2.65,
        },
    )

    def s_curve(t):
        v = np.zeros_like(t)
        m = (t >= 0.20) & (t < 0.40)
        v[m] = _smooth_transition(t[m], 0.20, 0.40, 0.0, vmax_10k)
        v[(t >= 0.40) & (t < 0.75)] = vmax_10k
        m = (t >= 0.75) & (t < 0.95)
        v[m] = _smooth_transition(t[m], 0.75, 0.95, vmax_10k, 0.0)
        m = (t >= 1.10) & (t < 1.50)
        v[m] = _smooth_transition(t[m], 1.10, 1.50, 0.0, -vmax_10k)
        v[(t >= 1.50) & (t < 1.85)] = -vmax_10k
        m = (t >= 1.85) & (t < 2.05)
        v[m] = _smooth_transition(t[m], 1.85, 2.05, -vmax_10k, 0.0)
        return v

    build(
        "jerk_limited_s_curve_10k",
        "Jerk-begrenzte S-Kurven bis +/-10.000 rpm",
        2.3,
        s_curve,
        {
            "acceleration_early": 0.25,
            "acceleration_peak": 0.30,
            "acceleration_late": 0.35,
            "positive_cruise": 0.58,
            "braking_peak": 0.85,
            "standstill": 1.02,
            "full_reversal_midpoint": 1.30,
            "negative_cruise": 1.68,
        },
    )

    def dc_full_power(t):
        dt = 1.0 / sample_rate_hz
        target = np.zeros_like(t)
        target[(t >= 0.20) & (t < 0.80)] = vmax_10k
        target[(t >= 0.80) & (t < 1.00)] = 0.0
        target[(t >= 1.00) & (t < 1.60)] = -vmax_10k
        target[(t >= 1.60)] = 0.0
        tau = 0.080
        v = np.zeros_like(t)
        for i in range(1, t.size):
            v[i] = v[i - 1] + (target[i - 1] - v[i - 1]) * dt / tau
        return v

    build(
        "dc_bldc_full_power_cycle",
        "Vollspannungs-ähnliche BLDC-Dynamik mit 80-ms-Zeitkonstante",
        2.1,
        dc_full_power,
        {
            "startup_high_acceleration": 0.22,
            "startup_tail": 0.42,
            "positive_near_cruise": 0.70,
            "active_braking": 0.84,
            "reverse_command": 1.02,
            "reverse_zero_crossing": 1.08,
            "negative_near_cruise": 1.50,
            "final_braking": 1.65,
        },
    )

    return profiles


def _align_to_truth(estimate: np.ndarray, truth: np.ndarray, anchor: int) -> np.ndarray:
    offset = round((truth[anchor] - estimate[anchor]) / 360.0) * 360.0
    return estimate + offset


def _model_outputs(
    t: np.ndarray,
    observed: np.ndarray,
    invalid: np.ndarray,
) -> dict[str, np.ndarray]:
    _, linear_u = interpolate_circular_linear(t, observed, invalid)
    cv: CircularInterpolationResult = interpolate_circular_inertial(
        t,
        observed,
        invalid,
        velocity_random_walk_std=50_000.0,
        measurement_std_deg=0.03,
        gate_sigma=8.0,
    )
    ca = interpolate_circular_constant_acceleration(
        t,
        observed,
        invalid,
        acceleration_random_walk_std=2_000_000.0,
        measurement_std_deg=0.03,
        gate_sigma=8.0,
    )
    hermite = interpolate_circular_boundary_hermite(t, observed, invalid)
    return {
        "linear_shortest_arc": linear_u,
        "constant_velocity_rts": cv.unwrapped_deg,
        "constant_acceleration_rts": ca.unwrapped_deg,
        "boundary_velocity_hermite": hermite.unwrapped_deg,
    }


def evaluate_motor_profiles(
    profiles: list[MotorProfile],
    *,
    gap_lengths_ms: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50, 100, 200, 400),
    trials: int = 8,
    sample_rate_hz: float = 1000.0,
    seed: int = 500,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for profile in profiles:
        for phase, center_s in profile.phase_centers_s.items():
            for gap_ms in gap_lengths_ms:
                by_model: dict[str, list[tuple[float, float, float, bool]]] = {}

                for trial in range(trials):
                    # Small center jitter prevents accidental alignment with a ramp knot.
                    jitter_s = rng.uniform(-0.002, 0.002)
                    start_s = center_s + jitter_s - gap_ms / 2000.0
                    start_s = max(0.002, min(start_s, profile.time_s[-1] - gap_ms / 1000.0 - 0.002))
                    observed, missing, start, end = add_single_held_dropout(
                        profile.measured_angle_deg,
                        sample_rate_hz=sample_rate_hz,
                        start_s=start_s,
                        gap_ms=float(gap_ms),
                    )
                    # Models only need a local window around the gap. This keeps
                    # the full test matrix fast while retaining ample history and future data.
                    margin_ms = max(60.0, min(250.0, 0.75 * gap_ms + 50.0))
                    margin = int(round(margin_ms * sample_rate_hz / 1000.0))
                    lo = max(0, start - margin)
                    hi = min(profile.time_s.size, end + margin)
                    local_t = profile.time_s[lo:hi]
                    local_observed = observed[lo:hi]
                    local_missing = missing[lo:hi]
                    local_truth = profile.angle_unwrapped_deg[lo:hi]
                    outputs = _model_outputs(local_t, local_observed, local_missing)
                    anchor = max(0, start - lo - 1)
                    check = min(end - lo, local_t.size - 1)

                    for model, estimate in outputs.items():
                        aligned = _align_to_truth(estimate, local_truth, anchor)
                        error = aligned[local_missing] - local_truth[local_missing]
                        rmse = float(np.sqrt(np.mean(error**2)))
                        max_error = float(np.max(np.abs(error)))
                        endpoint_error = float(abs(aligned[check] - local_truth[check]))
                        slip = endpoint_error > 180.0
                        by_model.setdefault(model, []).append(
                            (rmse, max_error, endpoint_error, slip)
                        )

                speed_at_center = float(
                    np.interp(center_s, profile.time_s, profile.velocity_deg_s)
                )
                accel_at_center = float(
                    np.interp(center_s, profile.time_s, profile.acceleration_deg_s2)
                )
                nominal_start = max(0.0, center_s - gap_ms / 2000.0)
                nominal_end = min(profile.time_s[-1], center_s + gap_ms / 2000.0)
                gap_kinematics = (profile.time_s >= nominal_start) & (profile.time_s <= nominal_end)
                gap_velocity = profile.velocity_deg_s[gap_kinematics]
                gap_acceleration = profile.acceleration_deg_s2[gap_kinematics]
                max_abs_speed_rpm = float(np.max(np.abs(gap_velocity)) / RPM_TO_DEG_S)
                velocity_span_rpm = float((np.max(gap_velocity) - np.min(gap_velocity)) / RPM_TO_DEG_S)
                max_abs_acceleration = float(np.max(np.abs(gap_acceleration)))
                sign_change = bool(np.min(gap_velocity) < -50 * RPM_TO_DEG_S and np.max(gap_velocity) > 50 * RPM_TO_DEG_S)
                acceleration_sign_changes = int(np.count_nonzero(np.diff(np.sign(gap_acceleration[np.abs(gap_acceleration) > 1_000])))) if np.any(np.abs(gap_acceleration) > 1_000) else 0
                if max_abs_speed_rpm < 50.0:
                    motion_class = "standstill"
                elif sign_change:
                    motion_class = "reversal"
                elif max_abs_acceleration < 20_000.0 and velocity_span_rpm < 200.0:
                    motion_class = "constant_speed"
                elif acceleration_sign_changes >= 2 or (np.min(np.abs(gap_velocity)) < 50 * RPM_TO_DEG_S and max_abs_speed_rpm > 1_000):
                    motion_class = "mixed_transition"
                else:
                    moving = np.abs(gap_velocity) > 50 * RPM_TO_DEG_S
                    power_like = gap_velocity[moving] * gap_acceleration[moving]
                    motion_class = "acceleration" if power_like.size and np.median(power_like) >= 0 else "braking"
                for model, values in by_model.items():
                    arr = np.asarray(values, dtype=float)
                    rows.append(
                        {
                            "profile": profile.name,
                            "profile_description": profile.description,
                            "phase": phase,
                            "center_speed_rpm": speed_at_center / RPM_TO_DEG_S,
                            "center_acceleration_deg_s2": accel_at_center,
                            "gap_max_abs_speed_rpm": max_abs_speed_rpm,
                            "gap_velocity_span_rpm": velocity_span_rpm,
                            "gap_max_abs_acceleration_deg_s2": max_abs_acceleration,
                            "gap_sign_change": sign_change,
                            "gap_motion_class": motion_class,
                            "gap_ms": gap_ms,
                            "model": model,
                            "median_unwrapped_rmse_deg": float(np.median(arr[:, 0])),
                            "p95_unwrapped_rmse_deg": float(np.percentile(arr[:, 0], 95)),
                            "median_max_abs_error_deg": float(np.median(arr[:, 1])),
                            "median_endpoint_error_deg": float(np.median(arr[:, 2])),
                            "winding_slip_percent": 100.0 * float(np.mean(arr[:, 3])),
                        }
                    )
    return rows


def select_empirical_winners(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["profile"]), str(row["phase"]), int(row["gap_ms"]))
        grouped.setdefault(key, []).append(row)

    winners = []
    for (profile, phase, gap_ms), candidates in grouped.items():
        # Slip is catastrophic, then p95 and median unwrapped error decide.
        best = min(
            candidates,
            key=lambda r: (
                float(r["winding_slip_percent"]),
                float(r["p95_unwrapped_rmse_deg"]),
                float(r["median_unwrapped_rmse_deg"]),
            ),
        )
        winners.append(
            {
                "profile": profile,
                "phase": phase,
                "gap_ms": gap_ms,
                "speed_rpm": best["center_speed_rpm"],
                "acceleration_deg_s2": best["center_acceleration_deg_s2"],
                "winner": best["model"],
                "winner_median_rmse_deg": best["median_unwrapped_rmse_deg"],
                "winner_p95_rmse_deg": best["p95_unwrapped_rmse_deg"],
                "winner_slip_percent": best["winding_slip_percent"],
            }
        )
    return winners


def summarize_by_motion_and_gap(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    gap_bands = {
        "1-2ms": lambda g: g <= 2,
        "3-5ms": lambda g: 3 <= g <= 5,
        "10-20ms": lambda g: 10 <= g <= 20,
        "50-100ms": lambda g: 50 <= g <= 100,
        "200-400ms": lambda g: g >= 200,
    }

    buckets: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        motion = str(row["gap_motion_class"])
        gap = int(row["gap_ms"])
        band = next(name for name, predicate in gap_bands.items() if predicate(gap))
        key = (motion, band, str(row["model"]))
        buckets.setdefault(key, []).append(row)

    summaries = []
    for (motion, band, model), bucket in buckets.items():
        summaries.append(
            {
                "motion_class": motion,
                "gap_band": band,
                "model": model,
                "median_of_median_rmse_deg": float(
                    np.median([float(r["median_unwrapped_rmse_deg"]) for r in bucket])
                ),
                "p95_of_p95_rmse_deg": float(
                    np.percentile([float(r["p95_unwrapped_rmse_deg"]) for r in bucket], 95)
                ),
                "mean_winding_slip_percent": float(
                    np.mean([float(r["winding_slip_percent"]) for r in bucket])
                ),
                "case_count": len(bucket),
            }
        )
    return summaries


def write_csv(rows: list[dict[str, object]], path: Path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_profile_examples(profiles: list[MotorProfile], path: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(profiles), 1, figsize=(11, 2.5 * len(profiles)), sharex=False)
    if len(profiles) == 1:
        axes = [axes]
    for ax, profile in zip(axes, profiles):
        ax.plot(profile.time_s, profile.velocity_deg_s / RPM_TO_DEG_S)
        for phase, center in profile.phase_centers_s.items():
            ax.axvline(center, alpha=0.25)
        ax.set_title(profile.description)
        ax.set_ylabel("rpm")
        ax.grid(True)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_method_comparison(rows: list[dict[str, object]], path: Path):
    import matplotlib.pyplot as plt

    # Aggregate all non-standstill cases by gap length.
    filtered = [
        r
        for r in rows
        if not (
            abs(float(r["center_speed_rpm"])) < 50
            and abs(float(r["center_acceleration_deg_s2"])) < 5_000
        )
    ]
    models = sorted({str(r["model"]) for r in filtered})
    gaps = sorted({int(r["gap_ms"]) for r in filtered})

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        y = []
        for gap in gaps:
            vals = [
                float(r["median_unwrapped_rmse_deg"])
                for r in filtered
                if str(r["model"]) == model and int(r["gap_ms"]) == gap
            ]
            y.append(float(np.median(vals)))
        ax.plot(gaps, y, marker="o", label=model)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Held-value gap [ms]")
    ax.set_ylabel("Median unwrapped RMSE across motor cases [deg]")
    ax.set_title("Motor profiles: interpolation error vs. gap length")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_slip_comparison(rows: list[dict[str, object]], path: Path):
    import matplotlib.pyplot as plt

    models = sorted({str(r["model"]) for r in rows})
    gaps = sorted({int(r["gap_ms"]) for r in rows})
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        y = []
        for gap in gaps:
            vals = [
                float(r["winding_slip_percent"])
                for r in rows
                if str(r["model"]) == model and int(r["gap_ms"]) == gap
            ]
            y.append(float(np.mean(vals)))
        ax.plot(gaps, y, marker="o", label=model)
    ax.set_xscale("log")
    ax.set_xlabel("Held-value gap [ms]")
    ax.set_ylabel("Mean winding-slip rate [%]")
    ax.set_title("Wrong revolution assignment across all motor cases")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_compact_summary(summaries: list[dict[str, object]]):
    order = ["standstill", "constant_speed", "acceleration", "braking", "reversal", "mixed_transition"]
    bands = ["1-2ms", "3-5ms", "10-20ms", "50-100ms", "200-400ms"]
    print("\nBest model by motion class and gap band")
    for motion in order:
        print(f"\n{motion}:")
        for band in bands:
            candidates = [
                r
                for r in summaries
                if r["motion_class"] == motion and r["gap_band"] == band
            ]
            if not candidates:
                continue
            best = min(
                candidates,
                key=lambda r: (
                    float(r["mean_winding_slip_percent"]),
                    float(r["p95_of_p95_rmse_deg"]),
                    float(r["median_of_median_rmse_deg"]),
                ),
            )
            print(
                f"  {band:10s} -> {best['model']:31s} "
                f"median={best['median_of_median_rmse_deg']:.3f} deg, "
                f"slip={best['mean_winding_slip_percent']:.1f}%"
            )


def run(output_dir: Path, trials: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = make_motor_profiles()
    rows = evaluate_motor_profiles(profiles, trials=trials)
    winners = select_empirical_winners(rows)
    summaries = summarize_by_motion_and_gap(rows)

    write_csv(rows, output_dir / "motor_profile_model_results.csv")
    write_csv(winners, output_dir / "motor_profile_winners.csv")
    write_csv(summaries, output_dir / "motor_profile_switching_summary.csv")
    plot_profile_examples(profiles, output_dir / "motor_profiles.png")
    plot_method_comparison(rows, output_dir / "motor_model_error_by_gap.png")
    plot_slip_comparison(rows, output_dir / "motor_model_winding_slips.png")
    print_compact_summary(summaries)

    print("\nFiles written:")
    for name in [
        "motor_profile_model_results.csv",
        "motor_profile_winners.csv",
        "motor_profile_switching_summary.csv",
        "motor_profiles.png",
        "motor_model_error_by_gap.png",
        "motor_model_winding_slips.png",
    ]:
        print(f"  {output_dir / name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.output_dir, args.trials)


if __name__ == "__main__":
    main()
