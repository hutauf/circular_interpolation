from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np


BENCHMARK_VERSION = "1.0.0"
RPM_TO_DEG_S = 6.0


@dataclass(frozen=True)
class GapWindow:
    label: str
    center_s: float
    motion_hint: str


@dataclass
class BenchmarkCase:
    case_id: str
    family: str
    description: str
    sample_rate_hz: float
    seed: int
    time_s: np.ndarray
    true_angle_unwrapped_deg: np.ndarray
    true_velocity_deg_s: np.ndarray
    true_acceleration_deg_s2: np.ndarray
    measured_angle_deg: np.ndarray
    gap_windows: tuple[GapWindow, ...]
    metadata: dict[str, object]

    @property
    def max_abs_speed_rpm(self) -> float:
        return float(np.max(np.abs(self.true_velocity_deg_s)) / RPM_TO_DEG_S)

    @property
    def max_abs_acceleration_deg_s2(self) -> float:
        return float(np.max(np.abs(self.true_acceleration_deg_s2)))

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(BENCHMARK_VERSION.encode("utf-8"))
        h.update(self.case_id.encode("utf-8"))
        for arr in (
            self.time_s,
            self.true_angle_unwrapped_deg,
            self.true_velocity_deg_s,
            self.measured_angle_deg,
        ):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()


def wrap_deg(values: np.ndarray) -> np.ndarray:
    return np.mod(values, 360.0)


def integrate_velocity(time_s: np.ndarray, velocity_deg_s: np.ndarray, initial_angle_deg: float) -> np.ndarray:
    angle = np.empty_like(velocity_deg_s, dtype=float)
    angle[0] = initial_angle_deg
    dt = np.diff(time_s)
    angle[1:] = initial_angle_deg + np.cumsum(
        0.5 * (velocity_deg_s[:-1] + velocity_deg_s[1:]) * dt
    )
    return angle


def smoothstep5(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def linear_segment(t: np.ndarray, start: float, end: float, v0: float, v1: float) -> np.ndarray:
    u = np.clip((t - start) / (end - start), 0.0, 1.0)
    return v0 + (v1 - v0) * u


def smooth_segment(t: np.ndarray, start: float, end: float, v0: float, v1: float) -> np.ndarray:
    u = (t - start) / (end - start)
    return v0 + (v1 - v0) * smoothstep5(u)


def first_order_response(time_s: np.ndarray, target_deg_s: np.ndarray, tau_s: float) -> np.ndarray:
    velocity = np.zeros_like(target_deg_s, dtype=float)
    dt = float(time_s[1] - time_s[0])
    alpha = min(1.0, dt / tau_s)
    for index in range(1, len(velocity)):
        velocity[index] = velocity[index - 1] + alpha * (
            target_deg_s[index - 1] - velocity[index - 1]
        )
    return velocity


def make_measurement(
    true_angle_unwrapped_deg: np.ndarray,
    *,
    seed: int,
    noise_std_deg: float,
    quantization_deg: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    measured = true_angle_unwrapped_deg + rng.normal(
        0.0, noise_std_deg, true_angle_unwrapped_deg.size
    )
    if quantization_deg > 0:
        measured = np.round(measured / quantization_deg) * quantization_deg
    return wrap_deg(measured)


def _build_case(
    *,
    case_id: str,
    family: str,
    description: str,
    duration_s: float,
    sample_rate_hz: float,
    seed: int,
    velocity_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    gap_windows: tuple[GapWindow, ...],
    noise_std_deg: float = 0.03,
    quantization_deg: float = 0.01,
    metadata: dict[str, object] | None = None,
) -> BenchmarkCase:
    count = int(round(duration_s * sample_rate_hz)) + 1
    time_s = np.arange(count, dtype=float) / sample_rate_hz
    rng = np.random.default_rng(seed)
    velocity = np.asarray(velocity_fn(time_s, rng), dtype=float)
    if velocity.shape != time_s.shape:
        raise ValueError(f"Velocity function for {case_id} returned wrong shape")
    acceleration = np.gradient(velocity, time_s)
    angle = integrate_velocity(time_s, velocity, initial_angle_deg=17.0 + seed % 100)
    measured = make_measurement(
        angle,
        seed=seed + 100_000,
        noise_std_deg=noise_std_deg,
        quantization_deg=quantization_deg,
    )
    return BenchmarkCase(
        case_id=case_id,
        family=family,
        description=description,
        sample_rate_hz=sample_rate_hz,
        seed=seed,
        time_s=time_s,
        true_angle_unwrapped_deg=angle,
        true_velocity_deg_s=velocity,
        true_acceleration_deg_s2=acceleration,
        measured_angle_deg=measured,
        gap_windows=gap_windows,
        metadata={
            "noise_std_deg": noise_std_deg,
            "quantization_deg": quantization_deg,
            **(metadata or {}),
        },
    )


def make_benchmark_corpus(sample_rate_hz: float = 1000.0) -> list[BenchmarkCase]:
    """Return the fixed, deterministic motor-angle interpolation benchmark corpus."""
    cases: list[BenchmarkCase] = []

    def add(**kwargs):
        cases.append(_build_case(sample_rate_hz=sample_rate_hz, **kwargs))

    # 1) Standstill and tiny positioning motion.
    add(
        case_id="idle_clean",
        family="standstill",
        description="True standstill with normal encoder noise and quantization",
        duration_s=2.0,
        seed=1001,
        velocity_fn=lambda t, rng: np.zeros_like(t),
        gap_windows=(
            GapWindow("early_idle", 0.55, "standstill"),
            GapWindow("late_idle", 1.45, "standstill"),
        ),
    )
    add(
        case_id="idle_dither_5rpm",
        family="standstill",
        description="Near-standstill servo dither, +/-5 rpm at 2 Hz",
        duration_s=2.0,
        seed=1002,
        velocity_fn=lambda t, rng: 5.0 * RPM_TO_DEG_S * np.sin(2.0 * np.pi * 2.0 * t),
        gap_windows=(
            GapWindow("positive_dither", 0.125, "slow_motion"),
            GapWindow("zero_crossing", 0.25, "reversal"),
            GapWindow("negative_dither", 0.375, "slow_motion"),
            GapWindow("later_zero_crossing", 1.25, "reversal"),
        ),
    )

    # 2) Constant-speed cases, deliberately spanning everyday to extreme speeds.
    for idx, rpm in enumerate((100, 500, 1500, 3000, 6000, 10000), start=1):
        add(
            case_id=f"constant_{rpm:05d}_rpm",
            family="constant_speed",
            description=f"Constant rotation at {rpm:,} rpm".replace(",", "."),
            duration_s=1.8,
            seed=1100 + idx,
            velocity_fn=lambda t, rng, rpm=rpm: np.full_like(t, rpm * RPM_TO_DEG_S),
            gap_windows=(
                GapWindow("cruise_a", 0.45, "constant_speed"),
                GapWindow("cruise_b", 0.90, "constant_speed"),
                GapWindow("cruise_c", 1.35, "constant_speed"),
            ),
            metadata={"nominal_rpm": rpm},
        )

    # 3) Linear ramps at several realistic scales.
    ramp_specs = (
        ("gentle_ramp_0500", 500, 1.0, 0.8),
        ("standard_ramp_3000", 3000, 0.5, 0.5),
        ("fast_ramp_6000", 6000, 0.25, 0.35),
        ("aggressive_ramp_10000", 10000, 0.10, 0.30),
    )
    for idx, (case_id, rpm, ramp_s, cruise_s) in enumerate(ramp_specs):
        start = 0.20
        ramp_end = start + ramp_s
        cruise_end = ramp_end + cruise_s
        brake_end = cruise_end + ramp_s
        duration = brake_end + 0.25

        def velocity_fn(t, rng, rpm=rpm, start=start, ramp_end=ramp_end, cruise_end=cruise_end, brake_end=brake_end):
            vmax = rpm * RPM_TO_DEG_S
            v = np.zeros_like(t)
            m = (t >= start) & (t < ramp_end)
            v[m] = linear_segment(t[m], start, ramp_end, 0.0, vmax)
            v[(t >= ramp_end) & (t < cruise_end)] = vmax
            m = (t >= cruise_end) & (t < brake_end)
            v[m] = linear_segment(t[m], cruise_end, brake_end, vmax, 0.0)
            return v

        add(
            case_id=case_id,
            family="linear_ramp",
            description=f"Linear 0 -> {rpm:,} rpm -> 0 profile".replace(",", "."),
            duration_s=duration,
            seed=1200 + idx,
            velocity_fn=velocity_fn,
            gap_windows=(
                GapWindow("ramp_early", start + 0.25 * ramp_s, "acceleration"),
                GapWindow("ramp_mid", start + 0.50 * ramp_s, "acceleration"),
                GapWindow("ramp_late", start + 0.85 * ramp_s, "acceleration"),
                GapWindow("cruise", ramp_end + 0.50 * cruise_s, "constant_speed"),
                GapWindow("brake_mid", cruise_end + 0.50 * ramp_s, "braking"),
                GapWindow("ramp_to_cruise_edge", ramp_end, "transition"),
                GapWindow("cruise_to_brake_edge", cruise_end, "transition"),
            ),
            metadata={"peak_rpm": rpm, "ramp_duration_s": ramp_s},
        )

    # 4) Smooth S-curves with low, medium and high peak speed.
    for idx, (rpm, ramp_s) in enumerate(((1500, 0.55), (6000, 0.30), (10000, 0.18))):
        start = 0.20
        up_end = start + ramp_s
        cruise_end = up_end + 0.35
        down_end = cruise_end + ramp_s
        duration = down_end + 0.25

        def velocity_fn(t, rng, rpm=rpm, start=start, up_end=up_end, cruise_end=cruise_end, down_end=down_end):
            vmax = rpm * RPM_TO_DEG_S
            v = np.zeros_like(t)
            m = (t >= start) & (t < up_end)
            v[m] = smooth_segment(t[m], start, up_end, 0.0, vmax)
            v[(t >= up_end) & (t < cruise_end)] = vmax
            m = (t >= cruise_end) & (t < down_end)
            v[m] = smooth_segment(t[m], cruise_end, down_end, vmax, 0.0)
            return v

        add(
            case_id=f"s_curve_{rpm:05d}_rpm",
            family="s_curve",
            description=f"Jerk-limited S-curve to {rpm:,} rpm and back".replace(",", "."),
            duration_s=duration,
            seed=1301 + idx,
            velocity_fn=velocity_fn,
            gap_windows=(
                GapWindow("accel_early", start + 0.20 * ramp_s, "acceleration"),
                GapWindow("accel_peak", start + 0.50 * ramp_s, "acceleration"),
                GapWindow("accel_late", start + 0.80 * ramp_s, "acceleration"),
                GapWindow("cruise", up_end + 0.18, "constant_speed"),
                GapWindow("brake_peak", cruise_end + 0.50 * ramp_s, "braking"),
                GapWindow("curve_boundary", up_end, "transition"),
            ),
            metadata={"peak_rpm": rpm, "ramp_duration_s": ramp_s},
        )

    # 5) Direction changes at several speeds.
    for idx, (rpm, transition_s) in enumerate(((500, 0.60), (3000, 0.35), (10000, 0.18))):
        start = 0.30
        transition_end = start + transition_s
        duration = transition_end + 0.55

        def velocity_fn(t, rng, rpm=rpm, start=start, transition_end=transition_end):
            vmax = rpm * RPM_TO_DEG_S
            v = np.full_like(t, vmax)
            m = (t >= start) & (t < transition_end)
            v[m] = smooth_segment(t[m], start, transition_end, vmax, -vmax)
            v[t >= transition_end] = -vmax
            return v

        zero_time = start + transition_s / 2.0
        add(
            case_id=f"reversal_{rpm:05d}_rpm",
            family="reversal",
            description=f"Smooth reversal from +{rpm:,} to -{rpm:,} rpm".replace(",", "."),
            duration_s=duration,
            seed=1401 + idx,
            velocity_fn=velocity_fn,
            gap_windows=(
                GapWindow("positive_cruise", 0.15, "constant_speed"),
                GapWindow("reversal_entry", start + 0.15 * transition_s, "transition"),
                GapWindow("zero_crossing", zero_time, "reversal"),
                GapWindow("reversal_exit", start + 0.85 * transition_s, "transition"),
                GapWindow("negative_cruise", transition_end + 0.25, "constant_speed"),
            ),
            metadata={"peak_rpm": rpm, "transition_duration_s": transition_s},
        )

    # 6) Periodic motion / positioning-like cases.
    add(
        case_id="sinusoidal_speed_0300_rpm",
        family="periodic",
        description="Sinusoidal speed, +/-300 rpm at 1 Hz",
        duration_s=2.2,
        seed=1501,
        velocity_fn=lambda t, rng: 300.0 * RPM_TO_DEG_S * np.sin(2.0 * np.pi * t),
        gap_windows=(
            GapWindow("positive_peak", 0.25, "constant_speed_like"),
            GapWindow("zero_crossing_down", 0.50, "reversal"),
            GapWindow("negative_peak", 0.75, "constant_speed_like"),
            GapWindow("zero_crossing_up", 1.00, "reversal"),
        ),
    )
    add(
        case_id="sinusoidal_speed_3000_rpm",
        family="periodic",
        description="Sinusoidal speed, +/-3.000 rpm at 1.5 Hz",
        duration_s=1.8,
        seed=1502,
        velocity_fn=lambda t, rng: 3000.0 * RPM_TO_DEG_S * np.sin(2.0 * np.pi * 1.5 * t),
        gap_windows=(
            GapWindow("positive_peak", 1.0 / 6.0, "constant_speed_like"),
            GapWindow("zero_crossing", 1.0 / 3.0, "reversal"),
            GapWindow("negative_peak", 0.50, "constant_speed_like"),
            GapWindow("later_zero_crossing", 2.0 / 3.0, "reversal"),
        ),
    )
    add(
        case_id="micro_positioning",
        family="periodic",
        description="Low-speed point-to-point positioning with short dwells",
        duration_s=2.4,
        seed=1503,
        velocity_fn=lambda t, rng: 120.0 * RPM_TO_DEG_S * np.sin(2.0 * np.pi * 0.8 * t) * (
            0.55 + 0.45 * np.sin(2.0 * np.pi * 0.2 * t) ** 2
        ),
        gap_windows=(
            GapWindow("move_a", 0.35, "slow_motion"),
            GapWindow("turnaround_a", 0.625, "reversal"),
            GapWindow("move_b", 1.00, "slow_motion"),
            GapWindow("turnaround_b", 1.25, "reversal"),
        ),
    )

    # 7) Random walks. Each case has its own fixed seed and clipping range.
    random_walk_specs = (
        ("random_walk_low", 500.0, 120.0, 1200.0, 1601),
        ("random_walk_medium", 3000.0, 500.0, 6000.0, 1602),
        ("random_walk_bidirectional", 0.0, 900.0, 4500.0, 1603),
    )
    for case_id, start_rpm, rw_std_rpm_sqrt_s, clip_rpm, seed in random_walk_specs:
        def velocity_fn(t, rng, start_rpm=start_rpm, rw_std=rw_std_rpm_sqrt_s, clip_rpm=clip_rpm):
            dt = float(t[1] - t[0])
            increments = rng.normal(0.0, rw_std * np.sqrt(dt), t.size - 1)
            rpm = np.empty_like(t)
            rpm[0] = start_rpm
            rpm[1:] = start_rpm + np.cumsum(increments)
            rpm = np.clip(rpm, -clip_rpm, clip_rpm)
            return rpm * RPM_TO_DEG_S

        add(
            case_id=case_id,
            family="random_walk",
            description=f"Fixed-seed velocity random walk, start {start_rpm:.0f} rpm",
            duration_s=3.0,
            seed=seed,
            velocity_fn=velocity_fn,
            gap_windows=(
                GapWindow("walk_20pct", 0.60, "random_walk"),
                GapWindow("walk_40pct", 1.20, "random_walk"),
                GapWindow("walk_60pct", 1.80, "random_walk"),
                GapWindow("walk_80pct", 2.40, "random_walk"),
            ),
            metadata={
                "start_rpm": start_rpm,
                "velocity_random_walk_std_rpm_sqrt_s": rw_std_rpm_sqrt_s,
                "clip_rpm": clip_rpm,
            },
        )

    # 8) Load disturbances and coast-down.
    def load_droop(t, rng):
        base = 3000.0 * RPM_TO_DEG_S
        v = np.full_like(t, base)
        # Smooth 30% speed droop and recovery under load.
        m = (t >= 0.55) & (t < 0.70)
        v[m] = smooth_segment(t[m], 0.55, 0.70, base, 0.70 * base)
        v[(t >= 0.70) & (t < 1.05)] = 0.70 * base
        m = (t >= 1.05) & (t < 1.25)
        v[m] = smooth_segment(t[m], 1.05, 1.25, 0.70 * base, base)
        return v

    add(
        case_id="load_droop_3000_rpm",
        family="load_disturbance",
        description="3.000 rpm motor with a 30% load-induced speed droop and recovery",
        duration_s=1.8,
        seed=1701,
        velocity_fn=load_droop,
        gap_windows=(
            GapWindow("pre_load", 0.35, "constant_speed"),
            GapWindow("droop_entry", 0.62, "braking"),
            GapWindow("loaded_cruise", 0.85, "constant_speed"),
            GapWindow("recovery", 1.15, "acceleration"),
            GapWindow("post_load", 1.48, "constant_speed"),
        ),
    )

    def coast_down(t, rng):
        start = 0.35
        v0 = 6000.0 * RPM_TO_DEG_S
        velocity = np.full_like(t, v0)
        m = t >= start
        velocity[m] = v0 * np.exp(-(t[m] - start) / 0.32)
        return velocity

    add(
        case_id="coast_down_6000_rpm",
        family="load_disturbance",
        description="Free coast-down from 6.000 rpm with exponential drag",
        duration_s=1.9,
        seed=1702,
        velocity_fn=coast_down,
        gap_windows=(
            GapWindow("pre_coast", 0.20, "constant_speed"),
            GapWindow("coast_fast", 0.48, "braking"),
            GapWindow("coast_medium", 0.80, "braking"),
            GapWindow("coast_slow", 1.35, "braking"),
        ),
    )

    def step_command(t, rng):
        target = np.zeros_like(t)
        target[(t >= 0.20) & (t < 0.65)] = 1500.0 * RPM_TO_DEG_S
        target[(t >= 0.65) & (t < 1.05)] = 3500.0 * RPM_TO_DEG_S
        target[(t >= 1.05) & (t < 1.45)] = -1200.0 * RPM_TO_DEG_S
        target[t >= 1.45] = 0.0
        return first_order_response(t, target, tau_s=0.06)

    add(
        case_id="first_order_step_commands",
        family="command_response",
        description="First-order motor response to several speed commands",
        duration_s=1.9,
        seed=1801,
        velocity_fn=step_command,
        gap_windows=(
            GapWindow("first_start", 0.22, "acceleration"),
            GapWindow("first_settle", 0.52, "constant_speed_like"),
            GapWindow("up_step", 0.68, "acceleration"),
            GapWindow("reverse_step", 1.08, "reversal"),
            GapWindow("final_stop", 1.48, "braking"),
        ),
    )

    def mixed_cycle(t, rng):
        target = np.zeros_like(t)
        target[(t >= 0.15) & (t < 0.60)] = 2500.0 * RPM_TO_DEG_S
        target[(t >= 0.60) & (t < 0.85)] = 1800.0 * RPM_TO_DEG_S
        target[(t >= 0.85) & (t < 1.15)] = 5000.0 * RPM_TO_DEG_S
        target[(t >= 1.15) & (t < 1.45)] = 0.0
        target[(t >= 1.45) & (t < 1.90)] = -3000.0 * RPM_TO_DEG_S
        target[t >= 1.90] = 0.0
        v = first_order_response(t, target, tau_s=0.085)
        # Add a small torque ripple that is typical enough to challenge local fits.
        v += 35.0 * RPM_TO_DEG_S * np.sin(2.0 * np.pi * 23.0 * t)
        return v

    add(
        case_id="mixed_industrial_duty_cycle",
        family="mixed_cycle",
        description="Mixed industrial duty cycle with load change, overspeed command and reversal",
        duration_s=2.35,
        seed=1802,
        velocity_fn=mixed_cycle,
        gap_windows=(
            GapWindow("startup", 0.18, "acceleration"),
            GapWindow("normal_cruise", 0.48, "constant_speed_like"),
            GapWindow("load_change", 0.63, "transition"),
            GapWindow("speed_up", 0.88, "acceleration"),
            GapWindow("stop", 1.18, "braking"),
            GapWindow("reverse_start", 1.48, "reversal"),
            GapWindow("reverse_cruise", 1.78, "constant_speed_like"),
            GapWindow("final_stop", 1.92, "braking"),
        ),
    )

    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate benchmark case IDs")
    return cases


def save_corpus(cases: list[BenchmarkCase], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    manifest: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []

    for case in cases:
        prefix = case.case_id
        arrays[f"{prefix}__time_s"] = case.time_s
        arrays[f"{prefix}__true_angle_unwrapped_deg"] = case.true_angle_unwrapped_deg
        arrays[f"{prefix}__true_velocity_deg_s"] = case.true_velocity_deg_s
        arrays[f"{prefix}__true_acceleration_deg_s2"] = case.true_acceleration_deg_s2
        arrays[f"{prefix}__measured_angle_deg"] = case.measured_angle_deg

        manifest.append(
            {
                "benchmark_version": BENCHMARK_VERSION,
                "case_id": case.case_id,
                "family": case.family,
                "description": case.description,
                "sample_rate_hz": case.sample_rate_hz,
                "duration_s": float(case.time_s[-1]),
                "sample_count": int(case.time_s.size),
                "seed": case.seed,
                "max_abs_speed_rpm": case.max_abs_speed_rpm,
                "max_abs_acceleration_deg_s2": case.max_abs_acceleration_deg_s2,
                "window_count": len(case.gap_windows),
                "sha256": case.digest(),
                "metadata_json": json.dumps(case.metadata, sort_keys=True),
            }
        )
        for window in case.gap_windows:
            windows.append(
                {
                    "benchmark_version": BENCHMARK_VERSION,
                    "case_id": case.case_id,
                    "family": case.family,
                    "window_label": window.label,
                    "center_s": window.center_s,
                    "motion_hint": window.motion_hint,
                }
            )

    npz_path = output_dir / f"motor_angle_benchmark_v{BENCHMARK_VERSION}.npz"
    np.savez_compressed(npz_path, **arrays)

    import csv

    manifest_path = output_dir / "benchmark_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    windows_path = output_dir / "benchmark_gap_windows.csv"
    with windows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0].keys()))
        writer.writeheader()
        writer.writerows(windows)

    return npz_path, manifest_path, windows_path


def validate_corpus(cases: list[BenchmarkCase]) -> None:
    for case in cases:
        n = case.time_s.size
        arrays = (
            case.true_angle_unwrapped_deg,
            case.true_velocity_deg_s,
            case.true_acceleration_deg_s2,
            case.measured_angle_deg,
        )
        if any(array.shape != (n,) for array in arrays):
            raise AssertionError(f"Shape mismatch in {case.case_id}")
        if not np.all(np.isfinite(case.time_s)) or not np.all(np.diff(case.time_s) > 0):
            raise AssertionError(f"Bad time axis in {case.case_id}")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise AssertionError(f"Non-finite data in {case.case_id}")
        if np.any((case.measured_angle_deg < 0) | (case.measured_angle_deg >= 360)):
            raise AssertionError(f"Wrapped measurement outside [0,360) in {case.case_id}")
        for window in case.gap_windows:
            if not (0.0 < window.center_s < case.time_s[-1]):
                raise AssertionError(f"Bad window {window.label} in {case.case_id}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    corpus = make_benchmark_corpus()
    validate_corpus(corpus)
    paths = save_corpus(corpus, out)
    print(f"Benchmark version: {BENCHMARK_VERSION}")
    print(f"Cases: {len(corpus)}")
    print(f"Test windows: {sum(len(case.gap_windows) for case in corpus)}")
    for path in paths:
        print(path)
