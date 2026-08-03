
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CircularInterpolationResult:
    """Result of the inertia-aware interpolation."""

    angle_deg: np.ndarray
    unwrapped_deg: np.ndarray
    angular_velocity_deg_s: np.ndarray
    invalid_mask: np.ndarray
    rejected_outliers: np.ndarray


def wrap_deg(x: np.ndarray | float, period: float = 360.0) -> np.ndarray:
    """Wrap values to [0, period)."""
    return np.mod(x, period)


def circular_difference_deg(
    a: np.ndarray | float,
    b: np.ndarray | float,
    period: float = 360.0,
) -> np.ndarray:
    """Signed shortest difference a-b in [-period/2, period/2)."""
    return (np.asarray(a) - np.asarray(b) + period / 2.0) % period - period / 2.0


def detect_held_samples(
    angle_deg: np.ndarray,
    *,
    atol_deg: float = 0.0,
    min_run: int = 2,
    period: float = 360.0,
) -> np.ndarray:
    """
    Detect samples that repeat their predecessor for at least ``min_run`` samples.

    Only the repeated samples are marked; the last genuinely measured sample
    immediately before the held run remains valid.

    Important:
        From angle values alone, a genuine standstill and a sensor that repeats
        its last sample are not always distinguishable. With a quantized sensor,
        keep ``atol_deg`` small and increase ``min_run``.
    """
    y = np.asarray(angle_deg, dtype=float)
    if y.ndim != 1:
        raise ValueError("angle_deg must be one-dimensional.")
    if min_run < 1:
        raise ValueError("min_run must be at least 1.")
    if atol_deg < 0:
        raise ValueError("atol_deg must be non-negative.")

    invalid = ~np.isfinite(y)
    if y.size < 2:
        return invalid

    repeated = np.zeros(y.size, dtype=bool)
    finite_pair = np.isfinite(y[1:]) & np.isfinite(y[:-1])
    repeated[1:] = finite_pair & (
        np.abs(circular_difference_deg(y[1:], y[:-1], period)) <= atol_deg
    )

    i = 1
    while i < y.size:
        if not repeated[i]:
            i += 1
            continue

        end = i
        while end < y.size and repeated[end]:
            end += 1

        if end - i >= min_run:
            invalid[i:end] = True

        i = end

    return invalid


def interpolate_circular_linear(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    invalid_mask: np.ndarray | None = None,
    *,
    period: float = 360.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baseline interpolation using NumPy only.

    1. Remove invalid samples.
    2. Unwrap the angular values with ``np.unwrap(..., period=period)``.
    3. Interpolate linearly with ``np.interp``.
    4. Wrap the result back to [0, period).

    This is excellent for short gaps, but the winding number becomes ambiguous
    when the actual motion across a gap is around or above half a revolution.
    """
    t, y, invalid = _validate_inputs(time_s, angle_deg, invalid_mask)
    valid = ~invalid
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid samples are required.")

    unwrapped_valid = np.unwrap(y[valid], period=period)
    interpolated_unwrapped = np.interp(t, t[valid], unwrapped_valid)
    return wrap_deg(interpolated_unwrapped, period), interpolated_unwrapped


def interpolate_circular_inertial(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    invalid_mask: np.ndarray | None = None,
    *,
    period: float = 360.0,
    velocity_random_walk_std: float = 200.0,
    measurement_std_deg: float = 0.05,
    gate_sigma: float = 6.0,
    preserve_valid_samples: bool = True,
) -> CircularInterpolationResult:
    """
    Interpolate circular angle data with an inertia model.

    The state is [unwrapped angle, angular velocity]. Angular velocity follows
    a random walk, which corresponds to a constant-velocity Kalman model with
    stochastic acceleration. A Rauch-Tung-Striebel backward pass uses samples
    on both sides of each gap.

    Wrapped measurements are mapped onto the revolution nearest to the predicted
    unwrapped angle. This is the part that uses inertia to avoid implausible
    180-degree jumps and to preserve the likely number of revolutions.

    Parameters
    ----------
    velocity_random_walk_std:
        Standard deviation of the angular-velocity change per sqrt(second),
        in deg/s/sqrt(s). Lower values mean stronger assumed inertia.
    measurement_std_deg:
        Expected one-sigma sensor noise in degrees.
    gate_sigma:
        Measurements whose innovation exceeds this normalized threshold are
        rejected as outliers. A single approximately 180-degree glitch is
        therefore normally ignored when the model uncertainty is small.
    """
    t, y, invalid = _validate_inputs(time_s, angle_deg, invalid_mask)
    n = t.size
    valid = ~invalid

    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid samples are required.")
    if velocity_random_walk_std <= 0:
        raise ValueError("velocity_random_walk_std must be positive.")
    if measurement_std_deg <= 0:
        raise ValueError("measurement_std_deg must be positive.")
    if gate_sigma <= 0:
        raise ValueError("gate_sigma must be positive.")

    first = int(np.flatnonzero(valid)[0])
    x_filtered = np.full((n, 2), np.nan)
    p_filtered = np.full((n, 2, 2), np.nan)
    x_predicted = np.full((n, 2), np.nan)
    p_predicted = np.full((n, 2, 2), np.nan)
    transitions = np.full((n, 2, 2), np.nan)
    rejected = np.zeros(n, dtype=bool)

    omega0 = _estimate_initial_velocity(t, y, valid, first, period)
    state = np.array([y[first], omega0], dtype=float)
    covariance = np.diag(
        [
            max(measurement_std_deg**2, 1e-10) * 4.0,
            max(velocity_random_walk_std**2, 1.0) * 4.0,
        ]
    )

    x_filtered[first] = state
    p_filtered[first] = covariance
    x_predicted[first] = state
    p_predicted[first] = covariance
    transitions[first] = np.eye(2)

    identity = np.eye(2)
    measurement_variance = measurement_std_deg**2
    process_intensity = velocity_random_walk_std**2

    for k in range(first + 1, n):
        dt = t[k] - t[k - 1]
        transition = np.array([[1.0, dt], [0.0, 1.0]])

        # Continuous white-noise acceleration / Brownian angular velocity.
        process_noise = process_intensity * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]
        )

        predicted_state = transition @ state
        predicted_covariance = (
            transition @ covariance @ transition.T + process_noise
        )

        transitions[k] = transition
        x_predicted[k] = predicted_state
        p_predicted[k] = predicted_covariance

        if valid[k]:
            innovation = float(
                circular_difference_deg(
                    y[k], wrap_deg(predicted_state[0], period), period
                )
            )
            innovation_variance = (
                predicted_covariance[0, 0] + measurement_variance
            )

            if innovation**2 <= gate_sigma**2 * innovation_variance:
                kalman_gain = predicted_covariance[:, 0] / innovation_variance
                state = predicted_state + kalman_gain * innovation

                # Joseph covariance update for numerical robustness.
                gain_times_h = np.array(
                    [[kalman_gain[0], 0.0], [kalman_gain[1], 0.0]]
                )
                covariance = (
                    (identity - gain_times_h)
                    @ predicted_covariance
                    @ (identity - gain_times_h).T
                    + np.outer(kalman_gain, kalman_gain) * measurement_variance
                )
            else:
                state = predicted_state
                covariance = predicted_covariance
                rejected[k] = True
        else:
            state = predicted_state
            covariance = predicted_covariance

        x_filtered[k] = state
        p_filtered[k] = covariance

    # Rauch-Tung-Striebel smoother: use future measurements as well.
    x_smoothed = x_filtered.copy()
    p_smoothed = p_filtered.copy()

    for k in range(n - 2, first - 1, -1):
        transition = transitions[k + 1]
        smoother_gain = np.linalg.solve(
            p_predicted[k + 1].T,
            (p_filtered[k] @ transition.T).T,
        ).T

        x_smoothed[k] = x_filtered[k] + smoother_gain @ (
            x_smoothed[k + 1] - x_predicted[k + 1]
        )
        p_smoothed[k] = p_filtered[k] + smoother_gain @ (
            p_smoothed[k + 1] - p_predicted[k + 1]
        ) @ smoother_gain.T

    # Backward extrapolation if the series starts with missing samples.
    for k in range(first - 1, -1, -1):
        dt = t[first] - t[k]
        x_smoothed[k, 0] = x_smoothed[first, 0] - x_smoothed[first, 1] * dt
        x_smoothed[k, 1] = x_smoothed[first, 1]

    model_angle = wrap_deg(x_smoothed[:, 0], period)
    output_angle = model_angle.copy()
    final_invalid = invalid | rejected

    if preserve_valid_samples:
        keep_original = valid & ~rejected
        output_angle[keep_original] = wrap_deg(y[keep_original], period)

    return CircularInterpolationResult(
        angle_deg=output_angle,
        unwrapped_deg=x_smoothed[:, 0],
        angular_velocity_deg_s=x_smoothed[:, 1],
        invalid_mask=final_invalid,
        rejected_outliers=rejected,
    )


def _validate_inputs(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    invalid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(angle_deg, dtype=float)

    if t.ndim != 1 or y.ndim != 1 or t.shape != y.shape:
        raise ValueError("time_s and angle_deg must be 1-D arrays of equal shape.")
    if t.size < 2:
        raise ValueError("At least two samples are required.")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("time_s must be finite and strictly increasing.")

    if invalid_mask is None:
        invalid = ~np.isfinite(y)
    else:
        invalid = np.asarray(invalid_mask, dtype=bool)
        if invalid.shape != y.shape:
            raise ValueError("invalid_mask must have the same shape as angle_deg.")
        invalid = invalid | ~np.isfinite(y)

    return t, y, invalid


def _estimate_initial_velocity(
    t: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    first: int,
    period: float,
    max_points: int = 50,
) -> float:
    indices = np.flatnonzero(valid & (np.arange(t.size) >= first))[:max_points]
    if indices.size < 2:
        return 0.0

    local_time = t[indices] - t[indices[0]]
    local_angle = np.unwrap(y[indices], period=period)
    if local_time[-1] <= 0:
        return 0.0

    return float(np.polyfit(local_time, local_angle, 1)[0])


def simulate_position_sensor(
    *,
    duration_s: float = 10.0,
    sample_rate_hz: float = 1000.0,
    initial_velocity_deg_s: float = 720.0,
    velocity_random_walk_std: float = 200.0,
    sensor_noise_std_deg: float = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a wrapped angular position whose velocity is a random walk."""
    rng = np.random.default_rng(seed)
    sample_count = int(round(duration_s * sample_rate_hz))
    time_s = np.arange(sample_count) / sample_rate_hz
    dt = 1.0 / sample_rate_hz

    velocity = np.empty(sample_count)
    velocity[0] = initial_velocity_deg_s
    velocity[1:] = initial_velocity_deg_s + np.cumsum(
        rng.normal(
            0.0,
            velocity_random_walk_std * np.sqrt(dt),
            sample_count - 1,
        )
    )

    unwrapped_angle = np.empty(sample_count)
    unwrapped_angle[0] = 30.0
    unwrapped_angle[1:] = unwrapped_angle[0] + np.cumsum(
        0.5 * (velocity[:-1] + velocity[1:]) * dt
    )

    measured_angle = wrap_deg(
        unwrapped_angle
        + rng.normal(0.0, sensor_noise_std_deg, sample_count)
    )
    return time_s, unwrapped_angle, velocity, measured_angle


def add_random_held_dropouts(
    angle_deg: np.ndarray,
    *,
    sample_rate_hz: float = 1000.0,
    mean_interval_s: float = 1.0,
    gap_range_ms: tuple[float, float] = (5.0, 80.0),
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace random blocks by the sample immediately before each block."""
    rng = np.random.default_rng(seed)
    observed = np.asarray(angle_deg, dtype=float).copy()
    missing = np.zeros(observed.size, dtype=bool)

    nominal_time = mean_interval_s
    while True:
        jittered_time = nominal_time + rng.uniform(
            -0.2 * mean_interval_s, 0.2 * mean_interval_s
        )
        start = max(1, int(round(jittered_time * sample_rate_hz)))
        if start >= observed.size:
            break

        low = max(1, int(round(gap_range_ms[0] * sample_rate_hz / 1000.0)))
        high = max(low + 1, int(round(gap_range_ms[1] * sample_rate_hz / 1000.0)) + 1)
        length = int(rng.integers(low, high))
        end = min(observed.size, start + length)

        observed[start:end] = observed[start - 1]
        missing[start:end] = True
        nominal_time += mean_interval_s

    return observed, missing


def add_single_held_dropout(
    angle_deg: np.ndarray,
    *,
    sample_rate_hz: float,
    start_s: float,
    gap_ms: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    observed = np.asarray(angle_deg, dtype=float).copy()
    missing = np.zeros(observed.size, dtype=bool)
    start = int(round(start_s * sample_rate_hz))
    length = max(1, int(round(gap_ms * sample_rate_hz / 1000.0)))
    end = min(observed.size, start + length)
    observed[start:end] = observed[start - 1]
    missing[start:end] = True
    return observed, missing, start, end


def add_angle_spikes(
    angle_deg: np.ndarray,
    *,
    spike_count: int = 20,
    magnitude_deg: float = 180.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject isolated positive or negative angular glitches."""
    rng = np.random.default_rng(seed)
    observed = np.asarray(angle_deg, dtype=float).copy()
    candidates = np.arange(100, observed.size - 100)
    spike_indices = rng.choice(candidates, size=spike_count, replace=False)
    signs = rng.choice((-1.0, 1.0), size=spike_count)
    observed[spike_indices] = wrap_deg(
        observed[spike_indices] + signs * magnitude_deg
    )
    spike_mask = np.zeros(observed.size, dtype=bool)
    spike_mask[spike_indices] = True
    return observed, spike_mask


def circular_rmse(
    estimated_deg: np.ndarray,
    true_unwrapped_deg: np.ndarray,
    mask: np.ndarray,
) -> float:
    error = circular_difference_deg(
        estimated_deg[mask], wrap_deg(true_unwrapped_deg[mask])
    )
    return float(np.sqrt(np.mean(error**2)))


def evaluate_gap_limits(
    *,
    trials: int = 20,
    gap_lengths_ms: tuple[int, ...] = (5, 10, 20, 50, 100, 250, 500, 1000),
    velocity_rw_values: tuple[int, ...] = (50, 200, 800),
) -> list[dict[str, float]]:
    """
    Monte-Carlo comparison.

    Initial speed is 720 deg/s. Therefore a 250 ms gap contains approximately
    half a revolution before random-walk variation is considered.
    """
    rows: list[dict[str, float]] = []

    for velocity_rw_std in velocity_rw_values:
        for gap_ms in gap_lengths_ms:
            linear_errors: list[float] = []
            inertial_errors: list[float] = []
            linear_slips: list[bool] = []
            inertial_slips: list[bool] = []

            for trial in range(trials):
                duration_s = max(4.0, 3.0 + gap_ms / 1000.0)
                t, truth, _, measured = simulate_position_sensor(
                    duration_s=duration_s,
                    initial_velocity_deg_s=720.0,
                    velocity_random_walk_std=float(velocity_rw_std),
                    sensor_noise_std_deg=0.02,
                    seed=1234 + trial,
                )
                observed, true_missing, start, end = add_single_held_dropout(
                    measured,
                    sample_rate_hz=1000.0,
                    start_s=1.5,
                    gap_ms=float(gap_ms),
                )
                detected = detect_held_samples(observed, min_run=2)

                linear_wrapped, linear_unwrapped = interpolate_circular_linear(
                    t, observed, detected
                )
                inertial = interpolate_circular_inertial(
                    t,
                    observed,
                    detected,
                    velocity_random_walk_std=float(velocity_rw_std),
                    measurement_std_deg=0.02,
                )

                linear_errors.append(
                    circular_rmse(linear_wrapped, truth, true_missing)
                )
                inertial_errors.append(
                    circular_rmse(inertial.angle_deg, truth, true_missing)
                )

                # Align branches immediately before the gap and test whether the
                # first valid point after it is off by at least half a revolution.
                check_index = min(end, t.size - 1)
                linear_alignment = round(
                    (truth[start - 1] - linear_unwrapped[start - 1]) / 360.0
                ) * 360.0
                inertial_alignment = round(
                    (truth[start - 1] - inertial.unwrapped_deg[start - 1]) / 360.0
                ) * 360.0

                linear_slips.append(
                    abs(
                        linear_unwrapped[check_index]
                        + linear_alignment
                        - truth[check_index]
                    )
                    > 180.0
                )
                inertial_slips.append(
                    abs(
                        inertial.unwrapped_deg[check_index]
                        + inertial_alignment
                        - truth[check_index]
                    )
                    > 180.0
                )

            rows.append(
                {
                    "velocity_rw_std": float(velocity_rw_std),
                    "gap_ms": float(gap_ms),
                    "linear_median_rmse_deg": float(np.median(linear_errors)),
                    "inertial_median_rmse_deg": float(np.median(inertial_errors)),
                    "linear_p95_rmse_deg": float(np.percentile(linear_errors, 95)),
                    "inertial_p95_rmse_deg": float(np.percentile(inertial_errors, 95)),
                    "linear_winding_slip_percent": 100.0 * float(np.mean(linear_slips)),
                    "inertial_winding_slip_percent": 100.0 * float(np.mean(inertial_slips)),
                }
            )

    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_limit_plot(rows: list[dict[str, float]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    rw_values = sorted({int(row["velocity_rw_std"]) for row in rows})
    for rw in rw_values:
        subset = [row for row in rows if int(row["velocity_rw_std"]) == rw]
        gaps = [row["gap_ms"] for row in subset]
        linear = [row["linear_median_rmse_deg"] for row in subset]
        inertial = [row["inertial_median_rmse_deg"] for row in subset]
        ax.plot(gaps, linear, marker="o", label=f"Linear, RW={rw}")
        ax.plot(gaps, inertial, marker="s", label=f"Inertia, RW={rw}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gap length [ms]")
    ax.set_ylabel("Median circular RMSE in gap [deg]")
    ax.set_title("Circular interpolation: linear baseline vs. inertia model")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_example_plot(path: Path) -> None:
    import matplotlib.pyplot as plt

    t, truth, _, measured = simulate_position_sensor(
        duration_s=3.0,
        initial_velocity_deg_s=720.0,
        velocity_random_walk_std=200.0,
        sensor_noise_std_deg=0.02,
        seed=2026,
    )
    observed, missing, start, end = add_single_held_dropout(
        measured,
        sample_rate_hz=1000.0,
        start_s=1.25,
        gap_ms=500.0,
    )
    detected = detect_held_samples(observed, min_run=2)
    _, linear_unwrapped = interpolate_circular_linear(t, observed, detected)
    inertial = interpolate_circular_inertial(
        t,
        observed,
        detected,
        velocity_random_walk_std=200.0,
        measurement_std_deg=0.02,
    )

    # Align all unwrapped curves to the truth immediately before the gap.
    linear_unwrapped = linear_unwrapped + round(
        (truth[start - 1] - linear_unwrapped[start - 1]) / 360.0
    ) * 360.0
    inertial_unwrapped = inertial.unwrapped_deg + round(
        (truth[start - 1] - inertial.unwrapped_deg[start - 1]) / 360.0
    ) * 360.0

    window = (t >= 1.1) & (t <= 1.9)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t[window], truth[window], label="True unwrapped angle")
    ax.plot(t[window], linear_unwrapped[window], label="NumPy unwrap + linear")
    ax.plot(t[window], inertial_unwrapped[window], label="Inertia model")
    ax.axvspan(t[start], t[end - 1], alpha=0.15, label="Held-sample gap")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Unwrapped angle [deg]")
    ax.set_title("Example: 500 ms held-value dropout at about 720 deg/s")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_demo(output_dir: Path, trials: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    t, truth, _, measured = simulate_position_sensor(
        duration_s=10.0,
        initial_velocity_deg_s=720.0,
        velocity_random_walk_std=200.0,
        sensor_noise_std_deg=0.02,
        seed=42,
    )
    observed, true_missing = add_random_held_dropouts(
        measured,
        gap_range_ms=(5.0, 80.0),
        seed=43,
    )
    detected = detect_held_samples(observed, min_run=2)

    linear_wrapped, _ = interpolate_circular_linear(t, observed, detected)
    inertial = interpolate_circular_inertial(
        t,
        observed,
        detected,
        velocity_random_walk_std=200.0,
        measurement_std_deg=0.02,
    )

    print("Random-dropout demo")
    print(f"  True missing samples: {np.count_nonzero(true_missing)}")
    print(f"  Detected samples:     {np.count_nonzero(detected)}")
    print(
        "  Linear circular RMSE: "
        f"{circular_rmse(linear_wrapped, truth, true_missing):.4f} deg"
    )
    print(
        "  Inertial RMSE:        "
        f"{circular_rmse(inertial.angle_deg, truth, true_missing):.4f} deg"
    )

    # Explicitly evaluate isolated 180-degree glitches.
    spike_observed, spike_mask = add_angle_spikes(measured, seed=44)
    spike_result = interpolate_circular_inertial(
        t,
        spike_observed,
        np.zeros(t.size, dtype=bool),
        velocity_random_walk_std=200.0,
        measurement_std_deg=0.02,
    )
    true_positives = np.count_nonzero(
        spike_result.rejected_outliers & spike_mask
    )
    false_positives = np.count_nonzero(
        spike_result.rejected_outliers & ~spike_mask
    )
    print("180-degree spike test")
    print(
        f"  Rejected injected spikes: {true_positives}/"
        f"{np.count_nonzero(spike_mask)}"
    )
    print(f"  False rejections:         {false_positives}")

    print(f"Running Monte-Carlo sweep with {trials} trials per case...")
    rows = evaluate_gap_limits(trials=trials)
    csv_path = output_dir / "circular_interpolation_evaluation.csv"
    limit_plot_path = output_dir / "circular_interpolation_limits.png"
    example_plot_path = output_dir / "circular_interpolation_example.png"
    write_csv(rows, csv_path)
    create_limit_plot(rows, limit_plot_path)
    create_example_plot(example_plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {limit_plot_path}")
    print(f"Wrote {example_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate interpolation of wrapped angular position data."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Monte-Carlo trials per gap/noise case.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for CSV and plots.",
    )
    args = parser.parse_args()
    run_demo(args.output_dir, args.trials)


if __name__ == "__main__":
    main()
