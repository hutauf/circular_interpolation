# circular_interpolation

Robust reconstruction of circular signals with dropouts, wraparound, held sensor values, and optional direct evaluation on a different time grid.

![A held-value dropout reconstructed on the physically plausible revolution](docs/gap_repair.svg)

Angles are not ordinary scalars: `359 deg` and `1 deg` are two degrees apart, and a gap may hide complete revolutions. The package keeps an unwrapped trajectory internally and wraps only the final output.

## Highlights

- **Automatic gap model selection** for standstill, short dynamic gaps, and longer moving gaps.
- **Winding-aware reconstruction** that can retain the likely number of full revolutions.
- **One optional resampling kwarg:** `target_time_s`.
- **Direct target-grid evaluation:** selected continuous trajectories are evaluated at the requested times instead of repairing a wrapped series and then resampling it.
- **Irregular grids and extrapolation** are supported.
- **Raw-stream wrapper** detects held-value plateaus while conservatively preserving real standstill and quantization plateaus.
- **Diagnostics** report the selected model and confidence for each source gap.

## Installation

```bash
pip install -e .
```

Python 3.9 or newer is required. Runtime dependencies are NumPy, SciPy, and Matplotlib.

## Recommended API: known invalid samples

Use `interpolate_circular_auto` when a CRC flag, missing value, quality channel, or another detector already tells you which source samples are invalid.

```python
import numpy as np
from circular_interpolation import interpolate_circular_auto

# Wrapped source measurements in degrees.
time_s = np.array([0.000, 0.010, 0.020, 0.030, 0.040])
angle_deg = np.array([350.0, 357.0, np.nan, 11.0, 18.0])
invalid_mask = ~np.isfinite(angle_deg)

result = interpolate_circular_auto(
    time_s=time_s,
    angle_deg=angle_deg,
    invalid_mask=invalid_mask,
)

print(result.time_s)        # Same as time_s by default.
print(result.angle_deg)     # Wrapped to [0, 360).
print(result.unwrapped_deg) # Continuous physical branch.
print(result.chosen_model_by_gap)
print(result.confidence_by_gap)
```

Omitting `target_time_s` preserves the historical source-grid behavior exactly.

## Repair and change the time grid in one call

Pass any finite, strictly increasing one-dimensional grid. It may be denser, sparser, irregular, shifted, or extend outside the measured time span.

```python
new_time_s = np.linspace(-0.005, 0.050, 500)

result = interpolate_circular_auto(
    time_s=time_s,
    angle_deg=angle_deg,
    invalid_mask=invalid_mask,
    target_time_s=new_time_s,
)

assert np.array_equal(result.time_s, new_time_s)
assert result.angle_deg.shape == new_time_s.shape
assert result.unwrapped_deg.shape == new_time_s.shape
```

![Gap repair and direct evaluation on a new time grid](docs/target_time_grid.svg)

The target-grid path does not interpolate already wrapped repaired values. It works on the selected unwrapped trajectories:

1. Valid source runs are assigned to consistent revolutions.
2. Each internal gap receives the same automatic model as on the source grid.
3. That continuous model is evaluated directly for target timestamps inside the gap. Linear and Hermite gaps use their analytic curves; short Kalman/RTS gaps insert the requested times as prediction-only states in the same state-space model.
4. Valid runs use shape-preserving interpolation on the unwrapped branch.
5. Outside the measured support, local boundary position, velocity, and acceleration are estimated. Acceleration is applied only over the local fitting horizon; farther extrapolation continues at constant velocity to avoid unbounded quadratic growth.

Extrapolation is necessarily less certain than interpolation between measurements, especially across reversals or abrupt control changes. Keep extrapolation horizons physically reasonable for the application.

## Raw stream processing

Use `interpolate_raw_circular_stream` when the sensor repeats its last value instead of supplying an explicit invalid flag. The classifier uses two-sided motion evidence and physical acceleration limits to separate likely held-value dropouts from true standstill, encoder quantization, and ambiguous reversals.

```python
from circular_interpolation import interpolate_raw_circular_stream

raw_result = interpolate_raw_circular_stream(
    time_s=time_s,
    angle_deg=angle_deg,
)

print(raw_result.angle_deg)
print(raw_result.decisions)
print(raw_result.repaired_mask)
```

The raw-stream wrapper deliberately keeps its classification masks and output on the sensor grid. To produce a different output grid after classification, call the automatic API with the detected mask:

```python
resampled = interpolate_circular_auto(
    time_s=time_s,
    angle_deg=angle_deg,
    invalid_mask=raw_result.repaired_mask,
    target_time_s=new_time_s,
)
```

`target_time_s` belongs to interpolation; plateau classification continues to describe the original samples.

## Automatic model selection

For each internal source gap, `interpolate_circular_auto` currently applies these empirical rules from the motor-profile test suite:

| Situation | Selected model |
|---|---|
| Near standstill | Shortest-arc linear trajectory |
| Moving, endpoint-to-endpoint gap at most 6 ms | Constant-acceleration Kalman/RTS trajectory |
| Longer moving gap | Boundary-velocity cubic Hermite trajectory |
| Leading or trailing missing source samples | Boundary kinematic extrapolation |

Long dynamic gaps are reconstructed but may receive medium or low confidence when boundary velocities disagree strongly or change direction.

## Lower-level algorithms

The lower-level functions remain available for experiments and applications that deliberately choose a fixed model:

```python
from circular_interpolation import (
    interpolate_circular_inertial,
    interpolate_circular_adaptive_minimum_jerk,
)
```

- `interpolate_circular_inertial`: constant-velocity Kalman filter with stochastic acceleration and an RTS backward pass.
- `interpolate_circular_adaptive_minimum_jerk`: winding search plus a smooth minimum-jerk correction with an exact linear fallback.

These lower-level APIs return values on the source grid. Direct `target_time_s` evaluation is exposed by the recommended automatic function, where model selection and output-grid semantics remain consistent.

## Result semantics

- `time_s`: timestamps corresponding to both output signal arrays.
- `angle_deg`: wrapped output in `[0, period)`.
- `unwrapped_deg`: continuous branch retaining complete revolutions.
- `chosen_model_by_gap`: `(start, end, model)` tuples using **source indices**; `end` is exclusive.
- `confidence_by_gap`: source-index ranges with `high`, `medium`, or `low` confidence.

The configurable `period` defaults to `360.0`.

## Development

Run the focused test suite:

```bash
pytest -q
```

Regenerate the README figures:

```bash
PYTHONPATH=src python docs/generate_readme_plots.py
```

Repository layout:

- `src/circular_interpolation/`: interpolation algorithms and wrappers.
- `tests/`: deterministic unit tests.
- `benchmarks/`: fixed motor-profile corpus and evaluation scripts.
- `docs/`: reproducible README figures.
