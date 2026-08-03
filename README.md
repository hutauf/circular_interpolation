# circular_interpolation

A Python package for robust circular (angular) interpolation.

## Features
- **Inertia Model**: Handles angular data dropouts and wraps using Kalman filtering and RTS smoothing.
- **Adaptive Minimum Jerk**: Smooth gap filling using minimum jerk trajectories.
- **Raw Stream Wrapper**: Detects and interpolates plateaus or dropouts in raw motor angle streams.

## Installation
```bash
pip install -e .
```

## Usage

Here are the primary ways to apply the library to your own datasets.

### 1. Raw Stream Processing (Most Common)
If you have a raw position sensor stream (e.g., from an encoder) that occasionally drops out and holds the last value, you can use the wrapper. It automatically detects plateaus, classifies them as true standstills or sensor dropouts, and interpolates the dropouts.

```python
import numpy as np
from circular_interpolation import interpolate_raw_circular_stream

# Your raw data: time in seconds, angle in degrees [0, 360)
time_s = np.array([0.0, 0.001, 0.002, 0.003, 0.004])
angle_deg = np.array([10.5, 11.0, 11.0, 11.0, 12.5]) # Note the plateau at 11.0

result = interpolate_raw_circular_stream(time_s, angle_deg)

print("Interpolated Angle:", result.angle_deg)
print("Repaired Indices:", result.repaired_mask)
print("Detected Standstills:", result.preserved_standstill_mask)
```

### 2. Inertial Interpolation (Advanced)
If you already know which samples are invalid (e.g., from CRC errors or predefined missing gaps) and want to apply the Kalman/RTS inertia model for robust unwrapping across wide gaps.

```python
from circular_interpolation import interpolate_circular_inertial

invalid_mask = np.isnan(angle_deg)

result = interpolate_circular_inertial(
    time_s=time_s,
    angle_deg=angle_deg,
    invalid_mask=invalid_mask,
    velocity_random_walk_std=200.0, # Adjust based on expected motor dynamics
    measurement_std_deg=0.05
)

print("Clean Angle:", result.angle_deg)
print("Estimated Angular Velocity:", result.angular_velocity_deg_s)
```

### 3. Adaptive Minimum Jerk
For smooth interpolation using polynomials that minimize jerk across gaps.

```python
from circular_interpolation import interpolate_circular_adaptive_minimum_jerk

result = interpolate_circular_adaptive_minimum_jerk(
    time_s=time_s,
    angle_deg=angle_deg,
    invalid_mask=invalid_mask
)
```

## Structure
- `src/circular_interpolation`: Core interpolation algorithms.
- `benchmarks/`: Evaluation scripts.
- `tests/`: Unit tests.
- `deprecated/`: Old outputs and zipped experiments (ignored by git).
