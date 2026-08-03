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

## Structure
- `src/circular_interpolation`: Core interpolation algorithms.
- `benchmarks/`: Evaluation scripts.
- `tests/`: Unit tests.
- `deprecated/`: Old outputs and zipped experiments (ignored by git).
