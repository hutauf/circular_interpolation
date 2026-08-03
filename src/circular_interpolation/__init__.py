"""
Circular interpolation package for unwrapping and filtering angle data.
"""

from .inertia import interpolate_circular_inertial, CircularInterpolationResult
from .adaptive_minimum_jerk import interpolate_circular_adaptive_minimum_jerk
from .raw_stream_wrapper import interpolate_raw_circular_stream

__all__ = [
    "interpolate_circular_inertial",
    "CircularInterpolationResult",
    "interpolate_circular_adaptive_minimum_jerk",
    "interpolate_raw_circular_stream",
]
