import numpy as np
import pytest
from circular_interpolation.adaptive_minimum_jerk import interpolate_circular_adaptive_minimum_jerk

def test_interpolate_circular_adaptive_minimum_jerk_basic():
    t = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # Constant velocity: 10 deg/s
    y = np.array([330.0, 340.0, 350.0, np.nan, 10.0, 20.0, 30.0])
    invalid = np.isnan(y)
    
    result = interpolate_circular_adaptive_minimum_jerk(
        time_s=t,
        angle_deg=y,
        invalid_mask=invalid,
    )
    
    # Mid-point should be interpolated properly across 0
    assert np.isclose(result.angle_deg[3], 0.0, atol=1.0)
    
def test_interpolate_circular_adaptive_minimum_jerk_no_gaps():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([10.0, 20.0, 30.0])
    invalid = np.array([False, False, False])
    
    result = interpolate_circular_adaptive_minimum_jerk(
        time_s=t,
        angle_deg=y,
        invalid_mask=invalid,
    )
    
    assert np.allclose(result.angle_deg, y)
