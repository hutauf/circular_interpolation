import numpy as np
from circular_interpolation.inertia import (
    wrap_deg,
    circular_difference_deg,
    interpolate_circular_linear,
    interpolate_circular_inertial,
)

def test_wrap_deg():
    assert np.isclose(wrap_deg(360.0), 0.0)
    assert np.isclose(wrap_deg(361.0), 1.0)
    assert np.isclose(wrap_deg(-1.0), 359.0)

def test_circular_difference_deg():
    assert np.isclose(circular_difference_deg(359.0, 1.0), -2.0)
    assert np.isclose(circular_difference_deg(1.0, 359.0), 2.0)
    assert np.isclose(abs(circular_difference_deg(180.0, 0.0)), 180.0)

def test_interpolate_circular_linear():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([350.0, np.nan, 10.0])
    invalid = np.isnan(y)
    wrapped, unwrapped = interpolate_circular_linear(t, y, invalid)
    
    assert np.isclose(wrapped[1], 0.0)
    assert np.isclose(unwrapped[0], 350.0)
    assert np.isclose(unwrapped[1], 360.0)
    assert np.isclose(unwrapped[2], 370.0)

def test_interpolate_circular_inertial():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # Constant velocity: 10 deg/s
    y = np.array([340.0, 350.0, np.nan, 10.0, 20.0])
    invalid = np.isnan(y)
    
    result = interpolate_circular_inertial(
        t, y, invalid,
        velocity_random_walk_std=10.0,
        measurement_std_deg=0.1
    )
    
    # Mid-point should be interpolated across the 0-degree boundary properly
    assert np.isclose(result.angle_deg[2], 0.0, atol=1.0)
    # Check velocity
    assert np.allclose(result.angular_velocity_deg_s, 10.0, atol=2.0)
