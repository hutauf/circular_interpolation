import numpy as np
from circular_interpolation.auto import interpolate_circular_auto

def test_interpolate_circular_auto_standstill():
    t = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    y = np.array([10.0, 10.0, np.nan, 10.0, 10.0])
    invalid = np.isnan(y)
    
    result = interpolate_circular_auto(t, y, invalid)
    
    assert np.isclose(result.angle_deg[2], 10.0)
    assert result.chosen_model_by_gap[0][2] == "linear_shortest_arc"

def test_interpolate_circular_auto_acceleration():
    t = np.array([0.0, 0.001, 0.002, 0.003, 0.004])
    y = np.array([10.0, 20.0, np.nan, 40.0, 50.0])
    invalid = np.isnan(y)
    
    result = interpolate_circular_auto(t, y, invalid)
    
    # Gap is 0.002s, which is <= 0.006, so it should choose constant_acceleration_rts
    assert result.chosen_model_by_gap[0][2] == "constant_acceleration_rts"
    # Should interpolate reasonably
    assert 20.0 < result.angle_deg[2] < 40.0
