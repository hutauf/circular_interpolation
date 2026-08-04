import numpy as np
import pytest

from circular_interpolation.auto import interpolate_circular_auto
from circular_interpolation.inertia import circular_difference_deg


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

    assert result.chosen_model_by_gap[0][2] == "constant_acceleration_rts"
    assert 20.0 < result.angle_deg[2] < 40.0


def test_target_grid_omitted_and_explicit_source_grid_are_identical():
    t = np.arange(0.0, 0.101, 0.001)
    truth = 350.0 + 720.0 * t
    y = np.mod(truth, 360.0)
    invalid = (t >= 0.040) & (t <= 0.060)
    y[invalid] = np.nan

    default = interpolate_circular_auto(t, y, invalid)
    explicit = interpolate_circular_auto(t, y, invalid, target_time_s=t.copy())

    assert np.array_equal(default.time_s, t)
    assert np.array_equal(explicit.time_s, t)
    assert np.array_equal(default.angle_deg, explicit.angle_deg)
    assert np.array_equal(default.unwrapped_deg, explicit.unwrapped_deg)
    assert default.chosen_model_by_gap == explicit.chosen_model_by_gap
    assert default.confidence_by_gap == explicit.confidence_by_gap


def test_irregular_target_grid_repairs_wraps_and_extrapolates():
    t = np.arange(0.0, 0.101, 0.001)
    truth = 350.0 + 720.0 * t
    y = np.mod(truth, 360.0)
    invalid = (t >= 0.040) & (t <= 0.060)
    y[invalid] = np.nan

    target = np.unique(
        np.r_[
            np.linspace(-0.020, 0.120, 431),
            np.array([0.0405, 0.0473, 0.0599]),
        ]
    )
    result = interpolate_circular_auto(
        t,
        y,
        invalid,
        target_time_s=target,
    )

    expected_unwrapped = 350.0 + 720.0 * target
    assert np.array_equal(result.time_s, target)
    assert result.angle_deg.shape == target.shape
    assert result.unwrapped_deg.shape == target.shape
    assert np.max(np.abs(result.unwrapped_deg - expected_unwrapped)) < 1e-6
    error = circular_difference_deg(result.angle_deg, np.mod(expected_unwrapped, 360.0))
    assert np.max(np.abs(error)) < 1e-6


def test_target_grid_can_be_entirely_outside_source_support():
    t = np.arange(0.0, 0.051, 0.001)
    y = np.mod(45.0 + 1200.0 * t, 360.0)
    invalid = np.zeros(t.size, dtype=bool)
    target = np.array([-0.050, -0.010, 0.080, 0.200])

    result = interpolate_circular_auto(t, y, invalid, target_time_s=target)

    assert np.all(np.isfinite(result.angle_deg))
    assert np.all(np.isfinite(result.unwrapped_deg))
    assert np.allclose(result.unwrapped_deg, 45.0 + 1200.0 * target, atol=1e-6)


@pytest.mark.parametrize(
    "target",
    [
        np.array([]),
        np.array([0.0, 0.0]),
        np.array([0.1, 0.0]),
        np.array([0.0, np.nan]),
        np.array([[0.0, 0.1]]),
    ],
)
def test_target_grid_validation(target):
    t = np.array([0.0, 0.1, 0.2])
    y = np.array([10.0, 20.0, 30.0])
    invalid = np.zeros(3, dtype=bool)

    with pytest.raises(ValueError, match="target_time_s"):
        interpolate_circular_auto(t, y, invalid, target_time_s=target)


def test_target_grid_handles_leading_and_trailing_missing_source_samples():
    t = np.arange(0.0, 0.101, 0.001)
    truth = 275.0 - 2400.0 * t
    y = np.mod(truth, 360.0)
    invalid = (t < 0.008) | (t > 0.092)
    y[invalid] = np.nan
    target = np.linspace(-0.02, 0.12, 281)

    result = interpolate_circular_auto(t, y, invalid, target_time_s=target)

    expected = 275.0 - 2400.0 * target
    branch_shift = round((expected[0] - result.unwrapped_deg[0]) / 360.0) * 360.0
    assert np.allclose(result.unwrapped_deg + branch_shift, expected, atol=1e-6)


def test_target_grid_with_multiple_gaps_and_single_requested_sample():
    t = np.arange(0.0, 0.151, 0.001)
    truth = 25.0 + 3600.0 * t
    y = np.mod(truth, 360.0)
    invalid = ((t >= 0.025) & (t <= 0.035)) | ((t >= 0.090) & (t <= 0.115))
    y[invalid] = np.nan

    result = interpolate_circular_auto(
        t,
        y,
        invalid,
        target_time_s=np.array([0.1037]),
    )

    assert np.isfinite(result.angle_deg[0])
    assert np.isclose(result.unwrapped_deg[0], 25.0 + 3600.0 * 0.1037, atol=1e-6)
    assert len(result.chosen_model_by_gap) == 2


def test_short_acceleration_gap_is_evaluated_directly_on_target_grid():
    t = np.arange(0.0, 0.0301, 0.001)
    truth = 20.0 + 5_000.0 * t + 0.5 * 400_000.0 * t**2
    y = np.mod(truth, 360.0)
    invalid = (t >= 0.014) & (t <= 0.017)
    y[invalid] = np.nan
    target = np.unique(
        np.r_[
            t,
            np.linspace(t[13], t[18], 101),
        ]
    )

    source = interpolate_circular_auto(t, y, invalid)
    result = interpolate_circular_auto(t, y, invalid, target_time_s=target)

    assert source.chosen_model_by_gap[0][2] == "constant_acceleration_rts"
    source_positions = np.searchsorted(target, t)
    assert np.allclose(
        result.unwrapped_deg[source_positions],
        source.unwrapped_deg,
        rtol=0.0,
        atol=1e-10,
    )
    expected = 20.0 + 5_000.0 * target + 0.5 * 400_000.0 * target**2
    assert np.max(np.abs(result.unwrapped_deg - expected)) < 0.002


def test_new_grid_uses_limited_extrapolation_even_at_old_invalid_timestamp():
    t = np.arange(0.0, 1.001, 0.01)
    truth = 10.0 + 100.0 * t + 50.0 * t**2
    y = np.mod(truth, 360.0)
    invalid = t < 0.5
    y[invalid] = np.nan

    result = interpolate_circular_auto(
        t,
        y,
        invalid,
        target_time_s=np.array([0.0]),
    )

    # The first 20 valid samples span 0.19 s. The fitted 100 deg/s^2
    # acceleration is used only over that local horizon, followed by constant
    # velocity extrapolation for the remaining 0.31 s. This deliberately
    # differs from extending a quadratic indefinitely merely because 0.0 was an
    # old (invalid) source timestamp.
    expected = (
        72.5
        + 150.0 * -0.19
        + 0.5 * 100.0 * 0.19**2
        + (150.0 - 100.0 * 0.19) * -0.31
    )
    assert np.isclose(result.unwrapped_deg[0], expected, atol=1e-9)
