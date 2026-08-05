# Synchronized multi-signal freeze repair

`interpolate_raw_circular_stream` accepts either one signal with shape
`(samples,)` or synchronized data with shape `(signals, samples)`. All rows use
one common source time axis.

For two-dimensional data, a repeated-value recorder-freeze candidate is created
only when **every row** repeats its own preceding value at the same source
timestamps. A local standstill, quantization plateau, or repeated value in only
one channel therefore does not trigger reconstruction in any row.

```python
import numpy as np

from circular_interpolation import interpolate_raw_circular_stream

# Rows are signals and columns are samples on time_s.
data = np.vstack([encoder_a, force, temperature, encoder_b])
new_time_s = np.linspace(time_s[0], time_s[-1], 20_000)

result = interpolate_raw_circular_stream(
    time_s,
    data,
    period=[360.0, None, None, 360.0],
    target_time_s=new_time_s,
    ambiguous_policy="repair",
    max_plateau_repair_duration_s=0.050,
)

assert result.angle_deg.shape == (4, new_time_s.size)
assert result.unwrapped_deg.shape == (4, new_time_s.size)
assert result.repaired_mask.shape == data.shape
assert result.joint_candidate_mask.shape == time_s.shape
assert result.joint_repair_mask.shape == time_s.shape
```

## Period configuration

A scalar period is broadcast to every row:

```python
period=360.0  # every signal is circular
period=None   # every signal is non-periodic
```

A sequence configures each row independently and must have length
`data.shape[0]`:

```python
period=[360.0, None, None, None, 360.0]
```

The value-domain thresholds can also be scalars or one value per signal. This is
useful when rows have different units and scales:

```python
result = interpolate_raw_circular_stream(
    time_s,
    data,
    period=[360.0, None, None],
    repeat_tolerance_deg=[0.0, 1e-6, 1e-3],
    standstill_speed_threshold_deg_s=[100.0, 0.01, 0.1],
    min_motion_evidence_deg=[0.15, 1e-5, 1e-4],
    max_abs_acceleration_deg_s2=[1e6, 10.0, 100.0],
)
```

The historical parameter names retain `deg`, but in `period=None` rows their
numerical units are simply the units of that signal.

## Detection and reconstruction are separate

The shared-channel condition decides whether a recorder-level candidate exists.
It does not force every signal to use the same interpolation curve.

1. Every row must repeat its own preceding value at the same timestamps.
2. The run is classified using the edge evidence from all rows.
3. `ambiguous_policy` and `max_plateau_repair_duration_s` decide whether the
   shared run is reconstructed.
4. The resulting source mask is applied to every row.
5. Each row is reconstructed independently with its own period and automatically
   selected gap model.
6. Every row is evaluated on the same optional `target_time_s` grid.

If at least one row has convincing two-sided motion evidence, the joint run is a
`held_dropout`. If all rows look locally stationary, the joint run remains
`ambiguous`: it could be a recorder freeze or a genuine whole-system standstill.
This lets an aggressive policy repair short shared freezes while the duration
guard protects longer system standstills.

```python
result = interpolate_raw_circular_stream(
    time_s,
    data,
    ambiguous_policy="repair",
    max_plateau_repair_duration_s=0.020,
)
```

The duration should come from the recorder or transport contract, such as the
longest credible frozen-value burst.

## Masks and diagnostics

For two-dimensional input:

- `joint_candidate_mask` is one-dimensional and marks every source timestamp at
  which all rows repeat simultaneously.
- `joint_repair_mask` is the subset selected after classification, policy, anchor
  handling, and the duration guard.
- `repaired_mask` has shape `(signals, source_samples)`. It contains the shared
  repair mask in every row plus row-specific explicit `NaN` or infinite samples.
- `preserved_standstill_mask` and `ambiguous_mask` are one-dimensional source
  masks for joint decisions.
- `decisions` contains `JointPlateauDecision` objects. Each decision retains the
  corresponding per-signal `PlateauDecision` objects in `signal_decisions`.
- `chosen_model_by_gap` and `interpolation_confidence_by_gap` contain one list per
  signal because reconstruction models may differ between rows.
- `period_by_signal` contains the normalized period configuration.

The first measured plateau value is preserved by default. Leading and trailing
joint plateaus are boundary plateaus and are never inferred as missing.

Explicit non-finite samples remain row-specific. A `NaN` in one row does not
silently invalidate the other rows.

## Remaining ambiguity

The shared-channel rule removes local standstills from consideration, but values
alone still cannot distinguish a short true whole-system standstill from a short
recorder freeze when every recorded signal is constant. Use the duration guard,
external recorder status, CRC information, or another quality channel whenever
such information is available.
