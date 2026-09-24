# Nested memory + System-1 smoke benchmark

This benchmark isolates adaptation, calibration, and return-to-regime retention. It is **not** evidence for awareness or consciousness.

| variant | accuracy | Brier ↓ | log loss ↓ | B-shift early acc. | A-return early acc. |
|---|---:|---:|---:|---:|---:|
| A_baseline | 0.7767 | 0.1645 | 0.5029 | 0.1000 | 0.3750 |
| B_calibrated_system1 | 0.7956 | 0.1528 | 0.4713 | 0.1500 | 0.4250 |
| C_nested_memory | 0.7933 | 0.1557 | 0.4814 | 0.1250 | 0.4000 |
| D_nested_plus_calibrated | 0.8122 | 0.1447 | 0.4505 | 0.1750 | 0.4250 |

Interpret improvements only relative to this synthetic task. The full D-RCS runtime must be evaluated separately.
