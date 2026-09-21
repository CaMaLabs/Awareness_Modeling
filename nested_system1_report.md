# Nested memory + System-1 smoke benchmark

This benchmark isolates adaptation, calibration, and return-to-regime retention. It is **not** evidence for awareness or consciousness.

| variant | accuracy | Brier ↓ | log loss ↓ | B-shift early acc. | A-return early acc. |
|---|---:|---:|---:|---:|---:|
| A_baseline | 0.7800 | 0.1634 | 0.5002 | 0.1250 | 0.3750 |
| B_calibrated_system1 | 0.7989 | 0.1515 | 0.4682 | 0.1750 | 0.4250 |
| C_nested_memory | 0.7933 | 0.1547 | 0.4788 | 0.1250 | 0.4000 |
| D_nested_plus_calibrated | 0.8156 | 0.1435 | 0.4475 | 0.2250 | 0.4500 |

Interpret improvements only relative to this synthetic task. The full D-RCS runtime must be evaluated separately.
