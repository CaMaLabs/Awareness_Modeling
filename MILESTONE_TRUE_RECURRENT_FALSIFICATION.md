# True Recurrent Dynamics Falsification Milestone

This phase tests whether the four-state rollout result survives after previous-state memory is moved **inside** state selection and the start-state prototype is removed from the final target vector.

The existing recurrent scripts are intentionally left unchanged as a historical baseline.

## Why this phase exists

The earlier rollout architecture classified the current probe before adding state inertia to the score. It also constructed each target from the starting state's feature prototype and only replaced alpha/chi coordinates. Those two choices can produce strong start-state-dependent endpoint maps even if recurrent memory is not actually changing the state decision.

This phase separates those effects.

## New model

For each candidate state `s`, the recurrent selector computes an energy from:

- emission similarity between the current feature probe and the state centroid
- state-specific inertia weighted by the previous state probability
- directed transition cost from the previous hard state

State probabilities are then computed with a softmax, and the winning state becomes the history input for the next rollout step.

The default state-energy form is:

```text
energy(s, t) =
    gaussian_similarity(features[t], centroid[s])
  + memory_scale * inertia[s] * P(s at t-1)
  - transition_scale * transition_cost(state[t-1], s)
```

Loop identity is retained as a state-conditioned readout/context variable using empirical loop priors from the input rows. It does not independently decide the state in this phase; REM loop diversification remains a later modeling target.

## Falsification matrix

`true_recurrent_dynamics.py` runs five conditions over the same alpha/chi grid:

1. `legacy_target_positive_control` — true recurrent selection, but the target still inherits the starting-state prototype.
2. `prototype_target_removed` — the main deconfounded condition; every starting state reaches the same final feature probe for a given alpha/chi cell.
3. `memory_off` — deconfounded target with state inertia removed.
4. `previous_state_shuffled` — deconfounded target with a deterministic derangement of previous-state history.
5. `transition_penalties_zero` — deconfounded target with directed transition costs removed.

The key metric is mutual information between starting state and final state (`history_mi_bits`). In the deconfounded condition, a nonzero value can no longer be explained by a different final feature vector across starting states.

## Reference run on the current balanced synthetic generator

Default parameters:

- 8 rollout steps
- softmax temperature 0.20
- emission width 2.0 standardized-distance units
- memory scale 1.0
- transition scale 1.0

Reference result:

| condition | history MI (bits) | same-state fraction |
|---|---:|---:|
| legacy_target_positive_control | ~1.539 | ~0.861 |
| prototype_target_removed | ~0.0091 | ~0.267 |
| memory_off | ~0.0028 | ~0.264 |
| previous_state_shuffled | ~0.0017 | ~0.242 |
| transition_penalties_zero | ~0.0026 | ~0.255 |

The deconfounded run retains only about **0.6%** of the legacy positive-control start/final mutual information.

## Interpretation

This is a useful falsification result, not a failure of the project.

The previous four-basin rollout map was dominated by start-state prototype information baked into the target trajectory. Once the final feature vector is held constant across starting states, most of that apparent history dependence disappears.

A smaller residual remains: the deconfounded recurrent condition carries more start/final information than memory-off, shuffled-history, or zero-transition controls. Under the current parameterization, however, that residual is weak and should not be described as a robust four-attractor recurrent system.

The next defensible modeling task is therefore **not** to tune inertia until the old basin map returns. It is to determine whether stronger recurrent structure can be justified from independent data or a more mechanistic transition model.

## Run it

```bash
python balanced_state_generator.py
python true_recurrent_dynamics.py --input balanced_states.csv --out-prefix true_recurrent
python -m unittest -v test_true_recurrent_dynamics.py
```

Outputs:

- `true_recurrent_rollouts.csv`
- `true_recurrent_summary.csv`
- `true_recurrent_by_start.csv`
- `true_recurrent_report.md`

## Guardrail

All results remain synthetic/model-internal unless evaluated against independent physiological data. This repository is experimental research software and is not intended for diagnosis, treatment, anesthesia, or medical decision-making.
