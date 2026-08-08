# Awareness Modeling

Adaptive neural regime modeling framework for mapping synthetic and real EEG-like dynamics into functional brain-state regimes using coherence, phase, slow-wave structure, and recurrent state-history tests.

## What this repo does

This project explores whether compact mesoscale features can separate wake-like, NREM-like, and REM-like regimes while tracking candidate loop families such as:

- `thalamo_cortical`
- `fronto_parietal`
- `dmn`
- `ct_cingulate`

The current pipeline supports:

- balanced synthetic wake/N2/REM/N3 generation
- synthetic augmentation and scoring
- regime-map generation in chi/alpha space
- heuristic sleep-state projection
- scoring of real EEG/MRI-style summary rows
- state-specific inertia and directed-transition experiments
- recurrent rollout maps
- deconfounded recurrent-state falsification and ablation tests

## Key files

- `balanced_state_generator.py` — balanced synthetic wake/N2/REM/N3 generator
- `asci_pipeline_nompl.py` — synthetic augmentation, scoring, and train/test split
- `regime_map_nompl.py` — binning and regime-map generation
- `sleep_projection.py` — wake/N1-N2/N3/REM projection from regime bins
- `real_data_ingest_nompl.py` — scoring for real or literature-derived summary rows
- `state_specific_inertia.py` — state-memory and directed-transition path tests
- `recurrent_rollout_map.py` — original recurrent rollout-map experiment
- `true_recurrent_dynamics.py` — recurrent state selection with prototype-target deconfounding and falsification controls
- `test_true_recurrent_dynamics.py` — unit tests for the recurrent falsification mechanics
- `MILESTONE_FOUR_STATE_RECURRENT.md` — historical four-state recurrent milestone
- `MILESTONE_TRUE_RECURRENT_FALSIFICATION.md` — deconfounded recurrence milestone and interpretation

## Quick start

```bash
python balanced_state_generator.py
python asci_pipeline_nompl.py --input asci_template.csv --out-prefix diversified
python regime_map_nompl.py
python sleep_projection.py
python state_specific_inertia.py
python recurrent_rollout_map.py
python true_recurrent_dynamics.py --input balanced_states.csv --out-prefix true_recurrent
python -m unittest -v test_true_recurrent_dynamics.py
```

## Current interpretation

The synthetic framework can generate separable wake, N2, REM, and N3 feature regimes. The original recurrent rollout map also produced strong start-state-dependent endpoint occupancy.

A later falsification test identified an important confound in that result: the original rollout target retained most of the starting state's feature prototype, while state classification occurred before recurrent inertia was added to the score.

`true_recurrent_dynamics.py` moves previous-state memory into the state-selection energy and tests a deconfounded condition in which every start state reaches the same final feature probe for a given alpha/chi cell. Under the current default synthetic parameterization, most of the original start/final dependence disappears after that control. A small recurrent residual remains and decreases further when memory, correct previous-state history, or transition penalties are removed.

The defensible current claim is therefore that the model has strong synthetic regime separation and a **weak residual history-dependent recurrent effect after deconfounding**, not yet a validated four-attractor neural dynamical system.

## Data format

Expected columns for ingest/scoring include:

- `subject_id`
- `state`
- `group`
- `candidate_loop`
- `L_eff_m`
- `v_eff_m_per_s`
- `tau_eff_s`
- `pl_local_aw`
- `pl_global_aw`
- `cfc`
- `w_prop`
- `s_struct`
- `peak_alpha_hz`
- `behavior_awareness_score`

## Disclaimer

This repository is experimental research software. Synthetic/model-internal results are not physiological validation. It is not intended for diagnosis, treatment, anesthesia, or any medical decision-making.
