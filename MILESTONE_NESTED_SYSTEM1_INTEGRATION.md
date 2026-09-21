# Nested plasticity + calibrated System-1 integration

## Goal

Add an experimental cognition layer without replacing or reinterpreting the existing recurrent-state model.

The layer combines two engineering ideas:

1. **HOPE/Nested-Learning-inspired plasticity**: several memory bands update at different frequencies, with surprise-gated writes and bounded meta-plasticity.
2. **Jev-inspired System-1 interface**: many small typed binary judgments return probabilities rather than free-form language. Each head is trained online and can use temperature calibration.

This is an architectural experiment, not a reproduction of Google HOPE or TypeSafe Jev.

## Files

- `nested_adaptive_cognition.py` — continuum memory, bounded self-modifying plasticity, calibrated judgment heads, persistence, teacher gate.
- `nested_recurrent_bridge.py` — maps the existing `true_recurrent_dynamics.FEATURES` schema into the additive layer.
- `nested_system1_experiment.py` — deterministic A/B/C/D drift-and-return smoke benchmark.
- `test_nested_adaptive_cognition.py` — memory-timescale, training-isolation, learning, and serialization tests.

## Memory bands

Default bands are intentionally separated by update frequency:

| band | update period | base LR | surprise threshold |
|---|---:|---:|---:|
| immediate | 1 tick | 0.35 | 0.00 |
| working | 4 ticks | 0.16 | 0.08 |
| episodic | 32 ticks | 0.07 | 0.18 |
| identity | 256 ticks | 0.025 | 0.30 |

Each band has a bounded plasticity multiplier. Surprise can raise or lower the multiplier, but it is clamped to prevent runaway self-modification.

## System-1 judgments

The default bank contains ten independent probabilistic heads:

- prediction_failure
- novel_event
- rewarding_state
- explore
- need_more_information
- teacher_needed
- sensory_conflict
- state_change
- memory_match
- action_success

The core API is deliberately typed and non-linguistic. Labels are optional and are only applied **after** the current-tick prediction has been produced, preventing same-tick target leakage.

## LLM teacher boundary

The layer does not call an LLM. It only exposes:

- `teacher_probability`
- `teacher_gate`

The host runtime decides whether and how to consult an external teacher. Teacher output should return through the normal sensory/learning path rather than directly overwriting recurrent state.

## Local verification

The module was compiled and the initial unit suite passed 6/6 tests before push.

A deterministic smoke benchmark (`seed=8776`, block size 300) produced:

| variant | accuracy | Brier ↓ | log loss ↓ | B-shift early accuracy | A-return early accuracy |
|---|---:|---:|---:|---:|---:|
| A baseline | 0.7800 | 0.1634 | 0.5002 | 0.1250 | 0.3750 |
| B calibrated System-1 | 0.7989 | 0.1515 | 0.4682 | 0.1750 | 0.4250 |
| C nested memory | 0.7933 | 0.1547 | 0.4788 | 0.1250 | 0.4000 |
| D nested + calibrated | 0.8156 | 0.1435 | 0.4475 | 0.2250 | 0.4500 |

These numbers only characterize the synthetic smoke task. They are not evidence for awareness, consciousness, or physiological validity.

## Run

```bash
python -m unittest -v test_nested_adaptive_cognition.py
python nested_system1_experiment.py --seed 8776 --block-size 300
```

## Next integration target

Wire `AdaptiveCognitionLayer` into the persistent D-RCS runtime so that:

1. raw sensory/recurrent features feed the layer every tick;
2. prediction error is supplied from the existing predictor;
3. real outcome labels train only the relevant heads;
4. continuum-memory state is saved in the existing checkpoint;
5. teacher escalation remains external and rate-limited;
6. baseline, System-1-only, nested-memory-only, and combined modes remain feature-gated for controlled A/B/C/D runs;
7. calibration, adaptation latency, catastrophic forgetting, action diversity, winner recurrence, and self-model stability are logged separately.
