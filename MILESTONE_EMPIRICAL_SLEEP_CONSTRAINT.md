# Empirical Sleep-Dynamics Constraint Milestone

This milestone tests the previously identified deconfounded, perturbation-stable recurrent region against external human sleep-stage dynamics **without refitting the 354 robust synthetic configurations**.

## Question

Do any model configurations that passed the synthetic recurrent parameter audit also reproduce coarse human sleep-stage transition structure, relative bout persistence, and a simple sleep-arousal ordering under the model's existing uniform alpha/chi probe driver?

## Empirical references

The direct temporal constraints are derived from published healthy-human sleep-stage dynamics:

- Kishi et al. (2011), untreated control nights: whole-night transition frequencies plus stage durations and continuous-run counts.
- Kishi et al. (2008), independent healthy cohort: whole-night sleep-stage transition frequencies and stage-duration distribution structure.
- Vallat et al. (2017): auditory arousal probabilities in N2, REM, and N3, used primarily as a qualitative perturbation-ordering check.

For compatibility with the repository's four-state model, N1+N2 are collapsed to `n2`; stages III+IV are collapsed to `n3` where required. Transitions internal to a collapsed macro-state are removed before row normalization.

## Frozen direct-match gates

The numerical criteria were fixed before checking which robust model configurations survived:

1. mean row-wise transition total-variation error <= **0.20**
2. dominant outgoing transition agrees for **all four** macro-states
3. dwell-shape log-RMSE <= **0.50** after fitting only one global minutes-per-model-step scale
4. N3 is less arousable than both N2 and REM under one common wake-directed perturbation
5. exploratory absolute arousal-rate RMSE <= **0.01**

The pre-existing shuffled-history and synthetic robustness requirements are not relaxed.

## Result

**Classification: `no_direct_temporal_match_current_driver`**

- robust synthetic configurations tested: **354**
- full frozen direct passes: **0**
- transition-TV passes: **0/354**
- four-of-four dominant-exit matches: **0/354**
- dwell-shape passes: **0/354**
- exploratory absolute arousal-rate passes: **0/354**
- qualitative N3 arousal-resistance ordering: **343/354**

Best transition fit:

- configuration: `m4_t0_s16_w4_T0.5_l1`
- mean row-wise TV error: **0.260808**
- frozen ceiling: **0.20**

Best dwell-shape fit:

- configuration: `m4_t4_s16_w4_T0.5_l0`
- log-RMSE: **0.649580**
- frozen ceiling: **0.50**

The number of correctly matched dominant outgoing edges per configuration was:

- 1/4 edges: **63** configurations
- 2/4 edges: **172** configurations
- 3/4 edges: **119** configurations
- 4/4 edges: **0** configurations

## Where the mismatch occurs

Dominant-exit agreement across the 354 robust configurations:

- wake: **222/354**
- N2: **73/354**
- REM: **354/354**
- N3: **115/354**

The strongest systematic mismatch is N2. The collapsed healthy-human reference favors **N2 -> N3**, while most robust synthetic configurations favor **N2 -> REM** under the current standardized probe ensemble.

REM is the cleanest agreement: every robust configuration sends REM predominantly toward N2. N3 is mixed; healthy data overwhelmingly favor N3 -> N2, while many model configurations instead favor wake or REM.

The nearest transition fit occurs at memory scale 4, but this is **not** an empirical estimate of the model's memory parameter. Minimum transition-TV error by tested memory scale is:

- memory 1: 0.3681
- memory 2: 0.3294
- memory 4: 0.2608
- memory 8: 0.2849

None meets the preregistered transition gate.

## Interpretation

This result does **not** erase the previous parameter-audit result. The synthetic model family still contains a finite region that satisfies the deconfounded recurrent, perturbation, and multi-seed gates.

What fails is the stronger step of treating those synthetic recurrent parameters as directly constrained by human temporal sleep architecture.

The current recurrent rollout is an attractor/classifier response to an externally specified alpha/chi probe path. It is not an autonomous hypnogram generator with empirically calibrated hazards, circadian/homeostatic drive, or stage-duration survival functions. Therefore spontaneous human transition probabilities and bout durations are not expected to identify a unique state-memory coefficient under the current architecture.

The direct-match null result should therefore be interpreted as a **temporal-driver / parameter-identifiability gap**.

The auditory perturbation result is partially encouraging: 343/354 robust configurations reproduce the qualitative ordering that N3 is harder to push toward wake than N2 or REM. However, the model softmax is not calibrated as an event-rate probability, so its absolute values must not be interpreted as literal arousal percentages.

## Consequence for the next model phase

The next useful extension is not a larger recurrence-strength sweep. It is a temporal generative layer that can be independently constrained by human recordings, for example:

- state-specific dwell-time / survival hazards
- at least second-order transition history
- circadian and homeostatic modulation
- a defined coupling between those temporal drivers and the existing recurrent attractor energy

Only after that layer reproduces held-out hypnogram statistics should the recurrent memory parameter be mapped to physiological persistence or hysteresis.

## Reproducibility

Run the parameter audit first so `recurrent_parameter_audit_passing_region.csv` exists, then:

```bash
python empirical_sleep_constraint_audit.py \
  --robust-configs recurrent_parameter_audit_passing_region.csv

python -m unittest -v test_empirical_sleep_constraint_audit.py
```

Primary outputs:

- `empirical_sleep_constraint_results.csv`
- `empirical_sleep_constraint_summary.json`
- `empirical_sleep_constraint_report.md`

## Status

The model has now passed through three distinct claims:

1. historical four-state recurrent appearance — **confounded**
2. deconfounded synthetic recurrent parameter region — **robust within the tested model family**
3. direct empirical temporal calibration to healthy-human sleep architecture — **not supported under the current driver**

That separation should be preserved in future interpretation.
