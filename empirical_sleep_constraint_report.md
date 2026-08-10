# Empirical sleep constraint audit

**Classification:** `no_direct_temporal_match_current_driver`

This audit tests the previously identified robust synthetic recurrent configurations against external human sleep-stage dynamics without refitting them.

## Frozen direct-match gates

- mean row-wise transition TV <= 0.2
- dominant outgoing transition agrees for all four macro-states
- dwell-shape log-RMSE <= 0.5 after fitting one global minutes-per-model-step scale
- N3 is less arousable than both N2 and REM under one common wake-directed perturbation
- exploratory absolute arousal-rate RMSE <= 0.01

## Result

- robust configurations tested: 354
- full frozen direct passes: **0**
- transition-TV gate passes: 0/354
- all-four dominant-exit gate passes: 0/354
- dwell-shape gate passes: 0/354
- exploratory absolute-arousal gate passes: 0/354
- best transition TV: 0.260808 (`m4_t0_s16_w4_T0.5_l1`)
- best dwell-shape log-RMSE: 0.649580 (`m4_t4_s16_w4_T0.5_l0`)
- qualitative N3 arousal-resistance ordering: 343/354

Dominant outgoing-transition agreement counts:

- wake: 222/354
- n2: 73/354
- rem: 354/354
- n3: 115/354

## Empirical four-state reference

N1+N2 are collapsed to the model's `n2`; stages III+IV are collapsed to `n3`. Internal transitions inside a collapsed macro-state are removed before row normalization.

Mean exit matrix across two untreated 2011 control nights plus an independent 2008 healthy cohort:

| from \ to | wake | n2 | REM | n3 |
|---|---:|---:|---:|---:|
| wake | 0.000 | 0.893 | 0.103 | 0.005 |
| n2 | 0.283 | 0.000 | 0.206 | 0.511 |
| rem | 0.357 | 0.643 | 0.000 | 0.000 |
| n3 | 0.068 | 0.930 | 0.001 | 0.000 |

Approximate continuous-bout durations derived from the two untreated 2011 control nights:

- wake: 0.614 min
- n2: 3.138 min
- rem: 5.233 min
- n3: 1.233 min

## Interpretation

The synthetic recurrent region does not directly reproduce human temporal sleep-stage dynamics under the existing uniform alpha/chi probe ensemble. Because that ensemble is not an autonomous time-resolved sleep driver, this result identifies a temporal-driver/identifiability gap rather than by itself falsifying the existence of the deconfounded recurrent attractor region.

The largest structural mismatch is the N2 exit pattern: healthy human data favor N2→N3, whereas most robust synthetic configurations favor N2→REM under the current standardized probe ensemble. The N3→N2 preference is also absent in many configurations.

The auditory perturbation ordering is more encouraging: nearly all robust configurations make N3 less likely than N2/REM to move toward wake, consistent with the human arousal experiment. Absolute softmax probabilities are not event-rate calibrated and should not be treated as literal arousal percentages.

## Sources

- Kishi A et al. *NREM Sleep Stage Transitions Control Ultradian REM Sleep Rhythm*. SLEEP 2011;34(10):1423-1432. DOI: 10.5665/SLEEP.1292.
- Kishi A et al. *Dynamics of sleep stage transitions in healthy humans and patients with chronic fatigue syndrome*. Am J Physiol Regul Integr Comp Physiol. 2008;294:R1980-R1987. DOI: 10.1152/ajpregu.00925.2007.
- Vallat R et al. *Increased Evoked Potentials to Arousing Auditory Stimuli during Sleep: Implication for the Understanding of Dream Recall*. Front Hum Neurosci. 2017.
- Yetton BD et al. *Quantifying sleep architecture dynamics and individual differences using big data and Bayesian networks*. PLoS ONE. 2018;13:e0194604. DOI: 10.1371/journal.pone.0194604.
