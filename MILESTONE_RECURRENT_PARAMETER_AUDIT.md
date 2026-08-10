# Recurrent Parameter-Space Audit Milestone

This milestone follows the true-recurrent falsification phase and asks a narrower question:

> Does the deconfounded model contain a finite, stable region of parameter space in which recurrent dynamics — rather than a start-state-specific final feature target — preserve four distinct history-dependent state basins?

## Preregistered design

The audit froze its grid and pass/fail criteria before inspecting the sweep result.

The primary rollout target is deconfounded. For each `(alpha, chi)` endpoint, every start state converges to the same final feature vector. The earlier start-state-specific target is retained only as a positive control.

The 2,430-point parameter grid varies:

- state-memory scale: `0, 0.5, 1, 2, 4, 8`
- directed-transition scale: `0, 0.5, 1, 2, 4`
- rollout depth: `4, 8, 16`
- emission width: `1, 2, 4`
- softmax temperature: `0.10, 0.25, 0.50`
- loop-state coupling scale: `0, 1, 4`

Every configuration is compared with:

- full recurrence
- memory off
- previous-state history shuffled
- transition penalties zero
- loop-state coupling off
- recurrence fully off
- legacy prototype-target positive control

The pass gates reject both single-state collapse and trivial frozen-history solutions. In particular, no state may occupy more than 70% of the endpoint map, every state must occupy at least 5%, final-state entropy must exceed 1.2 bits, history mutual information must exceed 0.10 bits, recurrence must beat recurrence-off and shuffled-history controls by preregistered margins, and more than 10% of endpoints must leave their starting state.

Candidate points are then tested under ±5% and ±10% local parameter perturbations. Points surviving at least 70% of those perturbations are rerun on the full alpha/chi endpoint grid across five synthetic seeds and must pass at least four of five seeds.

## Result

**Classification: `robust_four_state_region`**

- 2,430 preregistered configurations tested
- 445 passed the initial deconfounded four-state gates
- 401 survived the local perturbation requirement
- 354 survived both perturbation testing and the full-grid, multi-seed requirement
- fully robust points occupy about **14.6% of the preregistered grid**
- robust points form two connected components containing **349** and **5** grid points

The result therefore is not an isolated tuned point.

## Representative robust point

For description only, the runner chooses the fully robust grid point nearest the prior baseline by grid-index distance:

`m2_t1_s4_w4_T0.25_l1`

Parameters:

- memory scale: `2.0`
- transition scale: `1.0`
- rollout depth: `4`
- emission width: `4.0`
- temperature: `0.25`
- loop-state coupling: `1.0`

At seed 42 on the full endpoint grid:

- full recurrent history MI: **0.487 bits**
- recurrence-off history MI: **0.000 bits**
- shuffled-history MI: **0.031 bits**
- final-state entropy: **1.553 bits**
- endpoint occupancy range: **9.1% to 59.8%**
- same-start/same-final fraction: **49.3%**

The representative itself passes four of five preregistered seeds and 95.2% of its local perturbations.

## Main parameter dependence

The clearest boundary is state-memory strength.

Fully robust point counts by memory scale:

- `0.0`: 0
- `0.5`: 0
- `1.0`: 6
- `2.0`: 86
- `4.0`: 204
- `8.0`: 58

Robust configurations occur at every tested transition scale, rollout depth, emission width, temperature, and loop-coupling value.

This means stronger state-memory inertia is the main requirement for the robust region in the current model family. Directed transition penalties and loop coupling modify the dynamics but are not individually necessary. There are fully robust configurations with transition scale zero, loop coupling zero, and even both set to zero.

## Relationship to the previous falsification result

The earlier `true_recurrent_dynamics.py` default parameterization showed only weak residual history dependence after prototype-target removal. This audit does not invalidate that result.

Instead, it shows that the **default point is outside the main robust region**, while a finite region appears when recurrent state-memory strength is increased. The important distinction is therefore:

- the original apparent four-basin result was strongly confounded by start-state-specific target geometry;
- the deconfounded default recurrence is weak;
- the broader deconfounded model family nevertheless contains a reproducible four-state recurrent region under stronger memory parameters.

## Interpretation boundary

This is a model-internal dynamical result, not physiological validation.

The synthetic generator defines state-dependent feature populations, and the recurrent coefficients are not yet constrained by empirical neural measurements. The next scientific question is therefore not whether a robust region exists — this audit says it does — but whether measured neural transition persistence, hysteresis, or state-duration statistics place biologically plausible parameters inside that region.

No result in this milestone should be used for diagnosis, treatment, anesthesia, consciousness assessment, or other medical decision-making.
