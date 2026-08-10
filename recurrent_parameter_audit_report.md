# Recurrent parameter-space audit

The parameter grid and pass/fail gates were frozen before result inspection. Primary rollouts use a deconfounded common final feature target; the legacy prototype-preserving target is retained only as a positive control.

## Result

**Classification: `robust_four_state_region`**

- preregistered configurations: 2430
- basic passing configurations: 445
- locally perturbation-robust configurations: 401
- fully robust configurations (local perturbations + >=4/5 full-grid seeds): 354 (14.57% of the full grid)
- connected robust components: 2 with sizes [349, 5]

## Representative robust point

The descriptive representative is the fully robust grid point nearest the prior baseline by grid-index distance: `m2_t1_s4_w4_T0.25_l1`. This selection does not determine pass/fail.

- memory_scale: 2.0
- transition_scale: 1.0
- rollout_steps: 4
- emission_width: 4.0
- temperature: 0.25
- loop_scale: 1.0

At seed 42 on the full endpoint grid, this point gives 0.487 bits history MI versus 0.000 with recurrence fully off and 0.031 with shuffled history. Final occupancy spans 9.1% to 59.8%, and 49.3% of endpoints retain their starting state.

## Main parameter dependence

The robust region is dominated by stronger state-memory inertia: no fully robust configurations occur at memory scale 0 or 0.5; only a few occur at 1; most occur at 2-4. Robust points span every tested transition scale, rollout depth, emission width, temperature, and loop-coupling level. This means the existence of the robust region is primarily controlled by state-memory strength, while directed transition penalties and loop coupling shape the region but are not individually necessary.

## Preregistered guardrails

- `min_state_occupancy`: 0.05
- `max_state_occupancy`: 0.7
- `min_final_entropy_bits`: 1.2
- `min_history_mi_bits`: 0.1
- `min_mi_gain_vs_recurrence_off`: 0.08
- `min_mi_gain_vs_shuffled`: 0.05
- `min_mean_pairwise_js_bits`: 0.03
- `max_same_state_fraction`: 0.9
- `min_away_transition_fraction`: 0.1
- `min_seed_pass_fraction`: 0.8
- `min_perturb_pass_fraction`: 0.7
- `min_robust_component_size`: 3

Frozen initial-label memory locks (>90% same-state endpoints) and single-state collapses are explicitly rejected even if history MI is high.
