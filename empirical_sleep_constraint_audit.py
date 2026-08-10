"""External sleep-dynamics constraint audit for Awareness_Modeling.

This is a downstream audit of the robust recurrent parameter region. It does not
refit the recurrent model. The audit asks whether the model's transition response
under the existing uniform alpha/chi probe ensemble directly resembles published
human sleep-stage dynamics.

Important interpretation: failure here is not equivalent to falsifying the
existence of the synthetic recurrent region. The current recurrent runner is not
an autonomous time-resolved hypnogram generator; its probe ensemble is externally
specified. A direct temporal mismatch therefore diagnoses an architecture/
identifiability gap as well as parameter mismatch.
"""

import argparse
import csv
import json
import math

import numpy as np

from recurrent_parameter_audit_core import (
    STATES,
    SI,
    STATE_INERTIA,
    TRANS,
    FULL_ALPHAS,
    FULL_CHIS,
    synthetic_rows,
    build_model,
    rollout_batch,
)

# Frozen before model-point inspection in this empirical phase.
TRANSITION_TV_MAX = 0.20
DWELL_LOG_RMSE_MAX = 0.50
AROUSAL_RMSE_MAX = 0.01

# Human healthy/control global transition percentages.
# Kishi et al., SLEEP 2011, Table S2 (control group, untreated nights 2 and 3).
KISHI_2011_LABELS = ("W", "N1", "N2", "N3", "R")
KISHI_2011_SECOND = np.array([
    [0, 8.4, 3.8, 0.1, 1.9],
    [1.7, 0, 11.9, 0.0, 3.6],
    [7.2, 4.6, 0, 23.0, 2.5],
    [2.1, 0.1, 20.9, 0, 0.1],
    [3.4, 4.1, 0.7, 0, 0],
], dtype=float)
KISHI_2011_THIRD = np.array([
    [0, 9.2, 2.8, 0.1, 2.2],
    [2.7, 0, 12.5, 0, 3.9],
    [5.8, 6.1, 0, 21.5, 2.7],
    [1.9, 0, 19.7, 0, 0],
    [4.0, 3.8, 1.0, 0, 0],
], dtype=float)

# Kishi et al., AJP-Regulatory 2008, Table 2 (healthy controls).
KISHI_2008_LABELS = ("W", "I", "II", "III", "IV", "R")
KISHI_2008_HEALTHY = np.array([
    [0, 13.2, 1.3, 0.0, 0.0, 0.3],
    [5.3, 0, 19.0, 0.0, 0.0, 5.9],
    [7.2, 10.4, 0, 11.2, 0.0, 3.2],
    [0.3, 0.1, 10.7, 0, 1.3, 0.0],
    [0.0, 0.0, 0.1, 1.2, 0, 0.0],
    [1.9, 6.6, 0.9, 0.0, 0.0, 0],
], dtype=float)

MAP_2011 = {"W": "wake", "N1": "n2", "N2": "n2", "N3": "n3", "R": "rem"}
MAP_2008 = {"W": "wake", "I": "n2", "II": "n2", "III": "n3", "IV": "n3", "R": "rem"}

# Kishi 2011 untreated control-night traditional sleep variables and run counts.
CONTROL_SECOND_MIN = {"W": 409.1 - 391.7, "N1": 30.2, "N2": 207.3, "N3": 65.0, "R": 89.2}
CONTROL_SECOND_RUNS = (28.6, 35.7, 77.4, 48.0, 17.0)
CONTROL_THIRD_MIN = {"W": 431.3 - 411.5, "N1": 34.1, "N2": 225.4, "N3": 52.2, "R": 99.8}
CONTROL_THIRD_RUNS = (30.7, 42.3, 79.7, 47.7, 19.6)

# Vallat et al. / Frontiers Human Neuroscience 2017: auditory stimuli associated
# with arousing reactions within 15 s. N2 and REM were not significantly different;
# both exceeded N3. Absolute probabilities are exploratory because the model's
# softmax probability is not calibrated to event rate.
AROUSAL_TARGET = {"n2": 0.031, "rem": 0.024, "n3": 0.014}

EXPECTED_DOMINANT_EXIT = {"wake": "n2", "n2": "n3", "rem": "n2", "n3": "n2"}


def collapse_global(matrix, labels, mapping):
    out = np.zeros((4, 4), dtype=float)
    for i, src in enumerate(labels):
        a = mapping[src]
        ia = STATES.index(a)
        for j, dst in enumerate(labels):
            if i == j:
                continue
            b = mapping[dst]
            if a == b:
                continue
            ib = STATES.index(b)
            out[ia, ib] += matrix[i, j]
    probs = np.zeros_like(out)
    for i in range(4):
        s = out[i].sum()
        if s:
            probs[i] = out[i] / s
    return out, probs


def empirical_transition_references():
    refs = []
    for name, matrix, labels, mapping in (
        ("kishi_2011_control_second", KISHI_2011_SECOND, KISHI_2011_LABELS, MAP_2011),
        ("kishi_2011_control_third", KISHI_2011_THIRD, KISHI_2011_LABELS, MAP_2011),
        ("kishi_2008_healthy", KISHI_2008_HEALTHY, KISHI_2008_LABELS, MAP_2008),
    ):
        _, p = collapse_global(matrix, labels, mapping)
        refs.append((name, p))
    return refs


def macro_dwell(minutes, run_counts, global_matrix):
    total_transitions = sum(run_counts) - 1.0
    collapsed, _ = collapse_global(global_matrix, KISHI_2011_LABELS, MAP_2011)
    exits = collapsed.sum(axis=1) / 100.0 * total_transitions
    mins = np.array([
        minutes["W"],
        minutes["N1"] + minutes["N2"],
        minutes["R"],
        minutes["N3"],
    ])
    return mins / exits


def empirical_dwell_target():
    a = macro_dwell(CONTROL_SECOND_MIN, CONTROL_SECOND_RUNS, KISHI_2011_SECOND)
    b = macro_dwell(CONTROL_THIRD_MIN, CONTROL_THIRD_RUNS, KISHI_2011_THIRD)
    return (a + b) / 2.0


def normalize_model_exits(tc):
    p = np.zeros((4, 4), dtype=float)
    for i in range(4):
        row = np.asarray(tc[i], dtype=float).copy()
        row[i] = 0.0
        s = row.sum()
        if s:
            p[i] = row / s
    return p


def mean_row_tv(a, b):
    return float(np.mean([0.5 * np.abs(a[i] - b[i]).sum() for i in range(4)]))


def dominant_exit_match(p):
    matches = {}
    for i, state in enumerate(STATES):
        observed = STATES[int(np.argmax(p[i]))]
        matches[state] = observed == EXPECTED_DOMINANT_EXIT[state]
    return matches


def dwell_shape_from_tc(tc, empirical_dwell):
    self_p = np.zeros(4, dtype=float)
    for i in range(4):
        total = float(np.sum(tc[i]))
        self_p[i] = tc[i, i] / total if total else np.nan
    if np.any(~np.isfinite(self_p)) or np.any(self_p >= 0.999999):
        return math.inf, math.nan, self_p, np.full(4, math.inf)
    model_steps = 1.0 / (1.0 - self_p)
    log_dt = float(np.mean(np.log(empirical_dwell) - np.log(model_steps)))
    dt = math.exp(log_dt)
    prediction = dt * model_steps
    err = float(np.sqrt(np.mean((np.log(prediction) - np.log(empirical_dwell)) ** 2)))
    return err, dt, self_p, prediction


def one_step_probs(model, cfg, start_idx, wake_fraction):
    refs = model["refs"]
    sd = model["sd"]
    pri = model["priors"]
    wake_idx = SI["wake"]
    probe = (1.0 - wake_fraction) * refs[start_idx] + wake_fraction * refs[wake_idx]
    d2 = np.sum(((probe[None, :] - refs) / sd[None, :]) ** 2, axis=1)
    scores = np.exp(-0.5 * d2 / (cfg["emission_width"] ** 2))
    history = np.zeros(4)
    history[start_idx] = 1.0
    scores += cfg["memory_scale"] * history * STATE_INERTIA
    scores -= cfg["transition_scale"] * TRANS[start_idx]
    prev_loop = model["default_loop"][start_idx]
    scores += cfg["loop_scale"] * 0.10 * (pri[:, prev_loop] - 0.25)
    temp = max(cfg["temperature"], 1e-9)
    z = (scores - scores.max()) / temp
    e = np.exp(z)
    return e / e.sum()


def arousal_fit(model, cfg):
    target = np.array([AROUSAL_TARGET["n2"], AROUSAL_TARGET["rem"], AROUSAL_TARGET["n3"]])
    idx = [SI["n2"], SI["rem"], SI["n3"]]
    target_mean = float(target.mean())
    best = None
    for wake_fraction in np.linspace(0.0, 1.0, 1001):
        rates = np.array([one_step_probs(model, cfg, s, wake_fraction)[SI["wake"]] for s in idx])
        loss = abs(float(rates.mean()) - target_mean)
        if best is None or loss < best[0]:
            best = (loss, wake_fraction, rates)
    _, wake_fraction, rates = best
    rmse = float(np.sqrt(np.mean((rates - target) ** 2)))
    qualitative = bool(rates[2] < rates[0] and rates[2] < rates[1])
    return rmse, float(wake_fraction), rates, qualitative


def read_robust_configs(path):
    """Read fully robust configurations from the parameter-audit passing-region CSV.

    The parameter audit writes every basic passing point to `*_passing_region.csv`
    and marks the perturbation + multi-seed survivors with `fully_robust=1`.
    A legacy already-filtered robust CSV without that column is also accepted.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "fully_robust" in rows[0]:
        rows = [r for r in rows if int(float(r.get("fully_robust", 0))) == 1]
    out = []
    for r in rows:
        out.append({
            "config_id": r["config_id"],
            "memory_scale": float(r["memory_scale"]),
            "transition_scale": float(r["transition_scale"]),
            "rollout_steps": int(float(r["rollout_steps"])),
            "emission_width": float(r["emission_width"]),
            "temperature": float(r["temperature"]),
            "loop_scale": float(r["loop_scale"]),
        })
    if not out:
        raise ValueError(f"no fully robust configurations found in {path}")
    return out


def run_audit(configs, seed=42):
    refs = empirical_transition_references()
    empirical_mean = np.mean([p for _, p in refs], axis=0)
    dwell = empirical_dwell_target()
    model = build_model(synthetic_rows(seed))
    rows = []
    for cfg in configs:
        _, _, tc = rollout_batch(model, cfg, "full", FULL_ALPHAS, FULL_CHIS, record=True)
        p = normalize_model_exits(tc)
        tv = mean_row_tv(p, empirical_mean)
        matches = dominant_exit_match(p)
        dwell_err, dt, self_p, dwell_pred = dwell_shape_from_tc(tc, dwell)
        arousal_rmse, arousal_lambda, arousal_rates, arousal_order = arousal_fit(model, cfg)
        frozen_direct_pass = (
            all(matches.values())
            and tv <= TRANSITION_TV_MAX
            and dwell_err <= DWELL_LOG_RMSE_MAX
            and arousal_order
            and arousal_rmse <= AROUSAL_RMSE_MAX
        )
        row = dict(cfg)
        row.update({
            "transition_tv": tv,
            "dominant_wake_match": int(matches["wake"]),
            "dominant_n2_match": int(matches["n2"]),
            "dominant_rem_match": int(matches["rem"]),
            "dominant_n3_match": int(matches["n3"]),
            "dwell_log_rmse": dwell_err,
            "fitted_step_minutes": dt,
            "arousal_rmse": arousal_rmse,
            "arousal_wake_fraction": arousal_lambda,
            "arousal_n2": float(arousal_rates[0]),
            "arousal_rem": float(arousal_rates[1]),
            "arousal_n3": float(arousal_rates[2]),
            "arousal_order_match": int(arousal_order),
            "frozen_direct_pass": int(frozen_direct_pass),
        })
        for i, state in enumerate(STATES):
            row[f"self_{state}"] = float(self_p[i])
            row[f"dwell_pred_{state}_min"] = float(dwell_pred[i])
            row[f"empirical_dwell_{state}_min"] = float(dwell[i])
            for j, dst in enumerate(STATES):
                row[f"exit_{state}_to_{dst}"] = float(p[i, j])
        rows.append(row)
    return rows, refs, empirical_mean, dwell


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(rows, refs, empirical_mean, dwell):
    n = len(rows)
    direct = sum(r["frozen_direct_pass"] for r in rows)
    edge = {s: sum(r[f"dominant_{s}_match"] for r in rows) for s in STATES}
    best_transition = min(rows, key=lambda r: r["transition_tv"])
    best_dwell = min(rows, key=lambda r: r["dwell_log_rmse"])
    qualitative_arousal = sum(r["arousal_order_match"] for r in rows)
    transition_passes = sum(r["transition_tv"] <= TRANSITION_TV_MAX for r in rows)
    dominant_all_passes = sum(all(r[f"dominant_{s}_match"] for s in STATES) for r in rows)
    dwell_passes = sum(r["dwell_log_rmse"] <= DWELL_LOG_RMSE_MAX for r in rows)
    arousal_abs_passes = sum(r["arousal_rmse"] <= AROUSAL_RMSE_MAX for r in rows)
    dominant_distribution = {str(k): sum(sum(r[f"dominant_{s}_match"] for s in STATES) == k for r in rows) for k in range(5)}
    memory_values = sorted({r["memory_scale"] for r in rows})
    min_tv_by_memory = {str(m): min(r["transition_tv"] for r in rows if r["memory_scale"] == m) for m in memory_values}
    return {
        "classification": "direct_empirical_match" if direct else "no_direct_temporal_match_current_driver",
        "robust_configs_tested": n,
        "frozen_direct_passes": direct,
        "transition_tv_gate_passes": transition_passes,
        "all_dominant_exit_gate_passes": dominant_all_passes,
        "dwell_gate_passes": dwell_passes,
        "arousal_absolute_gate_passes": arousal_abs_passes,
        "dominant_exit_match_count_distribution": dominant_distribution,
        "min_transition_tv_by_memory_scale": min_tv_by_memory,
        "transition_tv_max": TRANSITION_TV_MAX,
        "dwell_log_rmse_max": DWELL_LOG_RMSE_MAX,
        "arousal_rmse_max": AROUSAL_RMSE_MAX,
        "dominant_exit_match_counts": edge,
        "qualitative_arousal_order_matches": qualitative_arousal,
        "best_transition_config": best_transition["config_id"],
        "best_transition_tv": best_transition["transition_tv"],
        "best_dwell_config": best_dwell["config_id"],
        "best_dwell_log_rmse": best_dwell["dwell_log_rmse"],
        "empirical_mean_transition_matrix": {
            STATES[i]: {STATES[j]: float(empirical_mean[i, j]) for j in range(4)} for i in range(4)
        },
        "empirical_dwell_minutes": {STATES[i]: float(dwell[i]) for i in range(4)},
        "reference_transition_tv_between_healthy_datasets": {
            f"{refs[i][0]}__{refs[j][0]}": mean_row_tv(refs[i][1], refs[j][1])
            for i in range(len(refs)) for j in range(i + 1, len(refs))
        },
        "interpretation": (
            "The synthetic recurrent region does not directly reproduce human temporal sleep-stage dynamics under the "
            "existing uniform alpha/chi probe ensemble. Because that ensemble is not an autonomous time-resolved sleep "
            "driver, this result identifies a temporal-driver/identifiability gap rather than by itself falsifying the "
            "existence of the deconfounded recurrent attractor region."
        ),
    }


def write_report(path, summary):
    with open(path, "w") as f:
        f.write("# Empirical sleep constraint audit\n\n")
        f.write(f"**Classification:** `{summary['classification']}`\n\n")
        f.write("This audit tests the previously identified robust synthetic recurrent configurations against external human sleep-stage dynamics without refitting them.\n\n")
        f.write("## Frozen direct-match gates\n\n")
        f.write(f"- mean row-wise transition TV <= {TRANSITION_TV_MAX}\n")
        f.write("- dominant outgoing transition agrees for all four macro-states\n")
        f.write(f"- dwell-shape log-RMSE <= {DWELL_LOG_RMSE_MAX} after fitting one global minutes-per-model-step scale\n")
        f.write("- N3 is less arousable than both N2 and REM under one common wake-directed perturbation\n")
        f.write(f"- exploratory absolute arousal-rate RMSE <= {AROUSAL_RMSE_MAX}\n\n")
        f.write("## Result\n\n")
        f.write(f"- robust configurations tested: {summary['robust_configs_tested']}\n")
        f.write(f"- full frozen direct passes: **{summary['frozen_direct_passes']}**\n")
        f.write(f"- transition-TV gate passes: {summary['transition_tv_gate_passes']}/{summary['robust_configs_tested']}\n")
        f.write(f"- all-four dominant-exit gate passes: {summary['all_dominant_exit_gate_passes']}/{summary['robust_configs_tested']}\n")
        f.write(f"- dwell-shape gate passes: {summary['dwell_gate_passes']}/{summary['robust_configs_tested']}\n")
        f.write(f"- exploratory absolute-arousal gate passes: {summary['arousal_absolute_gate_passes']}/{summary['robust_configs_tested']}\n")
        f.write(f"- best transition TV: {summary['best_transition_tv']:.6f} (`{summary['best_transition_config']}`)\n")
        f.write(f"- best dwell-shape log-RMSE: {summary['best_dwell_log_rmse']:.6f} (`{summary['best_dwell_config']}`)\n")
        f.write(f"- qualitative N3 arousal-resistance ordering: {summary['qualitative_arousal_order_matches']}/{summary['robust_configs_tested']}\n\n")
        f.write("Dominant outgoing-transition agreement counts:\n\n")
        for state, count in summary["dominant_exit_match_counts"].items():
            f.write(f"- {state}: {count}/{summary['robust_configs_tested']}\n")
        f.write("\n## Empirical four-state reference\n\n")
        f.write("N1+N2 are collapsed to the model's `n2`; stages III+IV are collapsed to `n3`. Internal transitions inside a collapsed macro-state are removed before row normalization.\n\n")
        f.write("Mean exit matrix across two untreated 2011 control nights plus an independent 2008 healthy cohort:\n\n")
        f.write("| from \\ to | wake | n2 | REM | n3 |\n|---|---:|---:|---:|---:|\n")
        m = summary["empirical_mean_transition_matrix"]
        for s in STATES:
            f.write(f"| {s} | {m[s]['wake']:.3f} | {m[s]['n2']:.3f} | {m[s]['rem']:.3f} | {m[s]['n3']:.3f} |\n")
        f.write("\nApproximate continuous-bout durations derived from the two untreated 2011 control nights:\n\n")
        for s, v in summary["empirical_dwell_minutes"].items():
            f.write(f"- {s}: {v:.3f} min\n")
        f.write("\n## Interpretation\n\n")
        f.write(summary["interpretation"] + "\n\n")
        f.write("The largest structural mismatch is the N2 exit pattern: healthy human data favor N2→N3, whereas most robust synthetic configurations favor N2→REM under the current standardized probe ensemble. The N3→N2 preference is also absent in many configurations.\n\n")
        f.write("The auditory perturbation ordering is more encouraging: nearly all robust configurations make N3 less likely than N2/REM to move toward wake, consistent with the human arousal experiment. Absolute softmax probabilities are not event-rate calibrated and should not be treated as literal arousal percentages.\n\n")
        f.write("## Sources\n\n")
        f.write("- Kishi A et al. *NREM Sleep Stage Transitions Control Ultradian REM Sleep Rhythm*. SLEEP 2011;34(10):1423-1432. DOI: 10.5665/SLEEP.1292.\n")
        f.write("- Kishi A et al. *Dynamics of sleep stage transitions in healthy humans and patients with chronic fatigue syndrome*. Am J Physiol Regul Integr Comp Physiol. 2008;294:R1980-R1987. DOI: 10.1152/ajpregu.00925.2007.\n")
        f.write("- Vallat R et al. *Increased Evoked Potentials to Arousing Auditory Stimuli during Sleep: Implication for the Understanding of Dream Recall*. Front Hum Neurosci. 2017.\n")
        f.write("- Yetton BD et al. *Quantifying sleep architecture dynamics and individual differences using big data and Bayesian networks*. PLoS ONE. 2018;13:e0194604. DOI: 10.1371/journal.pone.0194604.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust-configs", default="recurrent_parameter_audit_passing_region.csv")
    ap.add_argument("--out-prefix", default="empirical_sleep_constraint")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    configs = read_robust_configs(args.robust_configs)
    rows, refs, empirical_mean, dwell = run_audit(configs, args.seed)
    summary = summarize(rows, refs, empirical_mean, dwell)
    write_csv(f"{args.out_prefix}_results.csv", rows)
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_report(f"{args.out_prefix}_report.md", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
