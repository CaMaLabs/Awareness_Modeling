import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict

INPUT = "balanced_states.csv"
OUT_PREFIX = "true_recurrent"

FEATURES = [
    "peak_alpha_hz",
    "L_eff_m",
    "v_eff_m_per_s",
    "tau_eff_s",
    "pl_local_aw",
    "pl_global_aw",
    "cfc",
    "w_prop",
    "s_struct",
]
STATES = ["wake", "n2", "rem", "n3"]
LOOPS = ["thalamo_cortical", "fronto_parietal", "dmn", "ct_cingulate"]

STATE_INERTIA = {"wake": 0.18, "n2": 0.08, "rem": 0.10, "n3": 0.28}
TRANSITION_PENALTY = {
    ("wake", "n2"): 0.02,
    ("n2", "wake"): 0.015,
    ("n2", "n3"): 0.01,
    ("n3", "n2"): 0.03,
    ("wake", "rem"): 0.015,
    ("rem", "wake"): 0.02,
}
DEFAULT_TRANSITION_PENALTY = 0.015

ALPHA_MIN, ALPHA_MAX, ALPHA_STEP = 0.5, 14.0, 0.5
CHI_MIN, CHI_MAX, CHI_STEP = 0.20, 0.80, 0.04

# A fixed derangement makes the shuffled-history falsification deterministic.
SHUFFLED_STATE = {"wake": "n3", "n3": "rem", "rem": "n2", "n2": "wake"}

CONDITIONS = [
    "legacy_target_positive_control",
    "prototype_target_removed",
    "memory_off",
    "previous_state_shuffled",
    "transition_penalties_zero",
]


def safe_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    names = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)


def validate_rows(rows):
    if not rows:
        raise ValueError("input contains no rows")
    missing = [k for k in ["state", *FEATURES] if k not in rows[0]]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    present = {str(r.get("state", "")).lower() for r in rows}
    absent = [s for s in STATES if s not in present]
    if absent:
        raise ValueError(f"input is missing state rows for: {absent}")


def mean_feature_row(rows):
    return {k: statistics.mean(safe_float(r.get(k)) for r in rows) for k in FEATURES}


def build_state_refs(rows):
    refs = {}
    for state in STATES:
        subset = [r for r in rows if str(r.get("state", "")).lower() == state]
        refs[state] = mean_feature_row(subset)
    return refs


def build_norm_stats(rows):
    """Normalize from the full sample, not only four state centroids."""
    stats = {}
    for k in FEATURES:
        vals = [safe_float(r.get(k)) for r in rows]
        stats[k] = (statistics.mean(vals), statistics.pstdev(vals) or 1.0)
    return stats


def build_loop_priors(rows, pseudocount=1.0):
    counts = {s: Counter() for s in STATES}
    for r in rows:
        state = str(r.get("state", "")).lower()
        loop = str(r.get("candidate_loop", ""))
        if state in counts and loop in LOOPS:
            counts[state][loop] += 1

    priors = {}
    for state in STATES:
        total = sum(counts[state].values()) + pseudocount * len(LOOPS)
        priors[state] = {
            loop: (counts[state][loop] + pseudocount) / total for loop in LOOPS
        }
    return priors


def standardized_distance(row, ref, norm_stats):
    total = 0.0
    for k in FEATURES:
        _, sd = norm_stats[k]
        total += ((safe_float(row.get(k)) - ref[k]) / sd) ** 2
    return math.sqrt(total)


def emission_similarity(row, ref, norm_stats, width=2.0):
    """Gaussian centroid similarity on a transparent 0..1 scale."""
    d = standardized_distance(row, ref, norm_stats)
    return math.exp(-0.5 * (d / max(width, 1e-9)) ** 2)


def transition_cost(prev_state, state):
    if prev_state is None or prev_state == state:
        return 0.0
    return TRANSITION_PENALTY.get((prev_state, state), DEFAULT_TRANSITION_PENALTY)


def softmax(scores, temperature):
    temperature = max(temperature, 1e-9)
    peak = max(scores.values())
    exp_scores = {k: math.exp((v - peak) / temperature) for k, v in scores.items()}
    denom = sum(exp_scores.values())
    return {k: v / denom for k, v in exp_scores.items()}


def blend_features(a, b, t):
    return {k: a[k] * (1.0 - t) + b[k] * t for k in FEATURES}


def make_target_row(base, alpha_target, chi_target):
    row = dict(base)
    tau = max(1e-9, safe_float(row["tau_eff_s"]))
    v = max(1e-9, safe_float(row["v_eff_m_per_s"]))
    row["peak_alpha_hz"] = alpha_target
    row["L_eff_m"] = chi_target * v * tau
    return row


def bin_centers(lo, hi, step):
    n = int((hi - lo) / step)
    return [lo + (i + 0.5) * step for i in range(n)]


def shuffled_history(prev_state, prev_probs):
    shuffled_probs = {s: 0.0 for s in STATES}
    for state, prob in prev_probs.items():
        shuffled_probs[SHUFFLED_STATE[state]] = prob
    return SHUFFLED_STATE[prev_state], shuffled_probs


def choose_loop(state, loop_priors, prev_loop=None, stickiness=0.05):
    """Loop identity is a readout/context variable; it does not decide the state."""
    best_loop = None
    best_score = None
    for loop in LOOPS:
        score = math.log(max(loop_priors[state][loop], 1e-12))
        if prev_loop == loop:
            score += stickiness
        if best_score is None or score > best_score:
            best_loop = loop
            best_score = score
    return best_loop


def condition_config(name):
    if name == "legacy_target_positive_control":
        return {"target_mode": "legacy", "memory": True, "shuffle": False, "transitions": True}
    if name == "prototype_target_removed":
        return {"target_mode": "neutral", "memory": True, "shuffle": False, "transitions": True}
    if name == "memory_off":
        return {"target_mode": "neutral", "memory": False, "shuffle": False, "transitions": True}
    if name == "previous_state_shuffled":
        return {"target_mode": "neutral", "memory": True, "shuffle": True, "transitions": True}
    if name == "transition_penalties_zero":
        return {"target_mode": "neutral", "memory": True, "shuffle": False, "transitions": False}
    raise ValueError(f"unknown condition: {name}")


def rollout_to_target(
    start_state,
    alpha_target,
    chi_target,
    refs,
    global_ref,
    norm_stats,
    loop_priors,
    condition,
    rollout_steps=8,
    temperature=0.20,
    emission_width=2.0,
    memory_scale=1.0,
    transition_scale=1.0,
):
    cfg = condition_config(condition)
    start = dict(refs[start_state])
    target_base = refs[start_state] if cfg["target_mode"] == "legacy" else global_ref
    target = make_target_row(target_base, alpha_target, chi_target)

    prev_probs = {s: (1.0 if s == start_state else 0.0) for s in STATES}
    prev_state = start_state
    prev_loop = choose_loop(start_state, loop_priors)

    final_energy = None
    final_emission = None

    for step in range(rollout_steps):
        # Crucial deconfounding detail: interpolate from the original start to the
        # target directly. At the last step, neutral-target probes are identical
        # across all start states for a given (alpha, chi) cell.
        t = (step + 1) / rollout_steps
        probe = blend_features(start, target, t)

        history_state = prev_state
        history_probs = prev_probs
        if cfg["shuffle"]:
            history_state, history_probs = shuffled_history(prev_state, prev_probs)

        energies = {}
        emissions = {}
        for state in STATES:
            emission = emission_similarity(probe, refs[state], norm_stats, emission_width)
            energy = emission
            if cfg["memory"]:
                energy += memory_scale * STATE_INERTIA[state] * history_probs[state]
            if cfg["transitions"]:
                energy -= transition_scale * transition_cost(history_state, state)
            energies[state] = energy
            emissions[state] = emission

        probs = softmax(energies, temperature)
        next_state = max(STATES, key=lambda s: probs[s])
        next_loop = choose_loop(next_state, loop_priors, prev_loop=prev_loop)

        prev_probs = probs
        prev_state = next_state
        prev_loop = next_loop
        final_energy = energies[next_state]
        final_emission = emissions[next_state]

    return {
        "final_state": prev_state,
        "final_loop": prev_loop,
        "final_probability": prev_probs[prev_state],
        "final_energy": final_energy,
        "final_emission": final_emission,
    }


def mutual_information_bits(rows):
    starts = Counter(r["start_state"] for r in rows)
    finals = Counter(r["final_state"] for r in rows)
    joint = Counter((r["start_state"], r["final_state"]) for r in rows)
    total = len(rows)
    mi = 0.0
    for (start, final), n in joint.items():
        pxy = n / total
        px = starts[start] / total
        py = finals[final] / total
        mi += pxy * math.log2(pxy / (px * py))
    return mi


def summarize_condition(condition, rows):
    counts = Counter(r["final_state"] for r in rows)
    persistence = sum(r["start_state"] == r["final_state"] for r in rows) / len(rows)
    return {
        "condition": condition,
        "n": len(rows),
        "history_mi_bits": round(mutual_information_bits(rows), 9),
        "same_state_fraction": round(persistence, 9),
        "wake_final_fraction": round(counts["wake"] / len(rows), 9),
        "n2_final_fraction": round(counts["n2"] / len(rows), 9),
        "rem_final_fraction": round(counts["rem"] / len(rows), 9),
        "n3_final_fraction": round(counts["n3"] / len(rows), 9),
    }


def summarize_start(condition, start_state, rows):
    counts = Counter(r["final_state"] for r in rows)
    n = len(rows)
    return {
        "condition": condition,
        "start_state": start_state,
        "n": n,
        "wake": counts["wake"],
        "n2": counts["n2"],
        "rem": counts["rem"],
        "n3": counts["n3"],
        "wake_fraction": round(counts["wake"] / n, 6),
        "n2_fraction": round(counts["n2"] / n, 6),
        "rem_fraction": round(counts["rem"] / n, 6),
        "n3_fraction": round(counts["n3"] / n, 6),
    }


def run_experiment(
    rows,
    rollout_steps=8,
    temperature=0.20,
    emission_width=2.0,
    memory_scale=1.0,
    transition_scale=1.0,
):
    validate_rows(rows)
    refs = build_state_refs(rows)
    global_ref = mean_feature_row(rows)
    norm_stats = build_norm_stats(rows)
    loop_priors = build_loop_priors(rows)

    alphas = bin_centers(ALPHA_MIN, ALPHA_MAX, ALPHA_STEP)
    chis = bin_centers(CHI_MIN, CHI_MAX, CHI_STEP)

    rollouts = []
    for condition in CONDITIONS:
        for start_state in STATES:
            for alpha in alphas:
                for chi in chis:
                    result = rollout_to_target(
                        start_state=start_state,
                        alpha_target=alpha,
                        chi_target=chi,
                        refs=refs,
                        global_ref=global_ref,
                        norm_stats=norm_stats,
                        loop_priors=loop_priors,
                        condition=condition,
                        rollout_steps=rollout_steps,
                        temperature=temperature,
                        emission_width=emission_width,
                        memory_scale=memory_scale,
                        transition_scale=transition_scale,
                    )
                    rollouts.append({
                        "condition": condition,
                        "start_state": start_state,
                        "alpha_center_hz": round(alpha, 4),
                        "chi_center": round(chi, 4),
                        "final_state": result["final_state"],
                        "final_loop": result["final_loop"],
                        "final_probability": round(result["final_probability"], 9),
                        "final_energy": round(result["final_energy"], 9),
                        "final_emission": round(result["final_emission"], 9),
                    })

    summary = []
    start_summary = []
    for condition in CONDITIONS:
        subset = [r for r in rollouts if r["condition"] == condition]
        summary.append(summarize_condition(condition, subset))
        for start_state in STATES:
            by_start = [r for r in subset if r["start_state"] == start_state]
            start_summary.append(summarize_start(condition, start_state, by_start))

    return rollouts, summary, start_summary


def write_report(path, summary, start_summary, args):
    by_condition = {r["condition"]: r for r in summary}
    legacy_mi = by_condition["legacy_target_positive_control"]["history_mi_bits"]
    deconf_mi = by_condition["prototype_target_removed"]["history_mi_bits"]
    retention = deconf_mi / legacy_mi if legacy_mi > 0 else 0.0

    with open(path, "w") as f:
        f.write("# True recurrent dynamics falsification report\n\n")
        f.write("The legacy scripts are not modified by this experiment.\n\n")
        f.write("## Parameters\n\n")
        f.write(f"- rollout_steps: {args.rollout_steps}\n")
        f.write(f"- temperature: {args.temperature}\n")
        f.write(f"- emission_width: {args.emission_width}\n")
        f.write(f"- memory_scale: {args.memory_scale}\n")
        f.write(f"- transition_scale: {args.transition_scale}\n\n")
        f.write("## Condition-level results\n\n")
        f.write("| condition | history MI (bits) | same-state fraction | wake | N2 | REM | N3 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in summary:
            f.write(
                f"| {r['condition']} | {r['history_mi_bits']:.6f} | "
                f"{r['same_state_fraction']:.6f} | {r['wake_final_fraction']:.4f} | "
                f"{r['n2_final_fraction']:.4f} | {r['rem_final_fraction']:.4f} | "
                f"{r['n3_final_fraction']:.4f} |\n"
            )

        f.write("\n## Deconfounding diagnostic\n\n")
        f.write(
            "`legacy_target_positive_control` retains the old start-state-specific target construction. "
            "`prototype_target_removed` instead makes the final feature probe identical across start states "
            "for each alpha/chi cell. Therefore any remaining start-state dependence in that condition must "
            "come from path/history terms rather than a different final feature vector.\n\n"
        )
        f.write(f"- legacy history MI: {legacy_mi:.9f} bits\n")
        f.write(f"- deconfounded history MI: {deconf_mi:.9f} bits\n")
        f.write(f"- MI retained after prototype-target removal: {retention:.3%}\n\n")

        f.write("## Start-state occupancy\n\n")
        for condition in CONDITIONS:
            f.write(f"### {condition}\n\n")
            f.write("| start | wake | N2 | REM | N3 |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for r in start_summary:
                if r["condition"] != condition:
                    continue
                f.write(
                    f"| {r['start_state']} | {r['wake']} | {r['n2']} | {r['rem']} | {r['n3']} |\n"
                )
            f.write("\n")

        f.write("## Interpretation guardrail\n\n")
        f.write(
            "A large history MI in the legacy-target positive control is not evidence of recurrent dynamics by itself, "
            "because the final feature vector still depends on the start state. The deconfounded condition is the relevant "
            "test. If its history MI is small and collapses further under memory-off, shuffled-history, or zero-transition "
            "ablations, the defensible conclusion is that recurrence contributes a measurable but weak residual effect. "
            "Do not tune parameters solely to recover the original four-basin occupancy.\n"
        )


def main():
    ap = argparse.ArgumentParser(description="True recurrent state-selection and falsification suite")
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--out-prefix", default=OUT_PREFIX)
    ap.add_argument("--rollout-steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.20)
    ap.add_argument("--emission-width", type=float, default=2.0)
    ap.add_argument("--memory-scale", type=float, default=1.0)
    ap.add_argument("--transition-scale", type=float, default=1.0)
    args = ap.parse_args()

    rows = read_rows(args.input)
    rollouts, summary, start_summary = run_experiment(
        rows,
        rollout_steps=args.rollout_steps,
        temperature=args.temperature,
        emission_width=args.emission_width,
        memory_scale=args.memory_scale,
        transition_scale=args.transition_scale,
    )

    write_csv(f"{args.out_prefix}_rollouts.csv", rollouts)
    write_csv(f"{args.out_prefix}_summary.csv", summary)
    write_csv(f"{args.out_prefix}_by_start.csv", start_summary)
    write_report(f"{args.out_prefix}_report.md", summary, start_summary, args)

    print("=== TRUE RECURRENT FALSIFICATION ===")
    for row in summary:
        print(
            f"{row['condition']}: MI={row['history_mi_bits']:.6f} bits, "
            f"same_state={row['same_state_fraction']:.4f}"
        )
    print(f"Saved: {args.out_prefix}_rollouts.csv")
    print(f"Saved: {args.out_prefix}_summary.csv")
    print(f"Saved: {args.out_prefix}_by_start.csv")
    print(f"Saved: {args.out_prefix}_report.md")


if __name__ == "__main__":
    main()
