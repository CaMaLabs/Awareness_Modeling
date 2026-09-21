"""A/B/C/D smoke benchmark for nested memory + calibrated System-1 heads.

This is not a consciousness or awareness benchmark. It isolates the engineering
question introduced by the integration: do multiple plasticity timescales and
probability calibration change adaptation/retention on a recurring drifting
binary decision task?
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from nested_adaptive_cognition import AdaptiveCognitionLayer, brier_score, log_loss


@dataclass(frozen=True)
class Variant:
    name: str
    memory: bool
    calibrated: bool


VARIANTS = (
    Variant("A_baseline", memory=False, calibrated=False),
    Variant("B_calibrated_system1", memory=False, calibrated=True),
    Variant("C_nested_memory", memory=True, calibrated=False),
    Variant("D_nested_plus_calibrated", memory=True, calibrated=True),
)


def build_stream(seed: int, block_size: int) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    rows: List[Dict[str, float]] = []
    for block_index, regime in enumerate((1.0, -1.0, 1.0)):
        phase = ("A_initial", "B_shift", "A_return")[block_index]
        for i in range(block_size):
            signal = rng.uniform(-1.0, 1.0)
            cue = 0.45 * regime + rng.gauss(0.0, 0.65)
            latent = signal + 0.75 * regime + rng.gauss(0.0, 0.20)
            target = 1.0 if latent >= 0.0 else 0.0
            rows.append(
                {
                    "phase": phase,
                    "signal": signal,
                    "regime_cue": cue,
                    "target": target,
                    "phase_step": float(i),
                }
            )
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def run_variant(variant: Variant, stream: List[Dict[str, float]]) -> Dict[str, object]:
    layer = AdaptiveCognitionLayer(
        feature_names=("signal", "regime_cue"),
        judgment_names=("action_success", "teacher_needed"),
        enable_memory=variant.memory,
        enable_system1=True,
        calibrated_heads=variant.calibrated,
        teacher_threshold=0.72,
    )

    records: List[Dict[str, object]] = []
    previous_error = 0.5
    for row in stream:
        teacher_label = 1.0 if previous_error >= 0.45 else 0.0
        result = layer.step(
            {"signal": row["signal"], "regime_cue": row["regime_cue"]},
            prediction_error=previous_error,
            labels={
                "action_success": row["target"],
                "teacher_needed": teacher_label,
            },
        )
        p = float(result["judgments"]["action_success"])
        y = float(row["target"])
        prediction = 1.0 if p >= 0.5 else 0.0
        error = abs(p - y)
        records.append(
            {
                "phase": row["phase"],
                "phase_step": int(row["phase_step"]),
                "target": y,
                "probability": p,
                "correct": 1.0 if prediction == y else 0.0,
                "brier": brier_score(p, y),
                "log_loss": log_loss(p, y),
                "teacher_label": teacher_label,
                "teacher_gate": 1.0 if result["teacher_gate"] else 0.0,
                "novelty": float(result["intrinsic_novelty"]),
            }
        )
        previous_error = error

    phase_metrics: Dict[str, Dict[str, float]] = {}
    for phase in ("A_initial", "B_shift", "A_return"):
        phase_rows = [r for r in records if r["phase"] == phase]
        early = [r for r in phase_rows if int(r["phase_step"]) < 40]
        late = [r for r in phase_rows if int(r["phase_step"]) >= max(0, len(phase_rows) - 80)]
        phase_metrics[phase] = {
            "accuracy": mean(float(r["correct"]) for r in phase_rows),
            "brier": mean(float(r["brier"]) for r in phase_rows),
            "log_loss": mean(float(r["log_loss"]) for r in phase_rows),
            "early_accuracy": mean(float(r["correct"]) for r in early),
            "late_accuracy": mean(float(r["correct"]) for r in late),
        }

    tp = sum(1 for r in records if r["teacher_label"] == 1.0 and r["teacher_gate"] == 1.0)
    fp = sum(1 for r in records if r["teacher_label"] == 0.0 and r["teacher_gate"] == 1.0)
    fn = sum(1 for r in records if r["teacher_label"] == 1.0 and r["teacher_gate"] == 0.0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "variant": variant.name,
        "memory": variant.memory,
        "calibrated": variant.calibrated,
        "overall": {
            "accuracy": mean(float(r["correct"]) for r in records),
            "brier": mean(float(r["brier"]) for r in records),
            "log_loss": mean(float(r["log_loss"]) for r in records),
            "teacher_precision": precision,
            "teacher_recall": recall,
        },
        "phases": phase_metrics,
        "layer_metrics": layer.metrics(),
    }


def run(seed: int = 8776, block_size: int = 300) -> Dict[str, object]:
    stream = build_stream(seed=seed, block_size=block_size)
    results = [run_variant(variant, stream) for variant in VARIANTS]
    return {
        "seed": seed,
        "block_size": block_size,
        "samples": len(stream),
        "purpose": "mechanism smoke benchmark; not an awareness/consciousness measure",
        "results": results,
    }


def write_markdown(result: Dict[str, object], path: Path) -> None:
    rows = result["results"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Nested memory + System-1 smoke benchmark\n\n")
        f.write("This benchmark isolates adaptation, calibration, and return-to-regime retention. ")
        f.write("It is **not** evidence for awareness or consciousness.\n\n")
        f.write("| variant | accuracy | Brier ↓ | log loss ↓ | B-shift early acc. | A-return early acc. |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            overall = row["overall"]
            phases = row["phases"]
            f.write(
                f"| {row['variant']} | {overall['accuracy']:.4f} | {overall['brier']:.4f} | "
                f"{overall['log_loss']:.4f} | {phases['B_shift']['early_accuracy']:.4f} | "
                f"{phases['A_return']['early_accuracy']:.4f} |\n"
            )
        f.write("\nInterpret improvements only relative to this synthetic task. The full D-RCS runtime must be evaluated separately.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8776)
    parser.add_argument("--block-size", type=int, default=300)
    parser.add_argument("--json", default="nested_system1_results.json")
    parser.add_argument("--report", default="nested_system1_report.md")
    args = parser.parse_args()

    result = run(seed=args.seed, block_size=args.block_size)
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result, Path(args.report))
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
