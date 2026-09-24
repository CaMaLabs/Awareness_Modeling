"""Feature-gated adapter for persistent D-RCS runtime snapshots.

This module deliberately does not import or replace the D-RCS runtime. It accepts
plain snapshot/event dictionaries from the existing persistent runtime and feeds
an additive nested cognition layer. The host runtime remains authoritative for
recurrent state, action execution, checkpointing, teacher I/O, and sensory
processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Mapping, Optional, Sequence

from nested_adaptive_cognition import AdaptiveCognitionLayer, DEFAULT_JUDGMENTS


FEATURE_MODE_A = "A"
FEATURE_MODE_B = "B"
FEATURE_MODE_C = "C"
FEATURE_MODE_D = "D"


def mode_flags(mode: str) -> Dict[str, bool]:
    mode = mode.upper()
    if mode == FEATURE_MODE_A:
        return {"enable_layer": False, "enable_memory": False, "enable_system1": False, "calibrated": False}
    if mode == FEATURE_MODE_B:
        return {"enable_layer": True, "enable_memory": False, "enable_system1": True, "calibrated": True}
    if mode == FEATURE_MODE_C:
        return {"enable_layer": True, "enable_memory": True, "enable_system1": False, "calibrated": False}
    if mode == FEATURE_MODE_D:
        return {"enable_layer": True, "enable_memory": True, "enable_system1": True, "calibrated": True}
    raise ValueError(f"unknown nested D-RCS feature mode: {mode!r}")


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else value


def _stream_hash(name: str) -> float:
    total = 0
    for char in name:
        total = (total * 131 + ord(char)) % 1000003
    return (total / 1000003.0) * 2.0 - 1.0


@dataclass
class NestedDRCSConfig:
    mode: str = FEATURE_MODE_D
    max_streams: int = 8
    teacher_threshold: float = 0.72
    feature_names: Sequence[str] = field(default_factory=tuple)


class DRCSSnapshotFeatureExtractor:
    def __init__(self, *, max_streams: int = 8):
        self.max_streams = int(max_streams)
        if self.max_streams <= 0:
            raise ValueError("max_streams must be positive")
        self.stream_slots: Dict[str, int] = {}

    def feature_names(self) -> tuple[str, ...]:
        names = [
            "status.mean_prediction_error",
            "status.self_model_stability",
            "status.winner_recurrence",
            "status.recent_memory_continuity",
            "status.pending_feedback",
            "status.sensory_stream_count",
            "action.mean_prediction_error",
            "sensory.conflict",
        ]
        for index in range(self.max_streams):
            prefix = f"stream{index}"
            names.extend(
                [
                    f"{prefix}.identity_hash",
                    f"{prefix}.novelty",
                    f"{prefix}.prediction_error",
                    f"{prefix}.reconstruction_error",
                    f"{prefix}.drift",
                    f"{prefix}.prototype_count",
                ]
            )
        return tuple(names)

    def extract(self, snapshot: Mapping[str, object]) -> Dict[str, float]:
        status = snapshot.get("status") if isinstance(snapshot.get("status"), Mapping) else {}
        action = snapshot.get("action_loop") if isinstance(snapshot.get("action_loop"), Mapping) else {}
        sensory = snapshot.get("sensory_cortex") if isinstance(snapshot.get("sensory_cortex"), Mapping) else {}
        streams = sensory.get("streams") if isinstance(sensory.get("streams"), Mapping) else {}

        stream_names = sorted(str(name) for name in streams.keys())
        for name in stream_names:
            if name not in self.stream_slots and len(self.stream_slots) < self.max_streams:
                self.stream_slots[name] = len(self.stream_slots)

        prediction_errors = []
        novelties = []
        for name in stream_names:
            row = streams.get(name, {})
            if isinstance(row, Mapping):
                prediction_errors.append(_safe_float(row.get("prediction_error")))
                novelties.append(_safe_float(row.get("novelty")))
        conflict = _conflict(prediction_errors) + 0.5 * _conflict(novelties)

        features = {
            "status.mean_prediction_error": _safe_float(status.get("mean_prediction_error")),
            "status.self_model_stability": _safe_float(status.get("self_model_stability")),
            "status.winner_recurrence": _safe_float(status.get("winner_recurrence")),
            "status.recent_memory_continuity": _safe_float(status.get("recent_memory_continuity")),
            "status.pending_feedback": _clip(_safe_float(action.get("pending_feedback")) / 8.0, 0.0, 1.0),
            "status.sensory_stream_count": _clip(_safe_float(status.get("sensory_stream_count")) / max(1.0, self.max_streams), 0.0, 1.0),
            "action.mean_prediction_error": _safe_float(action.get("mean_prediction_error")),
            "sensory.conflict": _clip(conflict, 0.0, 1.0),
        }

        for index in range(self.max_streams):
            prefix = f"stream{index}"
            features[f"{prefix}.identity_hash"] = 0.0
            features[f"{prefix}.novelty"] = 0.0
            features[f"{prefix}.prediction_error"] = 0.0
            features[f"{prefix}.reconstruction_error"] = 0.0
            features[f"{prefix}.drift"] = 0.0
            features[f"{prefix}.prototype_count"] = 0.0

        for name, slot in self.stream_slots.items():
            if slot >= self.max_streams:
                continue
            row = streams.get(name, {})
            if not isinstance(row, Mapping):
                continue
            prefix = f"stream{slot}"
            features[f"{prefix}.identity_hash"] = _stream_hash(name)
            features[f"{prefix}.novelty"] = _safe_float(row.get("novelty"))
            features[f"{prefix}.prediction_error"] = _safe_float(row.get("prediction_error"))
            features[f"{prefix}.reconstruction_error"] = _safe_float(row.get("reconstruction_error"))
            features[f"{prefix}.drift"] = _safe_float(row.get("drift"))
            features[f"{prefix}.prototype_count"] = _clip(_safe_float(row.get("prototype_count")) / 16.0, 0.0, 1.0)

        return features

    def state_dict(self) -> Dict[str, object]:
        return {"max_streams": self.max_streams, "stream_slots": dict(self.stream_slots)}

    @classmethod
    def from_state_dict(cls, data: Mapping[str, object] | None) -> "DRCSSnapshotFeatureExtractor":
        if not isinstance(data, Mapping):
            return cls()
        extractor = cls(max_streams=int(data.get("max_streams", 8)))
        raw_slots = data.get("stream_slots", {})
        if isinstance(raw_slots, Mapping):
            extractor.stream_slots = {str(k): int(v) for k, v in raw_slots.items()}
        return extractor


class NestedDRCSRuntimeAdapter:
    def __init__(self, config: NestedDRCSConfig = NestedDRCSConfig()):
        self.config = config
        self.flags = mode_flags(config.mode)
        self.extractor = DRCSSnapshotFeatureExtractor(max_streams=config.max_streams)
        self.layer: Optional[AdaptiveCognitionLayer] = None
        if self.flags["enable_layer"]:
            names = tuple(config.feature_names) or self.extractor.feature_names()
            self.layer = AdaptiveCognitionLayer(
                feature_names=names,
                judgment_names=DEFAULT_JUDGMENTS,
                enable_memory=self.flags["enable_memory"],
                enable_system1=self.flags["enable_system1"],
                calibrated_heads=self.flags["calibrated"],
                teacher_threshold=config.teacher_threshold,
            )
        self.tick = 0
        self.latest: Dict[str, object] = {"mode": config.mode, "enabled": self.flags["enable_layer"]}
        self.teacher_log: list[Dict[str, object]] = []

    def step(
        self,
        snapshot: Mapping[str, object],
        *,
        outcome_labels: Optional[Mapping[str, float]] = None,
        reward: float = 0.0,
        learn: bool = True,
    ) -> Dict[str, object]:
        self.tick += 1
        if self.layer is None:
            self.latest = {"mode": self.config.mode, "enabled": False, "tick": self.tick}
            return self.latest

        if outcome_labels:
            self.layer.learn_from_labels(outcome_labels)

        features = self.extractor.extract(snapshot)
        prediction_error = _safe_float(features.get("status.mean_prediction_error"))
        result = self.layer.step(
            features,
            prediction_error=prediction_error,
            reward=reward,
            labels=None,
            learn=learn,
        )
        teacher_event = None
        if result.get("teacher_gate"):
            teacher_event = {
                "tick": self.tick,
                "probability": result.get("teacher_probability", 0.0),
                "reason": result.get("teacher_reason", ""),
            }
            self.teacher_log.append(teacher_event)
            self.teacher_log[:] = self.teacher_log[-128:]
        self.latest = {
            "mode": self.config.mode,
            "enabled": True,
            "tick": self.tick,
            "judgments": result.get("judgments", {}),
            "teacher_probability": result.get("teacher_probability", 0.0),
            "teacher_gate": result.get("teacher_gate", False),
            "teacher_reason": result.get("teacher_reason", ""),
            "memory_updates": result.get("memory_updates", {}),
            "memory_readout": result.get("memory_readout", []),
            "sensory_stream_slots": dict(self.extractor.stream_slots),
            "teacher_event": teacher_event,
        }
        return self.latest

    def telemetry(self) -> Dict[str, object]:
        if self.layer is None:
            return dict(self.latest)
        return {
            **self.latest,
            "metrics": self.layer.metrics(),
            "teacher_log": list(self.teacher_log),
            "extractor": self.extractor.state_dict(),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "config": {
                "mode": self.config.mode,
                "max_streams": self.config.max_streams,
                "teacher_threshold": self.config.teacher_threshold,
                "feature_names": list(self.config.feature_names),
            },
            "tick": self.tick,
            "extractor": self.extractor.state_dict(),
            "layer": self.layer.state_dict() if self.layer else None,
            "latest": self.latest,
            "teacher_log": list(self.teacher_log),
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, object] | None) -> "NestedDRCSRuntimeAdapter":
        if not isinstance(data, Mapping):
            return cls()
        raw_config = data.get("config", {})
        if not isinstance(raw_config, Mapping):
            raw_config = {}
        config = NestedDRCSConfig(
            mode=str(raw_config.get("mode", FEATURE_MODE_D)),
            max_streams=int(raw_config.get("max_streams", 8)),
            teacher_threshold=float(raw_config.get("teacher_threshold", 0.72)),
            feature_names=tuple(str(x) for x in raw_config.get("feature_names", []) or ()),
        )
        adapter = cls(config)
        adapter.tick = int(data.get("tick", 0))
        adapter.extractor = DRCSSnapshotFeatureExtractor.from_state_dict(data.get("extractor"))
        if adapter.flags["enable_layer"]:
            names = tuple(config.feature_names) or adapter.extractor.feature_names()
            adapter.layer = AdaptiveCognitionLayer.safe_from_state_dict(
                data.get("layer") if isinstance(data.get("layer"), Mapping) else None,
                feature_names=names,
                enable_memory=adapter.flags["enable_memory"],
                enable_system1=adapter.flags["enable_system1"],
                calibrated_heads=adapter.flags["calibrated"],
            )
        raw_latest = data.get("latest", {})
        adapter.latest = dict(raw_latest) if isinstance(raw_latest, Mapping) else {}
        raw_teacher = data.get("teacher_log", [])
        adapter.teacher_log = [dict(item) for item in raw_teacher if isinstance(item, Mapping)] if isinstance(raw_teacher, list) else []
        return adapter


def _conflict(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
