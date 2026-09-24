"""HOPE-inspired nested plasticity + calibrated System-1 judgments.

This module is deliberately additive: it does not replace the existing recurrent
state model. A host runtime supplies a numeric feature vector, prediction error,
and (optionally) delayed outcome labels. The layer maintains several surprise-
gated memory bands with different update periods and a bank of online binary
judgment heads whose probabilities are temperature calibrated.

The implementation is inspired by the principles of Google's Nested Learning /
HOPE work (multiple update timescales, self-modifying plasticity) and TypeSafe's
Jev interface (fast typed probabilistic decisions). It is not a reproduction of
either proprietary or research implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_JUDGMENTS = (
    "prediction_failure",
    "novel_event",
    "rewarding_state",
    "explore",
    "need_more_information",
    "teacher_needed",
    "sensory_conflict",
    "state_change",
    "memory_match",
    "action_success",
)


def _clip(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def brier_score(probability: float, target: float) -> float:
    target = 1.0 if target >= 0.5 else 0.0
    return (probability - target) ** 2


def log_loss(probability: float, target: float, eps: float = 1e-9) -> float:
    target = 1.0 if target >= 0.5 else 0.0
    p = _clip(probability, eps, 1.0 - eps)
    return -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p))


def vector_magnitude(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(float(v) * float(v) for v in values) / len(values))


@dataclass
class CalibrationBin:
    count: int = 0
    probability_sum: float = 0.0
    positive_sum: float = 0.0

    def observe(self, probability: float, target: float) -> None:
        self.count += 1
        self.probability_sum += _clip(float(probability), 0.0, 1.0)
        self.positive_sum += 1.0 if float(target) >= 0.5 else 0.0

    def state_dict(self) -> Dict[str, object]:
        return {
            "count": self.count,
            "probability_sum": self.probability_sum,
            "positive_sum": self.positive_sum,
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, object]) -> "CalibrationBin":
        return cls(
            count=int(data.get("count", 0)),
            probability_sum=float(data.get("probability_sum", 0.0)),
            positive_sum=float(data.get("positive_sum", 0.0)),
        )

    def metrics(self) -> Dict[str, float]:
        if not self.count:
            return {
                "count": 0.0,
                "mean_probability": 0.0,
                "empirical_rate": 0.0,
            }
        return {
            "count": float(self.count),
            "mean_probability": self.probability_sum / self.count,
            "empirical_rate": self.positive_sum / self.count,
        }


@dataclass
class TeacherGateState:
    """Bounded escalation state for an external teacher.

    The layer never calls a teacher and never applies teacher output directly.
    It only exposes whether escalation is currently permitted and why.
    """

    threshold: float = 0.72
    cooldown_ticks: int = 128
    min_persistent_ticks: int = 3
    cooldown_remaining: int = 0
    persistent_count: int = 0
    total_consultations: int = 0
    last_reason: str = ""
    last_tick: int = 0
    last_probability: float = 0.0

    def evaluate(
        self,
        *,
        tick: int,
        probability: float,
        prediction_error: float,
        novelty: float,
        sensory_conflict: float = 0.0,
        failed_exploration: float = 0.0,
    ) -> Tuple[bool, str]:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        probability = _clip(float(probability), 0.0, 1.0)
        reasons = []
        if probability >= self.threshold:
            reasons.append("teacher_probability")
        if abs(float(prediction_error)) >= 0.55:
            reasons.append("high_prediction_error")
        if float(novelty) >= 0.55:
            reasons.append("high_novelty")
        if float(sensory_conflict) >= 0.50:
            reasons.append("sensory_conflict")
        if float(failed_exploration) >= 0.50:
            reasons.append("failed_exploration")

        escalatable = probability >= self.threshold and len(reasons) >= 2
        self.persistent_count = self.persistent_count + 1 if escalatable else 0
        allowed = (
            escalatable
            and self.persistent_count >= self.min_persistent_ticks
            and self.cooldown_remaining == 0
        )
        reason = ",".join(reasons) if reasons else "none"
        self.last_tick = int(tick)
        self.last_probability = probability
        self.last_reason = reason
        if allowed:
            self.total_consultations += 1
            self.cooldown_remaining = max(0, int(self.cooldown_ticks))
        return allowed, reason

    def state_dict(self) -> Dict[str, object]:
        return {
            "threshold": self.threshold,
            "cooldown_ticks": self.cooldown_ticks,
            "min_persistent_ticks": self.min_persistent_ticks,
            "cooldown_remaining": self.cooldown_remaining,
            "persistent_count": self.persistent_count,
            "total_consultations": self.total_consultations,
            "last_reason": self.last_reason,
            "last_tick": self.last_tick,
            "last_probability": self.last_probability,
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, object] | None, *, threshold: float) -> "TeacherGateState":
        if not isinstance(data, Mapping):
            return cls(threshold=threshold)
        return cls(
            threshold=float(data.get("threshold", threshold)),
            cooldown_ticks=int(data.get("cooldown_ticks", 128)),
            min_persistent_ticks=int(data.get("min_persistent_ticks", 3)),
            cooldown_remaining=int(data.get("cooldown_remaining", 0)),
            persistent_count=int(data.get("persistent_count", 0)),
            total_consultations=int(data.get("total_consultations", 0)),
            last_reason=str(data.get("last_reason", "")),
            last_tick=int(data.get("last_tick", 0)),
            last_probability=float(data.get("last_probability", 0.0)),
        )


@dataclass
class MemoryBand:
    """One memory/plasticity timescale.

    ``update_every`` is measured in host ticks. ``meta_rate`` changes the band's
    own plasticity based on surprise, giving the layer a small self-modifying
    component without allowing unbounded learning-rate growth.
    """

    name: str
    width: int
    update_every: int
    learning_rate: float
    surprise_threshold: float
    meta_rate: float = 0.001
    min_plasticity: float = 0.25
    max_plasticity: float = 2.0
    state: List[float] = field(default_factory=list)
    plasticity: float = 1.0
    updates: int = 0
    observations: int = 0
    latest_write_surprise: float = 0.0
    last_write_tick: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.update_every <= 0:
            raise ValueError("update_every must be positive")
        if not self.state:
            self.state = [0.0] * self.width
        if len(self.state) != self.width:
            raise ValueError("state width mismatch")

    def should_update(self, tick: int, surprise: float) -> bool:
        return tick % self.update_every == 0 and surprise >= self.surprise_threshold

    def observe(self, vector: Sequence[float], tick: int, surprise: float) -> bool:
        if len(vector) != self.width:
            raise ValueError("memory observation width mismatch")
        self.observations += 1
        surprise = _clip(abs(float(surprise)), 0.0, 1.0)
        if not self.should_update(tick, surprise):
            return False

        effective_lr = _clip(
            self.learning_rate * self.plasticity * max(surprise, 0.05), 0.0, 1.0
        )
        for i, value in enumerate(vector):
            target = float(value)
            self.state[i] += effective_lr * (target - self.state[i])

        centered = surprise - self.surprise_threshold
        self.plasticity = _clip(
            self.plasticity + self.meta_rate * centered,
            self.min_plasticity,
            self.max_plasticity,
        )
        self.updates += 1
        self.latest_write_surprise = surprise
        self.last_write_tick = int(tick)
        return True

    def telemetry(self, current_tick: int) -> Dict[str, float]:
        return {
            "writes": float(self.updates),
            "update_frequency": float(self.updates) / max(1.0, float(current_tick)),
            "plasticity_multiplier": float(self.plasticity),
            "state_magnitude": vector_magnitude(self.state),
            "latest_write_surprise": float(self.latest_write_surprise),
            "ticks_since_latest_write": float(max(0, int(current_tick) - self.last_write_tick))
            if self.last_write_tick
            else float(current_tick),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "width": self.width,
            "update_every": self.update_every,
            "learning_rate": self.learning_rate,
            "surprise_threshold": self.surprise_threshold,
            "meta_rate": self.meta_rate,
            "min_plasticity": self.min_plasticity,
            "max_plasticity": self.max_plasticity,
            "state": list(self.state),
            "plasticity": self.plasticity,
            "updates": self.updates,
            "observations": self.observations,
            "latest_write_surprise": self.latest_write_surprise,
            "last_write_tick": self.last_write_tick,
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, object]) -> "MemoryBand":
        band = cls(
            name=str(data["name"]),
            width=int(data["width"]),
            update_every=int(data["update_every"]),
            learning_rate=float(data["learning_rate"]),
            surprise_threshold=float(data["surprise_threshold"]),
            meta_rate=float(data.get("meta_rate", 0.001)),
            min_plasticity=float(data.get("min_plasticity", 0.25)),
            max_plasticity=float(data.get("max_plasticity", 2.0)),
            state=[float(v) for v in data.get("state", [])],
        )
        band.plasticity = float(data.get("plasticity", 1.0))
        band.updates = int(data.get("updates", 0))
        band.observations = int(data.get("observations", band.updates))
        band.latest_write_surprise = float(data.get("latest_write_surprise", 0.0))
        band.last_write_tick = int(data.get("last_write_tick", 0))
        return band


class ContinuumMemory:
    """Surprise-gated memory with multiple update frequencies."""

    DEFAULT_BANDS = (
        ("immediate", 1, 0.35, 0.00, 0.0100),
        ("working", 4, 0.16, 0.08, 0.0050),
        ("episodic", 32, 0.07, 0.18, 0.0015),
        ("identity", 256, 0.025, 0.30, 0.0005),
    )

    def __init__(self, width: int, bands: Optional[Iterable[MemoryBand]] = None):
        self.width = int(width)
        if self.width <= 0:
            raise ValueError("width must be positive")
        self.bands = list(bands) if bands is not None else [
            MemoryBand(name, self.width, every, lr, threshold, meta)
            for name, every, lr, threshold, meta in self.DEFAULT_BANDS
        ]
        if not self.bands:
            raise ValueError("at least one memory band is required")
        if any(b.width != self.width for b in self.bands):
            raise ValueError("all memory bands must use the same width")
        self.tick = 0

    def observe(self, vector: Sequence[float], surprise: float) -> Dict[str, bool]:
        self.tick += 1
        return {
            band.name: band.observe(vector, self.tick, surprise)
            for band in self.bands
        }

    def read(self) -> List[float]:
        raw_weights = [1.0 / math.sqrt(i + 1.0) for i in range(len(self.bands))]
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights]
        return [
            sum(weight * band.state[i] for weight, band in zip(weights, self.bands))
            for i in range(self.width)
        ]

    def novelty(self, vector: Sequence[float]) -> float:
        if len(vector) != self.width:
            raise ValueError("novelty vector width mismatch")
        readout = self.read()
        mse = sum((float(v) - readout[i]) ** 2 for i, v in enumerate(vector)) / self.width
        return _clip(math.sqrt(mse), 0.0, 1.0)

    def state_dict(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "tick": self.tick,
            "bands": [band.state_dict() for band in self.bands],
        }

    def telemetry(self) -> Dict[str, Dict[str, float]]:
        return {band.name: band.telemetry(self.tick) for band in self.bands}

    @classmethod
    def from_state_dict(cls, data: Mapping[str, object]) -> "ContinuumMemory":
        bands = [MemoryBand.from_state_dict(x) for x in data["bands"]]
        memory = cls(int(data["width"]), bands=bands)
        memory.tick = int(data.get("tick", 0))
        return memory


@dataclass
class CalibratedBinaryHead:
    """Small online logistic head with online temperature calibration."""

    name: str
    width: int
    learning_rate: float = 0.03
    calibration_rate: float = 0.005
    l2: float = 1e-5
    weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    log_temperature: float = 0.0
    updates: int = 0
    cumulative_brier: float = 0.0
    cumulative_log_loss: float = 0.0
    bins: List[CalibrationBin] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [0.0] * self.width
        if len(self.weights) != self.width:
            raise ValueError("head width mismatch")
        if not self.bins:
            self.bins = [CalibrationBin() for _ in range(10)]

    @property
    def temperature(self) -> float:
        return math.exp(self.log_temperature)

    def raw_logit(self, vector: Sequence[float]) -> float:
        if len(vector) != self.width:
            raise ValueError("judgment input width mismatch")
        return self.bias + sum(w * float(x) for w, x in zip(self.weights, vector))

    def predict(self, vector: Sequence[float]) -> float:
        return _sigmoid(self.raw_logit(vector) / max(self.temperature, 1e-6))

    def update(self, vector: Sequence[float], target: float) -> Dict[str, float]:
        y = 1.0 if float(target) >= 0.5 else 0.0
        raw = self.raw_logit(vector)
        temp = max(self.temperature, 1e-6)
        calibrated_logit = raw / temp
        p = _sigmoid(calibrated_logit)

        error = p - y
        grad_raw = error / temp
        for i, x in enumerate(vector):
            self.weights[i] -= self.learning_rate * (
                grad_raw * float(x) + self.l2 * self.weights[i]
            )
        self.bias -= self.learning_rate * grad_raw

        grad_log_t = -error * calibrated_logit
        self.log_temperature = _clip(
            self.log_temperature - self.calibration_rate * grad_log_t,
            -2.0,
            2.0,
        )

        brier = brier_score(p, y)
        ll = log_loss(p, y)
        self.cumulative_brier += brier
        self.cumulative_log_loss += ll
        bin_index = min(len(self.bins) - 1, max(0, int(p * len(self.bins))))
        self.bins[bin_index].observe(p, y)
        self.updates += 1
        return {"probability": p, "brier": brier, "log_loss": ll}

    def metrics(self) -> Dict[str, float]:
        bin_metrics = [item.metrics() for item in self.bins]
        ece = 0.0
        for row in bin_metrics:
            if self.updates:
                ece += (row["count"] / self.updates) * abs(
                    row["mean_probability"] - row["empirical_rate"]
                )
        if not self.updates:
            return {
                "updates": 0.0,
                "mean_brier": 0.0,
                "mean_log_loss": 0.0,
                "temperature": self.temperature,
                "expected_calibration_error": 0.0,
                "bins": bin_metrics,
            }
        return {
            "updates": float(self.updates),
            "mean_brier": self.cumulative_brier / self.updates,
            "mean_log_loss": self.cumulative_log_loss / self.updates,
            "temperature": self.temperature,
            "expected_calibration_error": ece,
            "bins": bin_metrics,
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "width": self.width,
            "learning_rate": self.learning_rate,
            "calibration_rate": self.calibration_rate,
            "l2": self.l2,
            "weights": list(self.weights),
            "bias": self.bias,
            "log_temperature": self.log_temperature,
            "updates": self.updates,
            "cumulative_brier": self.cumulative_brier,
            "cumulative_log_loss": self.cumulative_log_loss,
            "bins": [item.state_dict() for item in self.bins],
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, object]) -> "CalibratedBinaryHead":
        head = cls(
            name=str(data["name"]),
            width=int(data["width"]),
            learning_rate=float(data.get("learning_rate", 0.03)),
            calibration_rate=float(data.get("calibration_rate", 0.005)),
            l2=float(data.get("l2", 1e-5)),
            weights=[float(v) for v in data.get("weights", [])],
            bias=float(data.get("bias", 0.0)),
            log_temperature=float(data.get("log_temperature", 0.0)),
        )
        head.updates = int(data.get("updates", 0))
        head.cumulative_brier = float(data.get("cumulative_brier", 0.0))
        head.cumulative_log_loss = float(data.get("cumulative_log_loss", 0.0))
        raw_bins = data.get("bins", [])
        if isinstance(raw_bins, list) and raw_bins:
            head.bins = [CalibrationBin.from_state_dict(item) for item in raw_bins]
        return head


class SystemOneBank:
    """Typed fast judgments over a shared context vector."""

    def __init__(
        self,
        width: int,
        names: Sequence[str] = DEFAULT_JUDGMENTS,
        calibrated: bool = True,
    ):
        self.width = int(width)
        self.calibrated = bool(calibrated)
        calibration_rate = 0.005 if self.calibrated else 0.0
        self.heads = {
            name: CalibratedBinaryHead(
                name=name, width=self.width, calibration_rate=calibration_rate
            )
            for name in names
        }

    def predict(self, vector: Sequence[float]) -> Dict[str, float]:
        return {name: head.predict(vector) for name, head in self.heads.items()}

    def update(
        self, vector: Sequence[float], labels: Mapping[str, float]
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for name, target in labels.items():
            head = self.heads.get(name)
            if head is not None:
                results[name] = head.update(vector, target)
        return results

    def metrics(self) -> Dict[str, Dict[str, float]]:
        return {name: head.metrics() for name, head in self.heads.items()}

    def state_dict(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "calibrated": self.calibrated,
            "heads": {name: head.state_dict() for name, head in self.heads.items()},
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, object]) -> "SystemOneBank":
        heads_data = data["heads"]
        bank = cls(
            width=int(data["width"]),
            names=list(heads_data.keys()),
            calibrated=bool(data.get("calibrated", True)),
        )
        bank.heads = {
            str(name): CalibratedBinaryHead.from_state_dict(head_data)
            for name, head_data in heads_data.items()
        }
        return bank


class AdaptiveCognitionLayer:
    """Additive D-RCS integration layer.

    Call ``step`` once per host tick. Judgments are computed before any label
    update, so logged calibration metrics are out-of-sample for that tick.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        judgment_names: Sequence[str] = DEFAULT_JUDGMENTS,
        enable_memory: bool = True,
        enable_system1: bool = True,
        calibrated_heads: bool = True,
        teacher_threshold: float = 0.72,
    ):
        self.feature_names = tuple(feature_names)
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty")
        self.enable_memory = bool(enable_memory)
        self.enable_system1 = bool(enable_system1)
        self.teacher_threshold = float(teacher_threshold)
        self.tick = 0

        n = len(self.feature_names)
        self.memory = ContinuumMemory(n)
        self.context_width = 2 * n + 3
        self.system1 = SystemOneBank(
            self.context_width, judgment_names, calibrated=calibrated_heads
        )
        self.teacher_gate = TeacherGateState(threshold=self.teacher_threshold)
        self.pending_training_context: Optional[List[float]] = None
        self.prediction_log: List[Dict[str, object]] = []

    def _vectorize(self, features: Mapping[str, float]) -> List[float]:
        return [math.tanh(float(features.get(name, 0.0))) for name in self.feature_names]

    def _context(
        self,
        raw: Sequence[float],
        prediction_error: float,
        reward: float,
        novelty: float,
    ) -> List[float]:
        memory = self.memory.read() if self.enable_memory else [0.0] * len(raw)
        return [
            *raw,
            *memory,
            _clip(float(prediction_error), -1.0, 1.0),
            _clip(float(reward), -1.0, 1.0),
            _clip(float(novelty), 0.0, 1.0),
        ]

    def step(
        self,
        features: Mapping[str, float],
        prediction_error: float,
        reward: float = 0.0,
        labels: Optional[Mapping[str, float]] = None,
        learn: bool = True,
    ) -> Dict[str, object]:
        self.tick += 1
        raw = self._vectorize(features)
        novelty = self.memory.novelty(raw) if self.enable_memory else 0.0
        context = self._context(raw, prediction_error, reward, novelty)
        judgments = self.system1.predict(context) if self.enable_system1 else {}

        teacher_probability = float(judgments.get("teacher_needed", 0.0))
        sensory_conflict = float(judgments.get("sensory_conflict", 0.0))
        teacher_gate, teacher_reason = self.teacher_gate.evaluate(
            tick=self.tick,
            probability=teacher_probability if self.enable_system1 else 0.0,
            prediction_error=prediction_error,
            novelty=novelty,
            sensory_conflict=sensory_conflict,
        )
        teacher_gate = self.enable_system1 and teacher_gate

        previous_training_context = (
            list(self.pending_training_context)
            if self.pending_training_context is not None
            else None
        )
        self.prediction_log.append(
            {
                "tick": self.tick,
                "judgments": dict(judgments),
                "teacher_gate": teacher_gate,
                "teacher_reason": teacher_reason,
            }
        )
        self.prediction_log[:] = self.prediction_log[-256:]

        head_updates: Dict[str, Dict[str, float]] = {}
        memory_updates: Dict[str, bool] = {}
        if learn:
            if self.enable_memory:
                memory_updates = self.memory.observe(raw, abs(float(prediction_error)))
            if self.enable_system1 and labels and previous_training_context is not None:
                head_updates = self.system1.update(previous_training_context, labels)
        self.pending_training_context = list(context)

        return {
            "tick": self.tick,
            "judgments": judgments,
            "teacher_probability": teacher_probability,
            "teacher_gate": teacher_gate,
            "teacher_reason": teacher_reason,
            "intrinsic_novelty": novelty,
            "memory_readout": self.memory.read() if self.enable_memory else [0.0] * len(raw),
            "memory_updates": memory_updates,
            "head_updates": head_updates,
        }

    def learn_from_labels(
        self,
        labels: Mapping[str, float],
        *,
        context: Optional[Sequence[float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Train selected heads from an already-logged prediction context."""

        if not self.enable_system1 or not labels:
            return {}
        selected = list(context) if context is not None else self.pending_training_context
        if selected is None:
            return {}
        return self.system1.update(selected, labels)

    def metrics(self) -> Dict[str, object]:
        return {
            "tick": self.tick,
            "memory": {
                band.name: band.telemetry(self.memory.tick)
                for band in self.memory.bands
            },
            "judgments": self.system1.metrics(),
            "teacher_gate": self.teacher_gate.state_dict(),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "enable_memory": self.enable_memory,
            "enable_system1": self.enable_system1,
            "teacher_threshold": self.teacher_threshold,
            "tick": self.tick,
            "memory": self.memory.state_dict(),
            "system1": self.system1.state_dict(),
            "teacher_gate": self.teacher_gate.state_dict(),
            "pending_training_context": list(self.pending_training_context or []),
            "prediction_log": list(self.prediction_log[-64:]),
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, object]) -> "AdaptiveCognitionLayer":
        system1_data = data["system1"]
        layer = cls(
            feature_names=[str(x) for x in data["feature_names"]],
            judgment_names=list(system1_data["heads"].keys()),
            enable_memory=bool(data.get("enable_memory", True)),
            enable_system1=bool(data.get("enable_system1", True)),
            calibrated_heads=bool(system1_data.get("calibrated", True)),
            teacher_threshold=float(data.get("teacher_threshold", 0.72)),
        )
        layer.tick = int(data.get("tick", 0))
        layer.memory = ContinuumMemory.from_state_dict(data["memory"])
        layer.system1 = SystemOneBank.from_state_dict(system1_data)
        layer.teacher_gate = TeacherGateState.from_state_dict(
            data.get("teacher_gate"), threshold=layer.teacher_threshold
        )
        pending = data.get("pending_training_context", [])
        if isinstance(pending, list) and pending:
            layer.pending_training_context = [float(value) for value in pending]
        raw_log = data.get("prediction_log", [])
        if isinstance(raw_log, list):
            layer.prediction_log = [dict(item) for item in raw_log if isinstance(item, Mapping)]
        return layer

    @classmethod
    def safe_from_state_dict(
        cls,
        data: Mapping[str, object] | None,
        *,
        feature_names: Optional[Sequence[str]] = None,
        judgment_names: Sequence[str] = DEFAULT_JUDGMENTS,
        enable_memory: bool = True,
        enable_system1: bool = True,
        calibrated_heads: bool = True,
    ) -> "AdaptiveCognitionLayer":
        try:
            if isinstance(data, Mapping):
                return cls.from_state_dict(data)
        except Exception:
            pass
        if feature_names is None:
            feature_names = ("bias",)
        return cls(
            feature_names=feature_names,
            judgment_names=judgment_names,
            enable_memory=enable_memory,
            enable_system1=enable_system1,
            calibrated_heads=calibrated_heads,
        )

    def dumps(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True)

    @classmethod
    def loads(cls, payload: str) -> "AdaptiveCognitionLayer":
        return cls.from_state_dict(json.loads(payload))
