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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


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
        return True

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

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [0.0] * self.width
        if len(self.weights) != self.width:
            raise ValueError("head width mismatch")

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
        self.updates += 1
        return {"probability": p, "brier": brier, "log_loss": ll}

    def metrics(self) -> Dict[str, float]:
        if not self.updates:
            return {
                "updates": 0.0,
                "mean_brier": 0.0,
                "mean_log_loss": 0.0,
                "temperature": self.temperature,
            }
        return {
            "updates": float(self.updates),
            "mean_brier": self.cumulative_brier / self.updates,
            "mean_log_loss": self.cumulative_log_loss / self.updates,
            "temperature": self.temperature,
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
        teacher_gate = self.enable_system1 and teacher_probability >= self.teacher_threshold

        head_updates: Dict[str, Dict[str, float]] = {}
        memory_updates: Dict[str, bool] = {}
        if learn:
            if self.enable_system1 and labels:
                head_updates = self.system1.update(context, labels)
            if self.enable_memory:
                memory_updates = self.memory.observe(raw, abs(float(prediction_error)))

        return {
            "tick": self.tick,
            "judgments": judgments,
            "teacher_probability": teacher_probability,
            "teacher_gate": teacher_gate,
            "intrinsic_novelty": novelty,
            "memory_readout": self.memory.read() if self.enable_memory else [0.0] * len(raw),
            "memory_updates": memory_updates,
            "head_updates": head_updates,
        }

    def metrics(self) -> Dict[str, object]:
        return {
            "tick": self.tick,
            "memory": {
                band.name: {
                    "updates": band.updates,
                    "plasticity": band.plasticity,
                }
                for band in self.memory.bands
            },
            "judgments": self.system1.metrics(),
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
        return layer

    def dumps(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True)

    @classmethod
    def loads(cls, payload: str) -> "AdaptiveCognitionLayer":
        return cls.from_state_dict(json.loads(payload))
