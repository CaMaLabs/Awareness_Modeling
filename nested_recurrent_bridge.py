"""Bridge the nested cognition layer to the existing recurrent feature schema.

The bridge intentionally keeps outcome labels outside the core recurrent state
selection. A caller may provide labels only after an outcome is observed, which
prevents the System-1 heads from leaking future targets into the recurrent model.
"""

from __future__ import annotations

from typing import Mapping, Optional

from nested_adaptive_cognition import AdaptiveCognitionLayer, DEFAULT_JUDGMENTS
from true_recurrent_dynamics import FEATURES


class RecurrentAdaptiveBridge:
    def __init__(
        self,
        enable_memory: bool = True,
        enable_system1: bool = True,
        calibrated_heads: bool = True,
        teacher_threshold: float = 0.72,
    ):
        self.layer = AdaptiveCognitionLayer(
            feature_names=FEATURES,
            judgment_names=DEFAULT_JUDGMENTS,
            enable_memory=enable_memory,
            enable_system1=enable_system1,
            calibrated_heads=calibrated_heads,
            teacher_threshold=teacher_threshold,
        )

    def step(
        self,
        row: Mapping[str, object],
        prediction_error: float,
        reward: float = 0.0,
        labels: Optional[Mapping[str, float]] = None,
        learn: bool = True,
    ):
        features = {}
        for name in FEATURES:
            try:
                features[name] = float(row.get(name, 0.0))
            except (TypeError, ValueError):
                features[name] = 0.0
        return self.layer.step(
            features,
            prediction_error=prediction_error,
            reward=reward,
            labels=labels,
            learn=learn,
        )

    def state_dict(self):
        return self.layer.state_dict()

    @classmethod
    def from_state_dict(cls, state):
        bridge = cls()
        bridge.layer = AdaptiveCognitionLayer.from_state_dict(state)
        return bridge
