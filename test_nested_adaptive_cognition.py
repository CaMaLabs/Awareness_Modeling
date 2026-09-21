import math
import unittest

from nested_adaptive_cognition import (
    AdaptiveCognitionLayer,
    CalibratedBinaryHead,
    ContinuumMemory,
)
from nested_drcs_runtime import NestedDRCSConfig, NestedDRCSRuntimeAdapter, mode_flags


class MemoryTests(unittest.TestCase):
    def test_bands_update_at_distinct_rates(self):
        memory = ContinuumMemory(2)
        for _ in range(300):
            memory.observe([1.0, -1.0], surprise=0.8)
        counts = {band.name: band.updates for band in memory.bands}
        self.assertGreater(counts["immediate"], counts["working"])
        self.assertGreater(counts["working"], counts["episodic"])
        self.assertGreater(counts["episodic"], counts["identity"])
        self.assertGreater(counts["identity"], 0)

    def test_slow_memory_changes_less_than_fast_memory(self):
        memory = ContinuumMemory(1)
        for _ in range(32):
            memory.observe([1.0], surprise=0.9)
        states = {band.name: band.state[0] for band in memory.bands}
        self.assertGreater(states["immediate"], states["working"])
        self.assertGreater(states["working"], states["episodic"])

    def test_identity_memory_resists_one_off_surprise(self):
        memory = ContinuumMemory(1)
        memory.observe([1.0], surprise=1.0)
        states = {band.name: band.state[0] for band in memory.bands}
        self.assertEqual(states["identity"], 0.0)
        self.assertGreater(states["immediate"], 0.0)

    def test_meta_plasticity_is_bounded(self):
        memory = ContinuumMemory(1)
        for _ in range(5000):
            memory.observe([1.0], surprise=1.0)
        for band in memory.bands:
            self.assertGreaterEqual(band.plasticity, band.min_plasticity)
            self.assertLessEqual(band.plasticity, band.max_plasticity)

    def test_memory_telemetry_contains_required_fields(self):
        memory = ContinuumMemory(2)
        memory.observe([0.5, -0.5], surprise=0.8)
        telemetry = memory.telemetry()["immediate"]
        for key in (
            "writes",
            "update_frequency",
            "plasticity_multiplier",
            "state_magnitude",
            "latest_write_surprise",
            "ticks_since_latest_write",
        ):
            self.assertIn(key, telemetry)


class HeadTests(unittest.TestCase):
    def test_online_head_learns_repeated_example(self):
        head = CalibratedBinaryHead("x", width=2, learning_rate=0.08)
        vector = [1.0, 0.5]
        before = head.predict(vector)
        for _ in range(100):
            head.update(vector, 1.0)
        after = head.predict(vector)
        self.assertGreater(after, before)
        self.assertGreater(after, 0.8)
        self.assertTrue(math.isfinite(head.temperature))

    def test_probability_range_and_calibration_metrics(self):
        head = CalibratedBinaryHead("x", width=1)
        for i in range(20):
            p = head.predict([float(i % 2)])
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
            head.update([float(i % 2)], float(i % 2))
        metrics = head.metrics()
        self.assertIn("expected_calibration_error", metrics)
        self.assertEqual(len(metrics["bins"]), 10)
        self.assertGreater(metrics["updates"], 0.0)


class LayerTests(unittest.TestCase):
    def test_no_labels_means_no_head_training(self):
        layer = AdaptiveCognitionLayer(["a", "b"])
        layer.step({"a": 1.0, "b": 0.5}, prediction_error=0.4)
        self.assertEqual(layer.system1.heads["teacher_needed"].updates, 0)

    def test_labels_train_only_named_heads(self):
        layer = AdaptiveCognitionLayer(["a", "b"])
        layer.step(
            {"a": 1.0, "b": -0.5},
            prediction_error=0.7,
            labels={"teacher_needed": 1.0, "prediction_failure": 1.0},
        )
        self.assertEqual(layer.system1.heads["teacher_needed"].updates, 1)
        self.assertEqual(layer.system1.heads["prediction_failure"].updates, 1)
        self.assertEqual(layer.system1.heads["explore"].updates, 0)

    def test_delayed_learning_uses_logged_previous_context(self):
        layer = AdaptiveCognitionLayer(["a"])
        result = layer.step({"a": 1.0}, prediction_error=0.6, labels=None)
        before = result["judgments"]["action_success"]
        layer.learn_from_labels({"action_success": 1.0})
        after = layer.step({"a": 1.0}, prediction_error=0.6, labels=None)["judgments"]["action_success"]
        self.assertEqual(layer.system1.heads["action_success"].updates, 1)
        self.assertGreater(after, before)

    def test_teacher_gate_is_rate_limited_and_logged(self):
        layer = AdaptiveCognitionLayer(["a"], teacher_threshold=0.1)
        head = layer.system1.heads["teacher_needed"]
        head.bias = 5.0
        gates = []
        for _ in range(8):
            gates.append(
                layer.step({"a": 1.0}, prediction_error=0.9, labels=None)["teacher_gate"]
            )
        self.assertTrue(any(gates))
        self.assertLess(sum(1 for item in gates if item), len(gates))
        self.assertIn("high_prediction_error", layer.teacher_gate.last_reason)

    def test_no_teacher_direct_state_overwrite(self):
        layer = AdaptiveCognitionLayer(["a"])
        before = layer.memory.state_dict()
        layer.learn_from_labels({"teacher_needed": 1.0})
        after = layer.memory.state_dict()
        self.assertEqual(before, after)

    def test_serialization_roundtrip(self):
        layer = AdaptiveCognitionLayer(["a", "b"])
        for _ in range(20):
            layer.step(
                {"a": 0.7, "b": -0.2},
                prediction_error=0.6,
                labels={"teacher_needed": 1.0},
            )
        payload = layer.dumps()
        restored = AdaptiveCognitionLayer.loads(payload)
        self.assertEqual(restored.tick, layer.tick)
        self.assertEqual(restored.memory.tick, layer.memory.tick)
        p1 = layer.step({"a": 0.7, "b": -0.2}, 0.2, learn=False)["judgments"]["teacher_needed"]
        p2 = restored.step({"a": 0.7, "b": -0.2}, 0.2, learn=False)["judgments"]["teacher_needed"]
        self.assertAlmostEqual(p1, p2, places=12)

    def test_legacy_checkpoint_missing_new_fields_loads(self):
        layer = AdaptiveCognitionLayer(["a"])
        state = layer.state_dict()
        state.pop("teacher_gate")
        state.pop("prediction_log")
        state.pop("pending_training_context")
        restored = AdaptiveCognitionLayer.from_state_dict(state)
        self.assertEqual(restored.tick, layer.tick)
        self.assertIsNotNone(restored.teacher_gate)

    def test_corrupted_optional_state_recovers_fail_safe(self):
        restored = AdaptiveCognitionLayer.safe_from_state_dict(
            {"system1": {"heads": "bad"}},
            feature_names=["a", "b"],
        )
        self.assertEqual(restored.feature_names, ("a", "b"))
        self.assertEqual(restored.tick, 0)

    def test_checkpoint_roundtrip_contains_new_state(self):
        layer = AdaptiveCognitionLayer(["a"])
        layer.step({"a": 1.0}, prediction_error=0.7, labels={"action_success": 1.0})
        state = layer.state_dict()
        self.assertIn("teacher_gate", state)
        self.assertIn("bins", state["system1"]["heads"]["action_success"])
        restored = AdaptiveCognitionLayer.from_state_dict(state)
        self.assertEqual(restored.state_dict()["teacher_gate"], state["teacher_gate"])


def snapshot(streams=None, mean_error=0.4):
    return {
        "status": {
            "mean_prediction_error": mean_error,
            "self_model_stability": 0.8,
            "winner_recurrence": 0.5,
            "recent_memory_continuity": 0.9,
            "sensory_stream_count": len(streams or {}),
        },
        "action_loop": {"pending_feedback": 1, "mean_prediction_error": mean_error},
        "sensory_cortex": {"streams": streams or {}},
    }


class RuntimeAdapterTests(unittest.TestCase):
    def test_feature_flag_modes(self):
        self.assertFalse(mode_flags("A")["enable_layer"])
        self.assertTrue(mode_flags("B")["enable_system1"])
        self.assertTrue(mode_flags("C")["enable_memory"])
        self.assertTrue(mode_flags("D")["calibrated"])

    def test_runtime_integration_and_stream_identity_preservation(self):
        adapter = NestedDRCSRuntimeAdapter(NestedDRCSConfig(mode="D", max_streams=4))
        out = adapter.step(
            snapshot(
                {
                    "audio": {"novelty": 0.4, "prediction_error": 0.2, "prototype_count": 2},
                    "wifi": {"novelty": 0.9, "prediction_error": 0.7, "prototype_count": 3},
                },
                mean_error=0.6,
            )
        )
        self.assertIn("audio", out["sensory_stream_slots"])
        self.assertIn("wifi", out["sensory_stream_slots"])
        self.assertIn("prediction_failure", out["judgments"])

    def test_adapter_checkpoint_roundtrip_and_restart_continuity(self):
        adapter = NestedDRCSRuntimeAdapter(NestedDRCSConfig(mode="D", max_streams=2))
        adapter.step(snapshot({"audio": {"novelty": 0.7, "prediction_error": 0.6}}))
        state = adapter.state_dict()
        restored = NestedDRCSRuntimeAdapter.from_state_dict(state)
        self.assertEqual(restored.tick, adapter.tick)
        first = adapter.step(snapshot({"audio": {"novelty": 0.7, "prediction_error": 0.6}}))
        second = restored.step(snapshot({"audio": {"novelty": 0.7, "prediction_error": 0.6}}))
        self.assertEqual(first["sensory_stream_slots"], second["sensory_stream_slots"])
        self.assertEqual(first["judgments"].keys(), second["judgments"].keys())

    def test_mode_a_does_not_activate_new_mechanisms(self):
        adapter = NestedDRCSRuntimeAdapter(NestedDRCSConfig(mode="A"))
        out = adapter.step(snapshot())
        self.assertFalse(out["enabled"])
        self.assertIsNone(adapter.layer)

    def test_adapter_outcome_labels_train_previous_prediction_only(self):
        adapter = NestedDRCSRuntimeAdapter(NestedDRCSConfig(mode="D"))
        adapter.step(snapshot(mean_error=0.6))
        before = adapter.layer.system1.heads["action_success"].updates
        adapter.step(snapshot(mean_error=0.2), outcome_labels={"action_success": 1.0})
        after = adapter.layer.system1.heads["action_success"].updates
        self.assertEqual(after, before + 1)

    def test_corrupted_adapter_state_recovers(self):
        adapter = NestedDRCSRuntimeAdapter.from_state_dict(
            {"config": {"mode": "D"}, "layer": {"bad": object()}}
        )
        out = adapter.step(snapshot())
        self.assertTrue(out["enabled"])


if __name__ == "__main__":
    unittest.main()
