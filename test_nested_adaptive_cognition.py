import math
import unittest

from nested_adaptive_cognition import (
    AdaptiveCognitionLayer,
    CalibratedBinaryHead,
    ContinuumMemory,
)


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


if __name__ == "__main__":
    unittest.main()
