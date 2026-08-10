import unittest
import numpy as np
import empirical_sleep_constraint_audit as e

class EmpiricalConstraintTests(unittest.TestCase):
    def test_collapse_removes_internal_light_sleep_transitions(self):
        raw, p = e.collapse_global(e.KISHI_2011_SECOND, e.KISHI_2011_LABELS, e.MAP_2011)
        self.assertAlmostEqual(raw[e.STATES.index('n2'), e.STATES.index('n3')], 23.0)
        self.assertAlmostEqual(p[e.STATES.index('n2')].sum(), 1.0)
        self.assertEqual(p[e.STATES.index('n2'), e.STATES.index('n2')], 0.0)

    def test_empirical_consensus_dominant_exits(self):
        refs = e.empirical_transition_references()
        mean = np.mean([p for _, p in refs], axis=0)
        matches = e.dominant_exit_match(mean)
        self.assertTrue(all(matches.values()))

    def test_healthy_reference_variation_below_frozen_tv_gate(self):
        refs = e.empirical_transition_references()
        distances = [e.mean_row_tv(refs[i][1], refs[j][1]) for i in range(3) for j in range(i+1, 3)]
        self.assertLess(max(distances), e.TRANSITION_TV_MAX)

    def test_dwell_target_order(self):
        d = e.empirical_dwell_target()
        by = dict(zip(e.STATES, d))
        self.assertGreater(by['rem'], by['n2'])
        self.assertGreater(by['n2'], by['n3'])
        self.assertGreater(by['n3'], by['wake'])

    def test_read_robust_configs_filters_passing_region(self):
        import csv, tempfile
        fields = ["config_id","memory_scale","transition_scale","rollout_steps","emission_width","temperature","loop_scale","fully_robust"]
        rows = [
            ["a",1,1,4,2,.25,1,0],
            ["b",2,1,8,4,.25,1,1],
        ]
        with tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv") as f:
            w = csv.writer(f); w.writerow(fields); w.writerows(rows); f.flush()
            got = e.read_robust_configs(f.name)
        self.assertEqual([r["config_id"] for r in got], ["b"])

    def test_arousal_target_requires_n3_resistance(self):
        self.assertLess(e.AROUSAL_TARGET['n3'], e.AROUSAL_TARGET['n2'])
        self.assertLess(e.AROUSAL_TARGET['n3'], e.AROUSAL_TARGET['rem'])

if __name__ == '__main__':
    unittest.main()
