import unittest

import true_recurrent_dynamics as trd


class TrueRecurrentDynamicsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"state": "wake", "candidate_loop": "thalamo_cortical", "peak_alpha_hz": 9.8, "L_eff_m": 0.030, "v_eff_m_per_s": 8.0, "tau_eff_s": 0.008, "pl_local_aw": 0.74, "pl_global_aw": 0.83, "cfc": 0.68, "w_prop": 0.84, "s_struct": 0.86},
            {"state": "n2", "candidate_loop": "dmn", "peak_alpha_hz": 7.2, "L_eff_m": 0.040, "v_eff_m_per_s": 6.1, "tau_eff_s": 0.012, "pl_local_aw": 0.58, "pl_global_aw": 0.70, "cfc": 0.52, "w_prop": 0.66, "s_struct": 0.75},
            {"state": "rem", "candidate_loop": "fronto_parietal", "peak_alpha_hz": 8.9, "L_eff_m": 0.034, "v_eff_m_per_s": 6.9, "tau_eff_s": 0.010, "pl_local_aw": 0.64, "pl_global_aw": 0.76, "cfc": 0.60, "w_prop": 0.74, "s_struct": 0.80},
            {"state": "n3", "candidate_loop": "ct_cingulate", "peak_alpha_hz": 2.0, "L_eff_m": 0.050, "v_eff_m_per_s": 4.0, "tau_eff_s": 0.020, "pl_local_aw": 0.42, "pl_global_aw": 0.30, "cfc": 0.25, "w_prop": 0.40, "s_struct": 0.55},
        ]
        self.refs = trd.build_state_refs(self.rows)
        self.global_ref = trd.mean_feature_row(self.rows)
        self.norm_stats = trd.build_norm_stats(self.rows)
        self.loop_priors = trd.build_loop_priors(self.rows)

    def test_requested_falsification_conditions_are_present(self):
        self.assertEqual(
            trd.CONDITIONS,
            [
                "legacy_target_positive_control",
                "prototype_target_removed",
                "memory_off",
                "previous_state_shuffled",
                "transition_penalties_zero",
            ],
        )

    def test_neutral_target_has_identical_final_probe(self):
        target = trd.make_target_row(self.global_ref, alpha_target=8.0, chi_target=0.50)
        wake_final = trd.blend_features(self.refs["wake"], target, 1.0)
        n3_final = trd.blend_features(self.refs["n3"], target, 1.0)
        self.assertEqual(wake_final, target)
        self.assertEqual(n3_final, target)
        self.assertEqual(wake_final, n3_final)

    def test_shuffled_history_is_a_derangement_and_preserves_probability(self):
        probs = {"wake": 0.55, "n2": 0.20, "rem": 0.15, "n3": 0.10}
        state, shuffled = trd.shuffled_history("wake", probs)
        self.assertNotEqual(state, "wake")
        self.assertAlmostEqual(sum(shuffled.values()), 1.0)
        for original in trd.STATES:
            self.assertNotEqual(trd.SHUFFLED_STATE[original], original)

    def test_memory_changes_recurrent_probability(self):
        common = dict(
            start_state="n3",
            alpha_target=7.5,
            chi_target=0.50,
            refs=self.refs,
            global_ref=self.global_ref,
            norm_stats=self.norm_stats,
            loop_priors=self.loop_priors,
            rollout_steps=8,
        )
        recurrent = trd.rollout_to_target(condition="prototype_target_removed", **common)
        no_memory = trd.rollout_to_target(condition="memory_off", **common)
        self.assertNotAlmostEqual(recurrent["final_probability"], no_memory["final_probability"], places=10)


if __name__ == "__main__":
    unittest.main()
