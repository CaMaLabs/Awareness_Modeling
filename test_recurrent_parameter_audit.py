import unittest
import recurrent_parameter_audit as r
class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.model=r.build_model(r.synthetic_rows(42,30)); cls.cfg={"memory_scale":1.0,"transition_scale":1.0,"rollout_steps":4,"emission_width":2.0,"temperature":.25,"loop_scale":1.0}
    def test_deconfounded_final_target_equal(self):
        refs=self.model['refs']; g=self.model['global']
        a=r.make_target(g,8,.44); b=r.make_target(g,8,.44); self.assertEqual(a,b)
        self.assertNotEqual(r.make_target(refs[r.SI['wake']],8,.44),r.make_target(refs[r.SI['n3']],8,.44))
    def test_recurrence_off_history_independent_at_final_probe(self):
        m=r.evaluate(self.model,self.cfg,'recurrence_fully_off',[8.0],[.44])
        self.assertAlmostEqual(m['history_mi_bits'],0.0,places=12)
    def test_collapse_rejected(self):
        full={"min_final_occupancy":0,"max_final_occupancy":1,"final_entropy_bits":0,"history_mi_bits":0,"mean_pairwise_js_bits":0,"same_state_fraction":.25,"away_transition_fraction":.75}
        off={"history_mi_bits":0}; shuf={"history_mi_bits":0}; self.assertFalse(r.basic_pass(full,off,shuf)[0])
    def test_frozen_identity_rejected(self):
        full={"min_final_occupancy":.25,"max_final_occupancy":.25,"final_entropy_bits":2,"history_mi_bits":2,"mean_pairwise_js_bits":1,"same_state_fraction":1,"away_transition_fraction":0}
        off={"history_mi_bits":0}; shuf={"history_mi_bits":0}; self.assertFalse(r.basic_pass(full,off,shuf)[0])
    def test_shuffled_history_reduces_dependence(self):
        cfg={"memory_scale":4.0,"transition_scale":1.0,"rollout_steps":4,"emission_width":4.0,"temperature":0.25,"loop_scale":1.0}
        full=r.evaluate(self.model,cfg,"full")
        shuffled=r.evaluate(self.model,cfg,"previous_state_shuffled")
        self.assertGreater(full["history_mi_bits"],shuffled["history_mi_bits"])
    def test_pass_classification_accepts_valid_metrics(self):
        full={"min_final_occupancy":0.10,"max_final_occupancy":0.45,"final_entropy_bits":1.6,"history_mi_bits":0.40,"mean_pairwise_js_bits":0.20,"same_state_fraction":0.60,"away_transition_fraction":0.40}
        off={"history_mi_bits":0.0}; shuffled={"history_mi_bits":0.10}
        self.assertTrue(r.basic_pass(full,off,shuffled)[0])
    def test_fixed_seed_deterministic(self):
        a=r.synthetic_rows(42,5);b=r.synthetic_rows(42,5);self.assertEqual(a,b)
if __name__=='__main__': unittest.main()
