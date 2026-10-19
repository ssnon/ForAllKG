import unittest,math
from pipeline_core.discovery.m30_sweep import oracle
class OracleTests(unittest.TestCase):
    def test_tasks_are_valid(self):
        for t in oracle.TASKS:
            p=oracle.public_problem(t)
            self.assertEqual(p['id'],t)
            self.assertEqual(p['science_truth_status'],'HIDDEN_SYNTHETIC_REFERENCE_NOT_EMPIRICAL')
    def test_sidon_checker(self):
        self.assertTrue(oracle.sidon_diagnostics({'set':[0,1,3,7]},25)['valid'])
        self.assertFalse(oracle.sidon_diagnostics({'set':[0,1,2]},25)['valid'])
    def test_dfa_counterexample(self):
        for k in oracle.DFA_TARGETS:
            self.assertIsNone(oracle.counterexample(oracle.DFA_TARGETS[k],oracle.DFA_TARGETS[k]))
            bad={'transitions':[[0,0]],'accepting':[]}
            self.assertIsNotNone(oracle.counterexample(bad,oracle.DFA_TARGETS[k]))
    def test_truth_dynamics(self):
        for t in ['dyn_predator','dyn_resource']:
            for p in oracle.dyn_protocols(t):
                v=oracle.simulate_truth(t,p)
                self.assertEqual(len(v),6)
                self.assertTrue(all(math.isfinite(z['x']+z['y']) for z in v))
    def test_expression_security(self):
        for bad in ['__import__("os")', {'var':'__class__'},{'op':'pow','args':[{'const':2},{'const':3}]}]:
            with self.assertRaises((ValueError,TypeError)):oracle.safe_calc(bad,{'x':1})
    def test_numeric_holdouts(self):
        for t in ['law_saturation','law_inhibition','chem_blocking','chem_dualpath']:
            train=oracle.numeric_rows(t);test=oracle.numeric_rows(t,True)
            self.assertNotEqual(train,test)
            self.assertGreater(len(test),0)
    def test_probe_deterministic(self):
        task='law_saturation';cand={'const':1.2};req={'point':{'x':1,'z':0}}
        self.assertEqual(oracle.experiment(task,req,cand),oracle.experiment(task,req,cand))
        self.assertEqual(oracle.experiment(task,req,cand)['status'],'EXECUTED')
    def test_invalid_scores_zero(self):
        for t in oracle.TASKS:
            self.assertEqual(oracle.final_score(t,{'invalid':'candidate'})['score'],0)
if __name__=='__main__':unittest.main()
