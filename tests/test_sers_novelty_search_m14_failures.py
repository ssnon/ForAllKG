"""Regression tests isolating M13-format failures and M14 claim conservatism."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from pipeline_core.discovery.sers_novelty_search_m14 import (
    make_plan, unique_works, parse_problems, problem_prompt,
    computational_null_probes, strategy)
from scripts.discovery.run_sers_novelty_search_m14 import (
    generation_validated, physics_validated, saved_model,
    _audit_error, load, freeze)
from test_sers_novelty_search_m14 import MockScience

class Failures(unittest.TestCase):
    def fixture(self):
        p=make_plan(generations=2,replicates=2,queries_per_candidate=3)
        f=MockScience()
        src=unique_works([f.lookup('openalex',q) for q in p['seed_queries']])
        problems=parse_problems(f.call('mock',problem_prompt(p,src))['text'],src)
        return p,problems,f
    def test_title_only_preserves_claim(self):
        p,problems,f=self.fixture()
        from pipeline_core.discovery.sers_novelty_search_m14 import candidate_prompt
        prompt=candidate_prompt(p,problems,1,1,'ADAPTIVE',[])
        d=json.loads(f.call('mock',prompt)['text']);original=copy.deepcopy(d)
        d['candidates'][0]['title']=None
        gen,corr=generation_validated(json.dumps(d),p,problems,1,1,'ADAPTIVE',[])
        self.assertEqual(gen['candidates'][0]['claim_atom'],original['candidates'][0]['claim_atom'])
        self.assertEqual(corr['title_label_only_derived_from_existing_claim_indices'],[0])
        self.assertTrue(gen['candidates'][0]['title'].startswith('UNTITLED_METADATA_ONLY'))
    def test_null_equations_missing_is_never_invented(self):
        p,problems,f=self.fixture()
        from pipeline_core.discovery.sers_novelty_search_m14 import candidate_prompt
        d=json.loads(f.call('mock',candidate_prompt(p,problems,1,1,'ADAPTIVE',[]))['text'])
        d['candidates'][0]['null_equations']=None
        with self.assertRaisesRegex(ValueError,'Missing scientific'):
            generation_validated(json.dumps(d),p,problems,1,1,'ADAPTIVE',[])
    def test_non_ascii_latex_control_escape_not_silently_accepted(self):
        p={'novelty_certified':False,'reviews':[{'candidate_id':'x','physical_status':'NON_IDENTIFIABLE',
            'strongest_null_equations':['x=1'],'counterexample_or_derivation':'alpha\t(beta)',
            'critical_flaw':'not identified','recommended_action':'branch'}]}
        with self.assertRaisesRegex(ValueError,'Ambiguous JSON control'):
            physics_validated(json.dumps(p),[{'candidate_id':'x'}])
    def test_unknown_status_never_relabelled(self):
        p={'novelty_certified':False,'reviews':[{'candidate_id':'x','physical_status':'IT_IS_NOVEL',
            'strongest_null_equations':['x=1'],'counterexample_or_derivation':'no',
            'critical_flaw':'unknown','recommended_action':'branch'}]}
        with self.assertRaisesRegex(ValueError,'Physics status invalid'):
            physics_validated(json.dumps(p),[{'candidate_id':'x'}])
    def test_both_unresolved_untestable_should_suspend(self):
        previous={'reviews':[{'lane':lane,'family':lane,'physical_status':'UNRESOLVED_UNTESTABLE',
          'prior_art_status':'UNRESOLVED_UNTESTABLE','retrieval_missing':False} for lane in ('MECHANISM_SEARCH','OPEN_FRONTIER')]}
        self.assertEqual(strategy(previous,[],True)['OPEN_FRONTIER'],'SUSPEND')
    def test_null_probe_is_real_math_not_experiment(self):
        p=computational_null_probes()
        self.assertFalse(p['novelty_certified'])
        self.assertAlmostEqual(p['examples'][0]['absolute_trajectory_difference_at_start_of_second_pulse'],0.4650883158696593)
        self.assertEqual(p['examples'][1]['predicted_Cov_XY'],1)
        self.assertIn('NOT_EMPIRICAL',p['status'])
    def test_paid_receipt_resumption_and_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prompt={'system':'x','user':'one'};callcount=[]
            def call(model,prompt):
                callcount.append(1)
                return {'text':'{}','finish_reason':'stop'}
            raw,new=saved_model(root,'generation',prompt,'mock',call)
            self.assertTrue(new)
            raw,new=saved_model(root,'generation',prompt,'mock',call)
            self.assertFalse(new);self.assertEqual(len(callcount),1)
            with self.assertRaises(RuntimeError):
                saved_model(root,'generation',{'system':'x','user':'two'},'mock',call)
            with self.assertRaises(RuntimeError):
                saved_model(root,'generation',prompt,'changed_model',call)
            self.assertEqual(len(callcount),1)
    def test_error_audit_is_idempotent_across_checkpoint_invocations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);freeze(root/'generation_RAW_RECEIPT.json',{'raw':'invalid','finish_reason':'stop'})
            first=_audit_error(root,'generation',ValueError('science missing'),5)
            second=_audit_error(root,'generation',ValueError('science missing'),0)
            self.assertEqual(first['new_model_calls'],5)
            self.assertEqual(second['new_model_calls'],0)
            self.assertNotIn('new_model_calls',load(root/'M14_GENERATION_INVALID.json'))

if __name__=='__main__':unittest.main()
