"""M14-C: deterministic physics, audit, evidence-policy and negative tests.
All tests are offline. Synthetic testbed is NOT empirical evidence.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest import mock

from pipeline_core.discovery.sers_evidence_search_m14c import (
    CASE_PHASE,CASE_INJECTION,SCHEMA,phase_probe,injection_probe,thermal_rate,execute_cases,
    effective_action,find_thermal_shift,canonical,sha,_record,STATUS_PHASE,STATUS_INJECTION)
from scripts.discovery.run_sers_evidence_search_m14c import (
    make_plan,check_preregistration,cmd_execute,cmd_verify,write_once,
    PLAN,ROWS,SUMMARIES,RECEIPT,CERT,FEEDBACK,load,policy_update)
from pipeline_core.discovery.sers_adaptive_policy_m14b import independent_evidence


class PhysicsTests(unittest.TestCase):
    def test_linear_classical_counterexample_exact(self):
        r,s=phase_probe();self.assertEqual(len(r),720)
        self.assertEqual(s['outcome'],STATUS_PHASE)
        self.assertTrue(s['equivalent_under_selected_observables'])
        self.assertEqual(s['max_abs_residual_vs_classical_hidden_oscillator'],0)
        self.assertFalse(s['can_certify_quantum_coherence'])

    def test_linear_phase_is_not_empirical(self):
        _,s=phase_probe();self.assertFalse(s['external_empirical_validation'])
        self.assertIn('linear',s['limitation'])

    def test_thermal_injection_restricted_diff(self):
        rows,s=injection_probe()
        self.assertEqual(len(rows),12)
        self.assertEqual(s['outcome'],STATUS_INJECTION)
        self.assertTrue(s['restricted_null_differentiated'])
        self.assertAlmostEqual(s['minimum_heldout_restricted_null_abs_difference'],.12)

    def test_expanded_thermal_is_observationally_possible(self):
        _,s=injection_probe()
        self.assertTrue(s['expanded_thermal_null_remains_viable'])
        self.assertLess(s['expanded_thermal_countermodel_max_temperature_shift_K'],10)
        self.assertLess(s['expanded_thermal_countermodel_max_abs_residual'],1e-9)
        self.assertFalse(s['causal_identification_in_real_SERS'])

    def test_rate_monotonic(self):
        self.assertGreater(thermal_rate(311,315),thermal_rate(309,313))
        self.assertGreater(find_thermal_shift(thermal_rate(310,310)+.1,310,310),0)

    def test_temperature_out_of_range_rejected(self):
        with self.assertRaises(ValueError):thermal_rate(100,300)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):_record('x','train','t','i',0,float('nan'),0,0)
        with self.assertRaises(ValueError):canonical({'x':float('inf')})

    def test_execution_reproducible_and_two_cases(self):
        a,sa=execute_cases();b,sb=execute_cases()
        self.assertEqual(a,b);self.assertEqual(sa,sb)
        self.assertEqual(sa['total_rows'],732)
        self.assertEqual(set(sa['case_results']),{CASE_PHASE,CASE_INJECTION})
        self.assertFalse(sa['observational_scientific_novelty_certified'])

    def test_unknown_or_unverified_evidence_never_changes_action(self):
        with self.assertRaises(ValueError):effective_action(CASE_PHASE,STATUS_PHASE,False)
        with self.assertRaises(ValueError):effective_action('fake',STATUS_PHASE,True)
        with self.assertRaises(ValueError):effective_action(CASE_PHASE,'CLAIM_NOVEL',True)

    def test_outcome_aware_override(self):
        self.assertEqual(effective_action(CASE_PHASE,STATUS_PHASE,True)['action'],'ESCAPE')
        self.assertEqual(effective_action(CASE_INJECTION,STATUS_INJECTION,True)['action'],'SUSPEND')


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'fixture';self.root.mkdir()
        self.output=Path(self.tmp.name)/'evidence'
        self.phase_id='m14:'+'a'*20;self.injection_id='m14:'+'b'*20
        self.cards=[{'problem_id':'gap:a','domain_family':'Vibration'},
                    {'problem_id':'gap:b','domain_family':'Catalysis'},
                    {'problem_id':'gap:c','domain_family':'Orientation'}]
        self.plan={'design':{'replicates':1,'generations':2}}
        self._save('M14_PLAN.json',self.plan)
        self._save('M14_PROVISIONAL_PROBLEMS.json',{'novelty_certified':False,'questions':self.cards})
        self._save('M14_RUN_SUMMARY.json',{'replicates':1,'generations':2})
        self.candidates=[{'lane':'MECHANISM_SEARCH','candidate_id':self.phase_id,'problem_id':'gap:a',
                          'family':'Vibration','claim_atom':'Phase coherence vibrational hidden Markov oscillation',
                          'strongest_null':'hidden classical Markov state'},
                         {'lane':'OPEN_FRONTIER','candidate_id':self.injection_id,'problem_id':'gap:b',
                          'family':'Catalysis','claim_atom':'Injection thermal plasmonic chemistry',
                          'strongest_null':'two temperature thermal null and injection'}]
        self.reviews=[{'lane':c['lane'],'candidate_id':c['candidate_id'],
                       'physical_status':'NON_IDENTIFIABLE','prior_art_status':'UNRESOLVED_TESTABLE',
                       'retrieval_missing':False} for c in self.candidates]
        for arm in ('ADAPTIVE','FIXED'):
            for g in (1,2):
                self._save(f'r01/g{g:02d}/{arm}/M14_GENERATION_RESULT.json',
                    {'replicate':1,'generation':g,'novelty_certified':False,
                     'candidates':copy.deepcopy(self.candidates),
                     'candidate_reviews':copy.deepcopy(self.reviews)})
        self.args=type('A',(),{'out_dir':self.output})()

    def _save(self,rel,data):
        p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(canonical(data))

    def prepare(self):
        p=make_plan(self.root,1,2,'ADAPTIVE')
        self.output.mkdir();write_once(self.output/PLAN,canonical(p))
        return p

    def execute(self):
        self.prepare();cmd_execute(self.args)

    def test_prepare_bound_ids(self):
        p=self.prepare()
        self.assertEqual(p['candidate_cases']['MECHANISM_SEARCH']['candidate_id'],self.phase_id)
        self.assertFalse(p['independent_empirical_data_supplied'])
        self.assertEqual(check_preregistration(self.output),p)

    def test_reject_unrelated_science(self):
        p=self.root/'r01/g02/ADAPTIVE/M14_GENERATION_RESULT.json'
        dat=json.loads(p.read_bytes());dat['candidates'][0]['claim_atom']='Unrelated gravitational waves'
        p.write_bytes(canonical(dat))
        with self.assertRaisesRegex(ValueError,'does not support'):
            make_plan(self.root,1,2,'ADAPTIVE')

    def test_reject_unmatched_id(self):
        p=self.root/'r01/g02/ADAPTIVE/M14_GENERATION_RESULT.json'
        dat=json.loads(p.read_bytes());dat['candidates'][0]['candidate_id']='incorrect'
        p.write_bytes(canonical(dat))
        with self.assertRaisesRegex(ValueError,'Bad candidate id'):
            make_plan(self.root,1,2,'ADAPTIVE')

    def test_fixed_rejected(self):
        with self.assertRaisesRegex(ValueError,'second experimental arm'):
            make_plan(self.root,1,2,'FIXED')

    def test_generator_cannot_certify_novelty(self):
        p=self.root/'r01/g02/ADAPTIVE/M14_GENERATION_RESULT.json'
        dat=json.loads(p.read_bytes());dat['candidates'][0]['novelty_certified']=True
        p.write_bytes(canonical(dat))
        with self.assertRaises(ValueError):make_plan(self.root,1,2,'ADAPTIVE')

    def test_source_mutation_rejected(self):
        self.prepare()
        p=self.root/'r01/g02/ADAPTIVE/M14_GENERATION_RESULT.json'
        p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'changed'):
            check_preregistration(self.output)

    def test_pre_registration_mutation_rejected(self):
        self.prepare();p=self.output/PLAN
        data=load(p);data['preregistered_code_sha256']='0'*64;p.write_bytes(canonical(data))
        with self.assertRaisesRegex(ValueError,'Numerical code changed'):
            check_preregistration(self.output)

    def test_execution_seals_raw_and_prevents_duplicate(self):
        self.execute()
        self.assertTrue((self.output/ROWS).exists())
        self.assertFalse((self.output/CERT).exists())
        with self.assertRaises(FileExistsError):cmd_execute(self.args)

    def test_verified_material_replays(self):
        from scripts.discovery.run_sers_evidence_search_m14c import verified_material
        self.execute();p,r,s=verified_material(self.output)
        self.assertEqual(r['observables_csv_sha256'],sha((self.output/ROWS).read_bytes()))
        self.assertEqual(s['total_rows'],732)
        self.assertFalse(r['novelty_certified'])

    def test_reject_tampered_observable(self):
        from scripts.discovery.run_sers_evidence_search_m14c import verified_material
        self.execute();path=self.output/ROWS
        path.write_bytes(path.read_bytes().replace(b'0.12',b'9.12',1))
        with self.assertRaisesRegex(ValueError,'do not replay exactly'):
            verified_material(self.output)

    def test_reject_tampered_summary(self):
        from scripts.discovery.run_sers_evidence_search_m14c import verified_material
        self.execute();p=self.output/SUMMARIES;p.write_bytes(p.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'do not replay exactly'):
            verified_material(self.output)

    def test_reject_tampered_receipt(self):
        from scripts.discovery.run_sers_evidence_search_m14c import verified_material
        self.execute();p=self.output/RECEIPT;obj=load(p);obj['source_generation_sha256']='f'*64
        p.write_bytes(canonical(obj))
        with self.assertRaisesRegex(ValueError,'does not match'):
            verified_material(self.output)

    def test_outcome_changes_policy_not_novelty(self):
        self.execute();cmd_verify(self.args)
        policy=load(self.output/FEEDBACK)
        a=policy['policy_by_lane']
        self.assertEqual(a['MECHANISM_SEARCH']['m14c_outcome_aware_action'],'ESCAPE')
        self.assertEqual(a['OPEN_FRONTIER']['baseline_m14b_action'],'ESCAPE')
        self.assertEqual(a['OPEN_FRONTIER']['m14c_outcome_aware_action'],'SUSPEND')
        self.assertTrue(a['OPEN_FRONTIER']['action_changed_from_baseline'])
        self.assertFalse(policy['novelty_certified'])
        self.assertFalse(policy['external_empirical_data'])
        self.assertTrue(policy['reproducibility_verified'])
        self.assertEqual(len(policy['evidence_references']),2)
        for e in policy['evidence_references']:
            self.assertTrue(independent_evidence([e],candidate_id=e['candidate_id']))
        cmd_verify(self.args) # deterministic idempotence

    def test_feedback_freezes_no_overwrite(self):
        self.execute();cmd_verify(self.args)
        p=self.output/FEEDBACK;p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Previously frozen'):
            cmd_verify(self.args)

    def test_can_read_full_audit_zip_instead_of_root_folder(self):
        files=[p for p in self.root.rglob('*') if p.is_file()]
        zpath=Path(self.tmp.name)/'input.zip'
        with zipfile.ZipFile(zpath,'w') as z:
            for p in files:z.write(p,p.relative_to(self.root).as_posix())
        registered=make_plan(zpath,1,2,'ADAPTIVE')
        self.assertEqual(registered['candidate_cases']['OPEN_FRONTIER']['candidate_id'],self.injection_id)

    def test_source_not_modified_by_pipeline(self):
        before={str(p.relative_to(self.root)):sha(p.read_bytes()) for p in self.root.rglob('*') if p.is_file()}
        self.execute();cmd_verify(self.args)
        after={str(p.relative_to(self.root)):sha(p.read_bytes()) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_missing_execution_not_implicitly_recalculated(self):
        self.prepare()
        with self.assertRaises(FileNotFoundError):cmd_verify(self.args)


if __name__=='__main__':unittest.main()
