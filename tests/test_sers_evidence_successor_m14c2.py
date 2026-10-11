import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_core.discovery.sers_evidence_successor_m14c2 import (prompt_for, test_model_vectors,LANE_ESCAPE,LANE_SUSPEND)
from scripts.discovery.run_sers_evidence_successor_m14c2 import (_validate_submission,cmd_ingest,cmd_verify,
    _load_contract,CONTRACT,RAW,REPORT,SCHEMA,canon)

class C2Test(unittest.TestCase):
    def contract(self):
        p={'problem_id':'gap:A','domain_family':'INTERFACE','source_anchors':[{'work_id':'OA:7','exact_quote':'A title'}], 'gap_confirmed':False}
        sig={x:'original_'+x for x in ('mechanism_family_id','causal_driver','core_assumption','predicted_observable','independent_intervention','strongest_null','falsification')}
        return {'schema':SCHEMA,'lanes':{
          LANE_ESCAPE:{'previous':{'problem_id':'gap:old','family':'VIBRATION','mechanism_signature':sig},'decision':{'schema':'m14b-adaptive-search-reconstruction-v1','action':'ESCAPE','problem_expansion_required':False,'escape_alternative_problem_ids':['gap:A']},'alternative_cards':[p]},
          LANE_SUSPEND:{'decision':{'schema':'m14b-adaptive-search-reconstruction-v1','action':'SUSPEND'},'previous_candidate_id':'X','next_evidence_required':'Measure temperature'}},
           'verified_c1_findings':{},'limitations':['synthetic']}
    def successor(self):
        sig={x:'new_'+x for x in ('mechanism_family_id','causal_driver','core_assumption','predicted_observable','independent_intervention','strongest_null','falsification')}
        return {'successor':{'problem_id':'gap:A','family':'INTERFACE','novelty_certified':False,'mechanism_signature':sig,
                    'conditional_predictions':['Observable changes with control'], 'independent_observables':['IR marker'],
                    'test_proposal':{'independent_variable':'x','null_coefficients':[1.0,0.0,0.0],
                    'hypothesis_coefficients':[1.0,0.2,0.0],'nuisance_envelope':0.1,
                    'units':'arbitrary','physical_scope_and_limitations':'Synthetic and not validated'}},
                'suspended_lane_successor':None}
    def test_valid(self):
        r=_validate_submission(self.contract(),self.successor())
        self.assertEqual(r['status'],'M14C2_ONE_SUCCESSOR_STRUCTURALLY_VALIDATED')
        self.assertEqual(r['prospective_test']['formal_grid_points_outside_assumed_envelope'],3)
        self.assertFalse(r['novelty_certified'])
    def test_suspended_lane_rejected(self):
        s=self.successor();s['suspended_lane_successor']={'x':1}
        with self.assertRaises(ValueError):_validate_submission(self.contract(),s)
    def test_same_problem_rejected(self):
        s=self.successor();s['successor']['problem_id']='gap:old'
        with self.assertRaises(ValueError):_validate_submission(self.contract(),s)
    def test_unanchored_problem_rejected(self):
        s=self.successor();s['successor']['problem_id']='gap:new'
        with self.assertRaises(ValueError):_validate_submission(self.contract(),s)
    def test_wrong_family_rejected(self):
        s=self.successor();s['successor']['family']='VIBRATION'
        with self.assertRaises(ValueError):_validate_submission(self.contract(),s)
    def test_same_causal_driver_rejected(self):
        s=self.successor();s['successor']['mechanism_signature']['causal_driver']='original_causal_driver'
        with self.assertRaises(ValueError):_validate_submission(self.contract(),s)
    def test_missing_observables_rejected(self):
        s=self.successor();s['successor']['independent_observables']=[]
        with self.assertRaises(ValueError):_validate_submission(self.contract(),s)
    def test_self_certification_rejected(self):
        s=self.successor();s['successor']['novelty_certified']=True
        with self.assertRaises(ValueError):_validate_submission(self.contract(),s)
    def test_nonfinite_coeff_rejected(self):
        s=self.successor()['successor']['test_proposal'];s['null_coefficients'][0]=float('nan')
        with self.assertRaises(ValueError):test_model_vectors(s)
    def test_zero_nuisance_rejected(self):
        s=self.successor()['successor']['test_proposal'];s['nuisance_envelope']=0
        with self.assertRaises(ValueError):test_model_vectors(s)
    def test_no_pseudo_empirical(self):
        r=_validate_submission(self.contract(),self.successor())
        self.assertIs(r['prospective_test']['real_world_observations'],False)
        self.assertIs(r['prospective_test']['independent_empirical_support'],False)
        self.assertIs(r['prospective_test']['nuisance_bounds_are_model_proposed_not_measured'],True)
    def test_ingest_and_idempotent_verify(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/CONTRACT).write_bytes(canon(self.contract()))
            proposed=folder/'submission.json';proposed.write_bytes(canon(self.successor()))
            args=type('Args',(),{'out_dir':folder,'response_json':proposed})
            cmd_ingest(args);cmd_verify(args);first=(folder/REPORT).read_bytes();cmd_verify(args)
            self.assertEqual(first,(folder/REPORT).read_bytes())
            with self.assertRaises(FileExistsError):cmd_ingest(args)
    def test_invalid_ingest_not_persisted(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/CONTRACT).write_bytes(canon(self.contract()))
            proposed=folder/'submission.json';proposed.write_bytes(canon({'successor':None,'suspended_lane_successor':None}))
            args=type('Args',(),{'out_dir':folder,'response_json':proposed})
            with self.assertRaises(ValueError):cmd_ingest(args)
            self.assertFalse((folder/RAW).exists())
    def test_prompt_contains_suspension(self):
        p=prompt_for(self.contract());self.assertIn('SUSPEND',p['user']);self.assertIn('exactly one',p['user'])
    def test_paid_must_be_explicitly_authorized(self):
        from scripts.discovery.run_sers_evidence_successor_m14c2 import cmd_paid
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/CONTRACT).write_bytes(canon(self.contract()))
            args=type('Args',(),{'out_dir':folder,'allow_paid':False,'i_authorize_model_calls':False})
            with self.assertRaises(RuntimeError):cmd_paid(args)
    def test_mock_paid_one_receipt_never_second_call(self):
        from scripts.discovery.run_sers_evidence_successor_m14c2 import cmd_paid
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/CONTRACT).write_bytes(canon(self.contract()))
            args=type('Args',(),{'out_dir':folder,'allow_paid':True,'i_authorize_model_calls':True,'model':'mock-local'})
            calls=[]
            def factory(a):
                def call(model,prompt):
                    calls.append(1)
                    return {'text':json.dumps(self.successor()),'finish_reason':'stop','usage':None}
                return call
            with patch('scripts.discovery.run_sers_evidence_successor_m14c2.source_call',side_effect=factory):
                cmd_paid(args)
                self.assertEqual(len(calls),1)
                self.assertTrue((folder/'m14c2_successor_RAW_RECEIPT.json').exists())
                self.assertTrue((folder/'m14c2_successor_INTENT.json').exists())
                with self.assertRaises(FileExistsError):cmd_paid(args)
                self.assertEqual(len(calls),1)
            cmd_verify(args)
    def test_mock_paid_invalid_receipt_retained(self):
        from scripts.discovery.run_sers_evidence_successor_m14c2 import cmd_paid
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/CONTRACT).write_bytes(canon(self.contract()))
            args=type('Args',(),{'out_dir':folder,'allow_paid':True,'i_authorize_model_calls':True,'model':'mock-local'})
            with patch('scripts.discovery.run_sers_evidence_successor_m14c2.source_call',return_value=lambda *_: {'text':'{bad', 'finish_reason':'stop','usage':None}):
                with self.assertRaises(ValueError):cmd_paid(args)
            self.assertTrue((folder/'m14c2_successor_RAW_RECEIPT.json').exists())
            self.assertTrue((folder/'M14C2_INVALID_SUBMISSION.json').exists())
            self.assertFalse((folder/RAW).exists())
    def test_report_is_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/CONTRACT).write_bytes(canon(self.contract()))
            (folder/RAW).write_bytes(canon(self.successor()))
            (folder/REPORT).write_text('corrupted')
            args=type('Args',(),{'out_dir':folder})
            with self.assertRaises(ValueError):cmd_verify(args)
    def test_synthetic_grid_has_five_rows(self):
        r=test_model_vectors(self.successor()['successor']['test_proposal'])
        self.assertEqual(len(r['rows']),5)
        self.assertEqual([x['x'] for x in r['rows']],[0.15,0.45,0.85,1.15,1.45])
    def test_non_numeric_coeff_rejected(self):
        s=self.successor()['successor']['test_proposal'];s['hypothesis_coefficients'][0]='0.1'
        with self.assertRaises(ValueError):test_model_vectors(s)

if __name__=='__main__':unittest.main()
