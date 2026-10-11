import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import os
import sys
import types
from unittest.mock import patch

from scripts.discovery import run_m15_computational_feedback_pilot as m


def fixture(base):
    """SYNTHETIC fixtures for contract tests only, not scientific evidence."""
    execution=base/'exec';execution.mkdir()
    requests=[];records=[];book={'cases':[]}
    for i in range(3):
        for side in ('control','intervention'):
            rid=f'TESTREQUEST{i}_{side}';hyp=f'TESTHYP{i}'
            req={'schema_version':'sers-meep-execution-request-v0', 'request_id':rid,
             'hypothesis_id':hyp,'source_numerical_plan_id':f'n{i}','source_numerical_plan_bundle_id':'nb',
             'source_solver_job_id':f'job{i}_{side}','source_solver_job_bundle_id':'jobs',
             'geometry_candidate_id':f'geo{i}_{side}','geometry_family':'core_shell_nanorod',
             'core_material':'Au','shell_material':'Ag','nanorod_length_nm':100+i*10,
             'nanorod_diameter_nm':35,'shell_thickness_nm':3+(side=='intervention'),
             'polarization':'parallel_to_nanorod_long_axis','surrounding_medium':'water',
             'field_component':'Ex','resolution_px_per_um':1200,'pml_thickness_um':0.2,
             'padding_um':0.3,'wavelength_min_nm':480,'wavelength_max_nm':780,
             'scientific_wavelength_min_nm':500,'scientific_wavelength_max_nm':750,
             'spectrum_frequency_points':20,'courant':0.25,'run_until_after_sources':100,
             'material_model':'meep_materials_library','background_index':1.33,
             'background_model':'water_nondispersive_n1p33','source_profile':'integrated_planewave_gaussian_pulse_v0',
             'coordinate_convention':'nanorod_x_incident_z','incident_propagation':'normal_to_nanorod_axis'}
            requests.append(req)
            worker=execution/'raw'/(rid+'.result.json');worker.parent.mkdir(exist_ok=True)
            spectrum=[]
            for j in range(20):
                spectrum.append({'wavelength_nm':500+10*j,
                     'scattering_cross_section_um2':(2 if side=='intervention' else 1)*(10/(1+abs(j-(7+i))))+.1})
            m.freeze(worker,{'schema_version':'sers-meep-worker-result-v0','worker_status':'completed',
                 'request_id':rid,'backend':'meep','backend_version':'test-fixture-meep-not-executed',
                 'field_component':'Ex','resolution_px_per_um':1200,'material_model':'meep_materials_library',
                 'evidence_eligibility':'screening_only','numerical_resolution_status':'coarse',
                 'spectrum':spectrum,'shadow_only':True,'scientific_interpretation_permitted':False,'hypothesis_rejection_authority':False})
            records.append({'request_id':rid,'execution_status':'completed','return_code':0,
                 'worker_result_path':str(worker),'worker_result_sha256':m.hashfile(worker),
                 'evidence_eligibility':'screening_only','numerical_resolution_status':'coarse'})
        book['cases'].append({'case_id':f'CASE{i}',
            'research_question':f'What field-selectivity difference remains after known plasmonic correction in nanorod system {i}?',
            'initial_hypothesis':f'Shell-thickness dependent scattering shows an observable interaction in chemistry context {i}.',
            'strong_null':f'Ordinary classical-EM scattering and geometric resonances can explain the changes in system {i}.',
            'control_request_id':f'TESTREQUEST{i}_control','intervention_request_id':f'TESTREQUEST{i}_intervention',
            'variable_under_test':'shell_thickness_nm'})
    request_path=execution/'sers_fdtd.meep_execution_requests.shadow.json'
    records_path=execution/'sers_fdtd.meep_execution_records.shadow.json'
    m.freeze(request_path,{'schema_version':'sers-meep-execution-request-bundle-v0','bundle_id':'B',
                            'request_count':len(requests),'requests':requests})
    m.freeze(records_path,{'schema_version':'sers-meep-execution-record-bundle-v0','source_request_bundle_id':'B',
                           'record_count':len(records),'records':records})
    m.freeze(execution/'sers_fdtd.meep_execution_shadow.manifest.json',
          {'execution_records_sha256':m.hashfile(records_path),'request_bundle_sha256':m.hashfile(request_path)})
    return execution,book


def candidate(k):
    return {key:f'This is a synthetic contract-test response for {key} in candidate {k} with detailed physical controls and no scientific claims.'
            for key in m.CANDIDATE_KEYS}


class M15Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.execution,self.book=fixture(self.root)
        self.inv=self.root/'inventory.json';self.casebook=self.root/'casebook.json';self.out=self.root/'pilot'
        m.freeze(self.casebook,self.book)
    def prepared(self):
        self.assertEqual(m.inventory([self.execution],self.inv)['completed_requests'],6)
        result=m.prepare(self.inv,self.casebook,self.out,1901)
        self.assertEqual(result['arm_calls'],6)
        return m.preflight(self.out)
    def receipts(self):
        plan,prompts=self.prepared()
        for p in prompts['prompts']:
            m.freeze(self.out/'paid'/(p['unit_id']+'.receipt.json'),{
               'schema':m.VERSION,'prompt_sha256':p['prompt_sha256'],'unit_id':p['unit_id'],
               'model':'synthetic-fixture-model','finish_reason':'stop',
               'raw':json.dumps(candidate(p['unit_id']))})
        return plan,prompts
    def test_matched_plan_and_prompt_separation(self):
        plan,prompts=self.prepared();self.assertEqual(len(plan['matched_cases']),3)
        self.assertEqual(len(prompts['prompts']),6)
        for item in plan['matched_cases']:
            self.assertEqual(item['observation']['numerical_gate'],'SCREENING_ONLY')
            self.assertFalse(item['observation']['independent_convergence_verified'])
        for a,b in zip(prompts['prompts'][::2],prompts['prompts'][1::2]):
            self.assertEqual(a['case_id'],b['case_id']);self.assertIn('numerical solver outcomes',a['prompt']['user'].lower())
            self.assertIn('sampled_peak_shift_nm',b['prompt']['user'])
            self.assertNotIn('sampled_peak_shift_nm',a['prompt']['user'])
    def test_does_not_promise_physics_or_novelty(self):
        self.prepared()
        x=m.load(self.out/m.PLAN);self.assertFalse(x['scientific_novelty_certified'])
        self.assertFalse(x['matched_cases'][0]['observation']['hypothesis_supported_or_falsified'])
    def test_refuses_source_tampering(self):
        self.prepared()
        p=self.execution/'raw/TESTREQUEST0_control.result.json'
        p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):m.preflight(self.out)
    def test_refuses_second_changed_physical_variable(self):
        self.prepared();by=m._read_sources(self.inv);a=by['TESTREQUEST0_control'];b=by['TESTREQUEST0_intervention']
        b['request_fields']['nanorod_length_nm']=999
        with self.assertRaisesRegex(ValueError,'MULTIPLE_PHYSICAL_VARIABLES_CHANGED'):m.compare(a,b,'shell_thickness_nm')
    def test_refuses_cross_hypothesis(self):
        self.prepared();by=m._read_sources(self.inv);a=by['TESTREQUEST0_control'];b=by['TESTREQUEST0_intervention'];b['request_fields']['hypothesis_id']='other'
        with self.assertRaisesRegex(ValueError,'NOT_SAME_INITIAL_HYPOTHESIS'):m.compare(a,b,'shell_thickness_nm')
    def test_exact_replay_disallows_change(self):
        self.prepared();(self.out/m.PROMPTS).write_text('{}')
        with self.assertRaises(Exception):m.preflight(self.out)
    def test_suggest_pairs_and_no_unearned_science_claim(self):
        m.inventory([self.execution],self.inv)
        path=self.root/'suggestions.json'
        result=m.suggest(self.inv,path)
        self.assertEqual(result['candidate_pair_count'],3)
        record=m.load(path)
        self.assertEqual(record['status'],'PAIR_SUGGESTIONS_NOT_READY_CASEBOOK')
        self.assertTrue(record['no_auto_hypothesis_generation'])
    def test_paid_requires_flags(self):
        self.prepared()
        with self.assertRaisesRegex(ValueError,'EXPLICIT_TWO_FLAGS_REQUIRED'):
            m.run_paid(self.out,'fixture-model','NO_KEY','https://invalid.test',1800,2,False,False)
        self.assertFalse((self.out/'M15_PAID_CALL_CONFIG.json').exists())
    def test_invalid_generated_json_is_rejected(self):
        with self.assertRaises(ValueError):m._valid_candidate('{"research_question":"not enough"}')
        with self.assertRaisesRegex(ValueError,'DUPLICATE_JSON_KEY'):m._valid_candidate('{"x":1,"x":2}')
    def test_blind_materialize_and_score(self):
        self.receipts();self.assertEqual(m.materialize(self.out)['blind_candidates'],6)
        self.assertFalse((self.out/m.REPORT).exists())
        with self.assertRaisesRegex(ValueError,'REVIEW_INCOMPLETE'):m.score(self.out)
        public=m.load(self.out/m.PUBLIC)
        reviewrows=[]
        for item in public['entries']:
            for rev in ('reviewer_alpha','reviewer_beta'):
                reviewrows.append({'blind_id':item['blind_id'],'reviewer_id':rev,
                        'prior_art_status':'uncertain','conceptual_delta':'2','null_discrimination':'2',
                        'independent_testability':'3','scientific_value':'2',
                        'rationale':'The proposed comparison does not substantiate novelty on the basis of the fixture alone.'})
        with (self.out/m.REVIEW).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=m.REVIEW_FIELDS);w.writeheader();w.writerows(reviewrows)
        report=m.score(self.out)
        self.assertEqual(report['cases'],3);self.assertTrue((self.out/m.REPORT).exists())
        saved=m.load(self.out/m.REPORT);self.assertFalse(saved['scientific_novelty_certified'])
        self.assertEqual(saved['mean_paired_B_minus_A']['conceptual_delta'],0)
    def test_requires_same_reviewers_for_every_candidate(self):
        self.receipts();m.materialize(self.out)
        public=m.load(self.out/m.PUBLIC);rows=[]
        for i,item in enumerate(public['entries']):
            for name in ('rA','rB' if i else 'rC'):
                rows.append({'blind_id':item['blind_id'],'reviewer_id':name,'prior_art_status':'uncertain',
                             'conceptual_delta':'2','null_discrimination':'2','independent_testability':'2',
                             'scientific_value':'2','rationale':'Insufficient external literature to verify any new scientific result in fixture.'})
        with (self.out/m.REVIEW).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=m.REVIEW_FIELDS);w.writeheader();w.writerows(rows)
        with self.assertRaisesRegex(ValueError,'REVIEWER_COVERAGE_IMBALANCE'):m.score(self.out)

    def test_paid_one_call_and_duplicate_no_retry(self):
        self.prepared()
        counter={'n':0}
        response=json.dumps(candidate('fake'))
        class FakeChat:
            def create(self,**kwargs):
                counter['n']+=1
                return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=response))],usage=None)
        class FakeOpenAI:
            def __init__(self,**kwargs):self.chat=types.SimpleNamespace(completions=FakeChat())
        with patch.dict(sys.modules,{'openai':types.SimpleNamespace(OpenAI=FakeOpenAI)}),patch.dict(os.environ,{'FAKE_API_KEY':'not-a-real-key'}):
            res=m.run_paid(self.out,'FAKE-MODEL','FAKE_API_KEY','https://invalid.example',1800,2,True,True)
            self.assertEqual(res['new_paid_calls'],2)
            self.assertEqual(res['completed_receipts'],2)
            res=m.run_paid(self.out,'FAKE-MODEL','FAKE_API_KEY','https://invalid.example',1800,2,True,True)
            self.assertEqual(res['new_paid_calls'],2)
            self.assertEqual(res['completed_receipts'],4)
        self.assertEqual(counter['n'],4)

    def test_provider_timeout_immutable_intent_no_second_charge(self):
        self.prepared()
        counter={'n':0}
        class FailingChat:
            def create(self,**kwargs):counter['n']+=1;raise RuntimeError('fake timeout')
        class FakeOpenAI:
            def __init__(self,**kwargs):self.chat=types.SimpleNamespace(completions=FailingChat())
        with patch.dict(sys.modules,{'openai':types.SimpleNamespace(OpenAI=FakeOpenAI)}),patch.dict(os.environ,{'FAKE_API_KEY':'not-a-real-key'}):
            one=m.run_paid(self.out,'FAKE-MODEL','FAKE_API_KEY','https://invalid.example',1800,2,True,True)
            self.assertEqual(one['status'],'PROVIDER_EXCEPTION_MAY_HAVE_CHARGED_NO_RETRY')
            two=m.run_paid(self.out,'FAKE-MODEL','FAKE_API_KEY','https://invalid.example',1800,2,True,True)
            self.assertEqual(two['status'],'BLOCKED_UNKNOWN_PAID_CALL_OUTCOME_NO_AUTOMATIC_RETRY')
        self.assertEqual(counter['n'],1)

    def test_double_prepare_is_blocked(self):
        self.prepared()
        with self.assertRaisesRegex(ValueError,'OUTPUT_DIR_EXISTS'):m.prepare(self.inv,self.casebook,self.out,1901)

if __name__=='__main__':unittest.main()
