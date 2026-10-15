"""Offline contract suite; mocks are fixtures, never research results."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
KIT=Path(os.environ.get('CASE01_REENTRY_KIT', str(REPO)))
SCRIPT=REPO/'scripts/discovery/run_case01_science_reentry.py'
spec=importlib.util.spec_from_file_location('case01_science_reentry',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
NATIVE_ZIP=Path(os.environ.get('CASE01_REENTRY_NATIVE_ZIP',str(KIT/'frozen/NATIVE_P0_INPUT_TEST_FIXTURE.zip')))
SCIENCE=Path(os.environ.get('CASE01_REENTRY_SCIENCE_DIR','/mnt/data/CASE01_OPEN_PROBLEM_DISCOVERY_PILOT/frozen_sources'))


def sample_review():
    return {'reviews':[{'idea_id':id,'proposed_causal_state':'Unidentified physical state; unspecified effective transition',
        'math_or_physics_check':'No supplied rate scale follows from the premise without measuring relaxation',
        'strongest_null_and_mimicry':'Heterogeneous independent site kinetics could reproduce that pattern',
        'independent_test':'Simultaneous independently controlled relevant measurement',
        'missing_evidence':'Time-resolved independent state variable has not been observed',
        'diagnosis':'INDETERMINATE','diagnosis_reason':'Observations are consistent with multiple physical states',
        'evidence_ids':[],'reviewer_abstained':True} for id in m.PARENTS]}

def sample_action(action='REOPEN_PROBLEM_ROOT'):
    return {'action':action,'target_idea_id':m.PARENTS[0] if action in ('CONTINUE_LINEAGE','DIVERSIFY_FROM_PARENT') else None,
            'diagnostic_idea_ids':list(m.PARENTS),'rationale':'The present causal state does not identify a discriminating mechanism',
            'abandoned_assumption':'Do not assume existing parent mechanistic state is identified',
            'research_question_for_next_step':'Which independent degree of freedom, if any, can explain the unresolved signal?',
            'evidence_requests':['Independent state measurement'] if action=='ACQUIRE_EVIDENCE' else [],
            'uncertainty':'The review does not constitute empirical falsification'}

class OfflineContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=Path(tempfile.mkdtemp(prefix='fakg_case01_reentry_test_'))
        cls.native=cls.base/'native';cls.native.mkdir()
        with zipfile.ZipFile(NATIVE_ZIP) as z:
            for name in m.FIELDS:(cls.native/name).write_bytes(z.read(name))
    @classmethod
    def tearDownClass(cls):shutil.rmtree(cls.base)
    def setUp(self):
        self.tmp=Path(tempfile.mkdtemp(dir=self.base));self.out=self.tmp/'out'
    def tearDown(self):shutil.rmtree(self.tmp)
    def _prepare(self):
        return m.prepare(self.native,SCIENCE,KIT/'frozen',self.out,'openai/gpt-5.6-luna',0.25,6500)
    def _review_fixture(self):
        self._prepare()
        plan,_=m.verify(self.out)
        rv=sample_review()
        receipt={'source_kind':'PROVIDER_FAKG_NATIVE_P0_SELF_REVIEW_EXPERIMENTAL','finish_reason':'stop',
           'prompt_sha256':plan['review_prompt_sha256'],'raw':json.dumps(rv),'model':plan['model'],
           'test_fixture_only':True}
        m.write_new(self.out/'responses/review.json',receipt)
        m.write_new(self.out/'parsed/review.json',rv)
        return m.freeze_action(self.out)
    def test_p0_and_sources_exact_match(self):
        x=m.preflight(self.native,SCIENCE,KIT/'frozen')
        self.assertEqual(x['status'],'NATIVE_P0_SCIENCE_READY')
        self.assertEqual(len(x['native']['p0.execution.json']['g4_population_nodes']),8)
    def test_prepared_input_excludes_earlier_scientist_answers(self):
        x=self._prepare();self.assertEqual(x['paid_calls'],0)
        m.verify(self.out)
        prompt=m.load(self.out/'REVIEW_PROMPT.json')
        self.assertNotIn('ABSTAIN_NO_DISTINCT_HYPOTHESIS',m.canon(prompt))
        self.assertNotIn('Trap-field Stark gating',m.canon(prompt))
        self.assertTrue('source' not in prompt or not prompt['type'].endswith('SIDECAR'))
    def test_source_byte_change_fails_closed(self):
        p=self.tmp/'mutable';shutil.copytree(self.native,p)
        (p/'p0.execution.json').write_bytes((p/'p0.execution.json').read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'NATIVE_SOURCE_MISSING_OR_CHANGED'):
            m.preflight(p,SCIENCE,KIT/'frozen')
    def test_prompt_mutation_fails_closed(self):
        self._prepare()
        p=self.out/'REVIEW_PROMPT.json';p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'FROZEN_SOURCES_OR_PROMPT_DRIFT'):m.verify(self.out)
    def test_paid_gate_blocks_before_api_call(self):
        self._prepare()
        cmd=[sys.executable,str(SCRIPT),'run-paid-review','--out',str(self.out)]
        p=subprocess.run(cmd,capture_output=True,text=True,env={**os.environ,'CASE01_REENTRY_ALLOW_PAID':'NO'})
        self.assertEqual(p.returncode,2);self.assertIn('PAID_MODEL_CALL_BLOCKED',p.stderr)
        self.assertFalse((self.out/'attempts/review.json').exists())
    def test_review_validation_rejects_wrong_identity(self):
        v=sample_review();v['reviews'][0]['idea_id']='fake'
        with self.assertRaisesRegex(ValueError,'REVIEW_PARENT_IDENTITY_MISMATCH'):
            m.parse_review(json.dumps(v))
    def test_review_validation_rejects_claimed_nonabstain(self):
        v=sample_review();v['reviews'][0]['diagnosis']='POTENTIALLY_DISTINCT'
        with self.assertRaisesRegex(ValueError,'ABSTENTION_MUST_BE_INDETERMINATE'):
            m.parse_review(json.dumps(v))
    def test_action_freeze_has_exact_review_binding(self):
        self._review_fixture()
        m.verify_action_prompt(self.out)
        self.assertEqual(m.load(self.out/'ACTION_PROMPT.json')['review_response_sha256'],m.sha(sample_review()))
        self.assertFalse(m.load(self.out/'ACTION_FREEZE.json')['native_sidecar_injected'])
    def test_altered_review_receipt_cannot_influence_action(self):
        self._review_fixture()
        p=self.out/'responses/review.json'
        a=m.load(p);a['raw']=a['raw'].replace('Unidentified physical state','Injected cause');p.write_text(json.dumps(a))
        with self.assertRaisesRegex(ValueError,'FROZEN_REVIEW_OR_ACTION_PROMPT_CHANGED'):
            m.verify_action_prompt(self.out)
    def test_reopen_stops_without_faking_native_root(self):
        self._review_fixture()
        freeze=m.load(self.out/'ACTION_FREEZE.json')
        act=sample_action()
        m.write_new(self.out/'responses/action.json',{'source_kind':'PROVIDER_FAKG_NATIVE_REENTRY_ACTION_EXPERIMENTAL',
            'finish_reason':'stop','prompt_sha256':freeze['action_prompt_sha256'],'raw':json.dumps(act),
            'model':'openai/gpt-5.6-luna','test_fixture_only':True})
        m.write_new(self.out/'parsed/action.json',act)
        r=m.route(self.out)
        self.assertEqual(r['status'],'NEEDS_NATIVE_ROOT_ENTRYPOINT_NOT_EXECUTED')
        self.assertFalse(m.load(self.out/'REENTRY_ROUTING_DECISION.json')['new_mechanism_generated'])
    def test_parent_diversify_does_not_execute_sis(self):
        self._review_fixture()
        freeze=m.load(self.out/'ACTION_FREEZE.json');act=sample_action('DIVERSIFY_FROM_PARENT')
        m.write_new(self.out/'responses/action.json',{'source_kind':'PROVIDER_FAKG_NATIVE_REENTRY_ACTION_EXPERIMENTAL',
             'finish_reason':'stop','prompt_sha256':freeze['action_prompt_sha256'],'raw':json.dumps(act),
             'test_fixture_only':True})
        m.write_new(self.out/'parsed/action.json',act)
        self.assertIn('NOT_EXECUTED',m.route(self.out)['status'])
    def test_invalid_m61_style_import_is_rejected(self):
        self._prepare();plan,_=m.verify(self.out)
        m.write_new(self.out/'responses/review.json',{'source_kind':'EXTERNAL_SIDECAR','finish_reason':'stop',
          'prompt_sha256':plan['review_prompt_sha256'],'raw':json.dumps(sample_review())})
        m.write_new(self.out/'parsed/review.json',sample_review())
        with self.assertRaisesRegex(ValueError,'NON_NATIVE_OR_UNVERIFIED_REVIEW_RECEIPT'):
            m.freeze_action(self.out)
    def test_action_rejects_fake_parent_root(self):
        x=sample_action();x['target_idea_id']=m.PARENTS[0]
        with self.assertRaisesRegex(ValueError,'ROOT_EVIDENCE_OR_STOP_CANNOT_CARRY_PARENT'):
            m.parse_action(json.dumps(x))
    def test_two_model_stages_are_separate(self):
        self._prepare()
        self.assertFalse((self.out/'ACTION_PROMPT.json').exists())
        self.assertFalse((self.out/'responses').exists())
        self.assertEqual(m.load(self.out/'PLAN.json')['planned_paid_calls'],2)

if __name__=='__main__':unittest.main()
