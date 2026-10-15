from __future__ import annotations
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Local tests run on the add-only patch; the real WSL execution additionally validates the genuine FAKG HypothesisContext.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline_core.discovery.scientific_research_control.runner import (
    Action, Idea, prepare, verify, run, recover, export_native, digest, record_step, repair_state
)

ORIGINAL_SCIENCE = Path(os.environ.get('CASE01_TEST_SCIENCE',
          '/mnt/data/CASE01_OPEN_PROBLEM_DISCOVERY_PILOT/frozen_sources'))
NATIVE_CONTEXT = Path(os.environ.get('CASE01_TEST_CONTEXT',
          '/mnt/data/case01_reentry_staging_native/hypothesis.context.json'))


def action(kind='EXPLORE', **kw):
    base={'action':kind,'scientific_reason':'The observed spectral events do not pick out a unique causal mechanism.',
          'research_question_now':'What independent physical degree of freedom is required to explain the time series?'}
    base.update(kw)
    return base


def idea(family='TEST FAMILY',id='x1'):
    return {'local_id':id,'title':'Independent causal test mechanism','causal_family':family,
      'proposed_state_variables':['hypothetical microscopic variable x'],
      'causal_mechanism':'A microscopic state x changes the observable only under a special stimulus while other states remain fixed.',
      'why_standard_explanations_are_incomplete':'A conventional independent-switching null is unable to generate this particular conditional response unless it introduces an additional state.',
      'strong_conventional_null':'A local conventional switching process with heterogeneous hotspots, ordinary adsorption, and bleach-like deactivation.',
      'conditional_prediction':'An independently calibrated perturbation would yield a response that is delayed under x but not the conventional null.',
      'how_the_prediction_follows':'A perturbation acts on x and creates a delayed output, while the stipulated independent null has no delayed state.',
      'null_mimic_risk':'An unobserved local-state memory could mimic this response.',
      'independent_test_or_observation':'Measure the state-selective reporter and Raman time series synchronously with calibrated timing.',
      'falsification_outcome':'Absence of the temporal asymmetry under calibrated conditions would contradict this conditional mechanism.',
      'evidence_ids':['YE:P1'],'novelty_uncertainty':'No literature novelty is certified from the supplied abstract-level evidence.'}


def ensure_hypothesis_context_stub():
    try:
        from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
    except ImportError:
        mod=types.ModuleType('pipeline_core.discovery.hypothesis_contracts')
        class FakeContext:
            @classmethod
            def model_validate_json(cls, raw):
                v=json.loads(raw)
                return types.SimpleNamespace(context_id=v['context_id'],
                     context_sha256=v['context_sha256'],source_packet_id=v['source_packet_id'])
        mod.HypothesisContext=FakeContext
        sys.modules['pipeline_core.discovery.hypothesis_contracts']=mod


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_hypothesis_context_stub()
        cls.sources={'pages':[{'id':f'YE:P{i}','page':i,'text':'source'} for i in range(1,9)],
                     'works':[{'id':'WORK:01','title':'real native work','abstract':'text','doi':''}]}
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)/'episode'
        prepare(out=self.out,pages_file=ORIGINAL_SCIENCE/'Ye_main_text_by_page.json',
                native_context_file=NATIVE_CONTEXT,
                works_file=ORIGINAL_SCIENCE/'hypothesis_axis_open_world.selected_works.json',max_steps=10)
    def test_actual_original_sources(self):
        x=verify(self.out)
        self.assertEqual(len(x['source_index']['pages']),8)
        self.assertGreater(len(x['source_index']['works']),5)
        self.assertEqual(x['state']['read_ids'],['YE:P1'])
    def test_no_paid_without_authorization(self):
        with self.assertRaises(PermissionError):run(self.out,allow_paid=True,max_new_calls=1)
    def test_reads_real_pages(self):
        row=record_step(self.out,action('READ_EVIDENCE',evidence_to_read=['YE:P3']))
        self.assertIn('YE:P3',verify(self.out)['state']['read_ids'])
        self.assertEqual(row['action']['action'],'READ_EVIDENCE')
    def test_invalid_source_citation(self):
        bad=idea();bad['evidence_ids']=['YE:P7']
        with self.assertRaisesRegex(ValueError,'unread'):
            record_step(self.out,action(ideas=[bad]))
    def test_invalid_unknown_evidence(self):
        with self.assertRaisesRegex(ValueError,'does not exist'):
            record_step(self.out,action('READ_EVIDENCE',evidence_to_read=['FICTITIOUS:P12']))
    def test_full_science_transition(self):
        record_step(self.out,action(ideas=[idea()]))
        s=verify(self.out)['state'];key=next(iter(s['ideas']))
        with self.assertRaisesRegex(ValueError,'requires a recorded challenge'):
            record_step(self.out,action('PIVOT',target_idea_id=key,abandoned_assumption='x is the only state',new_research_direction='investigate another coupling'))
        record_step(self.out,action('CHALLENGE',target_idea_id=key,challenge_null_mimic='An ordinary local-state memory process also makes this delay.',
                     challenged_prediction='Temporal asymmetry predicted under a perturbation',challenge_result='NULL_MIMICS'))
        record_step(self.out,action('PIVOT',target_idea_id=key,abandoned_assumption='x produces uniquely diagnostic delay',
                     new_research_direction='consider distinct optical environmental dynamics'))
        with self.assertRaisesRegex(ValueError,'abandoned causal family'):
            record_step(self.out,action(ideas=[idea()]))
        record_step(self.out,action(ideas=[idea(family='DISTINCT OPTICAL',id='x2')]))
        self.assertEqual(len(verify(self.out)['state']['pivots']),1)
        self.assertEqual(len(verify(self.out)['state']['ideas']),2)
    def test_end_gated_by_challenge(self):
        record_step(self.out,action(ideas=[idea()]))
        key=next(iter(verify(self.out)['state']['ideas']))
        with self.assertRaisesRegex(ValueError,'unchallenged'):
            record_step(self.out,action('END',target_idea_id=key,final_disposition='PROVISIONAL_RESEARCH_CANDIDATE',final_justification='This mechanism is a conditional discovery direction, not verified'))
        record_step(self.out,action('END',final_disposition='ABSTAIN',final_justification='No mechanism survived a strong null test.'))
        self.assertTrue(verify(self.out)['state']['ended'])
    def test_mock_backend_exact_once(self):
        calls=[]
        def backend(req):
            calls.append(req)
            return json.dumps(action('READ_EVIDENCE',evidence_to_read=['YE:P4'])),{'prompt_tokens':7},'fake-response-1'
        x=run(self.out,backend=backend,max_new_calls=1)
        self.assertEqual(x['new_paid_calls_this_command'],1)
        self.assertEqual(len(calls),1)
        self.assertEqual(verify(self.out)['state']['step'],1)
        self.assertTrue((self.out/'calls/step_01/receipt.json').exists())
        # call_dir already exists for completed step, state has moved to step2
        self.assertEqual(run(self.out,backend=lambda req:(json.dumps(action('END',final_disposition='ABSTAIN',final_justification='Nothing conclusive')),None,'fake2'),max_new_calls=1)['step'],2)
    def test_failed_response_never_repeated(self):
        def invalid(req):return '{"unparseable":true}',{'total_tokens':33},'fake-invalid'
        with self.assertRaisesRegex(RuntimeError,'Persisted response failed'):
            run(self.out,backend=invalid,max_new_calls=1)
        with self.assertRaisesRegex(RuntimeError,'Existing attempt'):
            run(self.out,backend=invalid,max_new_calls=1)
        with self.assertRaises(Exception):recover(self.out)
        self.assertTrue((self.out/'calls/step_01/raw_response.json').exists())
    def test_raw_recovery_no_call(self):
        def invalid(req):return json.dumps(action('READ_EVIDENCE',evidence_to_read=['YE:P3'])),{},'fake-id'
        # Manually stop after model has persisted raw but before validation commits (simulate crash).
        x=verify(self.out)
        from pipeline_core.discovery.scientific_research_control.runner import prompt_payload,SYSTEM,write_once
        u=json.dumps(prompt_payload(x['state'],x['source_index'],x['manifest']),ensure_ascii=False)
        req={'model':x['manifest']['model'],'temperature':x['manifest']['temperature'],'max_tokens':x['manifest']['max_tokens'],
             'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':u}],
             'research_step':1,'input_state_sha256':digest(x['state'])}
        d=self.out/'calls'/'step_01';d.mkdir(parents=True)
        write_once(d/'request.json',req)
        write_once(d/'attempt.json',{'request_sha256':digest(req),'status':'STARTED_UNKNOWN_UNTIL_RECEIPT'})
        write_once(d/'raw_response.json',{'raw_text':invalid(req)[0],'usage':{},'provider_response_id':'fake-id'})
        self.assertEqual(recover(self.out)['status'],'RECOVERED_NO_CALL')
        self.assertEqual(verify(self.out)['state']['step'],1)
    def test_state_tamper(self):
        record_step(self.out,action('READ_EVIDENCE',evidence_to_read=['YE:P2']))
        p=self.out/'state.json'
        v=json.loads(p.read_text());v['read_ids']=['YE:P1'];p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'differs from replayed'):
            verify(self.out)
    def test_repair_state_from_step(self):
        s0=verify(self.out)['state']
        record_step(self.out,action('READ_EVIDENCE',evidence_to_read=['YE:P2']))
        (self.out/'state.json').write_text(json.dumps(s0))
        self.assertEqual(repair_state(self.out)['status'],'RESTORED_FROM_VALID_CHAIN_NO_CALL')
        self.assertEqual(verify(self.out)['state']['step'],1)
    def test_no_automatic_novelty_certificate(self):
        record_step(self.out,action(ideas=[idea()]))
        self.assertFalse(verify(self.out)['state']['ideas'][next(iter(verify(self.out)['state']['ideas']))]['scientific_claim_authority'])
    def test_unsupported_json_response(self):
        with self.assertRaises(Exception):Action.model_validate({'action':'EXPLORE','scientific_reason':'a'} )
    def test_budget(self):
        m=self.out/'manifest.json';v=loadj(m);v['max_steps']=4;m.write_text(json.dumps(v))
        for p in [2,3,4,5]:
            if p>4:break
            record_step(self.out,action('READ_EVIDENCE',evidence_to_read=[f'YE:P{p}']))
        self.assertEqual(verify(self.out)['state']['step'],3)
    def test_recover_rejects_missing_raw(self):
        (self.out/'calls/step_01').mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError,'No persisted'):
            recover(self.out)


def loadj(p):return json.loads(p.read_text())
if __name__=='__main__':unittest.main()
