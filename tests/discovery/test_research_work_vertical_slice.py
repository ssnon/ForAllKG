"""Purely offline tests; no scientific claim is supported by the synthetic fixtures."""
from __future__ import annotations
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_core.discovery.scientific_research_control import research_work_runner as rw


def example_source():
    return {'pages':[{'id':f'YE:P{i}','page':i,'text':(
        'Blinking in surface-enhanced Raman spectrum: 10.1103/4yg7-1f85. Protocol fixture only.'
        if i==1 else f'Protocol fixture page {i}. No science is asserted.')} for i in range(1,9)],
        'works':[{'id':'WORK:01','title':'Synthetic citation for offline tests','doi':'',
                  'abstract':'This is a protocol fixture, not evidence of physics.',
                  'origin':'FAKG_OPEN_WORLD_SELECTED_WORK','source_support':'METADATA_ABSTRACT_ONLY'}]}


def request(kind='REASON'):
    return {'research_question':'Which unverified assumption should be examined next in this protocol test?',
            'target_idea_id':None,'gap_or_assumption':'The mock work has no validated conclusion.',
            'work_kind':kind,'work_instructions':'Produce a traceable, explicitly unverified assessment.',
            'available_inputs':['YE:P1'] if kind=='REASON' else [],
            'source_ids':['YE:P2'] if kind=='READ_FROZEN_SOURCE' else [],
            'expected_information':'A limitation or explicitly inconclusive mock result.',
            'stop_or_failure_condition':'Stop if source support is absent or premise unresolved.'}


def decision(action='DEEPEN', kind='REASON', interpretation=None, end=False):
    return {'action':'END' if end else action,
            'scientific_reason':'The mock protocol needs a bounded work item or an explicit research stop.',
            'research_question_now':'What does the new work result actually establish about our current uncertainty?',
            'work_request':None if end else request(kind),
            'interpretation':interpretation,'final_disposition':'ABSTAIN' if end else None,
            'final_justification':'The protocol fixture deliberately does not establish scientific progress.' if end else None}


def reason_response(result='INCONCLUSIVE'):
    return {'attempted_question':'Which unverified assumption should be examined next?',
            'premises_used':['Protocol fixture only, not actual science.'],
            'reasoning_or_operations':['Insufficient information to infer any physical mechanism.'],
            'candidate_consequence':None,'counterexample_or_obstruction':None,
            'limits_and_uncertainties':['The fixture contains no genuine experimental data.'],
            'work_conclusion':result}


def backend_for(obj, counter=None):
    def backend(_req):
        if counter is not None:
            counter.append(1)
        return json.dumps(obj,ensure_ascii=False), {'mock_calls':1},'offline-mock-response'
    return backend


class WorkVerticalSlice(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)/'new_independent_work_episode'
        self.out.mkdir()
        src=example_source()
        rw._check_sources(src)
        rw.write_new(self.out/'sources.json',src)
        rw.write_new(self.out/'manifest.json',{
            'schema_version':rw.PROTOCOL,'question':rw.QUESTION,'source_snapshot_sha256':rw.sha(src),
            'model':'test-model-no-network','max_decisions':10,'max_tokens':4000,'temperature':0,
            'research_lineage':'FRESH_NO_PRIOR_R01_R02_IDEAS'})
        rw.write_new(self.out/'state.json',rw.init_state(rw.QUESTION))

    def first_reason(self):
        return rw.decision(self.out,backend=backend_for(decision()))

    def performed_reason(self):
        self.first_reason()
        return rw.execute(self.out,backend=backend_for(reason_response()))

    def interpretation(self, assessment='NO_PROGRESS'):
        c=rw.verify(self.out)
        p=c['state']['pending_work']
        return {'work_id':p['work_id'],'product_sha256':p['product_sha256'],
                'assessment':assessment,
                'what_changed':'No validated physics has changed; this is a synthetic mock.',
                'supported_by_artifact_refs':[f'{p["work_id"]}:product'],
                'remaining_gap':'Original physical question remains genuinely unresolved.',
                'followup_reason':'Retain uncertainty and choose another bounded research question.',
                'external_scientific_validation':False}

    def test_full_reason_next_decision_requires_exact_product(self):
        self.performed_reason()
        payload=rw.decision_payload(rw.verify(self.out))
        self.assertEqual(payload['pending_interpretation']['work_result']['work_conclusion'],'INCONCLUSIVE')
        self.assertEqual(payload['pending_interpretation']['product_sha256'],
                         rw.verify(self.out)['state']['pending_work']['product_sha256'])
        rw.decision(self.out,backend=backend_for(decision(interpretation=self.interpretation())))
        c=rw.verify(self.out)
        self.assertEqual(len(c['state']['interpretations']),1)
        self.assertEqual(c['state']['pending_work']['status'],'REQUESTED')
        self.assertEqual(c['state']['decisions'][-1]['interpreted_prior'],c['state']['work_history'][-1]['work_id'])

    def test_source_read_actual_snapshot_and_feedback(self):
        rw.decision(self.out,backend=backend_for(decision('READ_EVIDENCE','READ_FROZEN_SOURCE')))
        self.assertEqual(rw.execute(self.out)['new_paid_calls_this_command'],0)
        c=rw.verify(self.out)
        self.assertIn('YE:P2',c['state']['read_ids'])
        self.assertEqual(c['state']['work_history'][-1]['content']['source_material'][0]['content']['text'],
                         example_source()['pages'][1]['text'])
        rw.decision(self.out,backend=backend_for(decision(interpretation=self.interpretation('EVIDENCE_ACQUIRED'))))
        self.assertEqual(len(rw.verify(self.out)['state']['interpretations']),1)

    def test_interpretation_changes_next_scientific_context(self):
        self.performed_reason()
        it1=self.interpretation('NO_PROGRESS')
        it2=self.interpretation('PROPOSED_COUNTEREXAMPLE')
        old_payload=rw.decision_payload(rw.verify(self.out))
        # Controlling only interpretation; the same work artifact is bound to both paths.
        one=copy.deepcopy(rw.verify(self.out)['state'])
        two=copy.deepcopy(one)
        c=rw.verify(self.out)
        d1=decision(interpretation=it1)
        d2=decision(interpretation=it2)
        s1=rw.apply_event(one,'DECISION',d1,manifest=c['manifest'],sources=c['sources'])
        s2=rw.apply_event(two,'DECISION',d2,manifest=c['manifest'],sources=c['sources'])
        self.assertNotEqual(rw.sha(s1),rw.sha(s2))
        self.assertEqual(s1['interpretations'][-1]['assessment'],'NO_PROGRESS')
        self.assertEqual(s2['interpretations'][-1]['assessment'],'PROPOSED_COUNTEREXAMPLE')
        self.assertEqual(old_payload['pending_interpretation']['product_sha256'],it1['product_sha256'])
        self.assertFalse(s2['science_certified'])

    def test_pending_work_cannot_be_skipped(self):
        self.first_reason()
        with self.assertRaisesRegex(ValueError,'work must be performed'):
            rw.decision_payload(rw.verify(self.out))

    def test_missing_interpretation_rejected_even_for_END(self):
        self.performed_reason()
        with self.assertRaises(RuntimeError):
            rw.decision(self.out,backend=backend_for(decision(end=True)))
        raw=self.out/'calls'/'decision_0002'/'raw_response.json'
        self.assertTrue(raw.is_file())
        with self.assertRaisesRegex(RuntimeError,'existing decision attempt'):
            rw.decision(self.out,backend=backend_for(decision(end=True)))

    def test_wrong_digest_rejected(self):
        self.performed_reason()
        it=self.interpretation()
        it['product_sha256']='0'*64
        with self.assertRaises(RuntimeError):
            rw.decision(self.out,backend=backend_for(decision(interpretation=it)))
        self.assertEqual(rw.verify(self.out)['state']['decision_count'],1)

    def test_wrong_source_rejected_without_science_claim(self):
        d=decision()
        d['work_request']['available_inputs']=['WORK:NEVER']
        with self.assertRaises(RuntimeError):
            rw.decision(self.out,backend=backend_for(d))
        self.assertFalse(rw.verify(self.out)['state']['science_certified'])

    def test_arbitrary_target_id_rejected(self):
        d=decision()
        d['work_request']['target_idea_id']='research_idea:invented'
        with self.assertRaises(RuntimeError):
            rw.decision(self.out,backend=backend_for(d))

    def test_tamper_work_product_detected(self):
        self.performed_reason()
        p=rw.verify(self.out)['state']['pending_work']['work_id']
        f=self.out/'work'/p/'product.json'
        f.write_text(f.read_text().replace('INCONCLUSIVE','DERIVATION_PROPOSED'))
        with self.assertRaisesRegex(ValueError,'linked artifact changed'):
            rw.verify(self.out)

    def test_tamper_raw_response_detected(self):
        self.performed_reason()
        p=rw.verify(self.out)['state']['pending_work']['work_id']
        f=self.out/'work'/p/'raw_response.json'
        f.write_text(f.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'linked artifact changed'):
            rw.verify(self.out)

    def test_tamper_frozen_source_detected(self):
        d=example_source();d['pages'][0]['text']='tampered 10.1103/4yg7-1f85 Blinking in surface-enhanced Raman spectrum'
        (self.out/'sources.json').write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'frozen source'):
            rw.verify(self.out)

    def test_recover_saved_reason_without_second_call(self):
        self.first_reason()
        call_count=[]
        with patch.object(rw,'_work_commit',side_effect=RuntimeError('crash after raw')):
            with self.assertRaises(RuntimeError):
                rw.execute(self.out,backend=backend_for(reason_response(),call_count))
        self.assertEqual(len(call_count),1)
        p=rw.verify(self.out)['state']['pending_work']['work_id']
        self.assertTrue((self.out/'work'/p/'raw_response.json').exists())
        with self.assertRaisesRegex(RuntimeError,'existing work attempt'):
            rw.execute(self.out,backend=backend_for(reason_response(),call_count))
        self.assertEqual(rw.recover(self.out)['status'],'RECOVERED_WORK_NO_CALL')
        self.assertEqual(len(call_count),1)
        self.assertEqual(len(rw.verify(self.out)['state']['work_history']),1)

    def test_recover_saved_decision_without_second_call(self):
        calls=[]
        with patch.object(rw,'_decision_commit',side_effect=RuntimeError('crash after raw')):
            with self.assertRaises(RuntimeError):
                rw.decision(self.out,backend=backend_for(decision(),calls))
        self.assertEqual(len(calls),1)
        with self.assertRaisesRegex(RuntimeError,'existing decision attempt'):
            rw.decision(self.out,backend=backend_for(decision(),calls))
        self.assertEqual(rw.recover(self.out)['status'],'RECOVERED_DECISION_NO_CALL')
        self.assertEqual(len(calls),1)
        self.assertEqual(rw.verify(self.out)['state']['decision_count'],1)

    def test_recover_after_product_written_before_event(self):
        self.first_reason()
        with patch.object(rw,'_record_event',side_effect=RuntimeError('crash after saved product')):
            with self.assertRaises(RuntimeError):
                rw.execute(self.out,backend=backend_for(reason_response()))
        p=rw.verify(self.out)['state']['pending_work']['work_id']
        before=(self.out/'work'/p/'product.json').read_bytes()
        self.assertEqual(rw.recover(self.out)['status'],'RECOVERED_WORK_NO_CALL')
        self.assertEqual((self.out/'work'/p/'product.json').read_bytes(),before)

    def test_state_recovery_after_committed_event(self):
        with patch.object(rw,'refresh_state',side_effect=RuntimeError('crash after event')):
            with self.assertRaises(RuntimeError):
                rw.decision(self.out,backend=backend_for(decision()))
        with self.assertRaisesRegex(ValueError,'state.json differs'):
            rw.verify(self.out)
        self.assertEqual(rw.recover(self.out)['status'],'STATE_RESTORED_FROM_VERIFIED_EVENTS_NO_CALL')
        self.assertEqual(rw.verify(self.out)['state']['decision_count'],1)

    def test_no_paid_call_without_explicit_approval(self):
        before=list(self.out.iterdir())
        with patch.dict('os.environ',{'FAKG_RESEARCH_ALLOW_PAID':''}):
            with self.assertRaises(PermissionError):
                rw.decision(self.out,allow_paid=True)
        self.assertEqual(list(self.out.iterdir()),before)

    def test_end_abstain_not_level_ii(self):
        rw.decision(self.out,backend=backend_for(decision(end=True)))
        state=rw.verify(self.out)['state']
        self.assertTrue(state['ended'])
        self.assertEqual(state['final_disposition'],'ABSTAIN')
        self.assertFalse(state['science_certified'])
        with self.assertRaises(ValueError):
            rw.decision_payload(rw.verify(self.out))

    def test_bad_work_raw_recover_does_not_bypass_validation(self):
        self.first_reason()
        with self.assertRaises(RuntimeError):
            rw.execute(self.out,backend=backend_for({'invalid':'no conclusion'}))
        with self.assertRaises(Exception):
            rw.recover(self.out)
        self.assertEqual(rw.verify(self.out)['state']['pending_work']['status'],'REQUESTED')


    def test_failure_receipt_is_explicit_not_fake_science(self):
        self.first_reason()
        result=rw.record_work_failure(self.out,execution_status='FAILED',
             reason='The selected analysis tool is not available in this prototype.')
        self.assertEqual(result['status'],'WORK_FAILED_NO_CALL')
        s=rw.verify(self.out)['state']
        self.assertEqual(s['work_history'][-1]['execution_status'],'FAILED')
        self.assertEqual(s['work_history'][-1]['origin'],'EXECUTION_STATUS_RECEIPT')
        self.assertFalse(s['science_certified'])
        rw.decision(self.out,backend=backend_for(decision(interpretation=self.interpretation('NO_PROGRESS'))))
        self.assertEqual(len(rw.verify(self.out)['state']['interpretations']),1)

    def test_blocked_work_remains_interpretable_and_not_paid(self):
        self.first_reason()
        rw.record_work_failure(self.out,execution_status='BLOCKED',
             reason='The research cannot proceed without a separately approved tool.')
        self.assertEqual(rw.verify(self.out)['state']['work_history'][-1]['execution_status'],'BLOCKED')
        with self.assertRaises(ValueError):
            rw.execute(self.out,backend=backend_for(reason_response()))
        rw.decision(self.out,backend=backend_for(decision(end=True,interpretation=self.interpretation('INSUFFICIENT_EVIDENCE'))))
        self.assertTrue(rw.verify(self.out)['state']['ended'])

    def test_paid_attempt_cannot_be_disguised_as_failure(self):
        self.first_reason()
        with patch.object(rw,'_work_commit',side_effect=RuntimeError('failure after provider saved')):
            with self.assertRaises(RuntimeError):
                rw.execute(self.out,backend=backend_for(reason_response()))
        with self.assertRaisesRegex(RuntimeError,'existing attempt'):
            rw.record_work_failure(self.out,execution_status='FAILED',
                   reason='Never erase saved paid provider responses as failed work.')

    def test_mock_decision_uses_real_product_content_to_branch(self):
        def branch(challenge):
            with tempfile.TemporaryDirectory() as t:
                src=example_source();out=Path(t)/'case';out.mkdir()
                rw.write_new(out/'sources.json',src)
                rw.write_new(out/'manifest.json',{
                    'schema_version':rw.PROTOCOL,'question':rw.QUESTION,
                    'source_snapshot_sha256':rw.sha(src),'model':'test','max_decisions':4,
                    'max_tokens':4000,'temperature':0})
                rw.write_new(out/'state.json',rw.init_state(rw.QUESTION))
                rw.decision(out,backend=backend_for(decision()))
                response=reason_response('COUNTEREXAMPLE_PROPOSED' if challenge else 'INCONCLUSIVE')
                if challenge:
                    response['counterexample_or_obstruction']='Synthetic contradictory premise, not real physics.'
                rw.execute(out,backend=backend_for(response))
                def choose(req):
                    payload=json.loads(req['messages'][1]['content'])
                    result=payload['pending_interpretation']['work_result']
                    old=rw.verify(out)['state']['pending_work']
                    self.assertEqual(payload['pending_interpretation']['product_sha256'],old['product_sha256'])
                    con=result['work_conclusion']
                    what={'work_id':old['work_id'],'product_sha256':old['product_sha256'],
                        'assessment':'PROPOSED_COUNTEREXAMPLE' if con=='COUNTEREXAMPLE_PROPOSED' else 'NO_PROGRESS',
                        'what_changed':'A synthetic conditional challenge.' if con=='COUNTEREXAMPLE_PROPOSED' else 'No scientific claim was established.',
                        'supported_by_artifact_refs':[f'{old["work_id"]}:product'],
                        'remaining_gap':'Test fixture has no actual physical evidence.',
                        'followup_reason':'Select a different mock method in light of the work.',
                        'external_scientific_validation':False}
                    d=decision(action='DEEPEN' if con=='COUNTEREXAMPLE_PROPOSED' else 'READ_EVIDENCE',
                               kind='REASON' if con=='COUNTEREXAMPLE_PROPOSED' else 'READ_FROZEN_SOURCE',
                               interpretation=what)
                    return json.dumps(d),None,'mock-conditional-selection'
                rw.decision(out,backend=choose)
                state=rw.verify(out)['state']
                return state['decisions'][-1]['action'],state['interpretations'][-1]['assessment']
        self.assertEqual(branch(False),('READ_EVIDENCE','NO_PROGRESS'))
        self.assertEqual(branch(True),('DEEPEN','PROPOSED_COUNTEREXAMPLE'))

    def test_final_decision_slot_must_not_open_uninterpretable_work(self):
        path=self.out/'manifest.json'
        m=rw.read(path);m['max_decisions']=2
        path.write_text(json.dumps(m))
        self.performed_reason()
        with self.assertRaisesRegex(ValueError,'reserve the final decision'):
            rw.apply_event(rw.verify(self.out)['state'],'DECISION',
                   decision(interpretation=self.interpretation()),
                   manifest=rw.verify(self.out)['manifest'],sources=rw.verify(self.out)['sources'])
        rw.decision(self.out,backend=backend_for(decision(interpretation=self.interpretation(),end=True)))
        self.assertTrue(rw.verify(self.out)['state']['ended'])

    def test_preparation_uses_only_clean_source_snapshot(self):
        import sys,types
        mod=types.ModuleType('pipeline_core.discovery.hypothesis_contracts')
        class FakeNativeContext:
            @staticmethod
            def model_validate_json(value):
                from types import SimpleNamespace
                parsed=json.loads(value)
                return SimpleNamespace(context_id=parsed['context_id'],
                    context_sha256=parsed['context_sha256'],source_packet_id=parsed['source_packet_id'])
        mod.HypothesisContext=FakeNativeContext
        d=Path(self.tmp.name)/'inputs';d.mkdir()
        pages=[{'page':i,'text':r['text']} for i,r in enumerate(example_source()['pages'],start=1)]
        works=[{'work':{'title':'Synthetic citation for offline tests','abstract':'Not evidence.','doi':''}}]
        (d/'pages.json').write_text(json.dumps(pages))
        (d/'works.json').write_text(json.dumps(works))
        (d/'native.json').write_text(json.dumps({'context_id':'fake','context_sha256':'f'*64,'source_packet_id':'fake'}))
        out=Path(self.tmp.name)/'new_prepare'
        with patch.dict(sys.modules,{'pipeline_core.discovery.hypothesis_contracts':mod}):
            rw.prepare(out=out,ye_pages=d/'pages.json',selected_works=d/'works.json',native_context=d/'native.json')
        c=rw.verify(out)
        self.assertEqual(c['state']['event_count'],0)
        self.assertNotIn('prior_research',c['sources'])
        self.assertEqual(c['manifest']['research_lineage'],'FRESH_NO_PRIOR_R01_R02_IDEAS')
        self.assertFalse(c['state']['science_certified'])

if __name__=='__main__':
    unittest.main(verbosity=2)
