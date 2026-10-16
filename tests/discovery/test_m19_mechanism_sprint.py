from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_mechanism_sprint as s

class MechanismSprintTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory(prefix='offline_',dir=s.root()/'experiments/m19_mechanism_sprint/runs')
        self.out=Path(self.t.name)/'study'
    def tearDown(self):self.t.cleanup()
    def backend(self,next_step='RETURN',invalid=False):
        def fake(req):
            assert req['model']=='openai/gpt-5.6-luna'
            out={'operation':'CONSTRUCT','research_delta':'Constructed a minimal candidate mechanism with conditional dynamics and an explicit competing null model.',
              'scientific_work_content':{'attempted_question':'Can a constrained interfacial process explain intermittent Raman modulation?',
                'premises_used':['This is a hypothetical physical model; empirical support is not established.'],
                'reasoning_or_operations':['Write dz/dt=k_on I (1-z)-k_off z and state that any prediction depends on model-specific coupling.'],
                'candidate_consequence':'A conditional relaxation time (k_on I+k_off)^(-1) follows from the stated one-state model.',
                'counterexample_or_obstruction':'An unmeasured field response with matching rates could mimic this relaxation.',
                'limits_and_uncertainties':['No actual optical marker observations available.'],
                'work_conclusion':'DERIVATION_PROPOSED'},
              'next_step':next_step,
              'next_research_action':'Develop an explicit mode-resolved null model for this result.' if next_step=='CONTINUE' else None,
              'why_next_step':'A precise comparison is useful if continuing; otherwise the current result is conditional.',
              'cited_source_ids':['IMPOSSIBLE:P9'] if invalid else ['YE:P2']}
            return json.dumps(out),{'total_tokens':110},'test-id'
        return fake
    def prepare(self):return s.prepare(out=self.out)
    def test_prepare_and_verify(self):
        q=self.prepare();self.assertIn('PREPARED',q['status'])
        st=s.verify(self.out);self.assertEqual(st['arms']['plain']['attempts'],0)
    def test_equal_frozen_sources(self):
        self.prepare();lk=s.base.read(self.out/'lock.json');p,_=s._bundle()
        a=s._request(lk,p,'plain',[],1);b=s._request(lk,p,'work',[],1)
        A=json.loads(a['messages'][1]['content']);B=json.loads(b['messages'][1]['content'])
        self.assertEqual(A['frozen_sources_and_prior_unverified_research'],B['frozen_sources_and_prior_unverified_research'])
        self.assertEqual(A['shared_research_assignment'],B['shared_research_assignment'])
        self.assertNotIn('work_request',A)
        self.assertIn('work_request',B)
        self.assertNotIn('M19-E4',a['messages'][1]['content'])
    def test_run_one_each_finalize(self):
        self.prepare()
        for arm in ('plain','work'):
            out=s.next_step(out=self.out,arm=arm,backend=self.backend())
            self.assertEqual(out['new_paid_calls_this_command'],0)
        f=s.finalize(out=self.out)
        self.assertIn('FINALIZED',f['status']);self.assertTrue(s.verify(self.out)['finalized'])
    def test_two_steps_then_return(self):
        self.prepare()
        s.next_step(out=self.out,arm='plain',backend=self.backend('CONTINUE'))
        self.assertEqual(s.preview(out=self.out,arm='plain')['step'],2)
        s.next_step(out=self.out,arm='plain',backend=self.backend('RETURN'))
        s.next_step(out=self.out,arm='work',backend=self.backend('RETURN'))
        self.assertTrue(s.finalize(out=self.out))
    def test_stop_not_overridden(self):
        self.prepare();s.next_step(out=self.out,arm='plain',backend=self.backend('RETURN'))
        with self.assertRaises(RuntimeError):s.preview(out=self.out,arm='plain')
    def test_arm_billing_and_env_gate(self):
        self.prepare()
        with self.assertRaises((PermissionError,RuntimeError)):
            s.next_step(out=self.out,arm='plain',allow_paid=False)
        self.assertEqual(s.verify(self.out)['arms']['plain']['attempts'],0)
    def test_unauthorized_location(self):
        with self.assertRaises(ValueError):s.prepare(out=Path(self.t.name)/'outside'/'../../../../tmp/splice')
    def test_raw_without_normalized_recovers(self):
        self.prepare();lk=s.base.read(self.out/'lock.json');p,_=s._bundle()
        d=s.step_dir(self.out,'plain',1);d.mkdir(parents=True)
        req=s._request(lk,p,'plain',[],1)
        s.base.write_new(d/'request.json',req)
        s.base.write_new(d/'attempt.json',{'request_sha256':s.sha(req),'previous_step_sha256':None,'status':'UNKNOWN','created_utc':'2026'})
        raw=self.backend()(req)[0];s.base.write_new(d/'raw_response.json',{'raw_text':raw,'usage':None,'response_id':'test'})
        self.assertEqual(s.verify(self.out)['arms']['plain']['incomplete'],'RAW_SAVED_OFFLINE_RECOVER_REQUIRED')
        self.assertEqual(s.recover(out=self.out,arm='plain')['status'],'RAW_REPLAYED_NO_PAID')
    def test_unknown_billing_no_raw(self):
        self.prepare();lk=s.base.read(self.out/'lock.json');p,_=s._bundle()
        d=s.step_dir(self.out,'plain',1);d.mkdir(parents=True)
        req=s._request(lk,p,'plain',[],1)
        s.base.write_new(d/'request.json',req)
        s.base.write_new(d/'attempt.json',{'request_sha256':s.sha(req),'previous_step_sha256':None})
        self.assertEqual(s.verify(self.out)['arms']['plain']['incomplete'],'UNKNOWN_BILLING_NO_RAW_NEVER_RETRY')
        with self.assertRaises(RuntimeError):s.preview(out=self.out,arm='plain')
    def test_packet_tamper_rejected(self):
        self.prepare();lk=s.base.read(self.out/'lock.json');lk['packet_file_sha256']='0'*64
        (self.out/'lock.json').write_text(json.dumps(lk))
        with self.assertRaises(ValueError):s.verify(self.out)
    def test_invalid_reference_preserved_not_certified(self):
        self.prepare();q=s.next_step(out=self.out,arm='plain',backend=self.backend(invalid=True))
        self.assertEqual(q['invalid_source_ids'],['IMPOSSIBLE:P9'])
        self.assertFalse(s.verify(self.out)['scientific_validation'])
    def test_finalize_requires_both_arms(self):
        self.prepare();s.next_step(out=self.out,arm='plain',backend=self.backend())
        with self.assertRaises(RuntimeError):s.finalize(out=self.out)

if __name__=='__main__':unittest.main()
