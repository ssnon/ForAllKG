"""Zero-network M19-E3 selected-work tests; native v1.8 integration checked in user's WSL prepare."""
import copy,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_core.discovery.scientific_research_control import m19_e3_deep_work as w
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as b

D4_WORK={'research_question':'Derive quantitative threshold and time scaling conditions for finite gold nanorod ensemble blinking',
         'gap_or_assumption':'Missing regime dependent invariants',
         'work_kind':'REASON','work_instructions':'Derive explicit scaling relations without forcing any mediator',
         'expected_information':'Conditional physics and counterexamples',
         'stop_or_failure_condition':'Cannot derive any constrained relation',
         'available_inputs':['work_0003_abc:product'],'source_ids':[], 'target_idea_id':None}

def output(step,choice,refs=None):
 return json.dumps({'operation':'DERIVE' if step==1 else 'DEVELOP',
   'research_delta':f'Derived conditional regime relation step {step} without asserting Ye confirmation',
   'provisional_work_content':{
       'attempted_question':'Obtain a conditional relation for multimode ensemble thresholds',
       'premises_used':['Explicit fixed site states are a model not a measured fact'],
       'reasoning_or_operations':['Define finite state emission and count boundary transitions under a specified rule'],
       'candidate_consequence':f'Conditional derived relation number {step} requires a specified threshold and rates',
       'counterexample_or_obstruction':'Alternative hidden state may reproduce same summary',
       'limits_and_uncertainties':['Not demonstrated for Ye time series'],
       'work_conclusion':'DERIVATION_PROPOSED'},
   'next_step':choice,'next_question_or_test':'Test a boundary case with a different threshold' if choice=='CONTINUE' else None,
   'reason_for_choice':'The conditional result has to be further analyzed' if choice=='CONTINUE' else 'Enough bounded research for current task; do not invent validation',
   'source_or_prior_work_refs':refs or []},ensure_ascii=False)

class TestE3DeepWork(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory()
  self.source=Path(self.temp.name)/'e3'
  self.episode=self.source/w.entry.FORK_SUBDIR
  self.episode.mkdir(parents=True)
  b.write_new(self.episode/'state.json',{'dummy':'E3 D4 immutable source'})
  self.out=w.repo_root()/'experiments'/'m19_e3_work'/'runs'/('TEST_'+Path(self.temp.name).name)
  self.state={'decision_count':4,'event_count':7,
              'pending_work':{'work_id':'work_0004_test','status':'REQUESTED','request':D4_WORK}}
  self.p0=patch.object(w,'D4_SHA',b.sha(self.state))
  self.p0.start()
  self.req={'model':'mock-luna','temperature':0,'max_tokens':5000,
            'input_state_sha256':w.D4_SHA,'role':'work',
            'messages':[{'role':'system','content':'Return ONLY a JSON object'},
                        {'role':'user','content':json.dumps({'work_request':D4_WORK,
                         'read_scientific_sources':[],
                         'read_work_products':[{'artifact_ref':'work_0003_abc:product'}]})}]}
  self.c={'state':self.state,'manifest':{'source_snapshot_sha256':'f'*64}}
  self.p1=patch.object(w.entry,'verify',side_effect=lambda x:{'status':'DECISION4_COMMITTED_VERIFIED',
      'fork_state_sha256':w.D4_SHA,'pending_work_kind':'REASON'})
  self.p2=patch.object(w.v18,'verify',side_effect=lambda x:copy.deepcopy(self.c))
  self.p3=patch.object(w.v18,'_make_call_request',return_value=self.req)
  for p in (self.p1,self.p2,self.p3):p.start()
 def tearDown(self):
  for p in (self.p3,self.p2,self.p1):p.stop()
  self.p0.stop()
  if self.out.exists():shutil.rmtree(self.out)
  tmp=self.out.with_name(self.out.name+'.preparing')
  if tmp.exists():shutil.rmtree(tmp)
  self.temp.cleanup()
 def prepare(self):return w.prepare(e3=self.source,out=self.out)
 def test_prepared_same_native_request(self):
  self.prepare();v=w.verify(self.out)
  self.assertEqual(v['completed_count'],0)
  self.assertEqual(b.read(self.out/'lock.json')['native_work_request'],self.req)
  self.assertTrue(w.preview(self.out)['json_mode_ready'])
 def test_two_steps_then_return_finalize(self):
  self.prepare()
  calls=iter([output(1,'CONTINUE'),output(2,'RETURN')])
  def backend(request):return next(calls),{'total_tokens':300},'mock-response'
  self.assertEqual(w.next_step(self.out,backend=backend)['next_choice'],'CONTINUE')
  self.assertEqual(w.next_step(self.out,backend=backend)['next_choice'],'RETURN')
  self.assertEqual(w.verify(self.out)['completed_count'],2)
  self.assertEqual(w.finalize(self.out)['status'],'CANDIDATE_WORK_PRODUCT_FINALIZED_NO_CALL')
  self.assertFalse(w.inspect(self.out)['native_episode_modified'])
  with self.assertRaisesRegex(RuntimeError,'FINALIZED'):w.preview(self.out)
 def test_early_return_is_not_overridden(self):
  self.prepare();w.next_step(self.out,backend=lambda req:(output(1,'RETURN'),{},'r'))
  with self.assertRaisesRegex(RuntimeError,'AUTONOMOUS_RETURN'):w.preview(self.out)
  self.assertEqual(w.finalize(self.out)['steps'],1)
 def test_unknown_reference_logged_but_not_approved(self):
  self.prepare();w.next_step(self.out,backend=lambda req:(output(1,'RETURN',['work_0999_fake']),{},'r'))
  r=w.verify(self.out)['steps'][0]['record']
  self.assertEqual(r['source_reference_audit']['invalid_ref_ids_not_accepted_as_evidence'],['work_0999_fake'])
 def test_billing_gate(self):
  self.prepare()
  with patch.dict('os.environ',{'FAKG_RESEARCH_ALLOW_PAID':'','OPENROUTER_API_KEY':''}):
   with self.assertRaises(PermissionError):w.next_step(self.out,allow_paid=False)
  self.assertEqual(w.verify(self.out)['attempt_count'],0)
 def test_saved_raw_offline_recovery(self):
  self.prepare()
  # Simulate a response persisted just before parser/recording.
  req=w._request(b.read(self.out/'lock.json'),[],1)
  sd=w.step_dir(self.out,1);sd.mkdir(parents=True)
  b.write_new(sd/'request.json',req)
  b.write_new(sd/'attempt.json',{'request_sha256':b.sha(req),'previous_step_sha256':None})
  b.write_new(sd/'raw_response.json',{'raw_text':output(1,'RETURN'),'usage':{},'response_id':'saved'})
  self.assertEqual(w.verify(self.out)['incomplete']['status'],'RAW_SAVED_OFFLINE_RECOVER_REQUIRED')
  self.assertEqual(w.recover(self.out)['status'],'SAVED_RAW_COMMITTED_NO_NEW_CALL')
  self.assertEqual(w.verify(self.out)['completed_count'],1)
 def test_no_duplicate_paid_after_empty_response(self):
  self.prepare();sd=w.step_dir(self.out,1);sd.mkdir(parents=True)
  req=w._request(b.read(self.out/'lock.json'),[],1)
  b.write_new(sd/'request.json',req)
  b.write_new(sd/'attempt.json',{'request_sha256':b.sha(req),'previous_step_sha256':None})
  self.assertEqual(w.verify(self.out)['incomplete']['status'],'UNKNOWN_BILLING_NO_RAW_NEVER_RETRY')
  with self.assertRaisesRegex(RuntimeError,'UNKNOWN_BILLING'):w.recover(self.out)
  with self.assertRaisesRegex(RuntimeError,'INCOMPLETE'):w.preview(self.out)
 def test_source_tamper_detected(self):
  self.prepare()
  (self.episode/'state.json').write_text('tampered')
  with self.assertRaisesRegex(ValueError,'E3_SOURCE_FORK_CHANGED'):w.verify(self.out)
 def test_no_external_output(self):
  with self.assertRaisesRegex(ValueError,'OUTSIDE_REPO'):
   w.prepare(e3=self.source,out=Path(self.temp.name)/'bad-output')
 def test_finalize_rejects_premature_continue(self):
  self.prepare();w.next_step(self.out,backend=lambda req:(output(1,'CONTINUE'),{},'r'))
  with self.assertRaisesRegex(RuntimeError,'CONTINUATION_SELECTED'):w.finalize(self.out)

if __name__=='__main__':unittest.main()
