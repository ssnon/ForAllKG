"""Offline safety/contract tests. No provider calls, no W08 mutations."""
from __future__ import annotations
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_core.discovery.scientific_research_control import m19_deep_work_e1 as m
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as b
from pipeline_core.discovery.scientific_research_control.research_work_v12_contracts import ResearchWorkProduct

REQUEST={
 'research_question':'Can ensemble blinking duration laws transfer under immobilization and finite site conditions?',
 'target_idea_id':None,
 'gap_or_assumption':'Transfer constraints not established',
 'work_kind':'REASON','work_instructions':'Construct and derive a finite ensemble model with thresholding.',
 'available_inputs':['YE:P2','WORK:01','work_0002_fake:product'],
 'source_ids':['WORK:01'],
 'expected_information':'A conditional theory for durations',
 'stop_or_failure_condition':'Stop if model assumptions cannot be bounded'}

def mock_content(tag):
 return {'attempted_question':REQUEST['research_question'],
  'premises_used':['Assume discrete independent response sites.'],
  'reasoning_or_operations':['Derived an explicitly conditional claim for '+tag+'.'],
  'candidate_consequence':'Different thresholds change detected event intervals under stated assumptions.',
  'counterexample_or_obstruction':None,
  'limits_and_uncertainties':['No original time-series or experimental verification.'],
  'work_conclusion':'DERIVATION_PROPOSED'}

def mock_output(tag,cont=False):
 return {'operation':'DERIVE' if not cont else 'EXPLORE',
  'research_delta':'Conditional finite ensemble detection reasoning for '+tag+' with assumptions.',
  'provisional_work_content':mock_content(tag),
  'next_step':'CONTINUE' if cont else 'RETURN',
  'next_question_or_test':'Explore alternative threshold regimes carefully' if cont else None,
  'reason_for_choice':'There are physical assumptions to test.' if cont else 'The bounded result is enough.',
  'source_or_prior_work_refs':['WORK:01']}

class M19E1Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.home=(Path(m.__file__).resolve().parents[3]/'experiments'/'m19_e1')
  cls.home.mkdir(parents=True,exist_ok=True)

 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='m19_test_',dir=self.home)
  self.addCleanup(self.temp.cleanup)
  self.out=Path(self.temp.name)/'run'
  self.w08=Path(self.temp.name)/'fixture_w08'
  wid=m.EXPECTED_WORK_ID
  (self.w08/'work'/wid).mkdir(parents=True)
  original={'model':'openai/gpt-5.6-luna','temperature':0.5,'max_tokens':4500,
    'messages':[{'role':'system','content':'Source policy (unverified)'},
        {'role':'user','content':json.dumps({
          'work_id':wid,'work_request':REQUEST,'read_scientific_sources':[{'id':'WORK:01','abstract':'Frozen abstract'}],
          'read_work_products':[{'work_id':'work_0002_fake','content':{'work_conclusion':'SOURCE_MATERIAL_OBTAINED'}}],
          'work_content_schema':{},'prior_W06_unverified_research_compact':{},
          'prior_W07_unverified_work':{},'current_scientific_focus':{}},sort_keys=True)}],
       'role':'work','input_state_sha256':'a'*64}
  b.write_new(self.w08/'work'/wid/'llm_request.json',original)
  self.pre={'pending_work':{'status':'REQUESTED','work_id':wid,'request':REQUEST}}
  self.context={'manifest':{'source_snapshot_sha256':'s'*64},
                'state':{'decision_count':3,'event_count':6},'sources':{},'prior':{},'prior_w07':{}}
  def fake_verify(x):return copy.deepcopy(self.context)
  with patch.object(m.v18,'verify',side_effect=fake_verify),\
       patch.object(m,'_reconstruct_prework',return_value=self.pre),\
       patch.object(m.v18,'_make_call_request',return_value=original),\
       patch.object(m,'EXPECTED_W08_SHA',b.sha(self.context['state'])):
   m.prepare(w08=self.w08,out=self.out)


 def backend(self,tag,cont=False):
  def func(req):return json.dumps(mock_output(tag,cont),ensure_ascii=False),{'prompt_tokens':42,'completion_tokens':58},'mock-'+tag
  return func

 def test_identical_initial_prompts_and_independent_B(self):
  qB=m.preview(self.out,'B')['request'];qC=m.preview(self.out,'C')['request']
  self.assertEqual(qB,qC)
  self.assertNotIn('m19_continuation',qB['messages'][1]['content'])
  m.next_step(self.out,'B',backend=self.backend('b1',cont=False))
  self.assertEqual(qB,m.preview(self.out,'B')['request'])
  m.next_step(self.out,'B',backend=self.backend('b2'))
  m.next_step(self.out,'B',backend=self.backend('b3'))
  self.assertEqual(m.finalize(self.out,'B')['candidates'],3)
  self.assertTrue(m.verify(self.out)['b_count']==3)

 def test_serial_context_and_return(self):
  m.next_step(self.out,'C',backend=self.backend('first',cont=True))
  req2=m.preview(self.out,'C')['request'];data=json.loads(req2['messages'][1]['content'])
  self.assertIn('m19_continuation',data)
  self.assertIn('first',data['m19_continuation']['previous_scientific_steps'][0]['research_delta'])
  m.next_step(self.out,'C',backend=self.backend('second'))
  self.assertEqual(m.finalize(self.out,'C')['candidates'],2)
  with self.assertRaises(RuntimeError):m.preview(self.out,'C')
  c=m.verify(self.out);product=b.read(self.out/'C'/'final_product.json')['entries'][-1]['product']
  ResearchWorkProduct.model_validate(product)
  self.assertEqual(c['c_count'],2)

 def test_unknown_ref_and_raw_recovery(self):
  bad=mock_output('raw');bad['source_or_prior_work_refs']=['NONEXISTENT:123']
  with self.assertRaises(RuntimeError):
   m.next_step(self.out,'C',backend=lambda req:(json.dumps(bad),None,'x'))
  self.assertEqual(m.verify(self.out)['incomplete']['C']['status'],'SAVED_RAW_NEEDS_RECOVER_NO_CALL')
  with self.assertRaises(ValueError):m.recover(self.out)
  with self.assertRaises(RuntimeError):m.preview(self.out,'C')

 def test_recover_persisted_valid_raw_without_recall(self):
  req=m.preview(self.out,'C')['request']
  sd=m._step_dir(self.out,'C',1);sd.mkdir(parents=True)
  b.write_new(sd/'request.json',req)
  b.write_new(sd/'attempt.json',{'status':'STARTED_UNKNOWN_UNTIL_RECEIPT',
    'request_sha256':b.sha(req),'parent_step_sha256':None})
  b.write_new(sd/'raw_response.json',{'raw_text':json.dumps(mock_output('persisted')),'usage':None,'response_id':'mock'})
  self.assertEqual(m.recover(self.out)['status'],'RECOVERED_SAVED_RAW_NO_CALL')
  self.assertEqual(m.verify(self.out)['c_count'],1)

 def test_unknown_billing_and_no_paid_without_double_key(self):
  with self.assertRaises(PermissionError):m.next_step(self.out,'B',allow_paid=False)
  req=m.preview(self.out,'B')['request']
  sd=m._step_dir(self.out,'B',1);sd.mkdir(parents=True)
  b.write_new(sd/'request.json',req)
  b.write_new(sd/'attempt.json',{'request_sha256':b.sha(req),'parent_step_sha256':None})
  with self.assertRaises(RuntimeError):m.recover(self.out)
  with self.assertRaises(RuntimeError):m.preview(self.out,'B')

 def test_tamper_record_detected(self):
  m.next_step(self.out,'B',backend=self.backend('a'))
  sd=m._step_dir(self.out,'B',1)
  obj=b.read(sd/'normalized.json');obj['research_delta']='tampered';(sd/'normalized.json').write_text(json.dumps(obj))
  with self.assertRaises(ValueError):m.verify(self.out)

 def test_no_root_or_broad_outputs(self):
  for bad in (Path('/tmp/m19'),m._safe_repo_root(self.out)/'experiments'/'other'):
   with self.assertRaises(ValueError):m._safe_repo_root(bad)

if __name__=='__main__':unittest.main()
