"""Offline checks: actual CTMC mathematics, autonomous tool choice, safe replay, cost guard."""
from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_core.discovery.scientific_research_control import science_act_probe as instrument
from pipeline_core.discovery.scientific_research_control import science_act_pilot as pilot


def toy(a=1,b=1,name='independent_site'):
    return {'name':name,'states':['00','01','10','11'],
     'transitions':[
      {'source':'00','target':'01','rate':a,'group':'on'},
      {'source':'00','target':'10','rate':a,'group':'on'},
      {'source':'01','target':'11','rate':a,'group':'on'},
      {'source':'10','target':'11','rate':a,'group':'on'},
      {'source':'01','target':'00','rate':b,'group':'off'},
      {'source':'10','target':'00','rate':b,'group':'off'},
      {'source':'11','target':'01','rate':b,'group':'off'},
      {'source':'11','target':'10','rate':b,'group':'off'}],
     'observables':[{'name':'any_bright','bright_states':['01','10','11']}]}


def req():return {'models':[toy()], 'sweep':{'rate_group':'on','multipliers':[0.5,1,2,5]}}


def output(step,arm='plain',probe=None,choice='CONTINUE'):
    return {'operation':'CONSTRUCT' if step==1 else 'DERIVE',
     'scientific_work_content':{
      'attempted_question':'Can two independent states generate a measurable departure?',
      'premises_used':['Toy transitions are assumed, not Ye observations.'],
      'reasoning_or_operations':['Constructed a finite-state causal model with explicit transitions.'],
      'candidate_consequence':'Conditional predictions are proposed for bright-set statistics.',
      'counterexample_or_obstruction':'Finite models cannot certify a molecular cause.',
      'limits_and_uncertainties':['The mechanism remains unverified.'],
      'work_conclusion':'DERIVATION_PROPOSED'},
     'research_delta':'A conditional finite-state model and a possible experiment were proposed.',
     'next_action':choice,'next_research_question':'How does changing activation alter observed episode rates?' if choice=='CONTINUE' else None,
     'why_next_action':'The next action can distinguish conditional model regimes.',
     'cited_source_ids':[],'probe_request':probe}


class ProbeMath(unittest.TestCase):
    def test_two_independent_sites(self):
        r=instrument.evaluate(req())['models'][0]['results']
        self.assertAlmostEqual(r[1]['observables'][0]['bright_occupancy'],.75)
        self.assertAlmostEqual(r[1]['observables'][0]['episode_entry_rate'],.5)
        self.assertAlmostEqual(r[1]['observables'][0]['mean_bright_episode_duration'],1.5)
        self.assertLess(r[0]['observables'][0]['episode_entry_rate'],r[1]['observables'][0]['episode_entry_rate'])
        self.assertLess(r[2]['observables'][0]['episode_entry_rate'],r[1]['observables'][0]['episode_entry_rate'])
        self.assertAlmostEqual(r[0]['observables'][0]['episode_entry_rate'],r[2]['observables'][0]['episode_entry_rate'])
    def test_asymmetric_two_state(self):
        model={'name':'two_state','states':['OFF','ON'],'transitions':[
          {'source':'OFF','target':'ON','rate':2,'group':'on'},
          {'source':'ON','target':'OFF','rate':3,'group':'off'}],
          'observables':[{'name':'signal','bright_states':['ON']}]}
        r=instrument.evaluate({'models':[model], 'sweep':{'rate_group':'','multipliers':[1.]}})['models'][0]['results'][0]
        self.assertAlmostEqual(r['stationary_probabilities']['ON'],2/5)
        self.assertAlmostEqual(r['observables'][0]['episode_entry_rate'],6/5)
        self.assertAlmostEqual(r['observables'][0]['mean_bright_episode_duration'],1/3)
    def test_invalid_generator(self):
        bad=req();bad['models'][0]['transitions'][0]['source']='NOT_A_STATE'
        with self.assertRaises(ValueError):instrument.evaluate(bad)
    def test_reducible(self):
        m=toy();m['transitions']=[e for e in m['transitions'] if e['source'] in ('00','01') and e['target'] in ('00','01')]
        with self.assertRaises(ValueError):instrument.evaluate({'models':[m],'sweep':{'rate_group':'','multipliers':[1.]}})
    def test_no_code_expressions(self):
        r=req();r['models'][0]['transitions'][0]['rate']='__import__("os").system("true")'
        with self.assertRaises(ValueError):instrument.evaluate(r)


class PilotEpisode(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        (root/'experiments/science_act_pilot/runs').mkdir(parents=True)
        self.p_root=patch.object(pilot,'root',lambda:root);self.p_root.start();self.addCleanup(self.p_root.stop)
        self.out=root/'experiments/science_act_pilot/runs/EPISODE_01'
        pilot.prepare(self.out)
    def test_prepare_preview_and_guard(self):
        self.assertEqual(pilot.verify(self.out)['arms']['plain']['steps'],0)
        self.assertEqual(pilot.preview(self.out,'plain')['step'],1)
        with self.assertRaises(PermissionError):pilot.next_step(self.out,'plain')
    def test_scientific_tool_selection_continuation_and_finalization(self):
        seen=[]
        def backend(request):
            payload=json.loads(request['messages'][1]['content'])
            step=payload['science_step'];arm='instrument' if ('optional_instrument_description' in payload or 'previous_instrument_result' in payload and payload['tool_available'] is False and 'tool' in request['messages'][0]['content']) else 'plain'
            seen.append((step,payload['previous_instrument_result']))
            use_tool=step==1 and payload['tool_available']
            obj=output(step,arm,req() if use_tool else None,choice='CONTINUE' if step==1 else 'RETURN')
            return (json.dumps(obj),{'fake':True},'fake-id')
        pilot.next_step(self.out,'plain',backend=backend)
        pilot.next_step(self.out,'instrument',backend=backend)
        v=pilot.verify(self.out)
        self.assertEqual(v['arms']['plain']['steps'],1)
        self.assertEqual(v['arms']['instrument']['steps'],1)
        self.assertEqual(v['arms']['instrument']['records'][0]['instrument_result']['status'],'COMPUTED')
        self.assertEqual(v['arms']['plain']['records'][0]['instrument_result'],None)
        for arm in ('plain','instrument'):
            pilot.next_step(self.out,arm,backend=backend)
        self.assertIsNotNone(seen[-1][1])
        self.assertEqual(pilot.finalize(self.out)['arms'],{'plain':2,'instrument':2})
        self.assertTrue(pilot.verify(self.out)['finalized'])
        with self.assertRaises(RuntimeError):pilot.preview(self.out,'plain')
    def test_return_without_forced_continuation(self):
        def backend(request):return (json.dumps(output(1,choice='RETURN')),None,'fake')
        pilot.next_step(self.out,'plain',backend=backend)
        pilot.next_step(self.out,'instrument',backend=backend)
        self.assertEqual(pilot.finalize(self.out)['arms'],{'plain':1,'instrument':1})
    def test_saved_raw_recovery_and_tamper(self):
        sd=self.out/'arms/plain/step_0001'
        pv=pilot.preview(self.out,'plain')
        sd.mkdir(parents=True)
        lock=pilot.read(self.out/'lock.json');pkt,_=pilot._packet()
        r=pilot._request(lock,pkt,'plain',1,[])
        pilot.write(sd/'request.json',r)
        pilot.write(sd/'attempt.json',{'request_sha256':pv['request_sha256'],
             'prev_record_sha256':None,'status':'BILLING_UNKNOWN_UNTIL_RAW_SAVED','created_utc':'fixed'})
        pilot.write(sd/'raw_response.json',{'raw_text':json.dumps(output(1,choice='RETURN')),
              'usage':None,'response_id':'fake'})
        self.assertEqual(pilot.recover(self.out,'plain')['status'],'STEP_COMMITTED_FROM_SAVED_RAW_NO_CALL')
        pilot.verify(self.out)
        path=sd/'normalized.json'
        obj=pilot.read(path);obj['research_delta']='tampered'
        path.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):pilot.verify(self.out)
    def test_unknown_source_rejected_without_retry(self):
        def backend(request):
            o=output(1,choice='RETURN');o['cited_source_ids']=['FAKE:P999']
            return (json.dumps(o),None,'fake')
        with self.assertRaises(RuntimeError):pilot.next_step(self.out,'plain',backend=backend)
        self.assertTrue((self.out/'arms/plain/step_0001/raw_response.json').exists())
        with self.assertRaises(ValueError):pilot.recover(self.out,'plain')
    def test_invalid_tool_model_is_feedback_not_fabricated(self):
        bad=req();bad['models'][0]['transitions'][0]['rate']=-1
        def backend(request):return (json.dumps(output(1,'instrument',bad)),None,'fake')
        pilot.next_step(self.out,'instrument',backend=backend)
        res=pilot.verify(self.out)['arms']['instrument']['records'][0]['instrument_result']
        self.assertEqual(res['status'],'REJECTED_INVALID_MODEL')

if __name__=='__main__':unittest.main()
