"""Offline M19-E2 tests with deterministic mock model, NEVER network."""
from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_e2_json_v2 as e
from pipeline_core.discovery.scientific_research_control import m19_e2_ctmc as c
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as base

SPEC={'on_rates':[1.,1.],'off_rates':[1.,1.],'weights':[1.,1.],'threshold':1.5}
INITIAL={'question_refined':'What exact ensemble threshold statistics survive two independent site switches?',
  'model_and_assumptions':['Two independent reversible sites each switch between zero and unit contributions',
                           'Each site switches at constant rates, threshold exceeds a single-site contribution'],
  'derived_claim':'The ensemble event duration is not automatically the same as the individual site duration.',
  'mathematical_reasoning':['The aggregate process is finite-state Markov, so its threshold-crossing times are first-passage times.'],
  'candidate_consequence':'Detection threshold can change inferred duration statistics without any shared state.',
  'limitations':['This toy model does not identify the Ye experimental mechanism.'],
  'proposed_calculator_spec':SPEC,'claim_to_check':'Ensemble mean bright duration is 1/2 when switching rates are one.',
  'next_scientific_question':'How does a changed detection threshold modify the inferred duration law?'}
FOLLOW={'refined_scientific_claim':'In the stated finite two-site model the mean duration differs with threshold and rate.',
 'changed_or_retained_why':'The stated assumptions determine a finite Markov first-passage problem, not a real SERS mechanism.',
 'mathematical_or_physical_reasoning':['I used the model constraints to condition the claim without adding experimental evidence.'],
 'remaining_uncertainties':['Actual site switching kinetics in Ye remain unknown.'],
 'new_conditional_prediction':'Changing threshold while holding site dynamics fixed changes observed bright-event statistics.',
 'next_research_action':'Study conditional duration distributions under alternative thresholds.',
 'scientific_validation':False}

class E2JSONV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.out=e.repo_root()/'experiments'/'m19_e2'/'runs'/('TEST_'+Path(self.tmp.name).name)
        if self.out.exists():raise RuntimeError('UNEXPECTED_TEST_COLLISION')
        e.prepare(self.out)
    def tearDown(self):
        import shutil
        shutil.rmtree(self.out)
        self.tmp.cleanup()
    @staticmethod
    def _mock_for(phase):
        def model(request):
            obj=INITIAL if phase=='initial' else FOLLOW
            return (json.dumps(obj),{'total_tokens':100},'fake-'+phase)
        return model
    def _initial(self):
        e.next_step(self.out,'initial',backend=self._mock_for('initial'))
        e.calculate(self.out)
    def test_json_mode_first_request_and_preview(self):
        req=e._request(self.out,'initial')
        self.assertIn('json',req['messages'][0]['content'].lower())
        self.assertTrue(e.preview(self.out,'initial')['json_mode_ready'])

    def test_json_mode_preflight_rejects_incompatible_request(self):
        bad={'model':'mock','messages':[{'role':'system','content':'Give a structured answer'},
                                        {'role':'user','content':'Give a structured answer'}]}
        with self.assertRaisesRegex(ValueError,'JSON_MODE_PREFLIGHT'):
            e._assert_json_mode_ready(bad)

    def test_json_mode_followups_are_checked(self):
        self._initial()
        for phase in ('control','tool'):
            self.assertTrue(e.preview(self.out,phase)['json_mode_ready'])
            e._assert_json_mode_ready(e._request(self.out,phase))

    def test_provider_error_does_not_automatically_retry(self):
        def error(_req):raise RuntimeError('simulated provider HTTP 400')
        with self.assertRaisesRegex(RuntimeError,'simulated provider HTTP 400'):
            e.next_step(self.out,'initial',backend=error)
        self.assertEqual(e.verify(self.out)['phases']['initial'],'UNKNOWN_BILLING_NO_RAW_NO_RETRY')
        with self.assertRaises(ValueError):e.preview(self.out,'initial')
        with self.assertRaises(ValueError):e.recover(self.out)

    def test_and_exact(self):
        z=c.evaluate(SPEC)
        self.assertAlmostEqual(z['mean_bright_interval'],.5)
        self.assertAlmostEqual(z['bright_stationary_fraction'],.25)
        self.assertAlmostEqual(z['stationary_bright_entry_rate'],.5)
    def test_or_exact(self):
        z=c.evaluate({**SPEC,'threshold':.5})
        self.assertAlmostEqual(z['bright_stationary_fraction'],.75)
        self.assertAlmostEqual(z['stationary_bright_entry_rate'],.5)
        self.assertAlmostEqual(z['mean_bright_interval'],1.5)
    def test_invalid_model_spec(self):
        with self.assertRaises(ValueError):c.evaluate({**SPEC,'weights':[1.,float('nan')]})
        with self.assertRaises(ValueError):c.evaluate({**SPEC,'threshold':0.})
        with self.assertRaises(ValueError):c.evaluate({**SPEC,'on_rates':[1.,1.,1.,1.]})
    def test_frozen_common_start_and_distinct_feedback(self):
        self._initial()
        p1=e.preview(self.out,'control')
        p2=e.preview(self.out,'tool')
        self.assertNotEqual(p1['request_sha256'],p2['request_sha256'])
        fromctrl=e._request(self.out,'control')['messages'][1]['content']
        fromtool=e._request(self.out,'tool')['messages'][1]['content']
        self.assertIn('NO_EXTERNAL_FEEDBACK_IN_THIS_ARM',fromctrl)
        self.assertIn('SEPARATELY_COMPUTED_MATHEMATICAL_CONSTRAINT',fromtool)
        self.assertEqual(e.verify(self.out)['attempt_count'],1)
    def test_paired_finalize_immutable(self):
        self._initial()
        e.next_step(self.out,'control',backend=self._mock_for('control'))
        e.next_step(self.out,'tool',backend=self._mock_for('tool'))
        e.finalize(self.out)
        self.assertTrue(e.verify(self.out)['report_exists'])
        with self.assertRaises(ValueError):e.preview(self.out,'initial')
        with self.assertRaises(FileExistsError):e.finalize(self.out)
    def test_paid_denied_without_double_authorization(self):
        with self.assertRaises(PermissionError):e.next_step(self.out,'initial',allow_paid=False)
        self.assertEqual(e.verify(self.out)['attempt_count'],0)
    def test_raw_recovered_without_new_call(self):
        d=self.out/'initial';d.mkdir()
        req=e._request(self.out,'initial')
        base.write_new(d/'request.json',req)
        base.write_new(d/'attempt.json',{'request_sha256':base.sha(req),'status':'PRETEND_STARTED'})
        base.write_new(d/'raw_response.json',{'raw_text':json.dumps(INITIAL),'response_id':'saved','usage':{}})
        self.assertEqual(e.verify(self.out)['phases']['initial'],'SAVED_RAW_OFFLINE_RECOVER_REQUIRED')
        self.assertEqual(e.recover(self.out)['status'],'RECOVERED_NO_PAID')
        self.assertEqual(e.verify(self.out)['phases']['initial'],'COMPLETE')
    def test_unsupported_spec_is_a_recorded_outcome_not_a_paid_parser_retry(self):
        modified={**INITIAL,'proposed_calculator_spec':{'unsupported_physics':'diffusion'},
                  'claim_to_check':'A diffusive state could produce a nonexponential waiting-time distribution.'}
        e.next_step(self.out,'initial',backend=lambda req:(json.dumps(modified),{},'fake-id'))
        self.assertEqual(e.calculate(self.out)['status'],'UNSUPPORTED_MODEL_SPEC')
        self.assertEqual(e.verify(self.out)['phases']['initial'],'COMPLETE')
    def test_omitted_tool_request_is_accepted_and_marked(self):
        modified={**INITIAL,'proposed_calculator_spec':None,'claim_to_check':None}
        e.next_step(self.out,'initial',backend=lambda req:(json.dumps(modified),{},'fake-id'))
        self.assertEqual(e.calculate(self.out)['status'],'NOT_REQUESTED')
        self.assertEqual(e.verify(self.out)['attempt_count'],1)
    def test_unknown_billing_no_retry(self):
        d=self.out/'initial';d.mkdir()
        req=e._request(self.out,'initial')
        base.write_new(d/'request.json',req)
        base.write_new(d/'attempt.json',{'request_sha256':base.sha(req),'status':'PRETEND_STARTED'})
        with self.assertRaises(ValueError):e.preview(self.out,'initial')
        with self.assertRaises(ValueError):e.recover(self.out)
    def test_tamper_detected(self):
        self._initial()
        p=self.out/'tool_check.json'
        row=base.read(p);row['result']['mean_bright_interval']=12
        p.write_text(json.dumps(row))
        with self.assertRaises(ValueError):e.verify(self.out)

if __name__=='__main__':unittest.main()
