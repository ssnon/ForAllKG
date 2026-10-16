"""No-network tests for post-hoc semantic fork and external CTMC feedback."""
from __future__ import annotations
import json,shutil,unittest,uuid
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_e2_semantic_fork as fork
from pipeline_core.discovery.scientific_research_control import m19_e2_json_v2 as prior
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as base

INITIAL={
    'question_refined':'What exact bright event statistics can three independent two-state sites generate?',
    'model_and_assumptions':[
        'Model each of three immobilized sites as independent two-state continuous-time Markov chains.',
        'Let the measured intensity be I(x)=Σ_i w_i x_i, and define a bright state by I(x)≥T; assume instantaneous detection, no photobleaching, no background, and stationarity.'
    ],
    'derived_claim':'Bright fraction and bright-entry rate are fixed by product stationary probabilities under this model.',
    'mathematical_reasoning':['A site has p_i=k_on,i/(k_on,i+k_off,i) and the joint distribution is a product.',
                             'Incorrectly claim R_B=28/45 and mean duration=17/28 for testing external discovery of arithmetic discrepancy.'],
    'candidate_consequence':'Aggregate event fraction alone does not identify an underlying shared physical state.',
    'limitations':['No mechanism in Ye has been empirically confirmed by this toy calculation.'],
    'proposed_calculator_spec':{'on_rates':[.2,.5,1.0],'off_rates':[.8,1.0,2.0],
       'intensity_weights':[1.,1.,2.],'total_intensity_threshold':2.},
    'claim_to_check':'Check the thresholded bright fraction and rate for this particular model.',
    'next_scientific_question':'Would multiple thresholds discriminate independent sites from a shared switch?'
}
FOLLOW={'refined_scientific_claim':'A conditional exact CTMC statement follows for bright episodes in the model.',
 'changed_or_retained_why':'The common initial statement is unverified, so it must be conditioned on this model.',
 'mathematical_or_physical_reasoning':['The stationary flux must be summed over all dark-to-bright state transitions.'],
 'remaining_uncertainties':['The actual Ye system has not been fitted or identified.'],
 'new_conditional_prediction':'Varying the threshold changes the bright fraction and mean residence time.',
 'next_research_action':'Consider a separate model with a shared state at identical numerical resolution.',
 'scientific_validation':False}

class ForkTests(unittest.TestCase):
    def setUp(self):
        self.source=prior.repo_root()/'experiments/m19_e2/runs'/('TEST_SRC_'+uuid.uuid4().hex[:12])
        self.out=prior.repo_root()/'experiments/m19_e2/runs'/('TEST_FORK_'+uuid.uuid4().hex[:12])
        prior.prepare(self.source)
        prior.next_step(self.source,'initial',backend=lambda r:(json.dumps(INITIAL),{'total_tokens':100},'test-id'))
        self.assertEqual(prior.calculate(self.source)['status'],'UNSUPPORTED_MODEL_SPEC')
        fork.prepare(self.out,self.source)
    def tearDown(self):
        shutil.rmtree(self.source,ignore_errors=True)
        shutil.rmtree(self.out,ignore_errors=True)
    def test_ge_exact_different_from_original_gt(self):
        v=base.read(self.out/'external_calculation.json')
        self.assertAlmostEqual(v['result']['bright_stationary_fraction'],17/45)
        self.assertAlmostEqual(v['result']['stationary_bright_entry_rate'],158/225)
        self.assertAlmostEqual(v['result']['mean_bright_interval'],85/158)
        self.assertAlmostEqual(v['adapter']['effective_strict_threshold'],1.5)
        self.assertEqual(v['initial_model_rule'],'I(x) >= T')
        self.assertEqual(prior.verify(self.source)['tool_check'],'UNSUPPORTED_MODEL_SPEC')
    def test_common_initial_pinned_different_feedback(self):
        a=fork._request(self.out,'control');b=fork._request(self.out,'tool')
        self.assertEqual(a['model'],b['model'])
        self.assertEqual(a['messages'][0],b['messages'][0])
        u=json.loads(a['messages'][1]['content']);v=json.loads(b['messages'][1]['content'])
        self.assertEqual(u['original_unverified_scientific_result'],v['original_unverified_scientific_result'])
        self.assertEqual(u['original_unverified_scientific_result'],INITIAL)
        self.assertIsNone(u['external_calculator_result'])
        self.assertEqual(u['external_calculator_label'],'NO_EXTERNAL_FEEDBACK_IN_THIS_ARM')
        self.assertEqual(v['external_calculator_result']['status'],
            'EXACT_CTMC_RESULT_AFTER_EXPLICIT_ALIAS_AND_THRESHOLD_ADAPTER')
        self.assertTrue(fork.preview(self.out,'control')['json_mode_ready'])
        self.assertTrue(fork.preview(self.out,'tool')['json_mode_ready'])
    def test_reject_implicit_comparison(self):
        changed={**INITIAL,'model_and_assumptions':['Three sites','Use some threshold']}
        with self.assertRaisesRegex(ValueError,'NO_EXPLICIT_GE_BRIGHT_RULE'):
            fork.checked_calculation(changed)
    def test_paid_denied_before_attempt(self):
        with self.assertRaises(PermissionError):fork.next_step(self.out,'control')
        self.assertEqual(fork.verify(self.out)['new_attempts'],0)
    def test_paired_mock_followups_finalize(self):
        for phase in ('control','tool'):
            z=fork.next_step(self.out,phase,backend=lambda r:(json.dumps(FOLLOW),{},'mock-'+phase))
            self.assertEqual(z['status'],'COMMITTED')
        self.assertEqual(fork.verify(self.out)['new_attempts'],2)
        self.assertEqual(fork.finalize(self.out)['status'],'FINALIZED_NO_CALL')
        with self.assertRaises(ValueError):fork.preview(self.out,'control')
        self.assertTrue(fork.verify(self.out)['report_exists'])
    def test_recover_saved_raw_no_paid(self):
        d=self.out/'tool';d.mkdir()
        request=fork._request(self.out,'tool')
        base.write_new(d/'request.json',request)
        base.write_new(d/'attempt.json',{'request_sha256':base.sha(request),'status':'ATTEMPT_STARTED'})
        base.write_new(d/'raw_response.json',{'raw_text':json.dumps(FOLLOW),'response_id':'saved','usage':{}})
        self.assertEqual(fork.verify(self.out)['phases']['tool'],'SAVED_RAW_OFFLINE_RECOVER_REQUIRED')
        self.assertEqual(fork.recover(self.out)['status'],'RECOVERED_SAVED_RAW_NO_CALL')
        self.assertEqual(fork.verify(self.out)['phases']['tool'],'COMPLETE')
    def test_tamper_feedback_detected(self):
        f=self.out/'external_calculation.json';z=base.read(f)
        z['result']['stationary_bright_entry_rate']=123
        f.write_text(json.dumps(z),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'EXTERNAL_CALCULATION_TAMPERED'):fork.verify(self.out)
    def test_original_source_tamper_detected(self):
        f=self.source/'initial'/'normalized.json';z=base.read(f)
        z['derived_claim']='A tampered scientific claim that should be rejected by the source verifier.'
        f.write_text(json.dumps(z),encoding='utf8')
        with self.assertRaises((ValueError,AssertionError)):fork.verify(self.out)
    def test_unknown_billing_no_retry(self):
        d=self.out/'control';d.mkdir()
        req=fork._request(self.out,'control')
        base.write_new(d/'request.json',req)
        base.write_new(d/'attempt.json',{'request_sha256':base.sha(req),'status':'ATTEMPT_STARTED'})
        self.assertEqual(fork.verify(self.out)['phases']['control'],'UNKNOWN_BILLING_NO_RAW_NO_RETRY')
        with self.assertRaisesRegex(ValueError,'UNRESOLVED_ATTEMPT'):fork.preview(self.out,'tool')
        with self.assertRaisesRegex(ValueError,'UNKNOWN_BILLING'):fork.recover(self.out)

if __name__=='__main__':unittest.main()
