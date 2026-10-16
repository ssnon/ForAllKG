"""Zero-network mathematical audit and replay tests. No calls to live LLMs."""
import copy,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_core.discovery.scientific_research_control import m19_e4_independent_audit as e
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as b

def make_fixture():
    product={'work_id':'work_0004_test','work_kind':'REASON','execution_status':'COMPLETED',
       'origin':'LLM_REASONING_UNVERIFIED','scientific_validation':False,
       'content':{'attempted_question':'What can finite independent sites distinguish?',
           'reasoning_or_operations':[e.TARGET_SENTENCE]}}
    trace={'work_id':'work_0004_test','scientific_validation':False,
       'native_work_event_committed':False,
       'final_product_sha256':b.sha(product),
       'source_d4_state_sha256':'2'*64,
       'micro_steps':[{'next_step':'RETURN',
           'why':'There is no more theoretical work despite this tentative regime claim.',
           'record_sha256':'3'*64}]}
    return product,trace


def answer(revise=True):
    return json.dumps({'original_claim_under_review':e.TARGET_SENTENCE,
       'disposition':'REVISE' if revise else 'UNRESOLVED',
       'mathematical_analysis':['A stationary CTMC can be analyzed by summing dark-to-bright boundary flux exactly.'],
       'explicit_assumptions':['Independent identical binary sites; fixed switching rates and strict stationarity.'],
       'counterexample_or_proof':'With F_N=1-(1-p)^N the flux R_N=Na(1-p)^N is nonmonotonic.' if revise else None,
       'corrected_conditional_claim':'At low occupancy the rate is approximately linear in N; after overlap, rate may decrease to zero, not just plateau.',
       'earlier_return_was_scientifically_sufficient':False,
       'return_assessment_reason':'An explicit non-monotonicity theorem and concrete finite-state countermodel were not derived in the earlier research.',
       'new_testable_consequence':'Measure dark-to-bright entry flux versus occupancy across number of independent sites.',
       'next_research_action':'Derive and test the threshold dependence across two distinct levels.',
       'limitations':['These are toy-model mathematical results, not validated physics of Ye.'],
       'scientific_validation':False},ensure_ascii=False)

class E4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.work=Path(self.tmp.name)/'finished_e3_work';self.work.mkdir()
        self.product,self.trace=make_fixture()
        b.write_new(self.work/'final_product.json',self.product)
        b.write_new(self.work/'deep_work_trace.json',self.trace)
        b.write_new(self.work/'lock.json',{'model':'mock-model'})
        self.out=e.repo_root()/'experiments'/'m19_e4'/'runs'/('TEST_'+Path(self.tmp.name).name)
        self.check=patch.object(e.source,'verify',return_value={
            'finalized':True,'completed_count':1,'next_choice':'RETURN','incomplete':None})
        self.check.start()
    def tearDown(self):
        self.check.stop()
        if self.out.exists():shutil.rmtree(self.out)
        pp=self.out.with_name(self.out.name+'.preparing')
        if pp.exists():shutil.rmtree(pp)
        self.tmp.cleanup()
    def prep(self):return e.prepare(e3_work=self.work,out=self.out)
    def audited(self):
        self.prep()
        return e.audit(self.out)
    def test_source_claim_frozen_and_same_product_hash(self):
        self.prep()
        self.assertEqual(e.verify(self.out)['audit_status'],'NOT_RUN')
        self.assertEqual(b.read(self.out/'frozen_final_product.json'),self.product)
        self.assertEqual(b.read(self.out/'frozen_deep_work_trace.json'),self.trace)
        self.assertFalse((self.out/'arms').exists())
    def test_audit_analytic_relation_and_independent_enumeration(self):
        self.audited()
        a=b.read(self.out/'external_audit.json')
        r={x['N']:x for x in a['computed_examples']}
        self.assertAlmostEqual(r[5]['bright_entry_rate'],0.32768)
        self.assertAlmostEqual(r[20]['bright_entry_rate'],0.0461168601843)
        self.assertGreater(r[20]['bright_occupancy'],r[5]['bright_occupancy'])
        self.assertLess(r[20]['bright_entry_rate'],r[5]['bright_entry_rate'])
        self.assertEqual(a['origin'],e.EVIDENCE_ORIGIN)
        self.assertEqual(e.verify(self.out)['audit_status'],'CONDITIONAL_COUNTERMODEL_FOUND')
    def test_blind_arm_does_not_receive_audit_evidence(self):
        self.audited()
        s=e._request(self.out,'self')['messages'][1]['content']
        t=e._request(self.out,'evidence')['messages'][1]['content']
        self.assertNotIn('externally_supplied_deterministic_audit_not_self_discovered',s)
        self.assertNotIn('specific_claim_for_review',s)
        self.assertIn('self_selection_rule',s)
        self.assertNotIn('4398046511104',s)
        self.assertIn('externally_supplied_deterministic_audit_not_self_discovered',t)
        self.assertIn('specific_claim_for_review',t)
        self.assertIn('4398046511104',t)
        self.assertNotEqual(b.sha(s),b.sha(t))
    def test_both_mock_calls_then_finalize(self):
        self.audited()
        backend=lambda req:(answer(),{'total_tokens':300},'mock-response')
        self.assertEqual(e.next_step(self.out,'self',backend=backend)['status'],'ARM_COMMITTED')
        self.assertEqual(e.next_step(self.out,'evidence',backend=backend)['status'],'ARM_COMMITTED')
        r=e.finalize(self.out)
        self.assertEqual(r['status'],'E4_FINALIZED_NO_CALL')
        v=e.verify(self.out)
        self.assertEqual(v['arms'],{'self':'COMPLETE','evidence':'COMPLETE'})
        self.assertTrue(v['finalized'])
        self.assertFalse(b.read(self.out/'final_report.json')['scientific_validation'])
        self.assertTrue(b.read(self.out/'final_report.json')['no_autonomous_continuation_credit'])
        self.assertEqual(self.product,b.read(self.work/'final_product.json'))
    def test_raw_saved_recovery_without_recall(self):
        self.audited()
        arm='self';ad=self.out/'arms'/arm;ad.mkdir(parents=True)
        req=e._request(self.out,arm)
        b.write_new(ad/'request.json',req)
        b.write_new(ad/'attempt.json',{'arm':arm,'request_sha256':b.sha(req)})
        b.write_new(ad/'raw_response.json',{'raw_text':answer(),'usage':{},'response_id':'saved'})
        self.assertEqual(e.verify(self.out)['arms'][arm],'RAW_SAVED_OFFLINE_RECOVER_REQUIRED')
        self.assertEqual(e.recover(self.out,arm)['status'],'SAVED_RAW_RECOVERED_OFFLINE_NO_CALL')
        self.assertEqual(e.verify(self.out)['arms'][arm],'COMPLETE')
    def test_unknown_billing_no_retry(self):
        self.audited()
        ad=self.out/'arms'/'self';ad.mkdir(parents=True)
        req=e._request(self.out,'self')
        b.write_new(ad/'request.json',req)
        b.write_new(ad/'attempt.json',{'arm':'self','request_sha256':b.sha(req)})
        self.assertEqual(e.verify(self.out)['arms']['self'],'UNKNOWN_BILLING_NO_RAW_NEVER_RETRY')
        with self.assertRaisesRegex(RuntimeError,'UNRESOLVED_ATTEMPT'):
            e.next_step(self.out,'evidence',backend=lambda req:(answer(),{},'id'))
        with self.assertRaisesRegex(RuntimeError,'UNKNOWN_BILLING'):e.recover(self.out,'self')
    def test_audit_tamper_detected(self):
        self.audited()
        p=self.out/'external_audit.json';v=b.read(p);v['status']='FAKE';p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'EXTERNAL_AUDIT_CHANGED'):e.verify(self.out)
    def test_source_change_detected(self):
        self.audited()
        p=self.work/'final_product.json';v=b.read(p);v['content']['attempted_question']='tampered'
        p.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError,'TRACE_PRODUCT_SHA_MISMATCH|E3_SOURCE_FILE_CHANGED'):e.verify(self.out)
    def test_paid_gate_no_new_attempt(self):
        self.audited()
        with patch.dict('os.environ',{'FAKG_RESEARCH_ALLOW_PAID':'','OPENROUTER_API_KEY':''}):
            with self.assertRaises(PermissionError):e.next_step(self.out,'self')
        self.assertEqual(e.verify(self.out)['attempts_total'],0)
    def test_attempt_cap_and_no_duplicate(self):
        self.audited()
        backend=lambda req:(answer(),{},'resp')
        e.next_step(self.out,'self',backend=backend)
        with self.assertRaisesRegex(RuntimeError,'CEILING'):
            e.next_step(self.out,'evidence',max_new_paid_attempts=1,backend=backend)
        with self.assertRaisesRegex(RuntimeError,'ALREADY_ATTEMPTED'):
            e.preview(self.out,'self')
    def test_cannot_finalize_only_one_arm(self):
        self.audited()
        e.next_step(self.out,'self',backend=lambda req:(answer(),{},'resp'))
        with self.assertRaisesRegex(RuntimeError,'BOTH_ARMS'):e.finalize(self.out)
    def test_no_output_outside_repo(self):
        with self.assertRaisesRegex(ValueError,'OUTSIDE_REPO'):
            e.prepare(e3_work=self.work,out=Path(self.tmp.name)/'wrong')
    def test_rejected_claim_not_in_source(self):
        v=self.product;v['content']['reasoning_or_operations']=['unrelated']
        with self.assertRaisesRegex(ValueError,'SOURCE_CLAIM_SELECTION_CHANGED'):e.expected_audit(v)
    def test_source_return_is_not_overridden(self):
        self.audited()
        self.assertEqual(b.read(self.out/'frozen_deep_work_trace.json')['micro_steps'][0]['next_step'],'RETURN')
        self.assertTrue(e.verify(self.out)['original_autonomous_return_unchanged'])

if __name__=='__main__':unittest.main()
