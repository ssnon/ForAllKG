"""All E4-B tests run without paid calls; original E4 trial is preserved."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_core.discovery.scientific_research_control import m19_e4b_target_only as t
from pipeline_core.discovery.scientific_research_control import m19_e4_independent_audit as e4
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as b


def fake_answer():
    return json.dumps({
        'original_claim_under_review': e4.TARGET_SENTENCE,
        'disposition': 'REVISE',
        'mathematical_analysis': ['For N independent reversible sites, compute stationary all-OFF to ON crossings directly.'],
        'explicit_assumptions': ['Sites are stationary, independent, with binary ON/OFF kinetics.'],
        'counterexample_or_proof': 'The stationary entry flux can have a maximum rather than a plateau.',
        'corrected_conditional_claim': 'A low-occupancy linear flux estimate need not produce a general monotone saturation at higher occupancy.',
        'earlier_return_was_scientifically_sufficient': False,
        'return_assessment_reason': 'Event rate behavior outside low occupancy should be mathematically checked before concluding.',
        'new_testable_consequence': 'Compare event entry and state occupancy at multiple concentrations.',
        'next_research_action': 'Calculate exact flux for three thresholds.',
        'limitations': ['This does not establish the molecular physics of Ye.'],
        'scientific_validation': False,
    })


class E4BTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name) / 'e3'
        self.work.mkdir()
        product = {'work_id': 'work_0004_test', 'work_kind': 'REASON',
                   'execution_status': 'COMPLETED', 'origin': 'LLM_REASONING_UNVERIFIED',
                   'scientific_validation': False,
                   'content': {'attempted_question': 'Can aggregation identify mechanism?',
                               'reasoning_or_operations': [e4.TARGET_SENTENCE]}}
        trace = {'work_id': product['work_id'], 'scientific_validation': False,
                 'native_work_event_committed': False,
                 'final_product_sha256': b.sha(product),
                 'source_d4_state_sha256': 'a'*64,
                 'micro_steps': [{'next_step': 'RETURN',
                                  'why': 'Existing source data do not settle mechanism identity.',
                                  'record_sha256': 'b'*64}]}
        b.write_new(self.work / 'final_product.json', product)
        b.write_new(self.work / 'deep_work_trace.json', trace)
        b.write_new(self.work / 'lock.json', {'model': 'mock-luna'})
        basename = 'TEST_' + Path(self.tmp.name).name
        self.e4dir = t.repo_root() / 'experiments' / 'm19_e4' / 'runs' / basename
        self.out = t.repo_root() / 'experiments' / 'm19_e4b' / 'runs' / basename
        self.patch = patch.object(e4.source, 'verify', return_value={
            'finalized': True, 'completed_count': 1, 'next_choice': 'RETURN', 'incomplete': None})
        self.patch.start()
        e4.prepare(e3_work=self.work, out=self.e4dir)
        e4.audit(self.e4dir)
        backend = lambda req: (fake_answer(), {'total_tokens': 5}, 'simulated')
        e4.next_step(self.e4dir, 'self', backend=backend)
        e4.next_step(self.e4dir, 'evidence', backend=backend)
        e4.finalize(self.e4dir)

    def tearDown(self):
        self.patch.stop()
        for p in (self.out, self.e4dir):
            if p.exists(): shutil.rmtree(p)
            preparing = p.with_name(p.name + '.preparing')
            if preparing.exists(): shutil.rmtree(preparing)
        self.tmp.cleanup()

    def prep(self):
        return t.prepare(e4_out=self.e4dir, out=self.out)

    def test_pinned_final_source_prepared(self):
        x = self.prep()
        self.assertEqual(x['status'], 'E4B_PREPARED_PINNED_FINAL_E4_NO_CALL')
        self.assertEqual(t.verify(self.out)['target_status'], 'NOT_STARTED')
        self.assertTrue(b.read(self.out/'lock.json')['scientific_validation'] is False)

    def test_prompt_is_exact_self_baseline_plus_only_target(self):
        self.prep()
        s = e4._request(self.e4dir, 'self')
        v = e4._request(self.e4dir, 'evidence')
        req = t._request(self.out)
        sr = json.loads(s['messages'][1]['content'])
        tr = json.loads(req['messages'][1]['content'])
        er = json.loads(v['messages'][1]['content'])
        self.assertEqual(req['messages'][0], s['messages'][0])
        self.assertEqual(req['temperature'], s['temperature'])
        self.assertEqual(req['max_tokens'], s['max_tokens'])
        self.assertEqual(req['model'], s['model'])
        self.assertEqual(set(tr)-set(sr), {'specific_claim_for_review'})
        self.assertEqual(set(sr)-set(tr), {'self_selection_rule'})
        self.assertEqual(tr['specific_claim_for_review'], e4.TARGET_SENTENCE)
        self.assertEqual(tr['review_kind'], t.REVIEW_KIND)
        self.assertNotIn('externally_supplied_deterministic_audit_not_self_discovered', tr)
        self.assertIn('externally_supplied_deterministic_audit_not_self_discovered', er)
        self.assertNotEqual(b.sha(req), b.sha(s))
        self.assertNotEqual(b.sha(req), b.sha(v))

    def test_no_outcome_leak_after_original_e4_finalized(self):
        self.prep()
        msg = json.loads(t._request(self.out)['messages'][1]['content'])
        # Output schema legitimately contains the enum RETAIN_WITH_PROOF.
        # Check only scientific input, excluding schema definition.
        msg.pop('output_schema')
        msg = json.dumps(msg)
        self.assertNotIn('RETAIN_WITH_PROOF', msg)
        self.assertNotIn('CONDITIONAL_COUNTERMODEL_FOUND', msg)
        self.assertNotIn('bright_entry_rate_exact', msg)
        self.assertNotIn('EXTERNAL_DETERMINISTIC_MATHEMATICAL_AUDIT_NOT_FAKG', msg)
        self.assertNotIn('computed_examples', msg)
        self.assertNotIn('source_e4_audit_sha256', msg)
        self.assertNotIn('N_*', msg)

    def test_mock_run_finalize_preserves_original(self):
        self.prep()
        original = {p: b.file_sha(self.e4dir/p) for p in ('final_report.json', 'external_audit.json')}
        r = t.next_step(self.out, backend=lambda req: (fake_answer(), {'total_tokens': 10}, 'id'))
        self.assertEqual(r['status'], 'TARGET_COMMITTED')
        self.assertEqual(t.finalize(self.out)['status'], 'E4B_FINALIZED_NO_CALL')
        v = t.verify(self.out)
        self.assertTrue(v['finalized'])
        self.assertEqual(v['target_status'], 'COMPLETE')
        report = b.read(self.out/'final_report.json')
        self.assertEqual(set(report['three_arm_comparison']), {'self', 'target', 'evidence'})
        self.assertTrue(report['no_autonomous_continuation_credit'])
        self.assertEqual(report['audit_credit'], 'EXTERNAL_NOT_FAKG')
        self.assertFalse(report['scientific_validation'])
        for p, expected in original.items():
            self.assertEqual(b.file_sha(self.e4dir/p), expected)

    def test_unknown_billing_never_retry(self):
        self.prep()
        ad = self.out/'target'
        ad.mkdir()
        req = t._request(self.out)
        b.write_new(ad/'request.json', req)
        b.write_new(ad/'attempt.json', {'phase': 'target', 'request_sha256': b.sha(req)})
        self.assertEqual(t.verify(self.out)['target_status'], 'UNKNOWN_BILLING_NO_RAW_NEVER_RETRY')
        with self.assertRaisesRegex(RuntimeError, 'ALREADY_ATTEMPTED'):
            t.next_step(self.out, backend=lambda req: (fake_answer(), {}, 'id'))
        with self.assertRaisesRegex(RuntimeError, 'UNKNOWN_BILLING'):
            t.recover(self.out)

    def test_offline_recovery_without_second_call(self):
        self.prep()
        ad = self.out/'target'
        ad.mkdir()
        req = t._request(self.out)
        b.write_new(ad/'request.json', req)
        b.write_new(ad/'attempt.json', {'phase': 'target', 'request_sha256': b.sha(req)})
        b.write_new(ad/'raw_response.json', {'raw_text': fake_answer(), 'usage': {}, 'response_id': 'saved'})
        self.assertEqual(t.verify(self.out)['target_status'], 'RAW_SAVED_OFFLINE_RECOVER_REQUIRED')
        self.assertEqual(t.recover(self.out)['status'], 'SAVED_RAW_COMMITTED_OFFLINE_NO_CALL')
        self.assertEqual(t.verify(self.out)['target_status'], 'COMPLETE')

    def test_tampered_e4_result_detected(self):
        self.prep()
        f = self.e4dir/'arms/self/normalized.json'
        j = b.read(f); j['disposition'] = 'UNRESOLVED'; f.write_text(json.dumps(j))
        with self.assertRaisesRegex(ValueError, 'ARM_NORMALIZED_MISMATCH|ORIGINAL_E4_FILES_CHANGED'):
            t.verify(self.out)

    def test_tampered_target_result_detected(self):
        self.prep()
        t.next_step(self.out, backend=lambda req: (fake_answer(), {}, 'id'))
        f = self.out/'target/normalized.json';j=b.read(f);j['disposition']='RETRACT';f.write_text(json.dumps(j))
        with self.assertRaisesRegex(ValueError, 'TARGET_NORMALIZED_REPLAY_MISMATCH'):
            t.verify(self.out)

    def test_source_original_return_still_unchanged(self):
        self.prep()
        self.assertTrue(t.verify(self.out)['source_original_return_unchanged'])
        self.assertEqual(b.read(self.e4dir/'frozen_deep_work_trace.json')['micro_steps'][0]['next_step'], 'RETURN')

    def test_no_paid_without_double_authorization(self):
        self.prep()
        with patch.dict('os.environ', {'FAKG_RESEARCH_ALLOW_PAID': '', 'OPENROUTER_API_KEY': ''}):
            with self.assertRaises(PermissionError):
                t.next_step(self.out)
        self.assertEqual(t.verify(self.out)['attempt_count'], 0)

    def test_no_duplicate_after_commit(self):
        self.prep()
        t.next_step(self.out, backend=lambda req: (fake_answer(), {}, 'id'))
        with self.assertRaisesRegex(RuntimeError, 'ALREADY_ATTEMPTED'):
            t.preview(self.out)
        with self.assertRaisesRegex(ValueError, 'MAX_NEW_PAID_ATTEMPTS'):
            t.next_step(self.out, max_new_paid_attempts=2, backend=lambda req: (fake_answer(), {}, 'id'))

    def test_cannot_finalize_incomplete(self):
        self.prep()
        with self.assertRaisesRegex(RuntimeError, 'TARGET_MUST_BE_COMPLETE'):
            t.finalize(self.out)

    def test_no_output_outside_repo(self):
        with self.assertRaisesRegex(ValueError, 'OUTSIDE_REPO'):
            t.prepare(e4_out=self.e4dir, out=Path(self.tmp.name)/'bad')

    def test_no_source_other_than_final_e4(self):
        with self.assertRaisesRegex(ValueError, 'SOURCE_OUTSIDE'):
            t.prepare(e4_out=self.work, out=self.out)

    def test_reject_existing_prepared_run(self):
        self.prep()
        with self.assertRaisesRegex(FileExistsError, 'E4B_ALREADY_EXISTS'):
            self.prep()


if __name__ == '__main__':
    unittest.main()
