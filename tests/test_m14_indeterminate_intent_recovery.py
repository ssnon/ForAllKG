"""No-provider-call safety tests for one explicitly paid M14 orphan-intent retry."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path

from pipeline_core.discovery.sers_novelty_search_m14 import make_plan, candidate_prompt, sha
from scripts.discovery.run_sers_novelty_search_m14 import create, freeze, discover, digest, run, load
from scripts.discovery.recover_sers_novelty_search_m14 import inspect, perform_explicit_retry
from test_sers_novelty_search_m14 import MockScience

class RecoveryTests(unittest.TestCase):
    def seed(self, root):
        p=make_plan(generations=2,replicates=2,queries_per_candidate=3)
        freeze(root/'M14_PLAN.json',p)
        f=MockScience()
        discover(p,root,'mock',f.call,f.lookup)
        problems=load(root/'M14_PROVISIONAL_PROBLEMS.json')
        prompt=candidate_prompt(p,problems,1,1,'ADAPTIVE',[])
        folder=root/'r01/g01/ADAPTIVE'
        create(folder/'generation_INTENT.json',{
            'prompt_sha256':sha(prompt),'model':'mock',
            'status':'PROVIDER_ATTEMPT_STARTED_NO_AUTO_RETRY'})
        return p,f,folder

    def test_inspect_offline_and_exact_sha(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p,f,folder=self.seed(root)
            before=len(f.calls)
            status,_,_=inspect(root/'M14_PLAN.json',1,1,'ADAPTIVE','mock')
            self.assertEqual(len(f.calls),before)
            self.assertTrue(status['original_intent_exists'])
            self.assertFalse(status['original_receipt_exists'])
            self.assertFalse(status['automatic_retry_safe'])
            with self.assertRaisesRegex(RuntimeError,'exact frozen prompt/model'):
                inspect(root/'M14_PLAN.json',1,1,'ADAPTIVE','another-model')

    def test_explicit_retry_preserves_intent_and_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p,f,folder=self.seed(root)
            original_hash=digest(folder/'generation_INTENT.json')
            with self.assertRaisesRegex(RuntimeError,'acknowledgement'):
                perform_explicit_retry(root/'M14_PLAN.json',1,1,'ADAPTIVE','mock',f.call,
                    accept_possible_duplicate=False,confirm_not_running=True)
            self.assertFalse((folder/'generation_RETRY_INTENT.json').exists())
            before=len(f.calls)
            out=perform_explicit_retry(root/'M14_PLAN.json',1,1,'ADAPTIVE','mock',f.call,
                accept_possible_duplicate=True,confirm_not_running=True)
            self.assertEqual(out['new_model_calls_this_invocation'],1)
            self.assertEqual(len(f.calls),before+1)
            self.assertEqual(original_hash,digest(folder/'generation_INTENT.json'))
            self.assertTrue((folder/'generation_RAW_RECEIPT.json').exists())
            self.assertTrue(load(folder/'generation_RAW_RECEIPT.json')['explicit_retry_of_unknown_original_charge'])
            self.assertTrue((folder/'M14_INDETERMINATE_RETRY_AUDIT.json').exists())
            with self.assertRaisesRegex(RuntimeError,'Original receipt already present'):
                perform_explicit_retry(root/'M14_PLAN.json',1,1,'ADAPTIVE','mock',f.call,
                    accept_possible_duplicate=True,confirm_not_running=True)
            self.assertEqual(len(f.calls),before+1)
            result=run(p,root,'mock','mock','mock',f.call,f.lookup,None)
            self.assertEqual(result['status'],'M14_ARMS_COMPLETE_POSSIBLE_DUPLICATE_CHARGE_DESCRIPTIVE_ONLY')
            self.assertFalse(result['matched_generation_and_review_calls'])
            self.assertFalse(result['actual_paid_budget_parity_certifiable'])
            self.assertTrue(result['equal_saved_generation_and_review_receipts'])
            self.assertEqual(result['ambiguous_original_provider_attempts_by_arm'],{'ADAPTIVE':1,'FIXED':0})
            self.assertEqual(result['candidates'],16)
            self.assertEqual(digest(folder/'generation_INTENT.json'),original_hash)
            self.assertEqual(len(list(root.glob('r*/g*/ADAPTIVE/*_RAW_RECEIPT.json'))),12)

    def test_failed_provider_never_autoretries(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p,f,folder=self.seed(root)
            def failure(*args):raise TimeoutError('ambiguous prior request')
            with self.assertRaisesRegex(RuntimeError,'DO NOT RECALL'):
                perform_explicit_retry(root/'M14_PLAN.json',1,1,'ADAPTIVE','mock',failure,
                    accept_possible_duplicate=True,confirm_not_running=True)
            self.assertTrue((folder/'generation_RETRY_INTENT.json').exists())
            self.assertFalse((folder/'generation_RAW_RECEIPT.json').exists())
            before=len(f.calls)
            with self.assertRaisesRegex(RuntimeError,'Retry intent already present'):
                perform_explicit_retry(root/'M14_PLAN.json',1,1,'ADAPTIVE','mock',f.call,
                    accept_possible_duplicate=True,confirm_not_running=True)
            self.assertEqual(len(f.calls),before)

    def test_original_intent_must_match_saved_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p,f,folder=self.seed(root)
            intent=load(folder/'generation_INTENT.json');intent['prompt_sha256']='bad'
            (folder/'generation_INTENT.json').write_text(json.dumps(intent))
            with self.assertRaisesRegex(RuntimeError,'exact frozen prompt/model'):
                inspect(root/'M14_PLAN.json',1,1,'ADAPTIVE','mock')
            self.assertFalse((folder/'generation_RETRY_INTENT.json').exists())

if __name__=='__main__':unittest.main()
