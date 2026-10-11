"""Offline no-paid-call tests: narrow M14 physics identity repair and deterministic replay."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from pipeline_core.discovery.sers_novelty_search_m14 import make_plan,canon
from scripts.discovery.run_sers_novelty_search_m14 import discover,run,freeze,load,digest
from scripts.discovery.recover_sers_novelty_search_m14_physics_ids import (
    locate,inspect,normal_form,build_audit,verified_replay,emit,data_sha,raw_sha,file_sha,AUDIT_LEAF)
from test_sers_novelty_search_m14 import MockScience

class IdentityTests(unittest.TestCase):
    def seed(self,broken):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);p=make_plan(generations=2,replicates=2,queries_per_candidate=3)
        freeze(root/'M14_PLAN.json',p)
        f=MockScience();discover(p,root,'mock',f.call,f.lookup)
        original=f.call
        def call(model,prompt):
            result=original(model,prompt)
            req=json.loads(prompt['user'])
            if 'decision_rule' in req and not (root/'r01/g01/ADAPTIVE/physics_RAW_RECEIPT.json').exists():
                d=json.loads(result['text']);d['reviews'][0]['candidate_id']=broken
                result['text']=json.dumps(d)
            return result
        status=run(p,root,'mock','mock','mock',call,f.lookup,None)
        self.assertEqual(status['status'],'M14_SAVED_MODEL_OUTPUT_INVALID_NO_AUTORECALL')
        folder=root/'r01/g01/ADAPTIVE'
        self.assertTrue((folder/'M14_PHYSICS_INVALID.json').exists())
        return root,p,f,folder,call
    def _mp(self,folder,doc,cs,order):
        return [{'review_index':i,'original_candidate_id':r.get('candidate_id'),
                 'source_review_sha256':data_sha(r),
                 'expected_candidate_id':cs[order[i]]['candidate_id'],
                 'human_confirmed_scientific_match':True} for i,r in enumerate(doc['reviews'])]
    def test_token_normalization_is_not_fuzzy(self):
        self.assertEqual(normal_form(' `m14:'+'a'*20+'` ',['m14:'+'a'*20]),'m14:'+'a'*20)
        self.assertEqual(normal_form('a'*20,['m14:'+'a'*20]),'m14:'+'a'*20)
        self.assertIsNone(normal_form('m14:'+'a'*19,['m14:'+'a'*20]))
        self.assertIsNone(normal_form('candidate_1',['m14:'+'a'*20]))
    def test_manual_audit_replay_and_full_resume(self):
        root,p,f,folder,call=self.seed('candidate_1')
        source=load(folder/'physics_RAW_RECEIPT.json');before=digest(folder/'physics_RAW_RECEIPT.json')
        cs=load(folder/'M14_GENERATION_PARSED.json')['candidates'];doc=json.loads(source['raw'])
        mapping=self._mp(folder,doc,cs,[0,1]);audit=build_audit(folder,doc,cs,mapping,'MANUAL_SCIENTIFIC_ALIGNMENT',p['plan_id'])
        emit(folder/AUDIT_LEAF,audit)
        text,proof=verified_replay(folder,source['raw'],cs)
        self.assertEqual([r['candidate_id'] for r in json.loads(text)['reviews']],[c['candidate_id'] for c in cs])
        self.assertEqual(digest(folder/'physics_RAW_RECEIPT.json'),before)
        # Zero new calls when replaying up to the checkpoint boundary.
        again=run(p,root,'mock','mock','mock',call,f.lookup,0)
        self.assertEqual(again['status'],'M14_CHECKPOINTED')
        self.assertTrue((folder/'M14_PHYSICS_CRITIQUE.json').exists())
        self.assertEqual(digest(folder/'physics_RAW_RECEIPT.json'),before)
        done=run(p,root,'mock','mock','mock',call,f.lookup,None)
        self.assertEqual(done['status'],'M14_MATCHED_ARMS_COMPLETE_INDEPENDENT_BLIND_REVIEW_REQUIRED')
        self.assertEqual(digest(folder/'physics_RAW_RECEIPT.json'),before)
    def test_auto_refuses_synthetic_ids(self):
        root,p,f,folder,call=self.seed('candidate_1')
        state=inspect(root/'M14_PLAN.json',digest(folder/'physics_RAW_RECEIPT.json'))
        self.assertFalse(state['token_normalization_unambiguous'])
        self.assertEqual(state['actual_review_entries'][0]['candidate_id'],'candidate_1')
        self.assertEqual(len(state['expected_candidates']),2)
    def test_manual_mismatched_source_refused(self):
        root,p,f,folder,call=self.seed('candidate_1')
        original=load(folder/'physics_RAW_RECEIPT.json');doc=json.loads(original['raw'])
        cs=load(folder/'M14_GENERATION_PARSED.json')['candidates']
        mapping=self._mp(folder,doc,cs,[0,1]);mapping[0]['source_review_sha256']='invalid'
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            build_audit(folder,doc,cs,mapping,'MANUAL_SCIENTIFIC_ALIGNMENT',p['plan_id'])
    def test_mapping_must_cover_exact_two_ids(self):
        root,p,f,folder,call=self.seed('candidate_1')
        orig=load(folder/'physics_RAW_RECEIPT.json');doc=json.loads(orig['raw'])
        cs=load(folder/'M14_GENERATION_PARSED.json')['candidates']
        mapping=self._mp(folder,doc,cs,[0,0])
        with self.assertRaisesRegex(ValueError,'exactly the two'):
            build_audit(folder,doc,cs,mapping,'MANUAL_SCIENTIFIC_ALIGNMENT',p['plan_id'])
    def test_raw_sha_guard(self):
        root,p,f,folder,call=self.seed('candidate_1')
        receipt=load(folder/'physics_RAW_RECEIPT.json');doc=json.loads(receipt['raw']);cs=load(folder/'M14_GENERATION_PARSED.json')['candidates']
        audit=build_audit(folder,doc,cs,self._mp(folder,doc,cs,[0,1]),'MANUAL_SCIENTIFIC_ALIGNMENT',p['plan_id'])
        emit(folder/AUDIT_LEAF,audit)
        with self.assertRaisesRegex(RuntimeError,'raw receipt drift'):
            verified_replay(folder,receipt['raw']+' ',cs)
        with self.assertRaisesRegex(RuntimeError,'not reproducible'):
            tampered=load(folder/AUDIT_LEAF);tampered['canonical_review_sha256']='wrong'
            (folder/AUDIT_LEAF).write_text(json.dumps(tampered))
            verified_replay(folder,receipt['raw'],cs)
if __name__=='__main__':unittest.main()
