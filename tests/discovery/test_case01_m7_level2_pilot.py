"""Offline contract tests. Use actual FAKG M7 on WSL to run preflight after install.

No network/API calls, no original source modification, no canonical E2E changes.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/discovery/run_case01_m7_level2_pilot.py'
spec=importlib.util.spec_from_file_location('case01_level2_under_test',SCRIPT)
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class FakeProgram:
    def __init__(self, obj): self.obj=obj
    @classmethod
    def parse(cls,obj):
        for k in ('idea_label','parent_idea_id','source_role','mechanism_model','formal_prediction'):
            if not obj.get(k): raise ValueError('missing '+k)
        if obj.get('novelty_certified') or obj.get('scientific_truth_authority') or obj.get('evidence_claim_authority'):
            raise ValueError('false authority required')
        return cls(obj)
    def to_dict(self): return dict(self.obj)
    def __getattr__(self,key): return self.obj[key]


def fake_prompt(task):
    return [{'role':'system','content':'M7 mocked scientific program schema'},
            {'role':'user','content':json.dumps({'task':task},ensure_ascii=False)}]


class TestCase01Level2Pilot(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.prev=root/'prior_source_depth_run';self.kit=root/'kit';self.out=root/'pilot'
        self.prev.mkdir();(self.prev/'parsed').mkdir();(self.prev/'responses').mkdir()
        (self.kit/'source').mkdir(parents=True)
        for p in pilot.FIGURES:
            (self.kit/f'source/Ye_page_{p:02d}.png').write_bytes(b'fake-image')
        self.task={'label':'CASE01_FROZEN_PORTFOLIO_M7','parent_idea_id':'hypothesis:sample','role':'OPEN_ALTERNATE_BRANCH',
                   'research_question':'Which causal mechanism survives a physically strong null?',
                   'original_kernel':{'all_cards_unverified':True}}
        self.parent={'idea_label':self.task['label'],'parent_idea_id':self.task['parent_idea_id'],
                     'source_role':self.task['role'],'source_of_draft':'MODEL_SPECULATIVE',
                     'mechanism_model':'proposed mechanism','formal_prediction':'hypothesis-specific test',
                     'novelty_certified':False,'scientific_truth_authority':False,'evidence_claim_authority':False}
        pilot.save_new(self.prev/'parsed/YE_FULLTEXT_FIGURES.json',self.parent)
        pilot.save_new(self.prev/'responses/YE_FULLTEXT_FIGURES.json',{'finish_reason':'stop'})
        pilot.save_new(self.prev/'RUN_MANIFEST.json',{'status':'READY_NO_PROVIDER_CALLS'})
        fake_bridge=types.SimpleNamespace(
            __file__=str(SCRIPT),
            check_frozen=lambda prev:(self.kit,{}),
            build_task=lambda kit,arm:self.task,
            build_evidence=lambda kit,arm:{'ye_source':'same full text and figures for both arms'},
            native_m7_module_sha=lambda:'m7-native-placeholder',
            verify_frozen=lambda kit:{'file_sha256':{'test':'hash'}},
        )
        self.native=(FakeProgram,fake_prompt,lambda prog:{'challenge':'structural-check-only'},fake_bridge)
        self.patcher=patch.object(pilot,'native',return_value=self.native)
        self.patcher.start();self.addCleanup(self.patcher.stop)

    def test_prepare_frozen_inputs_and_identical_source(self):
        result=pilot.prepare(self.prev,self.kit,self.out)
        self.assertEqual(result['max_new_calls'],4)
        m=pilot.verify(self.out)
        self.assertTrue(m['same_original_parent_and_fulltext_in_both_arms'])
        self.assertEqual(sorted(m['schedule']),sorted(pilot.STEPS))
        for arm in pilot.ARMS:
            messages=pilot.load(self.out/'prompts'/('REVIEW_'+arm+'.json'))['messages']
            self.assertIn('same full text and figures',messages[1]['content'])
            self.assertIn('proposed mechanism',messages[1]['content'])
        self.assertEqual(len(list((self.out/'attempts').glob('*'))) if (self.out/'attempts').exists() else 0,0)

    def test_immutable_and_separated_destination(self):
        pilot.prepare(self.prev,self.kit,self.out)
        with self.assertRaises(FileExistsError):pilot.prepare(self.prev,self.kit,self.out)
        inputpath=self.out/'inputs/parent.json'
        payload=pilot.load(inputpath);payload['mechanism_model']='tampered'
        inputpath.write_text(json.dumps(payload),encoding='utf-8')
        with self.assertRaises(ValueError):pilot.verify(self.out)

    def test_incomplete_paid_marker_blocks_continuation(self):
        pilot.prepare(self.prev,self.kit,self.out)
        first=pilot.step_name(self.out)
        pilot.save_new(self.out/'attempts'/(first+'.json'),{'started':True})
        with self.assertRaisesRegex(RuntimeError,'UNRESOLVED'):
            pilot.step_name(self.out)

    def test_blind_requires_two_valid_outputs(self):
        pilot.prepare(self.prev,self.kit,self.out)
        with self.assertRaises(FileNotFoundError):pilot.make_blind(self.out)
        for arm in pilot.ARMS:
            step='REVISE_'+arm
            pilot.save_new(self.out/'receipts'/(step+'.json'),{'finish_reason':'stop'})
            pilot.save_new(self.out/'parsed_revisions'/(step+'.json'),self.parent)
        result=pilot.make_blind(self.out)
        self.assertTrue((self.out/'BLIND_REVIEW_PUBLIC.json').is_file())
        public=pilot.load(self.out/'BLIND_REVIEW_PUBLIC.json')
        self.assertEqual(result['new_paid_calls'],0)
        self.assertNotIn('SCIENTIFIC_ADVERSARIAL_FEEDBACK',json.dumps(public))
        self.assertNotIn('STANDARD_FEEDBACK',json.dumps(public))
        self.assertFalse(public['scientific_novelty_certified'])
        self.assertEqual(len(public['items']),2)

    def test_four_step_end_to_end_fake_provider_no_network(self):
        pilot.prepare(self.prev,self.kit,self.out)
        parent=self.parent
        reviewer={'diagnosis':['check one','check two'],
                  'repair_priorities':['repair one','repair two'],
                  'abstain_boundary':'abstain if not distinguishable',
                  'overall_judgment':'unverified'}
        class FakeResponse:
            def __init__(self,content):
                self.choices=[types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=content))]
            def model_dump(self,mode='json'):return {'model':'fake','choices':[{'finish_reason':'stop'}]}
        class FakeCompletions:
            def create(self,**kwargs):
                is_review='research reviewer' in kwargs['messages'][0]['content']
                return FakeResponse(json.dumps(reviewer if is_review else parent))
        class FakeOpenAI:
            def __init__(self,**kwargs):self.chat=types.SimpleNamespace(completions=FakeCompletions())
        with patch.dict('sys.modules',{'openai':types.SimpleNamespace(OpenAI=FakeOpenAI)}), \
             patch.dict('os.environ',{'OPENROUTER_API_KEY':'unit-test-no-network'}):
            results=[pilot.run_one(self.out,'gpt-5.6-test','OPENROUTER_API_KEY','https://example.invalid',
                                   3200,8000,0.2) for _ in range(4)]
        self.assertEqual([r['remaining'] for r in results],[3,2,1,0])
        self.assertIsNone(pilot.step_name(self.out))
        self.assertEqual(pilot.report(self.out)['calls_recorded'],4)
        blind=pilot.make_blind(self.out)
        self.assertEqual(blind['status'],'LEVEL2_BLIND_REVIEW_READY')
        self.assertEqual(len(pilot.load(self.out/'BLIND_REVIEW_PUBLIC.json')['items']),2)

    def test_revision_prompt_uses_actual_m7_generation_contract(self):
        base=pilot.revision_messages(self.parent,{'paper':'original'},
            {'diagnosis':['A','B'],'repair_priorities':['C','D'],
             'abstain_boundary':'no unique prediction','overall_judgment':'unknown'},self.task,pilot.ARMS[1])
        self.assertEqual(len(base),2)
        self.assertIn('M7 mocked scientific program schema',base[0]['content'])
        self.assertIn('independent_reviewer_feedback_search_only',base[1]['content'])
        self.assertNotIn('N^2',json.dumps(base))


if __name__=='__main__':unittest.main(verbosity=2)
