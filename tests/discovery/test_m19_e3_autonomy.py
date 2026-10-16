"""Zero-network regression for M19-E3 isolated native Decision4 fork."""
from __future__ import annotations
import json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_core.discovery.scientific_research_control import m19_e3_autonomy as e
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as base


def state_initial():
    return {'decision_count':3,'event_count':6,'ended':False,
       'decisions':[{'number':1,'action':'READ_EVIDENCE'},
                    {'number':2,'action':'READ_EVIDENCE'},
                    {'number':3,'action':'DEEPEN'}],
       'pending_work':{'work_id':e.EXPECTED_WORK_3,'status':'EXECUTED',
                        'execution_status':'COMPLETED','request':{'work_kind':'REASON'}}}

class TestE3Autonomy(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.source=Path(self.tmp.name)/'W08'
        self.source.mkdir()
        (self.source/'events').mkdir()
        (self.source/'events'/'event_0006.json').write_text('{"old":true}',encoding='utf-8')
        self.state=state_initial()
        base.write_new(self.source/'state.json',self.state)
        base.write_new(self.source/'test_source.json',{'raw_science_unchanged':True})
        self.initial_sha=base.sha(self.state)
        self.out=e.repo_root()/'experiments'/'m19_e3'/'runs'/('TEST_'+Path(self.tmp.name).name)
        self.verify_patch=patch.object(e.v18,'verify',side_effect=lambda path:{'state':base.read(path/'state.json')})
        self.req={'model':'mock-gpt','temperature':0,'max_tokens':3000,
                  'messages':[{'role':'system','content':'Return a JSON object.'},
                              {'role':'user','content':'Self-selected future science.'}],
                  'role':'decision'}
        self.request_patch=patch.object(e.v18,'_make_call_request',return_value=self.req)
        self.sha_patch=patch.object(e,'EXPECTED_W08_SHA',self.initial_sha)
        self.verify_patch.start();self.request_patch.start();self.sha_patch.start()
    def tearDown(self):
        self.sha_patch.stop();self.request_patch.stop();self.verify_patch.stop()
        if self.out.exists():shutil.rmtree(self.out)
        stage=self.out.with_name(self.out.name+'.copying')
        if stage.exists():shutil.rmtree(stage)
        self.tmp.cleanup()
    def prepare(self):return e.prepare(source=self.source,out=self.out)
    def test_clone_preserves_source_and_choice_is_no_paid(self):
        before=e._tree_files(self.source)
        z=self.prepare()
        self.assertEqual(z['status'],'PREPARED_PINNED_W08_FORK_NO_PAID')
        self.assertEqual(e._tree_files(self.source),before)
        self.assertEqual(e.verify(self.out)['decision_count'],3)
        self.assertEqual(e.preview(self.out)['model'],'mock-gpt')
    def test_duplicate_prepare_rejected(self):
        self.prepare()
        with self.assertRaises(FileExistsError):self.prepare()
    def test_output_outside_repo_refused(self):
        with self.assertRaisesRegex(ValueError,'E3_OUTPUT_MUST'):
            e.prepare(source=self.source,out=Path(self.tmp.name)/'somewhere')
    def test_source_tampering_detected(self):
        self.prepare()
        (self.source/'test_source.json').write_text('{"tampered":true}')
        with self.assertRaisesRegex(ValueError,'SOURCE_W08_CHANGED'):e.verify(self.out)
    def test_frozen_clone_tampering_detected(self):
        self.prepare()
        f=self.out/e.FORK_SUBDIR/'test_source.json'
        f.write_text('{"tampered":true}')
        with self.assertRaisesRegex(ValueError,'FORK_FROZEN_FILE_MUTATED'):e.verify(self.out)
    def test_no_paid_without_double_authorization(self):
        self.prepare()
        with patch.dict('os.environ',{'FAKG_RESEARCH_ALLOW_PAID':'','OPENROUTER_API_KEY':'not-used'}):
            with self.assertRaises(PermissionError):e.decision(self.out,allow_paid=False)
        self.assertFalse((self.out/e.FORK_SUBDIR/'calls'/'decision_0004').exists())
    def test_native_post_decision4_readout(self):
        self.prepare()
        state=base.read(self.out/e.FORK_SUBDIR/'state.json')
        state['decision_count']=4;state['event_count']=7
        state['decisions'].append({'number':4,'action':'PIVOT'})
        state['pending_work']={'work_id':'work_new','status':'REQUESTED',
                               'request':{'work_kind':'REASON','research_question':'Investigate a new causal physical mechanism'}}
        (self.out/e.FORK_SUBDIR/'state.json').write_text(json.dumps(state))
        base.write_new(self.out/e.FORK_SUBDIR/'events'/'event_0007.json',{'new_event':'decision4'})
        z=e.verify(self.out)
        self.assertEqual(z['status'],'DECISION4_COMMITTED_VERIFIED')
        self.assertEqual(z['pending_work_kind'],'REASON')
        with self.assertRaisesRegex(RuntimeError,'DECISION4_ATTEMPT'):e.preview(self.out)
    def test_symlinks_in_source_refused(self):
        (self.source/'symlink').symlink_to(self.source/'state.json')
        with self.assertRaisesRegex(ValueError,'SYMLINK_IN_REFERENCE'):
            self.prepare()
    def test_absent_raw_not_retried(self):
        self.prepare()
        d=self.out/e.FORK_SUBDIR/'calls'/'decision_0004';d.mkdir(parents=True)
        base.write_new(d/'request.json',self.req)
        base.write_new(d/'attempt.json',{'request_sha256':base.sha(self.req)})
        self.assertEqual(e.verify(self.out)['status'],'UNKNOWN_BILLING_NO_RAW_NEVER_RETRY')
        with self.assertRaisesRegex(RuntimeError,'NO_SAVED_DECISION4_RAW'):
            e.recover(self.out)
        with self.assertRaisesRegex(RuntimeError,'DECISION4_ATTEMPT'):
            e.preview(self.out)

if __name__=='__main__': unittest.main()
