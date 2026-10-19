"""Offline integration test; uses the real native Case B request ledger with a fake provider."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_core.discovery.m30_sweep import runner

class FakeBackend:
    calls=0
    def complete(self,*,system,user,model,temperature,max_tokens):
        self.calls+=1
        d=json.loads(user)
        task=d['scientific_problem_and_public_evidence']['id']
        stage=d['phase']
        if task.startswith('sidon'): cand={'set':[0,1,3,7]}
        elif task.startswith('dfa'):cand={'transitions':[[0,0]],'accepting':[]}
        elif task.startswith('dyn'):cand={'dx':{'const':0},'dy':{'const':0}}
        else:cand={'const':1}
        if stage=='proposal':
            ans={'hypothesis':'Synthetic candidate for safe mock backend',
                 'substantive_reasoning':'Inspect observations and form a possible mechanism',
                 'candidate':cand,'null_or_alternative':'A weaker constant baseline',
                 'proposed_probe':{},'predicted_probe_result':'Prediction is uncertain',
                 'limitations':'No independent validation'}
        elif stage=='revision':
            ans={'what_changed_and_why':'No external data, retain baseline candidate',
                 'hypothesis':'Synthetic candidate remains a plausible placeholder',
                 'candidate':cand,'strongest_counterargument':'Hidden evaluation may reject it',
                 'proposed_probe':{},'predicted_probe_result':'Further data would help'}
        else:
            ans={'hypothesis':'Mock hypothesis with no scientific authority',
                 'candidate':cand,'core_consequence':'The current candidate is testable',
                 'strongest_remaining_obstacle':'Insufficient evidence for explanatory validity',
                 'update_from_evidence':'No substantive model updates',
                 'novelty_or_truth_certified':False}
        return ans,{'prompt_tokens':123,'completion_tokens':80,'provider_cost_credits':0,'served_model':model,'response_id':'offline-mock'}

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)/'run'
        runner.prepare(self.out,model='offline/mock')
    def tearDown(self):self.tmp.cleanup()
    def test_prepare_preview(self):
        self.assertEqual(runner.state(self.out)['episodes_total'],80)
        self.assertEqual(runner.preview_episode(self.out,'sidon_25','PLAIN',1)['stage'],'proposal')
        self.assertEqual(runner.state(self.out)['paid_attempts_recorded'],0)
    def test_tools_two_probes_and_reports(self):
        backend=FakeBackend()
        with patch.dict(os.environ,{'FAKG_M30_AUTHORIZE_PAID':'YES'}):
            for _ in range(3):runner.next_episode(self.out,'dyn_predator','FAKG_ADAPTER_TOOL',1,True,backend=backend)
        self.assertEqual(backend.calls,3)
        self.assertEqual(runner.preview_episode(self.out,'dyn_predator','FAKG_ADAPTER_TOOL',1)['status'],'COMPLETE')
        self.assertEqual(runner.state(self.out)['tool_probes_completed'],2)
        r=runner.report(self.out)
        self.assertTrue((self.out/'reports/scores.csv').is_file())
        self.assertEqual(r['episodes_completed'],1)
    def test_block_unapproved_call(self):
        with self.assertRaises(PermissionError):runner.next_episode(self.out,'sidon_25','PLAIN',1,True,FakeBackend())
        self.assertEqual(runner.state(self.out)['paid_attempts_recorded'],0)
    def test_recover_existing_raw_without_provider_call(self):
        backend=FakeBackend()
        with patch.dict(os.environ,{'FAKG_M30_AUTHORIZE_PAID':'YES'}):
            runner.next_episode(self.out,'sidon_25','PLAIN',1,True,backend=backend)
        d=runner.task_dir(self.out,'sidon_25','PLAIN',1)/'calls/proposal'
        (d/'receipt.json').unlink();(d/'response.json').unlink()
        self.assertEqual(runner.preview_episode(self.out,'sidon_25','PLAIN',1)['status'],'BLOCKED')
        self.assertEqual(runner.recover(self.out,'sidon_25','PLAIN',1)['status'],'RECOVERED_FROM_SAVED_RAW_NO_PAID_CALL')
        self.assertEqual(backend.calls,1)
    def test_refuses_problem_mutation(self):
        d=runner.task_dir(self.out,'sidon_25','PLAIN',1)/'problem.json'
        v=json.loads(d.read_text());v['scientific_problem']='touched';d.write_text(json.dumps(v))
        with self.assertRaises(ValueError):runner.state(self.out)
if __name__=='__main__':unittest.main()
