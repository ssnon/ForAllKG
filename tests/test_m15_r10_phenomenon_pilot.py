import csv
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.discovery import run_m15_r10_phenomenon_pilot as mod


def candidate():
    return {k:'a bounded scientific proposal with explicit caveats '+k for k in mod.RESPONSE_FIELDS}

class M15R10Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)/'R10'

    def prepare(self):return mod.prepare(self.out,'mock/provider',2,.3,6000)

    def run_mock(self):
        response={'choices':[{'message':{'content':json.dumps(candidate())}}],'id':'mock-receipt'}
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':'TEST_NOT_REAL'}),\
             patch.object(mod.r9,'invoke_openrouter',return_value=response) as mock:
            result=mod.run_paid(self.out,4,True,True)
            self.assertEqual(mock.call_count,4)
        return result

    def scored(self):
        path=self.out/'scored.csv'
        with (self.out/'M15R10_BLIND_RUBRIC.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            for idx,row in enumerate(rows):
                for k in mod.SCORES:row[k]=1 if idx%2 else 2
                row['prior_art_risk_low_med_high']='high'
                row['notes']='blinded science review (synthetic test)'
                writer.writerow(row)
        return path

    def test_small_size_rayleigh_agreement(self):
        x=.05;m=1.5
        exact=mod.mie_qsca(x,m);approx=mod.rayleigh_qsca(x,m)
        self.assertLess(abs(exact-approx)/exact,.003)

    def test_invisible_sphere(self):
        for x in [.1,1.,6.]:self.assertEqual(mod.mie_qsca(x,1.0),0.0)

    def test_nonnegative_mie(self):
        for x in [.2,2.,5.,7.,9.]:
            q=mod.mie_qsca(x,1.5)
            self.assertTrue(math.isfinite(q) and q>=0)

    def test_mie_rejects_invalid_range(self):
        with self.assertRaisesRegex(ValueError,'DOMAIN'):mod.mie_qsca(30,1.5)
        with self.assertRaisesRegex(ValueError,'DOMAIN'):mod.mie_qsca(2,.9)

    def test_genuine_nonmonotonic_training(self):
        d=mod.discover_pattern();v=d['selected_pattern']
        self.assertGreater(v['q_hi'],v['q_lo'])
        self.assertGreater(v['fractional_drop'],.05)
        self.assertGreater(v['x_lo'],v['x_hi'])
        self.assertEqual(len(d['full_training_points']),61)

    def test_prepare_freezes_training_and_holdout_only_as_commitment(self):
        p=self.prepare();self.assertEqual(p['planned_paid_calls'],4)
        self.assertFalse(p['heldout_values_evaluated'])
        saved=json.loads((self.out/'M15R10_PLAN_PRIVATE.json').read_text())
        for c in saved['calls']:
            self.assertNotIn('1.48',c['user_prompt'])
            self.assertNotIn('1.52',c['user_prompt'])
        self.assertEqual(mod.check(self.out)['saved_receipts'],0)

    def test_prepare_wont_overwrite(self):
        self.prepare()
        with self.assertRaisesRegex(FileExistsError,'NO_OVERWRITE'):self.prepare()

    def test_no_paid_without_explicit_flags(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'EXPLICIT_PAID'):
            mod.run_paid(self.out,4,False,True)

    def test_no_paid_without_key(self):
        self.prepare()
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':''}):
            with self.assertRaisesRegex(ValueError,'OPENROUTER_API_KEY_MISSING'):
                mod.run_paid(self.out,4,True,True)

    def test_changes_train_data_blocked(self):
        self.prepare()
        f=self.out/'M15R10_TRAIN_PHYSICS.json'
        obj=json.loads(f.read_text());obj['selected_pattern']['q_hi']=400
        f.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'TRAINING_PHYSICS_CHANGED'):
            mod.check(self.out)

    def test_prompt_change_blocked(self):
        self.prepare();f=self.out/'M15R10_PLAN_PRIVATE.json'
        obj=json.loads(f.read_text());obj['calls'][0]['user_prompt']+=' leakage'
        f.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'PROMPT_CHANGED'):
            mod.check(self.out)

    def test_json_validation(self):
        self.assertEqual(mod.parsed_candidate(json.dumps(candidate()))[1],'VALID_UNREVIEWED')
        self.assertEqual(mod.parsed_candidate('invalid')[1],'NON_JSON')
        self.assertEqual(mod.parsed_candidate(json.dumps({'research_question':'hi'}))[1],'MISSING_REQUIRED_FIELDS')

    def test_integration_paid_blind_reveal_check(self):
        self.prepare();self.assertEqual(self.run_mock()['status'],'R10_RESPONSES_COMPLETE')
        b=mod.blind(self.out);self.assertEqual(b['candidate_count'],4)
        self.assertEqual(mod.check(self.out)['saved_receipts'],4)
        private=json.loads((self.out/'M15R10_PRIVATE_ASSIGNMENT.json').read_text())
        pub=json.loads((self.out/'M15R10_BLIND_PACKET.json').read_text())
        self.assertNotIn('PHENOMENON_TREATMENT',json.dumps(pub))
        self.assertEqual(len(private['assignment']),4)
        output=mod.reveal(self.out,self.scored());self.assertEqual(output['status'],'R10_POST_UNBLIND_WITH_HELDOUT_MIE')
        saved=json.loads((self.out/'M15R10_POST_UNBLIND.json').read_text())
        self.assertEqual(len(saved['heldout']),2)
        self.assertTrue(saved['heldout_direction_pass'])
        self.assertFalse(saved['scientific_novelty_certified'])
        self.assertEqual(mod.check(self.out)['revealed'],True)

    def test_run_closed_after_blind(self):
        self.prepare();self.run_mock();mod.blind(self.out)
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':'TEST'}):
            with self.assertRaisesRegex(ValueError,'RUN_CLOSED'):
                mod.run_paid(self.out,1,True,True)

    def test_blind_requires_all(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'MISSING_PAID_RESPONSE'):mod.blind(self.out)

    def test_duplicate_blind_blocked(self):
        self.prepare();self.run_mock();mod.blind(self.out)
        with self.assertRaisesRegex(FileExistsError,'ALREADY_EXISTS'):mod.blind(self.out)

    def test_receipt_tampering_rejected(self):
        self.prepare();self.run_mock();mod.blind(self.out)
        receipt=next((self.out/'responses').glob('*.json'))
        data=json.loads(receipt.read_text());data['raw_provider_response']['id']='TAMPERED'
        receipt.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'RESPONSE_CHANGED_AFTER_BLIND'):
            mod.check(self.out)

    def test_incomplete_review_blocked_before_holdout(self):
        self.prepare();self.run_mock();mod.blind(self.out)
        scored=self.scored()
        s=scored.read_text().splitlines()
        scored.write_text('\n'.join(s[:-1])+'\n')
        with self.assertRaisesRegex(ValueError,'REVIEW_INCOMPLETE'):
            mod.reveal(self.out,scored)
        self.assertFalse((self.out/'M15R10_POST_UNBLIND.json').exists())

    def test_review_score_out_of_range(self):
        self.prepare();self.run_mock();mod.blind(self.out)
        scored=self.scored()
        with scored.open(newline='') as f:
            rows=list(csv.DictReader(f))
        rows[0]['prior_art_risk_low_med_high']='bad-risk'
        with scored.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with self.assertRaisesRegex(ValueError,'RISK_RATING_REQUIRED'):
            mod.reveal(self.out,scored)

    def test_holdout_recomputed_check(self):
        self.prepare();self.run_mock();mod.blind(self.out)
        mod.reveal(self.out,self.scored());p=self.out/'M15R10_POST_UNBLIND.json'
        obj=json.loads(p.read_text());obj['heldout'][0]['q_hi']=100
        p.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'HELDOUT_CHANGED'):mod.check(self.out)

if __name__=='__main__':unittest.main()
