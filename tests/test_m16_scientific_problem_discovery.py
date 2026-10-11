import csv
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.discovery import run_m16_scientific_problem_discovery as m


class M16Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.train=self.root/'M15R10_TRAIN_PHYSICS.json'
        self.post=self.root/'M15R10_POST_UNBLIND.json'
        self.out=self.root/'m16'
        train={'selected_pattern':{'x_hi':6.6,'x_lo':7.15,'q_hi':2.447958,
                                   'q_lo':1.735729,'fractional_drop':.290948},
               'n_relative':1.5,'material':'nondispersive dielectric, nonmetal',
               'science_novelty_certified':False}
        m.new_json(self.train,train)
        post={'schema_version':'m15-r10-phenomenon-driven-ab-pilot-v1',
              'status':'R10_POST_UNBLIND_WITH_HELDOUT_MIE','scientific_novelty_certified':False,
              'au_ag_science_gate':'BLOCKED','training_sha256':m.sha(self.train),
              'heldout_direction_pass':True,
              'heldout':[{'index':1.48,'direction_preserved':True},
                         {'index':1.52,'direction_preserved':True}]}
        m.new_json(self.post,post)

    def prep(self):return m.prepare(self.post,self.train,self.out,'mock/model',2,.3,6000)

    def fake_run(self):
        fields={k:'scientifically limited, independently testable description for '+k for k in m.REQUIRED}
        fields['decision']='PROPOSE'
        def provider(prompt,system,model,temp,max_tokens):
            self.assertEqual(system,m.SYSTEM)
            return {'choices':[{'message':{'content':json.dumps(fields)}}]}
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-token'}),\
             patch.object(m.r9,'invoke_openrouter',side_effect=provider):
            return m.run_paid(self.out,6,True,True)

    def scored(self):
        p=self.root/'completed.csv'
        packet=json.loads((self.out/'M16_BLIND_PACKET.json').read_text())
        with p.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['blind_id',*m.SCORE_COLS,*m.CSV_TAIL])
            for row in packet['items']:
                w.writerow([row['blind_id'],*[2]*len(m.SCORE_COLS),'high','no',
                            'Specific observation and uncertainty commentary.'])
        return p

    def test_source_and_prepare_six_calls(self):
        r=self.prep()
        self.assertEqual(r['planned_paid_calls'],6)
        self.assertFalse(r['novelty_certified'])
        p=m.load_plan(self.out)
        self.assertEqual(set(x['arm'] for x in p['calls']),set(m.ARMS))
        self.assertEqual(len(p['calls']),6)
        self.assertEqual(len(set(x['blind_id'] for x in p['calls'])),6)

    def test_three_distinct_arms_and_shared_literature(self):
        self.prep();p=m.load_plan(self.out)
        text={a:next(c['prompt'] for c in p['calls'] if c['arm']==a) for a in m.ARMS}
        self.assertEqual(len(set(text.values())),3)
        for s in text.values():
            for r in m.REFS:self.assertIn(r['doi'],s)
            self.assertIn('novel',s.lower())
        self.assertNotIn('2.447958',text['BASELINE'])
        self.assertNotIn('2.447958',text['KNOWN_PHYSICS'])
        self.assertIn('2.447958',text['NUMERIC_PHENOMENON'])
        self.assertIn('not',text['NUMERIC_PHENOMENON'].lower())

    def test_reject_source_hash_tampering(self):
        self.prep();t=json.loads(self.train.read_text());t['material']='gold';m_path=self.train
        m_path.write_text(json.dumps(t))
        with self.assertRaisesRegex(ValueError,'R10_TRAINING_SHA_MISMATCH'):
            m.load_plan(self.out)

    def test_reject_forged_qualified_sers(self):
        p=json.loads(self.post.read_text());p['au_ag_science_gate']='PASS'
        self.post.write_text(json.dumps(p))
        with self.assertRaisesRegex(ValueError,'R10_AUTHORITY_INVALID'):
            self.prep()

    def test_explicit_paid_authorization(self):
        self.prep()
        with self.assertRaisesRegex(ValueError,'AUTHORIZATION_REQUIRED'):
            m.run_paid(self.out,6,False,True)
        with self.assertRaisesRegex(ValueError,'AUTHORIZATION_REQUIRED'):
            m.run_paid(self.out,6,True,False)

    def test_mock_provider_calls_and_blind(self):
        self.prep();result=self.fake_run()
        self.assertEqual(result['new_paid_calls'],6)
        self.assertEqual(result['invalid_structured_responses'],0)
        r=m.blind(self.out)
        self.assertEqual(r['candidate_count'],6)
        pkt=json.loads((self.out/'M16_BLIND_PACKET.json').read_text())
        self.assertNotIn('assignment',pkt)
        self.assertFalse(pkt['scientific_novelty_certified'])
        self.assertEqual(m.check(self.out)['responses_saved'],6)
        with self.assertRaisesRegex(ValueError,'NO_MODEL_CALLS_AFTER_BLIND'):
            m.run_paid(self.out,6,True,True)

    def test_invalid_json_cannot_silently_blind(self):
        self.prep()
        plan=m.load_plan(self.out)
        for c in plan['calls']:
            m.new_json(self.out/'responses'/(c['call_id']+'.json'),
                       {'prompt_sha256':c['prompt_sha256'],'blind_id':c['blind_id'],
                        'parse_status':'NON_JSON','candidate':None})
        with self.assertRaisesRegex(ValueError,'MODEL_OUTPUT_INVALID_NO_SILENT_RETRY'):
            m.blind(self.out)

    def test_unblind_three_arm_results(self):
        self.prep();self.fake_run();m.blind(self.out)
        res=m.unblind(self.out,self.scored())
        self.assertEqual(res['numeric_minus_known_physics'],0)
        self.assertEqual(set(res['means_out_of_12']),set(m.ARMS))
        self.assertEqual(m.check(self.out)['status'],'M16_CHECK_OK')
        with self.assertRaisesRegex(FileExistsError,'POST_UNBLIND_EXISTS'):
            m.unblind(self.out,self.scored())

    def test_reject_review_incomplete(self):
        self.prep();self.fake_run();m.blind(self.out)
        p=self.scored()
        with p.open() as f:rows=list(csv.DictReader(f))
        rows[0][m.SCORE_COLS[2]]=''
        with p.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        with self.assertRaisesRegex(ValueError,'INCOMPLETE_0_TO_2'):
            m.unblind(self.out,p)

    def test_reject_review_missing_item(self):
        self.prep();self.fake_run();m.blind(self.out)
        p=self.scored()
        with p.open() as f:rows=list(csv.DictReader(f))[1:]
        with p.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        with self.assertRaisesRegex(ValueError,'REVIEW_INCOMPLETE'):
            m.unblind(self.out,p)

    def test_reject_private_assignment_after_blind_tamper(self):
        self.prep();self.fake_run();m.blind(self.out)
        p=self.out/'M16_PRIVATE_ASSIGNMENT.json'
        private=json.loads(p.read_text());private['blind_packet_sha256']='bad'
        p.write_text(json.dumps(private))
        with self.assertRaisesRegex(ValueError,'UNBLIND_FREEZE_HASH_MISMATCH'):
            m.unblind(self.out,self.scored())

    def test_reject_response_after_blind_mutation(self):
        self.prep();self.fake_run();m.blind(self.out)
        resp=next((self.out/'responses').glob('*.json'))
        resp.write_text(resp.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'RECEIPT_CHANGED_AFTER_BLIND'):
            m.unblind(self.out,self.scored())

    def test_duplicate_prepare_not_overwrite(self):
        self.prep()
        with self.assertRaisesRegex(FileExistsError,'OUT_EXISTS_NO_OVERWRITE'):
            self.prep()

    def test_parser_propose_or_abstain(self):
        data={k:'value for '+k for k in m.REQUIRED};data['decision']='ABSTAIN'
        parsed,status=m.parse_candidate('```json\n'+json.dumps(data)+'\n```')
        self.assertEqual(status,'VALID_UNREVIEWED');self.assertEqual(parsed['decision'],'ABSTAIN')
        data['decision']='NOVEL_DISCOVERY_CERTIFIED'
        self.assertEqual(m.parse_candidate(json.dumps(data))[1],'INVALID_DECISION')

    def test_no_sers_authority_after_review(self):
        self.prep();self.fake_run();m.blind(self.out);m.unblind(self.out,self.scored())
        r=json.loads((self.out/'M16_POST_UNBLIND.json').read_text())
        self.assertFalse(r['scientific_novelty_certified'])
        self.assertFalse(r['independent_literature_novelty_search_performed'])
        self.assertFalse(r['actual_new_scientific_observation_obtained'])
        self.assertEqual(r['au_ag_science_gate'],'BLOCKED')

    def test_reject_prompt_after_prepare(self):
        self.prep()
        plan_path=self.out/'M16_PLAN_PRIVATE.json'
        plan=json.loads(plan_path.read_text())
        plan['calls'][0]['prompt']+=' extra fabricated claim'
        plan_path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'M16_PROMPT_CHANGED'):
            m.load_plan(self.out)

    def test_reject_post_report_changed(self):
        self.prep();self.fake_run();m.blind(self.out)
        m.unblind(self.out,self.scored())
        report=self.out/'M16_POST_UNBLIND_REPORT.md'
        report.write_text(report.read_text()+'arbitrary modification')
        with self.assertRaisesRegex(ValueError,'POST_REPORT_HASH_MISMATCH'):
            m.check(self.out)

    def test_unblind_must_not_run_before_blind(self):
        self.prep()
        with self.assertRaisesRegex(ValueError,'BLIND_PACKET_MISSING'):
            m.unblind(self.out,self.root/'fake.csv')

    def test_legacy_r10_invalid_domain(self):
        source=json.loads(self.train.read_text())
        source['material']='Au core shell metal'
        m.new_json(self.root/'othertrain.json',source)
        post=json.loads(self.post.read_text());post['training_sha256']=m.sha(self.root/'othertrain.json')
        m.new_json(self.root/'otherpost.json',post)
        with self.assertRaisesRegex(ValueError,'R10_NOT_KNOWN_DIELECTRIC_SPHERE'):
            m.r10_source(self.root/'otherpost.json',self.root/'othertrain.json')

    def test_scores_fail_on_domain_misuse_invalid_flag(self):
        self.prep();self.fake_run();m.blind(self.out)
        p=self.scored()
        with p.open() as f:rows=list(csv.DictReader(f))
        rows[0]['misuses_mie_as_sers_evidence_yes_no']='perhaps'
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        with self.assertRaisesRegex(ValueError,'DOMAIN_MISUSE_FLAG'):
            m.unblind(self.out,p)

if __name__=='__main__':unittest.main()
