import copy
import json
import tempfile
import unittest
from pathlib import Path

from pipeline_core.discovery.sers_novelty_search_m14 import sha as m14sha
from pipeline_core.discovery.sers_evidence_successor_m14c2 import canon, SCHEMA
from scripts.discovery.run_sers_evidence_successor_m14c2 import CONTRACT, RAW, cmd_verify
from scripts.discovery.recover_sers_evidence_successor_m14c2_family import (
    RECEIPT, INTENT, PROMPT, TEMPLATE, AUDIT, inspection, make_review, recover,
)

class RecoverTest(unittest.TestCase):
    def make(self, d, *, family='Specific interface occupancy dynamics', problem='gap:A', raw=None, finish='stop'):
        path=Path(d)
        sig={x:'old_'+x for x in ('mechanism_family_id','causal_driver','core_assumption',
             'predicted_observable','independent_intervention','strongest_null','falsification')}
        card={'problem_id':'gap:A','domain_family':'INTERFACE','source_anchors':[{'title':'Independent surface study'}]}
        old={'problem_id':'gap:old','family':'VIBRATION','mechanism_signature':sig}
        contract={'schema':SCHEMA,'lanes':{'MECHANISM_SEARCH':{'previous':old,
            'decision':{'schema':'m14b-adaptive-search-reconstruction-v1','action':'ESCAPE',
              'escape_alternative_problem_ids':['gap:A'],'problem_expansion_required':False},
            'alternative_cards':[card,{'problem_id':'gap:B','domain_family':'ORIENTATION','source_anchors':[{'title':'Other work'}]}]},
            'OPEN_FRONTIER':{'decision':{'schema':'m14b-adaptive-search-reconstruction-v1','action':'SUSPEND'},
              'next_evidence_required':'Temperature'}}}
        nsig={k:'genuinely different '+k for k in sig}
        submitted={'successor':{'problem_id':problem,'family':family,'novelty_certified':False,
             'mechanism_signature':nsig,'conditional_predictions':['Observed response independent of background'],
             'independent_observables':['SERS and IR'],
             'test_proposal':{'independent_variable':'x','null_coefficients':[1,0,0],
                'hypothesis_coefficients':[1,.2,0], 'nuisance_envelope':0.1,
                'units':'arbitrary', 'physical_scope_and_limitations':'synthetic only'}},
                'suspended_lane_successor':None}
        p={'system':'SERS','user':'choose card'}
        (path/CONTRACT).write_bytes(canon(contract))
        (path/PROMPT).write_bytes(canon(p))
        (path/RECEIPT).write_bytes(canon({'prompt_sha256':m14sha(p),'model':'mock','finish_reason':finish,
                                        'raw':json.dumps(submitted) if raw is None else raw}))
        (path/INTENT).write_bytes(canon({'prompt_sha256':m14sha(p),'model':'mock'}))
        return submitted
    def review(self,d):
        p=Path(d)/TEMPLATE
        make_review(Path(d)); r=json.loads(p.read_text())
        r['scientific_alignment_confirmed']=True
        r['independent_review_of_causal_driver_and_problem']=True
        r['alignment_reason']='Reviewed the selected card and causal driver: specific mechanism lies within the interface question family; nothing else edited.'
        reviewed=Path(d)/'reviewed.json';reviewed.write_bytes(canon(r));return reviewed
    def test_success_with_original_receipt_unchanged_and_original_verify(self):
        with tempfile.TemporaryDirectory() as d:
            sub=self.make(d)
            r0=(Path(d)/RECEIPT).read_bytes(); contract0=(Path(d)/CONTRACT).read_bytes()
            data=inspection(Path(d))
            self.assertEqual(data['status'],'M14C2_FAMILY_LABEL_REVIEW_POSSIBLE_NO_MODEL_CALLS')
            self.assertEqual(data['anchored_card_domain_family'],'INTERFACE')
            review=self.review(d)
            result=recover(Path(d),review,True)
            self.assertEqual(result['new_model_calls'],0)
            self.assertTrue((Path(d)/AUDIT).is_file())
            self.assertEqual((Path(d)/RECEIPT).read_bytes(),r0)
            self.assertEqual((Path(d)/CONTRACT).read_bytes(),contract0)
            parsed=json.loads((Path(d)/RAW).read_bytes())
            self.assertEqual(parsed['successor']['family'],'INTERFACE')
            parsed['successor']['family']=sub['successor']['family']
            self.assertEqual(parsed,sub)
            cmd_verify(type('Args',(),{'out_dir':Path(d)}))
            with self.assertRaises(FileExistsError):recover(Path(d),review,True)
    def test_no_confirmation_no_write(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d);r=self.review(d)
            with self.assertRaises(ValueError):recover(Path(d),r,False)
            self.assertFalse((Path(d)/RAW).exists())
    def test_no_scientific_review_no_write(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d); make_review(Path(d))
            with self.assertRaises(ValueError):recover(Path(d),Path(d)/TEMPLATE,True)
            self.assertFalse((Path(d)/AUDIT).exists())
    def test_receipt_tamper_detected(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d);review=self.review(d)
            receipt=Path(d)/RECEIPT
            original=json.loads(receipt.read_text());original['raw']+=' '
            receipt.write_bytes(canon(original))
            with self.assertRaises(ValueError):recover(Path(d),review,True)
    def test_wrong_problem_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d,problem='gap:fake')
            out=inspection(Path(d));self.assertEqual(out['status'],'M14C2_NO_AUTOMATIC_SCIENTIFIC_REPAIR')
            self.assertFalse((Path(d)/RAW).exists())
    def test_follows_other_card_family_no_repair(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d,family='ORIENTATION')
            out=inspection(Path(d));self.assertIn('matches another card',out['reason'])
    def test_invalid_json_no_repair(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d,raw='{not-json')
            out=inspection(Path(d));self.assertEqual(out['status'],'M14C2_NO_AUTOMATIC_SCIENTIFIC_REPAIR')
    def test_truncated_receipt_no_repair(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d,finish='length')
            out=inspection(Path(d));self.assertIn('incomplete',out['reason'])
    def test_wrong_mechanism_after_repair_no_repair(self):
        with tempfile.TemporaryDirectory() as d:
            submitted=self.make(d)
            receipt=Path(d)/RECEIPT
            data=json.loads(receipt.read_text())
            submitted['successor']['mechanism_signature']['causal_driver']='old_causal_driver'
            data['raw']=json.dumps(submitted);receipt.write_bytes(canon(data))
            out=inspection(Path(d));self.assertEqual(out['status'],'M14C2_NO_AUTOMATIC_SCIENTIFIC_REPAIR')
    def test_existing_submission_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d);review=self.review(d)
            (Path(d)/RAW).write_text('original frozen')
            with self.assertRaises(FileExistsError):recover(Path(d),review,True)
            self.assertEqual((Path(d)/RAW).read_text(),'original frozen')
    def test_wrong_card_review_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d);review=self.review(d)
            r=json.loads(review.read_text());r['selected_problem_id']='gap:B';review.write_bytes(canon(r))
            with self.assertRaises(ValueError):recover(Path(d),review,True)
    def test_no_auto_broad_family_when_current_exact_match(self):
        with tempfile.TemporaryDirectory() as d:
            self.make(d,family='INTERFACE')
            out=inspection(Path(d));self.assertEqual(out['status'],'M14C2_NO_AUTOMATIC_SCIENTIFIC_REPAIR')

if __name__=='__main__':unittest.main()
