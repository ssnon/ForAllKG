import copy
import json
import tempfile
import unittest
from pathlib import Path

from pipeline_core.discovery.sers_novelty_search_m14 import sha as m14sha
from pipeline_core.discovery.sers_evidence_successor_m14c2 import canon, prompt_for, SCHEMA
from scripts.discovery.recover_sers_evidence_successor_m14c2_promptaware import (
    CONTRACT, RAW, RECEIPT, INTENT, PROMPT, AUDIT, TEMPLATE,
    inspection, make_review, recover,
)

class PromptOrderRecoveryTest(unittest.TestCase):
    def fixture(self, out:Path, *, disagree='none'):
        sig={k:'different '+k for k in (
            'mechanism_family_id','causal_driver','core_assumption',
            'predicted_observable','independent_intervention','strongest_null','falsification')}
        prev={'candidate_id':'m14:old','problem_id':'gap:old','family':'VIBRATION',
              'mechanism_signature':{k:'old_'+k for k in sig}}
        card_a={'problem_id':'gap:A','domain_family':'INTERFACE','source_anchors':[{'title':'independent paper'}]}
        card_b={'problem_id':'gap:B','domain_family':'ORIENTATION','source_anchors':[{'title':'other paper'}]}
        contract={'schema':SCHEMA,'root':'/frozen/science','source_generation_sha256':'old_sha',
                  'verified_c1_findings':{'OPEN_FRONTIER':{'case_id':'2'},'MECHANISM_SEARCH':{'case_id':'1'}},
                  'limitations':['model does not establish novelty'],
                  'lanes':{'MECHANISM_SEARCH':{
                      'previous':prev,
                      'decision':{'schema':'m14b-adaptive-search-reconstruction-v1','action':'ESCAPE',
                                  'escape_alternative_problem_ids':['gap:A','gap:B'],'problem_expansion_required':False},
                      'alternative_cards':[card_a,card_b]},
                      'OPEN_FRONTIER':{'decision':{'schema':'m14b-adaptive-search-reconstruction-v1','action':'SUSPEND'},
                                       'next_evidence_required':'independent measurement'}}}
        sub={'successor':{'problem_id':'gap:A','family':'Specific interface occupancy dynamics',
              'novelty_certified':False,'mechanism_signature':sig,
              'conditional_predictions':['independent physical effect'],
              'independent_observables':['spectral observable'],
              'test_proposal':{'independent_variable':'x','null_coefficients':[1,0,0],
                  'hypothesis_coefficients':[1,.3,0], 'nuisance_envelope':.1,
                  'units':'arb','physical_scope_and_limitations':'simulated only'}},
             'suspended_lane_successor':None}
        # Exactly the real bug: prepare uses in-memory (insertion order),
        # while paid call reloads a contract serialized with sort_keys=True.
        frozen=prompt_for(contract)
        (out/CONTRACT).write_bytes(canon(contract))
        reloaded=json.loads((out/CONTRACT).read_bytes())
        actually_sent=prompt_for(reloaded)
        self.assertNotEqual(m14sha(frozen),m14sha(actually_sent))
        (out/PROMPT).write_bytes(canon(frozen))
        if disagree=='instruction':
            broken=copy.deepcopy(frozen);broken['system']+=' unapproved addition'
            (out/PROMPT).write_bytes(canon(broken))
        if disagree=='payload':
            broken=copy.deepcopy(frozen)
            broken['user']=broken['user'].replace('model does not establish novelty','model establishes novelty')
            (out/PROMPT).write_bytes(canon(broken))
        if disagree=='request':
            request_sha=m14sha(frozen)
        else:
            request_sha=m14sha(actually_sent)
        (out/INTENT).write_bytes(canon({'prompt_sha256':request_sha,'model':'mock'}))
        (out/RECEIPT).write_bytes(canon({'prompt_sha256':request_sha,'model':'mock',
                                         'finish_reason':'stop','raw':json.dumps(sub)}))
        return frozen,actually_sent,sub

    def test_semantic_key_order_provenance_and_scientific_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);frozen,request,sub=self.fixture(out)
            before={n:(out/n).read_bytes() for n in [CONTRACT,PROMPT,INTENT,RECEIPT]}
            info=inspection(out)
            self.assertEqual(info['status'],'M14C2_FAMILY_LABEL_REVIEW_POSSIBLE_NO_MODEL_CALLS')
            self.assertNotEqual(info['prompt_provenance']['frozen_prompt_sha256'],info['prompt_provenance']['paid_request_prompt_sha256'])
            self.assertTrue(info['prompt_provenance']['frozen_semantically_identical_to_paid_request'])
            make_review(out)
            review=json.loads((out/TEMPLATE).read_bytes())
            review['scientific_alignment_confirmed']=True
            review['independent_review_of_causal_driver_and_problem']=True
            review['alignment_reason']='Independent examination: the causal mechanism described corresponds to the selected interface card, not the alternate orientation card.'
            reviewed=out/'reviewed.json';reviewed.write_bytes(canon(review))
            result=recover(out,reviewed,True)
            self.assertEqual(result['status'],'M14C2_REVIEWED_FAMILY_LABEL_RECOVERED_NO_MODEL_CALLS')
            self.assertTrue((out/AUDIT).exists());self.assertTrue((out/RAW).exists())
            for name,prior in before.items():self.assertEqual((out/name).read_bytes(),prior)
            corrected=json.loads((out/RAW).read_bytes());self.assertEqual(corrected['successor']['family'],'INTERFACE')
            corrected['successor']['family']=sub['successor']['family'];self.assertEqual(corrected,sub)

    def test_changed_science_payload_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.fixture(out,disagree='payload')
            result=inspection(out)
            self.assertEqual(result['status'],'M14C2_NO_AUTOMATIC_SCIENTIFIC_REPAIR')
            self.assertIn('genuine instruction or scientific payload mismatch',result['reason'])
            self.assertFalse((out/RAW).exists())

    def test_changed_instructions_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.fixture(out,disagree='instruction')
            result=inspection(out)
            self.assertEqual(result['status'],'M14C2_NO_AUTOMATIC_SCIENTIFIC_REPAIR')
            self.assertFalse((out/RAW).exists())

    def test_receipt_from_frozen_not_actual_request_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.fixture(out,disagree='request')
            result=inspection(out)
            self.assertIn('executable generation prompt',result['reason'])

    def test_tampered_intent_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.fixture(out)
            intent=json.loads((out/INTENT).read_bytes())
            intent['prompt_sha256']='tampered'
            (out/INTENT).write_bytes(canon(intent))
            self.assertIn('intent, model and receipt',inspection(out)['reason'])

    def test_unsupported_frozen_payload_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.fixture(out)
            frozen=json.loads((out/PROMPT).read_bytes())
            frozen['user']=frozen['user'].replace('Coefficient separation has ZERO novelty or empirical-certification value. ',
                                                'Coefficient separation has scientific proof value. ')
            (out/PROMPT).write_bytes(canon(frozen))
            self.assertIn('payload boundary',inspection(out)['reason'])

if __name__=='__main__': unittest.main()
