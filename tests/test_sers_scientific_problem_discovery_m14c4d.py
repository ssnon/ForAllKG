import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from pipeline_core.discovery import sers_scientific_problem_discovery_m14c4d as c
from scripts.discovery import run_sers_scientific_problem_discovery_m14c4d as r


def works():
    # Clearly labeled synthetic fixture: text here is NOT scientific evidence.
    context=['molecular orientation distributions','water adsorption changes','charge transfer changes','nanogap field maps',
             'single molecule fluctuations','vibrational lifetime estimates','Raman intensity calibration','spectral intermittency',
             'electrochemical interface potentials','nanoparticle surface curvature','isotope-sensitive molecular modes',
             'surface coverage heterogeneity']
    out=[]
    for i,subject in enumerate(context):
        sentence=('In a synthetic TEST fixture only, surface enhanced Raman spectroscopy examines '
                  +subject+' groupmark'+str(i//2)+'aaa. Independent experimental controls and variable interpretation remain necessary. '
                  +'Synthetic benchmark observations are not results and may reflect alternative optical mechanisms. '
                  +' '.join('testfactor'+str(i)+'k'+str(k) for k in range(32)))
        out.append({'work_id':f'TEST:W{i}', 'title':f'SERS fixture {i} {subject} calibration',
                    'abstract':sentence,'doi':f'10.invalid/{i}','year':2020+i%5,'url':'https://example.invalid/'+str(i)})
    return out


def valid_candidate(docs, i=0):
    return {'research_question':'How can '+['adsorbatewater','plasmoniccharge','vibrationalphase','molecularalignment'][i//2]+f' mechanism family {i} be distinguished using extra intervention {i} in crossmodal spectral data?',
        'hypothesis':f'A competing mechanism in synthetic test scenario {i} changes an independently measurable response.',
        'strongest_existing_null':f'An optical orientation and sampling variation model {i} may explain the same signal without new physical chemistry.',
        'discriminating_prediction':f'The two mechanisms predict different conditional values under independent experimental setting {i}.',
        'independent_measurement':f'Measure calibrated coverage and enhancement with independent instruments for intervention {i}.',
        'falsification':f'The hypothesis is rejected if calibrated alternative null predicts all heldout outcomes for setting {i}.',
        'evidence_anchors':[{'source_id':d['source_id'],'exact_quote':d['excerpt'][:90]} for d in docs[i*2:i*2+2]],
        'possible_prior_art_overlap':'Prior work on selection rules and laser enhanced Raman spectroscopy may overlap substantially.',
        'why_this_is_not_just_a_known_problem':'Potentially a new combination of independent tests, but no novelty verified from these abstracts.',
        'next_action':'SEARCH_PRIOR_ART','new_scientific_discovery_certified':False}

class TestScientificProblemDiscovery(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root=Path(self.t.name)
        self.m14=self.root/'M14'; (self.m14/'common_discovery').mkdir(parents=True)
        r.freeze(self.m14/'common_discovery/M14_SOURCE_SNAPSHOT.json',{'works':works()})
        r.freeze(self.m14/'M14_PROVISIONAL_PROBLEMS.json',{'questions':[
            {'scientific_question':'Original M14 toy problem about vibration dynamics'},
            {'scientific_question':'Original M14 toy problem about spectral selection'},
            {'scientific_question':'Original M14 toy problem about interfacial reaction'},
            {'scientific_question':'Original M14 toy problem about optical fields'}]})
        self.out=self.root/'new'

    def prepared(self):
        res=r.prepare(self.out,self.m14,desired_candidates=5)
        self.assertEqual(res['paid_calls'],0)
        return r.context(self.out)

    def raw_receipt(self, docs, candidates=None):
        plan,_,_,prompt=r.context(self.out)
        cs=candidates or [valid_candidate(docs,0),valid_candidate(docs,2)]
        r.freeze(self.out/r.RECEIPT,{'prompt_sha256':c.sha(c.canonical(prompt)),
           'model':'TEST-NO-BILL','finish_reason':'stop','raw':json.dumps({'candidates':cs})})

    def test_prepare_and_replay_no_model(self):
        plan,docs,briefs,prompt=self.prepared()
        self.assertGreaterEqual(len(briefs),3)
        self.assertFalse(plan['paid_calls_authorized_by_plan'])
        self.assertEqual(r.inspect(self.out)['scientific_gaps_proven'],0)
        self.assertEqual(r.verify(self.out)['status'],'M14C4D_PREPARED_ONLY_NO_GENERATION_YET')

    def test_real_user_data_is_optional(self):
        self.prepared()
        with self.assertRaises(FileExistsError):r.prepare(self.out,self.m14)

    def test_source_dedup_by_title_and_doi(self):
        rows=works();rows.append({**rows[0],'work_id':'TEST:DIFFERENT','title':'Other title'});
        docs,excluded=c.documents_from_snapshot({'works':rows})
        self.assertEqual(len(docs),len(works()))
        self.assertIn('DUPLICATE_PUBLICATION_ID_OR_TITLE',{x['reason'] for x in excluded})

    def test_non_sers_source_excluded(self):
        d=works()[0];d['title']='Black hole gravity cosmology';d['abstract']='This work describes stationary black hole cosmology without spectroscopy. '*3
        docs,excluded=c.documents_from_snapshot({'works':[d]})
        self.assertEqual(len(docs),0)
        self.assertEqual(excluded[0]['reason'],'OUT_OF_SCOPE_ABSTRACT')

    def test_harvest_network_requires_explicit_opt_in(self):
        with self.assertRaises(ValueError):r.harvest(self.out,False)
        self.assertFalse(self.out.exists())

    def test_valid_candidates_and_replay(self):
        _,docs,_,_=self.prepared()
        self.raw_receipt(docs)
        result=r.validate(self.out)
        self.assertEqual(result['validated_candidate_questions'],2)
        self.assertEqual(r.verify(self.out)['status'],'M14C4D_REPLAY_VERIFIED_OPEN_PROSPECTS_ONLY')
        self.assertFalse(r.load(self.out/r.CANDIDATES)['scientific_novelty_certified'])
        self.assertFalse(r.load(self.out/r.AUDIT)['independent_fulltext_prior_art_review_done'])

    def test_fabricated_quote_rejected(self):
        _,docs,_,_=self.prepared()
        ca=valid_candidate(docs,0)
        ca['evidence_anchors'][0]['exact_quote']='Invented 110% verified optical measurement'
        self.raw_receipt(docs,[ca,valid_candidate(docs,2)])
        with self.assertRaisesRegex(ValueError,'Unmatched exact'):
            r.validate(self.out)
        self.assertFalse((self.out/r.CANDIDATES).exists())

    def test_same_source_twice_rejected(self):
        _,docs,_,_=self.prepared();ca=valid_candidate(docs,0)
        ca['evidence_anchors'][1]=ca['evidence_anchors'][0].copy()
        self.raw_receipt(docs,[ca,valid_candidate(docs,2)])
        with self.assertRaisesRegex(ValueError,'distinct'):r.validate(self.out)

    def test_claim_of_novelty_rejected(self):
        _,docs,_,_=self.prepared();ca=valid_candidate(docs,0)
        ca['new_scientific_discovery_certified']=True
        self.raw_receipt(docs,[ca,valid_candidate(docs,2)])
        with self.assertRaisesRegex(ValueError,'certification'):r.validate(self.out)

    def test_duplicate_key_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate JSON key'):r._strict_json('{"candidates":[],"candidates":[]}')

    def test_source_file_tamper_fails_closed(self):
        self.prepared()
        p=self.m14/'M14_PROVISIONAL_PROBLEMS.json'
        p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'SOURCE_HASH'):r.inspect(self.out)

    def test_prompt_tamper_fails_closed(self):
        self.prepared();p=self.out/r.PROMPT
        p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'REPLAY_MISMATCH'):r.inspect(self.out)

    def test_receipt_prompt_mismatch_fails(self):
        _,docs,_,_=self.prepared();self.raw_receipt(docs)
        rec=r.load(self.out/r.RECEIPT);rec['prompt_sha256']='0'*64
        (self.out/r.RECEIPT).write_bytes(c.canonical(rec))
        with self.assertRaisesRegex(ValueError,'RAW_RECEIPT_INVALID'):r.validate(self.out)

    def test_paid_refusal_without_two_explicit_flags(self):
        self.prepared()
        with self.assertRaisesRegex(ValueError,'both'):
            r.paid(self.out,'test','TEST_KEY','https://example.invalid',1000,True,False)
        self.assertFalse((self.out/r.INTENT).exists())

    def test_saved_intent_blocks_paid_duplicate(self):
        self.prepared();r.freeze(self.out/r.INTENT,{'attempt':'test'})
        with self.assertRaisesRegex(RuntimeError,'NO_AUTOMATIC_RETRY'):
            r.paid(self.out,'test','TEST_KEY','https://example.invalid',1000,True,True)

    def test_paid_one_call_receipt_replay_without_repeat(self):
        _,docs,_,_=self.prepared()
        model_reply=json.dumps({'candidates':[valid_candidate(docs,0),valid_candidate(docs,2)]})
        counter={'n':0}
        class FakeChat:
            def create(self,**kwargs):
                counter['n']+=1
                choice=types.SimpleNamespace(finish_reason='stop',message=types.SimpleNamespace(content=model_reply))
                return types.SimpleNamespace(choices=[choice],usage=None)
        class FakeOpenAI:
            def __init__(self,**kwargs):self.chat=types.SimpleNamespace(completions=FakeChat())
        with patch.dict(sys.modules,{'openai':types.SimpleNamespace(OpenAI=FakeOpenAI)}),patch.dict(os.environ,{'TEST_KEY':'test-only'}):
            result=r.paid(self.out,'TEST-MODEL','TEST_KEY','https://example.invalid',1000,True,True)
            self.assertEqual(result['new_paid_calls'],1)
            result2=r.paid(self.out,'TEST-MODEL','TEST_KEY','https://example.invalid',1000,True,True)
            self.assertEqual(result2['paid_calls'],0)
            self.assertEqual(counter['n'],1)
        self.assertEqual(r.validate(self.out)['validated_candidate_questions'],2)

    def test_failed_model_call_keeps_intent(self):
        self.prepared()
        class FakeFail:
            def __init__(self,**kwargs):self.chat=types.SimpleNamespace(completions=self)
            def create(self,**kwargs):raise RuntimeError('provider timeout')
        with patch.dict(sys.modules,{'openai':types.SimpleNamespace(OpenAI=FakeFail)}),patch.dict(os.environ,{'TEST_KEY':'test-only'}):
            with self.assertRaisesRegex(RuntimeError,'NO automatic second call'):
                r.paid(self.out,'TEST-MODEL','TEST_KEY','https://example.invalid',1000,True,True)
        self.assertTrue((self.out/r.INTENT).exists())
        self.assertFalse((self.out/r.RECEIPT).exists())

    def test_prior_science_data_remains_untouched(self):
        before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.m14.rglob('*.json')}
        self.prepared()
        after={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.m14.rglob('*.json')}
        self.assertEqual(before,after)

if __name__=='__main__':unittest.main()
