import copy
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from pipeline_core.discovery import sers_scientific_problem_discovery_m14c4d as c
from scripts.discovery import run_sers_scientific_problem_discovery_m14c4d as r
from scripts.discovery import recover_sers_scientific_problem_discovery_m14c4d as fix


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


def candidate(docs, i=0):
    return {'research_question':(['Can time-resolved isotope response distinguish transient surface water adsorption from electrochemical charge transfer mechanisms?', 'Can polarized laser-field reference measurements distinguish nanoscale enhancement from molecular alignment at gold nanogaps?'][0 if i==0 else 1]),
        'hypothesis':f'Adsorbed molecules in chemical regime {i} may show a physically different population response.',
        'strongest_existing_null':f'Laser enhancement and adsorption sampling variation in regime {i} may mimic chemical redistribution.',
        'discriminating_prediction':f'An independent isotope response and calibrated surface occupancy would change according to regime {i}.',
        'independent_measurement':f'Measure optical enhancement and surface coverage without analyte spectroscopy in regime {i}.',
        'falsification':f'Independent coverage calibration and exact null replication remove the predicted difference in regime {i}.',
        'evidence_anchors':[{'source_id':d['source_id'],'exact_quote':d['excerpt'][:95]} for d in docs[i*2:i*2+2]],
        'possible_prior_art_overlap':f'Existing reports on interfacial optical sampling for mechanism {i} may already explain this.',
        'why_this_is_not_just_a_known_problem':f'The conditional independent controls for mechanism {i} might connect observations but novelty remains unverified.',
        'next_action':'SEARCH_PRIOR_ART','new_scientific_discovery_certified':False}

class TestOfflineReviewedRecovery(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        self.home=Path(t.name);self.m14=self.home/'M14';(self.m14/'common_discovery').mkdir(parents=True)
        r.freeze(self.m14/'common_discovery/M14_SOURCE_SNAPSHOT.json',{'works':works()})
        r.freeze(self.m14/'M14_PROVISIONAL_PROBLEMS.json',{'questions':[{'scientific_question':'Historical question '+str(i)} for i in range(4)]})
        self.emp=self.home/'C4C.json'
        r.freeze(self.emp,{'fold_count':39,'protocols':[{'protocol':[f'cAg@{i}', 'cAg',str(532+i)],
            'n_evaluated_folds':6,'below_rows':10,'above_rows':20,
            'median_out_of_range_band_area_rmse_by_model':{'H0':0.8,'H1':0.5,'H2':0.55},
            'physical_mechanism_identified':False,'novelty_certified':False} for i in range(6)]})
        self.out=self.home/'out'
        r.prepare(self.out,self.m14,c4c_actions=self.emp,desired_candidates=4)
        self.plan,self.docs,self.briefs,self.prompt=r.context(self.out)
        self.pub=[x for x in self.docs if x['kind']=='published_abstract']
        self.empdoc=next(x for x in self.docs if x['kind']=='published_data_secondary_analysis')
    def receipt(self, candidates):
        r.freeze(self.out/r.RECEIPT,{'schema':c.SCHEMA, 'model':'TEST-ONLY','prompt_sha256':c.sha(c.canonical(self.prompt)),
             'finish_reason':'stop','raw':json.dumps({'candidates':candidates}),
             'new_scientific_discovery_certified':False})
    def bad_primary(self):
        ca=candidate(self.pub,0);cb=candidate(self.pub,2)
        ca['evidence_anchors'].append({'source_id':self.empdoc['source_id'],'exact_quote':
            'The published SERS data proved a new chemical mechanism in every laboratory.'})
        return [ca,cb]
    def approve(self):
        fix.prepare_review(self.out)
        review=r.load(self.out/fix.TEMPLATE)
        review['reviewed_scientific_dependence_on_each_omitted_secondary_anchor']=True
        review['authorized_optional_secondary_anchor_omission']=True
        review['reviewer_reason']='The omitted secondary source did not substantiate the hypothesized mechanism, and the two original papers remain cited exactly; no physical mechanism or scientific novelty is claimed.'
        r.freeze(self.out/fix.REVIEWED,review)
    def test_reviewer_projection_is_independently_replayable_and_immutable(self):
        self.receipt(self.bad_primary());before=r.digest(self.out/r.RECEIPT)
        with self.assertRaisesRegex(ValueError,'Unmatched exact'):r.validate(self.out)
        d=fix.inspect(self.out)
        self.assertEqual(d['status'],'M14C4D_OPTIONAL_SECONDARY_ANCHOR_REVIEW_POSSIBLE')
        self.assertEqual(len(d['proposed_optional_secondary_anchor_omissions']),1)
        self.assertEqual(len(d['blocking_reasons']),0)
        self.approve()
        output=fix.recover_reviewed(self.out,True)
        self.assertFalse(output['original_submission_validated'])
        self.assertEqual(output['candidate_count'],2)
        self.assertEqual(fix.verify_recovered(self.out)['status'],'M14C4D_REVIEWED_PROJECTION_REPLAY_VERIFIED')
        self.assertEqual(before,r.digest(self.out/r.RECEIPT))
        self.assertFalse((self.out/r.CANDIDATES).exists())
        self.assertEqual(r.verify(self.out)['status'],'M14C4D_PAID_RECEIPT_PRESENT_VALIDATION_PENDING')
        self.assertEqual(fix.recover_reviewed(self.out,True),output) # idempotent
    def test_scientific_review_required(self):
        self.receipt(self.bad_primary());fix.prepare_review(self.out)
        with self.assertRaises(FileNotFoundError):fix.recover_reviewed(self.out,True)
        with self.assertRaisesRegex(ValueError,'EXPLICIT_HUMAN'):fix.recover_reviewed(self.out,False)
    def test_no_two_primary_papers_blocks(self):
        ca,cb=self.bad_primary();ca['evidence_anchors'][1]={'source_id':self.empdoc['source_id'],'exact_quote':'Secondary unsupported explanatory quote'}
        self.receipt([ca,cb]);d=fix.inspect(self.out)
        self.assertIn('FEWER_THAN_TWO_VALID_INDEPENDENT_PUBLISHED_PAPERS',' '.join(d['blocking_reasons']))
        with self.assertRaisesRegex(ValueError,'REVIEW_TEMPLATE_BLOCKED'):fix.prepare_review(self.out)
    def test_unmatched_primary_blocks(self):
        ca,cb=self.bad_primary();ca['evidence_anchors'][0]['exact_quote']='Completely fabricated Raman quote in published paper.'
        self.receipt([ca,cb]);d=fix.inspect(self.out)
        self.assertIn('NOT_EXACTLY_IN_FROZEN_EXCERPT',' '.join(d['blocking_reasons']))
    def test_no_invalid_secondary_does_not_create_fake_recovery(self):
        ca,cb=self.bad_primary();ca['evidence_anchors'][-1]['exact_quote']=self.empdoc['excerpt'][:90]
        self.receipt([ca,cb]);self.assertEqual(r.validate(self.out)['validated_candidate_questions'],2)
        d=fix.inspect(self.out)
        self.assertIn('NO_ELIGIBLE_SECONDARY_EVIDENCE_ANCHOR_TO_REVIEW',d['blocking_reasons'])
    def test_other_semantic_validation_failure_blocks(self):
        ca,cb=self.bad_primary();cb['research_question']=ca['research_question']
        self.receipt([ca,cb]);d=fix.inspect(self.out)
        self.assertIn('FURTHER_ORIGINAL_VALIDATOR_FAILURE',' '.join(d['blocking_reasons']))
    def test_quote_over_250_can_be_reviewed_but_never_silently_changed(self):
        ca,cb=self.bad_primary();ca['evidence_anchors'][-1]['exact_quote']=self.empdoc['excerpt'][:280]
        self.receipt([ca,cb]);d=fix.inspect(self.out)
        self.assertEqual(d['proposed_optional_secondary_anchor_omissions'][0]['reason'],'QUOTE_TOO_LONG')
    def test_tampered_original_prompt_blocks_before_any_recovery(self):
        self.receipt(self.bad_primary())
        (self.out/r.PROMPT).write_text('{}')
        with self.assertRaisesRegex(ValueError,'REPLAY_MISMATCH'):fix.inspect(self.out)
    def test_review_file_cannot_change_source_omission(self):
        self.receipt(self.bad_primary());self.approve()
        rev=r.load(self.out/fix.REVIEWED);rev['proposed_omissions'][0]['source_id']='TEST:DIFFERENT'
        (self.out/fix.REVIEWED).write_bytes(c.canonical(rev))
        with self.assertRaisesRegex(ValueError,'REVIEW_FILE_MUST_MATCH'):fix.recover_reviewed(self.out,True)

if __name__=='__main__':unittest.main()
