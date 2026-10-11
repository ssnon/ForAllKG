import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from pipeline_core.discovery.sers_adaptive_policy_m14b import (
    SCHEMA, normalize, independent_evidence, choose, policy_for_next,
    validate_transition, proposed_next_step,
)
from scripts.discovery.run_sers_adaptive_policy_m14b import replay, science_rows, task

LANES=('MECHANISM_SEARCH','OPEN_FRONTIER')
PIDS=('gap_thermal','gap_vibrational','gap_adsorption','gap_orientation')


def candidate(lane, generation, *, family='SERS MIXED Case', problem='gap_thermal', id_=None):
    return {'candidate_id':id_ or f'{lane}-{generation}', 'lane':lane,
            'family':family,'problem_id':problem,'requested_action':'BRANCH'}


def make_history(n=1, statuses=None, family='SERS MIXED Case'):
    statuses=statuses or [('NON_IDENTIFIABLE','UNRESOLVED_TESTABLE')]*n
    h=[]
    for g in range(1,n+1):
        candidates=[candidate(l,g, family=(family if g%2 else 'sers mixed case') if l==LANES[0] else 'DIFFERENT physical family',
                              problem='gap_thermal' if l==LANES[0] else 'gap_vibrational') for l in LANES]
        reviews=[{'candidate_id':c['candidate_id'],'lane':c['lane'],'family':c['family'],
                  'physical_status':statuses[g-1][0], 'prior_art_status':statuses[g-1][1],
                  'retrieval_missing':False} for c in candidates]
        h.append({'generation':g,'replicate':1,'candidates':candidates,
                  'candidate_reviews':reviews,'novelty_certified':False})
    return h


def cards():
    return [{'problem_id':p, 'domain_family':'DIFFERENT '+p} for p in PIDS]


def signature(*, driver='interfacial carriers', assumption='carriers outrun heating',
              family='CARRIER_CHEMISTRY'):
    return {'mechanism_family_id':family,'causal_driver':driver,'core_assumption':assumption,
            'predicted_observable':'product branching','independent_intervention':'electrode barrier',
            'strongest_null':'spatial two temperature model',
            'falsification':'no residual effect under independently matched optical absorption'}


def research_candidate(problem='gap_thermal',family='Plasmonic',sig=None):
    return {'problem_id':problem,'family':family,'mechanism_signature':sig or signature(),
            'novelty_certified':False}


def decision(action):
    return {'schema':SCHEMA,'action':action, 'problem_expansion_required':False}


class TestScientificPolicy(unittest.TestCase):
    def test_normalize_nfkc_whitespace_case(self):
        self.assertEqual(normalize('  ＳＥＲＳ\tMixed  CASE '),'sers mixed case')

    def test_discover(self):
        self.assertEqual(choose([],LANES[0])['action'],'DISCOVER')

    def test_first_nonidentifiable_branches(self):
        self.assertEqual(choose(make_history(),LANES[0],problem_cards=cards())['action'],'BRANCH')

    def test_second_repeated_family_escalates_escape_casefold(self):
        dec=choose(make_history(2),LANES[0],problem_cards=cards())
        self.assertEqual(dec['action'],'ESCAPE')
        self.assertEqual(dec['normalized_family_visits'],2)

    def test_cross_lane_repetition_triggers_only_if_matching_family(self):
        h=make_history(1)
        self.assertEqual(choose(h,LANES[0],problem_cards=cards())['action'],'BRANCH')
        h[0]['candidates'][1]['family']='sErS mIxEd cAsE'
        self.assertEqual(choose(h,LANES[0],problem_cards=cards())['action'],'ESCAPE')

    def test_invalid_science_escape(self):
        h=make_history(statuses=[('PHYSICALLY_INVALID','UNRESOLVED_TESTABLE')])
        self.assertEqual(choose(h,LANES[0],problem_cards=cards())['action'],'ESCAPE')

    def test_prior_art_overlap_escape_not_certified(self):
        h=make_history(statuses=[('KNOWN_PHYSICS_PLAUSIBLE','PRIOR_ART_SUSPECTED')])
        d=choose(h,LANES[0],problem_cards=cards())
        self.assertEqual(d['action'],'ESCAPE')
        self.assertFalse(d['novelty_certified'])

    def test_unresolvable_unmeasurable_suspend(self):
        h=make_history(statuses=[('UNRESOLVED_UNTESTABLE','UNRESOLVED_UNTESTABLE')])
        self.assertEqual(choose(h,LANES[0],problem_cards=cards())['action'],'SUSPEND')

    def test_missing_retrieval_suspend(self):
        h=make_history();h[0]['candidate_reviews'][0]['retrieval_missing']=True
        self.assertEqual(choose(h,LANES[0],problem_cards=cards())['action'],'SUSPEND')

    def test_testable_promising_refine(self):
        h=make_history(statuses=[('UNRESOLVED_TESTABLE','UNRESOLVED_TESTABLE')])
        self.assertEqual(choose(h,LANES[0],problem_cards=cards())['action'],'REFINE')

    def test_no_alternative_problem_requires_discovery(self):
        h=make_history(statuses=[('PHYSICALLY_INVALID','UNRESOLVED_TESTABLE')])
        d=choose(h,LANES[0],problem_cards=[{'problem_id':'gap_thermal','domain_family':'SERS MIXED Case'}])
        self.assertEqual(d['action'],'SUSPEND')
        self.assertTrue(d['problem_expansion_required'])

    def test_evidence_whitelist(self):
        c='MECHANISM_SEARCH-1'
        complete={'candidate_id':c,'kind':'EXECUTED_NUMERICAL_MODEL','verified':True,
                  'independent_of_generation':True,'artifact_id':'sim:1','artifact_sha256':'a'*64}
        invalids=[{**complete,'kind':'LITERATURE_METADATA'},
                  {**complete,'verified':False},
                  {**complete,'independent_of_generation':False},
                  {**complete,'candidate_id':'DIFFERENT'},
                  {**complete,'artifact_sha256':'unverifiable'}]
        self.assertEqual(independent_evidence(invalids,candidate_id=c),[])
        self.assertEqual(independent_evidence([complete,*invalids],candidate_id=c),['sim:1'])

    def test_evidence_can_rescue_stagnant_refine_but_not_certify(self):
        h=make_history(2,statuses=[('NON_IDENTIFIABLE','UNRESOLVED_TESTABLE'),
                                   ('UNRESOLVED_TESTABLE','UNRESOLVED_TESTABLE')])
        c=h[-1]['candidates'][0]['candidate_id']
        e={'candidate_id':c,'kind':'INDEPENDENT_DATA_ANALYSIS','verified':True,
           'independent_of_generation':True,'artifact_id':'data:1','artifact_sha256':'a'*64}
        d=choose(h,LANES[0],evidence=[e],problem_cards=cards())
        self.assertEqual(d['action'],'REFINE')
        self.assertFalse(d['external_scientific_evidence_certified'])

    def test_no_future_leakage(self):
        h=make_history(2)
        h[1]['generation']=9
        with self.assertRaises(ValueError):choose(h,LANES[0])

    def test_reviewer_candidate_mismatch_rejected(self):
        h=make_history();h[0]['candidate_reviews'][0]['candidate_id']='WRONG'
        with self.assertRaises(ValueError):choose(h,LANES[0])

    def test_lane_order_does_not_change_policy(self):
        h=make_history(2)
        before=choose(h,LANES[0],problem_cards=cards())
        for x in h:
            x['candidates'].reverse()
            x['candidate_reviews'].reverse()
        after=choose(h,LANES[0],problem_cards=cards())
        self.assertEqual(before,after)

    def test_bad_status_rejected(self):
        h=make_history();h[0]['candidate_reviews'][0]['physical_status']='NOVELTY_CERTIFIED'
        with self.assertRaises(ValueError):choose(h,LANES[0])

    def test_policy_two_lanes_and_contract(self):
        h=make_history()
        p=policy_for_next(h,problem_cards=cards())
        self.assertEqual(set(p['lanes']),set(LANES))
        self.assertEqual(proposed_next_step(h,cards())['no_model_calls'],True)


class TestStructuralTransition(unittest.TestCase):
    def setUp(self):
        self.old=research_candidate()

    def test_refine_keeps_assumption(self):
        new=copy.deepcopy(self.old)
        new['mechanism_signature']['predicted_observable']='different quantitative isotope slope'
        self.assertTrue(validate_transition(self.old,new,decision('REFINE'))['accepted'])

    def test_refine_must_not_return_identical_proposal(self):
        with self.assertRaises(ValueError):validate_transition(self.old,copy.deepcopy(self.old),decision('REFINE'))

    def test_refine_cannot_rebrand_core(self):
        new=copy.deepcopy(self.old)
        new['mechanism_signature']['core_assumption']='not the same premise'
        with self.assertRaises(ValueError):validate_transition(self.old,new,decision('REFINE'))

    def test_branch_requires_mechanism_shift(self):
        with self.assertRaises(ValueError):validate_transition(self.old,copy.deepcopy(self.old),decision('BRANCH'))
        new=copy.deepcopy(self.old)
        new['mechanism_signature']['core_assumption']='a new causal premise'
        self.assertTrue(validate_transition(self.old,new,decision('BRANCH'))['accepted'])

    def test_escape_rejects_relabel_only(self):
        new=research_candidate(problem='gap_adsorption',family='Different family')
        with self.assertRaises(ValueError):validate_transition(self.old,new,decision('ESCAPE'))

    def test_escape_requires_problem_driver_assumption_family(self):
        new=research_candidate(problem='gap_adsorption',family='Adsorption',
             sig=signature(driver='site exchange',assumption='exchange controls SERS selection',family='ADSORPTION_EXCHANGE'))
        v=validate_transition(self.old,new,decision('ESCAPE'))
        self.assertFalse(v['scientific_semantic_identity_verified'])
        self.assertTrue(v['accepted'])

    def test_escape_cannot_invent_new_problem_card(self):
        new=research_candidate(problem='GAP_INVENTED',family='Adsorption',
             sig=signature(driver='site exchange',assumption='new premise',family='ADSORPTION_EXCHANGE'))
        d=decision('ESCAPE');d['escape_alternative_problem_ids']=['gap_adsorption']
        with self.assertRaises(ValueError):validate_transition(self.old,new,d)

    def test_escape_cannot_bypass_gap_expansion(self):
        new=research_candidate(problem='gap_adsorption',family='Adsorption',
             sig=signature(driver='site exchange',assumption='new premise',family='ADSORPTION_EXCHANGE'))
        d=decision('ESCAPE');d['problem_expansion_required']=True
        with self.assertRaises(ValueError):validate_transition(self.old,new,d)

    def test_suspend_has_no_successor(self):
        self.assertTrue(validate_transition(self.old,None,decision('SUSPEND'))['accepted'])
        with self.assertRaises(ValueError):validate_transition(self.old,research_candidate(),decision('SUSPEND'))

    def test_novelty_certification_blocked(self):
        new=research_candidate();new['novelty_certified']=True
        with self.assertRaises(ValueError):validate_transition(None,new,decision('DISCOVER'))

    def test_incomplete_signature_blocked(self):
        new=research_candidate();del new['mechanism_signature']['strongest_null']
        with self.assertRaises(ValueError):validate_transition(None,new,decision('DISCOVER'))


class TestReplay(unittest.TestCase):
    def fake_root(self,tmp):
        root=Path(tmp)/'source';root.mkdir()
        plan={'design':{'replicates':2,'generations':3}}
        problems={'novelty_certified':False,'questions':cards()}
        summary={'replicates':2,'generations':3}
        for name,d in [('M14_PLAN.json',plan),('M14_PROVISIONAL_PROBLEMS.json',problems),
                       ('M14_RUN_SUMMARY.json',summary)]:
            (root/name).write_text(json.dumps(d))
        for rep in (1,2):
            for arm in ('ADAPTIVE','FIXED'):
                h=make_history(3)
                for g,record in enumerate(h,1):
                    record['replicate']=rep
                    p=root/f'r{rep:02d}'/f'g{g:02d}'/arm
                    p.mkdir(parents=True)
                    (p/'M14_GENERATION_RESULT.json').write_text(json.dumps(record))
        return root

    def test_replay_offline_determinism(self):
        with tempfile.TemporaryDirectory() as td:
            src=self.fake_root(td);out=Path(td)/'out'
            replay(src,out)
            summary=json.loads((out/'M14B_POLICY_REPLAY_SUMMARY.json').read_text())
            self.assertEqual(summary['model_calls'],0)
            self.assertEqual(summary['frozen_history_decision_rows'],16)
            self.assertFalse(summary['semantic_novelty_certified'])
            with self.assertRaises(FileExistsError):replay(src,out)

    def test_replay_does_not_modify_science_input(self):
        with tempfile.TemporaryDirectory() as td:
            src=self.fake_root(td)
            before={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*.json')}
            replay(src,Path(td)/'audit')
            after={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*.json')}
            self.assertEqual(before,after)

    def test_archive_replay(self):
        with tempfile.TemporaryDirectory() as td:
            src=self.fake_root(td)
            zpath=Path(td)/'source.zip'
            with zipfile.ZipFile(zpath,'w') as z:
                for p in src.rglob('*.json'):z.write(p,p.relative_to(src))
            out=Path(td)/'out';replay(zpath,out)
            self.assertEqual(json.loads((out/'M14B_POLICY_REPLAY_SUMMARY.json').read_text())['model_calls'],0)

    def test_draft_contract_is_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            src=self.fake_root(td);out=Path(td)/'contract.json'
            task(src,out,1,2,'ADAPTIVE')
            self.assertEqual(json.loads(out.read_text())['requested_generation'],2)
            with self.assertRaises(FileExistsError):task(src,out,1,2,'ADAPTIVE')


if __name__ == '__main__':unittest.main()
