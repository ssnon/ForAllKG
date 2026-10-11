"""Offline and zero-provider-call M14 science-search protocol regression tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from pipeline_core.discovery.sers_novelty_search_m14 import (
    make_plan, frozen, unique_works, problem_prompt, parse_problems,
    candidate_prompt, parse_candidates, strategy)
from scripts.discovery.run_sers_novelty_search_m14 import (
    discover, run, physics_validated, _repair_invalid_json_escapes,
    load, freeze, digest)

class MockScience:
    def __init__(self):self.calls=[];self.searches=[]
    def lookup(self,provider,q):
        self.searches.append((provider,q))
        # Unique anchored title for each broad query. Metadata is NOT asserted as a gap.
        name=f'{provider} study of SERS optical field kinetics and dynamic mechanism: {q}'
        wid=('OA:' if provider=='openalex' else 'EPMC:')+str(len(self.searches))
        return {'provider':provider,'query':q,'works':[{'work_id':wid,'title':name,
             'abstract':'We measure spectral changes and report a conditional dependence on environment.',
             'doi':'','url':'https://example.invalid/synthetic','year':2026}], 'success':True}
    def call(self,model,prompt):
        self.calls.append((model,prompt))
        req=json.loads(prompt['user'])
        if 'seed_M12_prior_art_context' in req:raise AssertionError('M13 prompt should not be used')
        if 'required_output' in req and 'questions' in req['required_output']:
            works=req['works']
            q=[{'title':f'Distinct topic {i}',
                'scientific_question':f'Under what controlled conditions does SERS mechanism {i} differ?',
                'competing_explanations':['null physical mechanism','candidate physical mechanism'],
                'critical_unknown':'Which mechanism survives control?',
                'differential_observable':'Time dependent response under independent measurements',
                'domain_family':f'topic_{i}',
                'source_anchors':[{'work_id':works[i]['work_id'],'exact_quote':works[i]['title'][:60]}],
                'why_still_uncertain':'The source does not verify that this research question is unresolved.',
                'not_claimed_novel':True} for i in range(4)]
            return self.reply({'questions':q,'novelty_certified':False})
        if 'policy_by_lane' in req:
            qn=len(req['required_output']['candidates'][0]['prior_art_queries'])
            gen=req['generation'];policy=req['policy_by_lane'];problems=req['provisional_problem_cards']
            candidates=[]
            for i,lane in enumerate(['MECHANISM_SEARCH','OPEN_FRONTIER']):
                previous=req['prior_generations'][-1]['candidates'][i] if req['prior_generations'] else None
                if previous and policy[lane] in ('ESCAPE','SUSPEND'):
                    gap=problems[(i+1)%4]['problem_id'];family=f'escape_{i}_{gen}'
                else:
                    gap=previous['problem_id'] if previous else problems[i]['problem_id']
                    family=previous['family'] if previous else f'family_{i}'
                candidates.append({'lane':lane,'problem_id':gap,'title':f'Synthetic M14 idea {lane} g{gen}',
                 'claim_atom':'A specified mechanism generates a distinct conditional response',
                 'mechanism':'A system with coupled parameters generates a measurable stochastic response',
                 'family':family,'equations':['dx/dt = -k*x + drive'],
                 'strongest_null':'A known relaxation model with a hidden variable',
                 'null_equations':['dq/dt = -(q - V)/tau'],
                 'conditional_predictions':['The response changes when drive changes'],
                 'independent_observables':['Direct displacement measurement'],
                 'interventions':['Vary temperature holding drive fixed'],
                 'falsification':'Matched null reproduces all conditional predictions',
                 'known_physics_overlap':'Strong null relaxation and field weighting are known',
                 'prior_art_queries':[f'{lane} gap {gap} physical test distinct query number {j} mechanism' for j in range(qn)]})
            return self.reply({'candidates':candidates,'novelty_certified':False})
        if 'decision_rule' in req:
            statuses=req['statuses']
            return self.reply({'novelty_certified':False,'reviews':[
                {'candidate_id':c['candidate_id'],
                 'physical_status':'PHYSICALLY_INVALID' if c['lane']=='MECHANISM_SEARCH' else 'NON_IDENTIFIABLE',
                 'strongest_null_equations':['dx/dt = -x + V'],
                 'counterexample_or_derivation':'Markovian null reproduces history-dependent changes',
                 'discriminating_experiment_or_why_impossible':'Measure state independently',
                 'critical_flaw':'Predictions are compatible with hidden variables',
                 'recommended_action':'Escape or branch to nonidentical mechanism'}
                for c in req['candidates']]})
        if 'not_a_human_literature_review' in req:
            return self.reply({'novelty_certified':False,'reviews':[
                {'candidate_id':c['candidate_id'],'prior_art_status':'UNRESOLVED_TESTABLE',
                 'evidence':[], 'precise_known_component':'',
                 'potential_distinct_residue':'Unverified',
                 'claim_by_claim_reasoning':'Abstract evidence alone cannot decide novelty.',
                 'further_query':'More precise mechanism query',
                 'feedback_for_next_generation':'Seek independent observables and a new explanation'}
                for c in req['candidates']]})
        raise AssertionError('Unexpected M14 prompt: '+str(list(req)))
    @staticmethod
    def reply(payload):return {'text':json.dumps(payload),'finish_reason':'stop','usage':{'mock':True}}

class M14Tests(unittest.TestCase):
    def test_frozen_checksum_and_required_design(self):
        p=make_plan(generations=2,replicates=2,queries_per_candidate=3)
        frozen(p)
        p['design']['generations']=5
        with self.assertRaises(ValueError):frozen(p)
    def test_science_problem_is_not_novelty(self):
        f=MockScience();p=make_plan(generations=2,replicates=2)
        source=unique_works([f.lookup('openalex',x) for x in p['seed_queries']])
        raw=f.call('mock',problem_prompt(p,source))['text']
        parsed=parse_problems(raw,source)
        self.assertEqual(len(parsed['questions']),4)
        d=json.loads(raw);d['questions'][0]['source_anchors'][0]['exact_quote']='fabricated statement'
        with self.assertRaisesRegex(ValueError,'quote'):parse_problems(json.dumps(d),source)
        d=json.loads(raw);d['questions'][0]['not_claimed_novel']=False
        with self.assertRaises(ValueError):parse_problems(json.dumps(d),source)
    def test_policy_and_escape_hard_gate(self):
        prev={'reviews':[{'lane':'MECHANISM_SEARCH','family':'f','physical_status':'PHYSICALLY_INVALID',
                           'prior_art_status':'UNRESOLVED_TESTABLE','retrieval_missing':False},
                          {'lane':'OPEN_FRONTIER','family':'h','physical_status':'NON_IDENTIFIABLE',
                           'prior_art_status':'UNRESOLVED_TESTABLE','retrieval_missing':False}]}
        self.assertEqual(strategy(prev,[],True)['MECHANISM_SEARCH'],'ESCAPE')
        self.assertEqual(strategy(prev,[],True)['OPEN_FRONTIER'],'BRANCH')
        self.assertEqual(strategy(prev,[],False)['MECHANISM_SEARCH'],'REFINE')
        self.assertEqual(strategy(prev,['f','f'],True)['MECHANISM_SEARCH'],'ESCAPE')
    def test_no_false_prior_art_coverage(self):
        with self.assertRaises(ValueError):unique_works([{'success':False}])
    def test_safe_escape_repair_narrow(self):
        raw=r'{"reviews":[{"equation":"\\nu and \\tau"}]}'
        fixed,edits=_repair_invalid_json_escapes(raw)
        self.assertEqual(edits,[])
        self.assertEqual(fixed,raw)
        broken=r'{"formula":"\beta x + \xi"}'
        fixed,edits=_repair_invalid_json_escapes(broken)
        self.assertEqual(len(edits),1) # \b is valid but \x is not
        self.assertEqual(json.loads(fixed)['formula'], '\x08eta x + \\xi')
    def test_composite_status_audited(self):
        c=[{'candidate_id':'x'}]
        p={'novelty_certified':False,'reviews':[{'candidate_id':'x',
           'physical_status':'KNOWN_PHYSICS_PLAUSIBLE_BUT_NON_IDENTIFIABLE',
           'strongest_null_equations':['x = 1'],
           'counterexample_or_derivation':'hidden variable counterexample',
           'critical_flaw':'cannot identify mechanism','recommended_action':'measure state'}]}
        parsed,a=physics_validated(json.dumps(p),c)
        self.assertEqual(parsed['reviews'][0]['physical_status'],'NON_IDENTIFIABLE')
        self.assertEqual(len(a['exact_composite_status_mappings']),1)
    def test_full_mock_discovery_and_matched_run_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=make_plan(generations=2,replicates=2,queries_per_candidate=3)
            freeze(root/'M14_PLAN.json',plan)
            f=MockScience();x=discover(plan,root,'mock',f.call,f.lookup)
            self.assertEqual(x['problems'],4)
            self.assertEqual(x['new_model_calls'],1)
            first=run(plan,root,'mock','mock','mock',f.call,f.lookup,2)
            self.assertEqual(first['status'],'M14_CHECKPOINTED')
            self.assertEqual(first['new_model_calls'],2)
            before=len(f.calls)
            complete=run(plan,root,'mock','mock','mock',f.call,f.lookup,None)
            self.assertEqual(complete['status'],'M14_MATCHED_ARMS_COMPLETE_INDEPENDENT_BLIND_REVIEW_REQUIRED')
            self.assertEqual(complete['candidates'],16)
            self.assertEqual(complete['calls_by_arm'],{'ADAPTIVE':12,'FIXED':12})
            self.assertEqual(len(f.calls),25)
            sha_before={str(p):digest(p) for p in root.rglob('*_RAW_RECEIPT.json')}
            rerun=run(plan,root,'mock','mock','mock',f.call,f.lookup,0)
            self.assertEqual(rerun['new_model_calls_this_invocation'],0)
            self.assertEqual(len(f.calls),25)
            self.assertEqual(sha_before,{str(p):digest(p) for p in root.rglob('*_RAW_RECEIPT.json')})
            self.assertEqual(load(root/'M14_BLIND_CANDIDATES.json')['arm_assignment_exposed'],False)
            self.assertEqual(len(load(root/'M14_PRIVATE_ARM_KEY.json')),16)
            self.assertEqual(load(root/'r01/g02/ADAPTIVE/M14_GENERATION_PARSED.json')['policy']['MECHANISM_SEARCH'],'ESCAPE')
            self.assertEqual(load(root/'r01/g02/FIXED/M14_GENERATION_PARSED.json')['policy']['MECHANISM_SEARCH'],'REFINE')
    def test_unsupported_policy_not_truth(self):
        previous={'reviews':[{'lane':'MECHANISM_SEARCH','family':'x','physical_status':'NONSENSE','prior_art_status':'UNRESOLVED_TESTABLE'},
           {'lane':'OPEN_FRONTIER','family':'y','physical_status':'NON_IDENTIFIABLE','prior_art_status':'UNRESOLVED_TESTABLE'}]}
        with self.assertRaises(ValueError):strategy(previous,[],True)

if __name__=='__main__':unittest.main()
