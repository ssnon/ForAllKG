"""M14 evidence-grounded problem discovery and adaptive novelty exploration.

Research directions may be imaginative; novelty, truth and unexploredness are NEVER
certified by literature retrieval, LLMs or internal scores. No production mutations.
"""
from __future__ import annotations
import hashlib
import math
import json
import re
from typing import Any

SCHEMA = 'm14-evidence-driven-adaptive-novelty-v1'
ARMS = ('ADAPTIVE', 'FIXED')
LANES = ('MECHANISM_SEARCH', 'OPEN_FRONTIER')
ACTIONS = ('DISCOVER', 'REFINE', 'BRANCH', 'ESCAPE', 'SUSPEND')
PHYSICS = frozenset(('PHYSICALLY_INVALID','NON_IDENTIFIABLE','KNOWN_PHYSICS_PLAUSIBLE','UNRESOLVED_TESTABLE','UNRESOLVED_UNTESTABLE'))
PRIOR = frozenset(('PRIOR_ART_SUSPECTED','KNOWN_PHYSICS_PLAUSIBLE','UNRESOLVED_TESTABLE','UNRESOLVED_UNTESTABLE'))
DEFAULT_QUERIES = (
    'surface enhanced Raman spectroscopy unresolved molecular dynamics anomalies',
    'single molecule SERS conflicting polarization orientation mechanisms',
    'electrochemical SERS potential dependent Raman spectral dynamics controls',
    'time resolved SERS nonthermal vibrational excitation pump probe',
    'plasmonic nanogap SERS fluctuations hidden hot spot dynamics',
    'Raman vibrational energy transfer isotope selective mechanisms SERS',
    'SERS ion specific interfacial adsorption structural dynamics',
    'SERS optical field mechanical coupling independent measurements',
)
DEFAULT_FAILURES = (
 'A history-dependent Raman response can arise from Markovian RC or hidden state relaxation.',
 'A Raman polarization intensity change is not by itself molecular reorientation.',
 'Stokes/anti-Stokes correlation and vibrational pumping are established research areas.',
 'Observation matching after model fitting is not independent model identification.',
 'Absence of retrieved prior art, or an exact quote, cannot certify novelty.',
)

def canon(x:Any)->str:
    return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))

def sha(x:Any)->str:
    return hashlib.sha256(canon(x).encode('utf-8')).hexdigest()

def nonempty(v:Any)->bool:
    if isinstance(v,str):return bool(v.strip())
    if isinstance(v,list):return bool(v) and all(nonempty(x) for x in v)
    return False

def frozen(plan:dict)->dict:
    if not isinstance(plan,dict) or plan.get('schema')!=SCHEMA:
        raise ValueError('Wrong M14 schema')
    ident='m14:'+sha({k:v for k,v in plan.items() if k!='plan_id'})[:20]
    if plan.get('plan_id')!=ident:raise ValueError('Frozen M14 plan mutated')
    d=plan['design']
    if d['arms']!=list(ARMS) or d['lanes']!=list(LANES):raise ValueError('Comparison protocol changed')
    if not 2<=d['generations']<=24 or not 2<=d['replicates']<=12:raise ValueError('Bad experiment dimensions')
    if not 3<=d['queries_per_candidate']<=8:raise ValueError('Bad literature depth')
    if plan['authority']['novelty_certified'] is not False:raise ValueError('Novelty authority prohibited')
    return plan

def make_plan(*,generations:int=6,replicates:int=4,queries_per_candidate:int=5,source_commit:str='',queries:list[str]|None=None)->dict:
    q=list(queries) if queries is not None else list(DEFAULT_QUERIES)
    if len(q)<6 or len(q)>20 or any(not isinstance(x,str) or len(x.strip())<16 for x in q):
        raise ValueError('Need 6..20 specific seed queries')
    obj={'schema':SCHEMA,'domain':'SERS Au/Ag, broader mechanistic scientific possibilities',
         'source_branch':'newSchema_proxy','source_commit':source_commit,
         'seed_queries':q,'common_known_failure_patterns':list(DEFAULT_FAILURES),
         'design':{'generations':generations,'replicates':replicates,
             'arms':list(ARMS),'lanes':list(LANES),'queries_per_candidate':queries_per_candidate,
             'model_calls_per_paired_generation':6,
             'fixed_budget_model_calls':6*generations*replicates+1,
             'common_problem_discovery_calls':1,
             'common_retrieval_providers':['openalex','europepmc'],
             'matched_model_budget':True,
             'adaptive_policy_is_deterministic':True,
             'independent_blind_science_review_required':True},
         'authority':{'novelty_certified':False,'scientific_truth_certified':False,
             'metadata_is_not_fulltext':True,'search_miss_proves_nothing':True,
             'production_graph_mutations':False}}
    obj['plan_id']='m14:'+sha(obj)[:20]
    return frozen(obj)

def unique_works(receipts:list[dict],limit:int=40)->dict:
    by={};source=[]
    for r in receipts:
        if r.get('success') is not True:raise ValueError('Failed retrieval cannot indicate a gap')
        if r.get('provider') not in ('openalex','europepmc') or not r.get('query'):
            raise ValueError('Malformed literature receipt')
        source.append({'provider':r['provider'],'query':r['query'],'n':len(r['works'])})
        for item in r['works']:
            if not item.get('work_id') or not item.get('title'):continue
            doi=str(item.get('doi') or '').lower().replace('https://doi.org/','')
            key='doi:'+doi if doi else item['work_id']
            if key not in by:
                w={k:item.get(k) for k in ('work_id','title','abstract','doi','url','year')}
                w['abstract']=str(w.get('abstract') or '')[:1700]
                by[key]=w
    return {'works':list(by.values())[:limit], 'searches':source,
            'availability':'METADATA_AND_ABSTRACT_NOT_FULLTEXT',
            'gap_certified':False,'novelty_certified':False}

def problem_prompt(plan:dict,source:dict)->dict:
    frozen(plan)
    return {'system':(
       'You are a skeptical scientist exploring SERS. Propose candidate unanswered scientific questions, NOT proven gaps. '
       'For each question anchor an exact <=150-character literal substring from provided paper titles or abstracts. '
       'A source mentioning a topic is not evidence it is unresolved; make the uncertainty explicit. '
       'Seek multiple physically distinct question families, not simply reformulations. '
       'Do not claim novelty, full-text coverage or empirical evidence. Return strict JSON. '
       'Use plain ASCII math without LaTex backslashes.'),
     'user':canon({'known_failures':plan['common_known_failure_patterns'],
       'known_countermodel_probes':computational_null_probes(),
       'works':source['works'],
       'required_output':{'questions':[{'title':'','scientific_question':'',
         'competing_explanations':['',''], 'critical_unknown':'',
         'differential_observable':'','domain_family':'',
         'source_anchors':[{'work_id':'','exact_quote':''}],
         'why_still_uncertain':'','not_claimed_novel':True} for _ in range(4)],
         'novelty_certified':False},
       'requirements':'Exactly four distinct physical families; each question must reference at least one exact, valid title/abstract quote. A research question is provisional, not proven missing in literature.'})}

def parse_problems(raw:str,source:dict)->dict:
    d=json.loads(raw.strip())
    if d.get('novelty_certified') is not False:raise ValueError('Problem generator asserted novelty')
    rows=d.get('questions')
    if not isinstance(rows,list) or len(rows)!=4:raise ValueError('Expected four research questions')
    lookup={w['work_id']:w for w in source['works']}
    families=set();out=[]
    for row in rows:
        for k in ('title','scientific_question','critical_unknown','differential_observable','domain_family','why_still_uncertain'):
            if not nonempty(row.get(k)):raise ValueError('Incomplete scientific question: '+k)
        if row.get('not_claimed_novel') is not True:raise ValueError('Gap portrayed as certified')
        if not nonempty(row.get('competing_explanations')) or len(row['competing_explanations'])<2:
            raise ValueError('No competing explanations')
        anchors=row.get('source_anchors')
        if not isinstance(anchors,list) or not anchors:raise ValueError('No scholarly anchor')
        for a in anchors:
            paper=lookup.get(a.get('work_id'))
            quote=a.get('exact_quote')
            if not paper or not isinstance(quote,str) or not 5<=len(quote)<=150 or not any(quote in str(paper.get(k) or '') for k in ('title','abstract')):
                raise ValueError('Invented or unsupported exact literature quote')
        fam=row['domain_family'].strip().lower()
        if fam in families:raise ValueError('Problem families not distinct')
        families.add(fam)
        out.append({**row,'problem_id':'m14gap:'+sha({'title':row['title'],'question':row['scientific_question']})[:16],
                    'gap_confirmed':False,'novelty_certified':False})
    return {'questions':out,'novelty_certified':False,'evidence_level':'TITLE_ABSTRACT_ANCHORED_OPEN_QUESTION_CANDIDATES_NOT_VERIFIED_GAPS'}

def computational_null_probes()->dict:
    """Auditable mathematical counterexamples, NOT experimental science data."""
    tau=1.0;duration=1.0;hold=1.0
    history_difference=2*(1-math.exp(-duration/tau))*math.exp(-hold/tau)
    return {'status':'ILLUSTRATIVE_PHYSICALLY_PLAUSIBLE_COUNTERMODELS_NOT_EMPIRICAL_EVIDENCE',
       'novelty_certified':False,
       'examples':[
         {'name':'Markovian_RC_history',
          'equation':'dq/dt = -(q - V)/tau',
          'protocol':'From q=0, V=+1 or -1 for duration=1; then V=0 for hold=1; tau=1.',
          'absolute_trajectory_difference_at_start_of_second_pulse':history_difference,
          'implication':'History-dependent spectroscopic response can occur with entirely Markovian hidden charge.'},
         {'name':'Common_cause_covariance',
          'equation':'X = U + ex; Y = U + ey; Cov(ex,ey)=0, ex and ey independent of U',
          'assumed_Var_U':1.0,'predicted_Cov_XY':1.0,
          'implication':'Cross-band signal covariance alone does not establish direct intermode coupling.'}]}


def strategy(previous:dict|None,previous_family_history:list[str],is_adaptive:bool)->dict:
    if previous is None:return {'MECHANISM_SEARCH':'DISCOVER','OPEN_FRONTIER':'DISCOVER','basis':'COMMON_INIT'}
    if not is_adaptive:return {'MECHANISM_SEARCH':'REFINE','OPEN_FRONTIER':'BRANCH','basis':'FIXED_M13_STYLE'}
    reviews=previous['reviews']
    actions={}
    for lane in LANES:
        r=next(x for x in reviews if x['lane']==lane)
        ps=r['physical_status'];ls=r['prior_art_status']
        if ps not in PHYSICS or ls not in PRIOR:raise ValueError('Invalid critic statuses in policy')
        if r.get('retrieval_missing'):
            action='SUSPEND'
        elif ps=='PHYSICALLY_INVALID' or (ps=='KNOWN_PHYSICS_PLAUSIBLE' and ls in ('PRIOR_ART_SUSPECTED','KNOWN_PHYSICS_PLAUSIBLE')):
            action='ESCAPE'
        elif ps=='UNRESOLVED_UNTESTABLE' and ls=='UNRESOLVED_UNTESTABLE':
            action='SUSPEND'
        elif ps=='NON_IDENTIFIABLE':
            action='BRANCH'
        elif ls=='PRIOR_ART_SUSPECTED':
            action='ESCAPE'
        elif ps=='UNRESOLVED_TESTABLE' and ls=='UNRESOLVED_TESTABLE':
            action='REFINE'
        else:
            action='BRANCH'
        # A repeated family without independent supporting evidence should trigger escape.
        family=r.get('family','').strip().lower()
        if previous_family_history.count(family)>=2 and action in ('REFINE','BRANCH'):
            action='ESCAPE'
        actions[lane]=action
    actions['basis']='EXPLICIT_CRITIC_STATUSES_AND_FAMILY_STAGNATION_HEURISTIC_NOT_SCIENTIFIC_TRUTH'
    return actions

def candidate_prompt(plan:dict,problems:dict,rep:int,g:int,arm:str,history:list[dict])->dict:
    frozen(plan)
    if arm not in ARMS or not 1<=g<=plan['design']['generations']:raise ValueError('Invalid round')
    if len(history)!=g-1:raise ValueError('Incomplete previous history')
    prev=history[-1] if history else None
    families=[h['family'] for entry in history for h in entry.get('candidates',[])]
    previous_review=prev.get('candidate_reviews') if prev else None
    policy=strategy({'reviews':previous_review} if previous_review else None,families,arm=='ADAPTIVE')
    qn=plan['design']['queries_per_candidate']
    model_template=[{'lane':lane,'problem_id':'choose only from provided problems',
      'title':'','claim_atom':'','mechanism':'','family':'','equations':[''],
      'strongest_null':'','null_equations':[''],'conditional_predictions':[''],
      'independent_observables':[''],'interventions':[''],'falsification':'',
      'known_physics_overlap':'','prior_art_queries':[f'specific search query {i}' for i in range(qn)]}
      for lane in LANES]
    prior=[{'generation':h['generation'],'candidates':[{'lane':c['lane'],'problem_id':c['problem_id'],
             'family':c['family'],'claim_atom':c['claim_atom']} for c in h['candidates']],
             'critic_feedback':h['candidate_reviews']} for h in history]
    return {'system':(
       'You are an imaginative but physically precise SERS scientist. Novelty is a goal, never a certified result. '
       'Generate two mechanistically distinct ideas based on provisional research questions. '
       'Search policy is binding: REFINE a defensible claim, BRANCH into a different mechanism, '
       'ESCAPE from the former mechanism and choose another scientific problem, '
       'SUSPEND a poorly evidenced path and start elsewhere, DISCOVER openly. '
       'Do not claim a known effect as an original discovery. Output JSON only; write equations in plain ASCII without LaTex backslashes. '
       'All formulas, nulls, independent observables and falsifiers must be scientifically substantive.'),
      'user':canon({'domain':plan['domain'],'replicate':rep,'generation':g,
        'provisional_problem_cards':problems['questions'],
        'known_failure_patterns':plan['common_known_failure_patterns'],
        'known_countermodel_probes':computational_null_probes(),
        'policy_by_lane':{x:policy[x] for x in LANES},
        'prior_generations':prior,
        'required_output':{'candidates':model_template,'novelty_certified':False},
        'rules':f'Provide >= {qn} distinct specific literature queries per candidate; use ONLY listed problem IDs. '
              'For ESCAPE or SUSPEND, move away from the prior problem and family if alternate problems exist. '
              'Scientific difference cannot be asserted merely by renaming a variable or measurement.'})}

def parse_candidates(raw:str,plan:dict,problems:dict,rep:int,g:int,arm:str,history:list[dict])->dict:
    d=json.loads(raw.strip())
    if d.get('novelty_certified') is not False:raise ValueError('Generator asserted novelty')
    rows=d.get('candidates')
    if not isinstance(rows,list) or len(rows)!=2 or {x.get('lane') for x in rows if isinstance(x,dict)}!=set(LANES):
        raise ValueError('Need exactly two independent lanes')
    valid={x['problem_id'] for x in problems['questions']}
    old={x['lane']:x for x in history[-1]['candidates']} if history else {}
    p=strategy({'reviews':history[-1]['candidate_reviews']} if history else None,
        [c['family'] for h in history for c in h['candidates']],arm=='ADAPTIVE')
    qn=plan['design']['queries_per_candidate'];out=[]
    for c in rows:
        fields=('title','claim_atom','mechanism','family','equations','strongest_null','null_equations',
                'conditional_predictions','independent_observables','interventions','falsification','known_physics_overlap')
        if any(not nonempty(c.get(key)) for key in fields):raise ValueError('Missing scientific specification')
        if c.get('problem_id') not in valid:raise ValueError('Invented problem id')
        qs=c.get('prior_art_queries')
        if not isinstance(qs,list) or len(qs)<qn or any(not isinstance(q,str) or len(q.strip())<12 for q in qs[:qn]):
            raise ValueError('Insufficient specific prior-art queries')
        if len({q.strip().casefold() for q in qs[:qn]})<qn:raise ValueError('Duplicate prior-art queries')
        if any(any(ord(z)<32 for z in s) for s in (c['claim_atom'],*c['equations'],*c['null_equations'])):
            raise ValueError('Control characters suggest corrupted formula')
        prior=old.get(c['lane'])
        if prior and p[c['lane']] in ('ESCAPE','SUSPEND'):
            if prior['problem_id']==c['problem_id'] or prior['family'].strip().lower()==c['family'].strip().lower():
                raise ValueError('Search policy escape/suspend not obeyed: same problem or family')
        item={**c,'prior_art_queries':qs[:qn],
              'candidate_id':'m14:'+sha({'rep':rep,'generation':g,'arm':arm,'candidate':c})[:20],
              'requested_action':p[c['lane']]}
        out.append(item)
    if out[0]['family'].strip().lower()==out[1]['family'].strip().lower():raise ValueError('Two lanes repeated same mechanism family')
    return {'replicate':rep,'generation':g,'candidates':out,'policy':p,'novelty_certified':False}

def feedback(candidates:list[dict],phy:dict,lit:dict,evidence:dict)->list[dict]:
    p={x['candidate_id']:x for x in phy['reviews']};l={x['candidate_id']:x for x in lit['reviews']}
    rows=[]
    for c in candidates:
        a=p[c['candidate_id']];b=l[c['candidate_id']]
        rows.append({'candidate_id':c['candidate_id'],'lane':c['lane'],'family':c['family'],
          'problem_id':c['problem_id'],'physical_status':a['physical_status'],
          'prior_art_status':b['prior_art_status'],
          'physics_countermodel':a['counterexample_or_derivation'],
          'known_overlap':b.get('precise_known_component'),
          'potential_distinct_residue':b.get('potential_distinct_residue'),
          'next_direction':b.get('feedback_for_next_generation'),
          'retrieval_missing':not bool(evidence['work_ids']),
          'scientific_truth_certified':False})
    return rows

def blind_rows(rounds:list[dict],plan:dict)->tuple[list[dict],dict]:
    rows=[];key={}
    for r in rounds:
        for c in r['generation']['candidates']:
            blind='BLIND_'+sha({'plan':plan['plan_id'],'candidate':c['candidate_id']})[:16]
            if blind in key:raise ValueError('Colliding blind identities')
            key[blind]={'replicate':r['replicate'],'generation':r['number'],'arm':r['arm'],
                  'lane':c['lane'],'candidate_id':c['candidate_id'],'requested_action':c['requested_action']}
            rows.append({'blind_id':blind,**{k:c[k] for k in ('title','claim_atom','mechanism','family',
               'equations','strongest_null','null_equations','conditional_predictions',
               'independent_observables','interventions','falsification','known_physics_overlap')},
               'novelty_certified':False})
    return sorted(rows,key=lambda x:sha({'shuffle':plan['plan_id'],'id':x['blind_id']})),key
