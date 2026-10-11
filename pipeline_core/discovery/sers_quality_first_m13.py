"""M13 quality-first scientific feedback vs matched-budget no-feedback search.

This is an opt-in analysis overlay on the user's locally installed M12. Scientific
judgments are reviewer hypotheses, never evidence of true or original physics.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

ARMS = ('FEEDBACK', 'CONTROL')
LANES = ('REFINE_OR_BRANCH', 'OPEN_FRONTIER')
SCHEMA = 'm13-quality-first-matched-search-v1'
CLAIM_STATUSES = ('PRIOR_ART_SUSPECTED','KNOWN_PHYSICS_PLAUSIBLE','PHYSICALLY_INVALID',
                  'NON_IDENTIFIABLE','UNRESOLVED_TESTABLE','UNRESOLVED_UNTESTABLE')

def canon(x:Any)->str:
    return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sh(x:Any)->str:
    return hashlib.sha256(canon(x).encode()).hexdigest()
def file_sha(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p:Path)->Any:
    return json.loads(p.read_text(encoding='utf-8'))
def create(p:Path,v:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    t=v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)
    with p.open('x',encoding='utf-8') as f:f.write(t+'\n' if not t.endswith('\n') else t)
def zipjson(p:Path,leaf:str):
    with zipfile.ZipFile(p) as z:
        matches=[x for x in z.namelist() if x.rsplit('/',1)[-1]==leaf]
        if len(matches)!=1:raise ValueError('Missing ambiguous packet entry '+leaf)
        return json.loads(z.read(matches[0]))

def prepare(m12:Path,adjudication:Path,generations:int=4,queries_per_candidate:int=4,replicates:int=3)->dict:
    if not 2<=generations<=24:raise ValueError('Human-auditable experiment requires 2..24 generations')
    if not 2<=replicates<=12:raise ValueError('Replicates must be 2..12 for meaningful matched comparison')
    if not 3<=queries_per_candidate<=8:raise ValueError('Search-depth range 3..8, not optimized for cost')
    r=zipjson(m12,'M12_ALL_ROUNDS.json')
    review=zipjson(adjudication,'M12_SYSTEM_SCIENCE_CAPABILITY_VERDICT.json')
    if len(r.get('rounds',[]))!=2 or review.get('human_claim_novelty_certified')!=0:
        raise ValueError('Source M12/independent review not the expected frozen result')
    seeds=[{'title':c['title'],'novelty_atom':c['novelty_atom'],'differential_prediction':c['differential_prediction'],
            'strong_null':c['strong_null'],'model_equations':c['model_equations']}
            for run in r['rounds'] for c in run['generation']['candidates']]
    unresolved=[{'id':key,'disposition':value} for key,value in review['manual_review_dispositions'].items()]
    p={'schema':SCHEMA,'source':{'M12_SHA256':file_sha(m12),'adjudication_SHA256':file_sha(adjudication)},
       'domain':'SERS Au/Ag orientation and neighboring SERS mechanistic possibilities',
       'design':{'generations':generations,'replicates':replicates,'arms':list(ARMS),'lanes':list(LANES),
          'queries_per_candidate':queries_per_candidate,'generation_calls_per_arm':1,
          'physics_critic_calls_per_arm':1,'literature_critic_calls_per_arm':1,
          'comparative_call_budget':6*generations*replicates,
          'providers':['openalex','europepmc'],
          'actual_bibliography_per_candidate_max':queries_per_candidate*2*4,
          'generation_rule':'Complete BOTH arms with matched model and search protocol before advancing',
          'quality_first_not_minimum_cost':True,
          'science_novelty_certification':False,'production_mutations':False},
       'common_seed':seeds,
       'common_manual_verdicts_search_context_only':unresolved,
       'common_failure_patterns':[
          'A novel phrase with no matching title is not evidence of novel physics',
          'Known photon Stokes anti-Stokes correlations cannot be repackaged as newly observed phonon memory',
          'Double-layer hysteresis does not imply non-Markovian dynamics if hidden state is not conditioned',
          'Observed lag asymmetry need not uniquely establish adsorption cycling; account for detector delays',
          'Hidden hotspot mixture may mimic interaction even when individual sites have zero within-site interaction',
          'A proposed discriminating signal may be produced by a more flexible known-physics null',
          'A paper identified by DOI may not support a citation claim: verify specific text',
          'Post-hoc change of a claim creates a NEW claim, not a rescued success'],
       'authority':{'novelty_certified':False,'truth_verified':False,
                    'original_M12_ideas_not_promoted':True,'human_blind_review_required':True}}
    p['plan_id']='m13:'+sh(p)[:20]
    return p

def checked(p:dict)->dict:
    if p.get('schema')!=SCHEMA or p.get('plan_id')!='m13:'+sh({k:v for k,v in p.items() if k!='plan_id'})[:20]:
        raise ValueError('M13 plan checksum altered')
    d=p['design']
    if d['arms']!=list(ARMS) or d['lanes']!=list(LANES) or d['comparative_call_budget']!=d['generations']*6*d['replicates']:
        raise ValueError('Matched budget protocol was modified')
    if p['authority']['novelty_certified'] is not False:raise ValueError('Unauthorized novelty authority')
    return p

def generate_prompt(p:dict,arm:str,g:int,prior_feedback:list[dict],replicate:int=1)->dict:
    checked(p)
    if arm not in ARMS or not 1<=g<=p['design']['generations'] or not 1<=replicate<=p['design']['replicates']:
        raise ValueError('Invalid generation or arm')
    if arm=='CONTROL' and prior_feedback:raise ValueError('CONTROL may not consume previous critic feedback')
    if arm=='FEEDBACK' and len(prior_feedback)!=g-1:raise ValueError('Feedback chain length mismatch')
    system=('You are a scientific ResearchIdea creator, not a novelty certifier. Propose two mechanistically distinct SERS research ideas. '
            'Search very broadly but distinguish novel phrasing from novel physics. Do not fabricate data, citation, or novelty. '
            'Do not defend invalid previous claims; propose a new testable claim or abstain. '
            'Two lanes must be physically different. Output JSON only. You may use known physics in the null, not as a claimed discovery.')
    req={'domain':p['domain'],'generation':g,'replicate':replicate,'seed_M12_prior_art_context':p['common_seed'],
         'M12_prior_results_not_evidence':p['common_manual_verdicts_search_context_only'],
         'shared_known_failures':p['common_failure_patterns'],
         'independent_search_lanes':list(LANES),
         'feedback':prior_feedback if arm=='FEEDBACK' else [],
         'required_output':{'candidates':[{'lane':'REFINE_OR_BRANCH','title':'','claim_atom':'',
          'mechanism':'','equations':[''], 'strongest_null':'','null_equations':[''],
          'conditional_predictions':[''], 'independent_observables':[''],
          'interventions':[''], 'falsification':'','known_physics_overlap':'',
          'prior_art_queries':[f'distinct science query {i}' for i in range(1,p['design']['queries_per_candidate']+1)]},
          {'lane':'OPEN_FRONTIER','title':'','claim_atom':'','mechanism':'','equations':[''],
          'strongest_null':'','null_equations':[''], 'conditional_predictions':[''],
          'independent_observables':[''], 'interventions':[''], 'falsification':'',
          'known_physics_overlap':'','prior_art_queries':[f'distinct science query {i}' for i in range(1,p['design']['queries_per_candidate']+1)]}],
          'abstain_reason':None,'novelty_certified':False},
         'important':f'Provide AT LEAST {p["design"]["queries_per_candidate"]} distinct specific search queries per candidate; do NOT cherry-pick results. Model may abstain with empty candidates.'}
    # No treatment labels shown in prompt; only scientific feedback varies.
    return {'system':system,'user':canon(req)}

def _nonempty(v)->bool:
    if isinstance(v,str):return bool(v.strip())
    if isinstance(v,(list,dict)):return bool(v) and all(_nonempty(x) for x in (v.values() if isinstance(v,dict) else v))
    return False

def parse_generation(raw:str,g:int,p:dict,replicate:int=1,arm:str='FEEDBACK')->dict:
    o=json.loads(raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
    if not isinstance(o,dict) or o.get('novelty_certified') is not False:raise ValueError('Generation model claimed novelty')
    cs=o.get('candidates')
    if not isinstance(cs,list):raise ValueError('Invalid candidates')
    if not cs:
        if not _nonempty(o.get('abstain_reason')):raise ValueError('Incomplete abstention')
        return {'replicate':replicate,'generation':g,'candidates':[],'abstained_reason':o['abstain_reason'],'novelty_certified':False}
    if len(cs)!=2 or set(x.get('lane') for x in cs)!=set(LANES):raise ValueError('Wrong lane count')
    cleaned=[]
    for c in cs:
        fields=('title','claim_atom','mechanism','equations','strongest_null','null_equations',
                'conditional_predictions','independent_observables','interventions','falsification','known_physics_overlap')
        if any(not _nonempty(c.get(key)) for key in fields):raise ValueError('Scientific content missing')
        q=c.get('prior_art_queries')
        if not isinstance(q,list) or len(q)<p['design']['queries_per_candidate'] or any(not isinstance(x,str) or len(x.strip())<12 for x in q[:p['design']['queries_per_candidate']]):
            raise ValueError('Insufficient literature search queries')
        row={**c,'prior_art_queries':[x.strip() for x in q[:p['design']['queries_per_candidate']]]}
        row['candidate_id']='m13:'+sh({'replicate':replicate,'generation':g,'arm':arm,'lane':c['lane'],'science':row})[:20]
        cleaned.append(row)
    return {'replicate':replicate,'generation':g,'candidates':cleaned,'novelty_certified':False}

# --- M13 generator scientific-invalid receipt quarantine (separate from literature) ---
GENERATION_REQUIRED_FIELDS=(
    'title','claim_atom','mechanism','equations','strongest_null','null_equations',
    'conditional_predictions','independent_observables','interventions','falsification',
    'known_physics_overlap',
)

def _empty_subpaths(value, path, depth=0):
    if depth>5:return [path+':TOO_DEEP']
    if value is None:return [path+':NULL']
    if isinstance(value,str):return [] if value.strip() else [path+':EMPTY_STRING']
    if isinstance(value,(dict,list)):
        if not value:return [path+':EMPTY_CONTAINER']
        entries=value.items() if isinstance(value,dict) else enumerate(value)
        invalid=[]
        for key,child in entries:
            invalid.extend(_empty_subpaths(child,path+'.'+str(key),depth+1))
        return invalid
    return [path+':UNSUPPORTED_TYPE_'+type(value).__name__]

def diagnose_generation_receipt(raw:str,g:int,p:dict,replicate:int,arm:str)->dict:
    """Auditable field-level failure analysis; does not repair, promote or infer scientific fields."""
    issues=[];summaries=[]
    try:
        text=raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        obj=json.loads(text)
    except (ValueError,TypeError) as e:
        return {'status':'M13_GENERATION_RAW_JSON_INVALID','error_type':type(e).__name__,
                'novelty_certified':False,'raw_receipt_must_remain_unchanged':True,
                'replicate':replicate,'generation':g,'arm':arm}
    if not isinstance(obj,dict):issues.append('root:EXPECTED_OBJECT')
    elif obj.get('novelty_certified') is not False:
        issues.append('novelty_certified:NOT_EXPLICIT_FALSE')
    cs=obj.get('candidates') if isinstance(obj,dict) else None
    if not isinstance(cs,list):issues.append('candidates:EXPECTED_LIST');cs=[]
    if not cs:issues.append('candidates:EMPTY_OR_ABSTENTION')
    if len(cs)!=2:issues.append(f'candidates:COUNT_{len(cs)}_EXPECTED_2')
    seen=[]
    for i,c in enumerate(cs):
        if not isinstance(c,dict):
            issues.append(f'candidate[{i}]:NOT_OBJECT');continue
        lane=c.get('lane');seen.append(lane)
        item={'lane':lane,'title_present':bool(str(c.get('title') or '').strip()),
              'field_types':{k:type(c.get(k)).__name__ for k in GENERATION_REQUIRED_FIELDS},
              'invalid_paths':[]}
        for key in GENERATION_REQUIRED_FIELDS:
            item['invalid_paths']+=_empty_subpaths(c.get(key),'candidate['+str(i)+'].'+key)
        queries=c.get('prior_art_queries')
        if not isinstance(queries,list):item['invalid_paths'].append('prior_art_queries:EXPECTED_LIST')
        elif len(queries)<p['design']['queries_per_candidate']:
            item['invalid_paths'].append('prior_art_queries:INSUFFICIENT_COUNT')
        elif any(not isinstance(q,str) or len(q.strip())<12 for q in queries[:p['design']['queries_per_candidate']]):
            item['invalid_paths'].append('prior_art_queries:INVALID_QUERY_LENGTH')
        summaries.append(item)
        issues+=item['invalid_paths']
    if len(seen)!=len(set(seen)) or set(seen)!={'REFINE_OR_BRANCH','OPEN_FRONTIER'}:
        issues.append('lanes:EXPECTED_ONE_EACH_REFINE_OR_BRANCH_OPEN_FRONTIER')
    # A strict parser failure with no identified issue must not auto-pass.
    return {'status':'M13_GENERATION_SCIENCE_INVALID_UNPROMOTED',
        'replicate':replicate,'generation':g,'arm':arm,
        'issues':issues,'candidate_diagnostics':summaries,
        'raw_receipt_must_remain_unchanged':True,
        'replacement_is_new_generation_attempt_not_saved_original_success':True,
        'novelty_certified':False}

def replacement_generation_prompt(p:dict,arm:str,g:int,prior_feedback:list[dict],
                                  replicate:int,raw:str,defect:dict)->dict:
    """A new separately-paid generation, not an autofill of invalid scientific content."""
    original=generate_prompt(p,arm,g,prior_feedback,replicate)
    try:d=json.loads(raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
    except ValueError:d={}
    exclusion=[]
    if isinstance(d,dict):
        for c in d.get('candidates',[]) if isinstance(d.get('candidates'),list) else []:
            if isinstance(c,dict):exclusion.append({'lane':c.get('lane'),
                'original_title_not_to_reuse':str(c.get('title') or '')[:400],
                'original_claim_not_to_reuse':str(c.get('claim_atom') or '')[:750]})
    system=original['system']+' CRITICAL: the previous raw model response was scientifically incomplete and is INVALID. Make two genuinely new, independently testable scientific proposals; do NOT silently fill in the old claims or count them as successes. Only actual substantive equations, strong nulls, independent observables and risky falsification criteria are acceptable. You may ABSTAIN explicitly if no defensible idea. No claims of novelty certification.'
    user=json.loads(original['user'])
    user['previous_attempt_declared_invalid_not_evidence']={
        'invalid_science_fields':defect.get('issues',[]),
        'previous_claims_to_avoid_reusing':exclusion,
        'original_raw_receipt_sha256':hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        'scientific_success_of_original':False}
    return {'system':system,'user':canon(user)}

def verify_new_claims_not_literal_rescue(original_raw:str, regenerated:dict)->None:
    """Reject exact atom/title reuse; not a substitute for independent semantic review."""
    try:
        original=json.loads(original_raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
    except (ValueError,TypeError):return
    old={c.get('lane'):c for c in original.get('candidates',[]) if isinstance(c,dict)} if isinstance(original,dict) else {}
    for c in regenerated.get('candidates',[]):
        prior=old.get(c['lane'])
        if not prior:continue
        # A corrected null alone is not a new scientific idea when the atomic claim is unchanged.
        old_atom=' '.join(str(prior.get('claim_atom') or '').lower().split())
        new_atom=' '.join(str(c.get('claim_atom') or '').lower().split())
        if old_atom and old_atom==new_atom:
            raise ValueError('REPLACEMENT_REPEATS_INVALID_ORIGINAL_CLAIM_ATOM_'+str(c['lane']))
        old_title=' '.join(str(prior.get('title') or '').lower().split())
        new_title=' '.join(str(c.get('title') or '').lower().split())
        if old_title and old_title==new_title:
            raise ValueError('REPLACEMENT_REPEATS_INVALID_ORIGINAL_TITLE_'+str(c['lane']))


def compress_evidence(results:list[dict])->dict:
    bydoi={};queries=[]
    for r in results:
        if not r.get('success'):raise ValueError('Retrieval failure cannot be treated as negative result')
        queries.append({'provider':r['provider'],'query':r['query'],'work_count':len(r['works'])})
        for w in r['works']:
            doi=(w.get('doi') or '').strip().lower().replace('https://doi.org/','')
            key=('doi:'+doi) if doi else w['work_id']
            if key not in bydoi:
                w=dict(w)
                w['abstract']=str(w.get('abstract') or '')[:5000]
                bydoi[key]=w
    works=list(bydoi.values())
    return {'works':works,'work_ids':[w['work_id'] for w in works],
        'search_queries':queries,'reliability':'METADATA_AND_ABSTRACT_NOT_FULLTEXT_CLAIM_SUPPORT',
        'absence_proves_novelty':False}

def physics_prompt(candidates:list[dict])->dict:
    return {'system':('You are an adversarial applied physics reviewer. Attack TWO scientific SERS claims using explicit '
               'equations and countermodels. Construct the STRONGEST physically plausible null, not a straw man. '
               'Never use absence-of-evidence as evidence of novelty. A model can be nonidentifiable without being false. '
               'If proposed new mechanism reduces to known physics, document exactly. Output JSON only.'),
     'user':canon({'candidates':candidates,'required_output':{'reviews':[
        {'candidate_id':c['candidate_id'],'physical_status':'NON_IDENTIFIABLE',
         'strongest_null_equations':[''], 'counterexample_or_derivation':'',
         'discriminating_experiment_or_why_impossible':'','critical_flaw':'',
         'recommended_action':'REJECT_OR_NEW_IDEA'} for c in candidates], 'novelty_certified':False},
       'statuses':['PHYSICALLY_INVALID','NON_IDENTIFIABLE','KNOWN_PHYSICS_PLAUSIBLE','UNRESOLVED_TESTABLE','UNRESOLVED_UNTESTABLE'],
       'decision_rule':'A candidate is not promoted merely because equations are syntactically plausible.'})}

def literature_prompt(candidates:list[dict], evidence:dict)->dict:
    return {'system':('You are a prior-art analyst, not an invention certifier. Compare each smallest scientific claim '
              'with the actual retrieved titles and abstract text. For EVERY work used, cite work_id plus an exact '
              'verbatim <=150-character excerpt from its abstract or title. If no exact passage, do not cite as evidence. '
              'A search miss proves nothing. Correct DOI/abstract content matters more than matching keywords. '
              'Return JSON only.'),
        'user':canon({'candidates':candidates,'works':evidence['works'],
          'required_output':{'reviews':[{'candidate_id':c['candidate_id'],
              'prior_art_status':'UNRESOLVED_TESTABLE',
              'evidence':[{'work_id':'from works','exact_quote':'literal substring'}],
              'precise_known_component':'','potential_distinct_residue':'',
              'claim_by_claim_reasoning':'','further_query':'',
              'feedback_for_next_generation':''} for c in candidates], 'novelty_certified':False},
          'status_choices':['PRIOR_ART_SUSPECTED','KNOWN_PHYSICS_PLAUSIBLE','UNRESOLVED_TESTABLE','UNRESOLVED_UNTESTABLE'],
          'not_a_human_literature_review':True})}

def parse_physics(raw:str,candidates:list[dict])->dict:
    d=json.loads(raw)
    ids={c['candidate_id'] for c in candidates}
    if d.get('novelty_certified') is not False or not isinstance(d.get('reviews'),list) or {r.get('candidate_id') for r in d['reviews']}!=ids or len(d['reviews'])!=len(ids):
        raise ValueError('Physics critique identity invalid')
    for row in d['reviews']:
        if row.get('physical_status') not in ('PHYSICALLY_INVALID','NON_IDENTIFIABLE','KNOWN_PHYSICS_PLAUSIBLE','UNRESOLVED_TESTABLE','UNRESOLVED_UNTESTABLE'):
            raise ValueError('Physics status invalid')
        if any(not _nonempty(row.get(k)) for k in ('strongest_null_equations','counterexample_or_derivation','critical_flaw','recommended_action')):
            raise ValueError('Weak physical critic output')
    return {'reviews':d['reviews'],'novelty_certified':False,'independent_empirical_truth':False}

def parse_literature(raw:str,candidates:list[dict], evidence:dict)->dict:
    d=json.loads(raw)
    ids={c['candidate_id'] for c in candidates}
    if d.get('novelty_certified') is not False or not isinstance(d.get('reviews'),list) or len(d['reviews'])!=len(ids) or {r.get('candidate_id') for r in d['reviews']}!=ids:
        raise ValueError('Literature review identity invalid')
    lookup={w['work_id']:w for w in evidence['works']}
    for row in d['reviews']:
        if row.get('prior_art_status') not in ('PRIOR_ART_SUSPECTED','KNOWN_PHYSICS_PLAUSIBLE','UNRESOLVED_TESTABLE','UNRESOLVED_UNTESTABLE'):
            raise ValueError('Literature status invalid')
        ev=row.get('evidence',[])
        if not isinstance(ev,list):raise ValueError('Evidence not a list')
        if row['prior_art_status'] in ('PRIOR_ART_SUSPECTED','KNOWN_PHYSICS_PLAUSIBLE') and not ev:
            raise ValueError('Prior art asserted without anchored evidence')
        for cite in ev:
            w=lookup.get(cite.get('work_id'))
            if not w:raise ValueError('Fabricated work ID')
            quote=cite.get('exact_quote')
            if not isinstance(quote,str) or not 5<=len(quote)<=150 or not any(quote in str(w.get(key) or '') for key in ('title','abstract')):
                raise ValueError('Citation quote not in retrieved title/abstract: '+str(cite))
        if not _nonempty(row.get('claim_by_claim_reasoning')) or not _nonempty(row.get('feedback_for_next_generation')):
            raise ValueError('Literature review does not contain actionable reasoning')
    return {'reviews':d['reviews'],'novelty_certified':False,
        'claim_support_verified':'EXCERPT_SUBSTRING_ONLY_NOT_SCIENTIFIC_ENTAILMENT'}


def parse_literature_with_quarantine(raw:str,candidates:list[dict], evidence:dict)->tuple[dict,dict|None]:
    """Resume a paid raw receipt without promoting unanchored literature claims.

    Strict validator remains unchanged. Only evidence anchoring failures are
    quarantined, at *whole review row* granularity; scientific content is not
    repaired or inferred. The original raw receipt is never changed.
    """
    try:
        return parse_literature(raw,candidates,evidence),None
    except ValueError as exc:
        if not any(str(exc).startswith(prefix) for prefix in (
            'Citation quote not in retrieved title/abstract:',
            'Fabricated work ID',
            'Prior art asserted without anchored evidence',
        )):
            raise
    data=json.loads(raw)
    ids={c['candidate_id'] for c in candidates}
    if (not isinstance(data,dict) or data.get('novelty_certified') is not False
        or not isinstance(data.get('reviews'),list)
        or len(data['reviews'])!=len(ids)
        or {r.get('candidate_id') for r in data['reviews']}!=ids):
        raise ValueError('Cannot quarantine invalid literature review identity')
    works={w['work_id']:w for w in evidence['works']}
    allowed=('PRIOR_ART_SUSPECTED','KNOWN_PHYSICS_PLAUSIBLE',
             'UNRESOLVED_TESTABLE','UNRESOLVED_UNTESTABLE')
    repaired=[];issues=[]
    for orig in data['reviews']:
        if orig.get('prior_art_status') not in allowed:
            raise ValueError('Invalid source literature status, not recoverable')
        if any(not _nonempty(orig.get(k)) for k in (
                'claim_by_claim_reasoning','feedback_for_next_generation')):
            raise ValueError('Scientifically incomplete source, not recoverable')
        citations=orig.get('evidence',[])
        if not isinstance(citations,list):raise ValueError('Malformed evidence list')
        row_issues=[]
        for cite in citations:
            if not isinstance(cite,dict):raise ValueError('Malformed cited entry')
            wid=cite.get('work_id')
            w=works.get(wid)
            quote=cite.get('exact_quote')
            if w is None:
                problem='WORK_ID_NOT_IN_RETRIEVED_EVIDENCE'
            elif (not isinstance(quote,str) or not 5<=len(quote)<=150
                  or not any(quote in str(w.get(key) or '') for key in ('title','abstract'))):
                problem='UNANCHORED_QUOTE_NOT_VERBATIM'
            else:
                continue
            row_issues.append({'work_id':wid,'exact_quote':quote,
                               'problem':problem,'candidate_id':orig['candidate_id']})
        if (orig['prior_art_status'] in ('PRIOR_ART_SUSPECTED','KNOWN_PHYSICS_PLAUSIBLE')
             and not citations):
            row_issues.append({'candidate_id':orig['candidate_id'],
                               'problem':'ASSERTED_KNOWN_WITH_NO_EVIDENCE'})
        row=dict(orig)
        if row_issues:
            # Whole-row quarantine, even if one cited excerpt was valid:
            # status and prose may have depended on the bad work.
            row['prior_art_status']='UNRESOLVED_TESTABLE'
            row['evidence']=[]
            row['precise_known_component']='NOT_ESTABLISHED_BY_VERIFIED_CITATION'
            row['potential_distinct_residue']='UNVERIFIED_NOT_CERTIFIED'
            row['claim_by_claim_reasoning']=(
                'Original prior-art analysis not accepted: at least one citation '
                'was not grounded in retrieved title/abstract. See raw receipt and quarantine audit.')
            row['feedback_for_next_generation']=(
                'PRIOR_ART_UNVERIFIED: neither novelty nor knownness established. '
                'Recheck claim-level original papers and strongest known-physics null; '
                'do not reuse this unanchored assertion as established literature.')
            issues+=row_issues
        repaired.append(row)
    if not issues:
        raise ValueError('Strict literature validator failed for a non-quarantinable reason')
    sanitized={'reviews':repaired,'novelty_certified':False}
    valid=parse_literature(json.dumps(sanitized,ensure_ascii=False),candidates,evidence)
    audit={'status':'M13_UNANCHORED_CITATIONS_QUARANTINED_NO_PROVIDER_CALLS',
           'original_reviews_unmodified_in_raw_receipt':True,
           'evidence_issues':issues,
           'quarantined_candidate_ids':sorted({x['candidate_id'] for x in issues}),
           'novelty_certified':False,
           'note':'Whole-row downgrade prevents unsupported literature judgments from propagating as fact.'}
    valid['citation_quarantine_active']=True
    valid['claim_support_verified']='PARTIAL_EXCERPT_CHECK_ONLY_SOME_ROWS_UNSUPPORTED'
    return valid,audit


def science_feedback(gen:dict,phy:dict,lit:dict)->dict:
    p={r['candidate_id']:r for r in phy['reviews']};l={r['candidate_id']:r for r in lit['reviews']}
    return {'replicate':gen['replicate'],'generation':gen['generation'],'science_feedback_not_novelty_authority':True,
       'candidate_feedback':[{'previous_idea_id':c['candidate_id'],'previous_claim':c['claim_atom'],
           'physics_status':p[c['candidate_id']]['physical_status'],
           'physics_countermodel':p[c['candidate_id']]['counterexample_or_derivation'],
           'literature_status':l[c['candidate_id']]['prior_art_status'],
           'specific_known_overlap':l[c['candidate_id']].get('precise_known_component'),
           'potential_residue':l[c['candidate_id']].get('potential_distinct_residue'),
           'next_direction':l[c['candidate_id']]['feedback_for_next_generation'],
           'reviewers_are_llms_not_expert_confirmation':True} for c in gen['candidates']]}

def blinded_rows(rounds:list[dict],plan:dict)->list[dict]:
    # Stable IDs conceal arm and generation. CSV mapping deliberately NOT included in blinded packet.
    out=[]
    for run in rounds:
        for c in run['generation']['candidates']:
            out.append({'blind_id':'BLIND_'+sh({'plan':plan['plan_id'],'idea':c['candidate_id']})[:16],
              'title':c['title'],'claim_atom':c['claim_atom'],'mechanism':c['mechanism'],
              'equations':c['equations'],'strongest_null':c['strongest_null'],
              'null_equations':c['null_equations'],'conditional_predictions':c['conditional_predictions'],
              'independent_observables':c['independent_observables'], 'falsification':c['falsification'],
              'known_physics_overlap':c['known_physics_overlap']})
    return sorted(out,key=lambda row:sh({'k':plan['plan_id'],'id':row['blind_id']}))

def blind_csv(rows:list[dict])->str:
    columns=['blind_id','scientific_coherence_0_4','strong_null_discrimination_0_4',
       'independent_measurement_0_4','conceptual_change_0_4',
       'prior_art_distinctness_0_4','critical_flaw','independent_reviewer_disposition','reviewer_notes']
    s=io.StringIO();w=csv.DictWriter(s,fieldnames=columns);w.writeheader()
    for row in rows:w.writerow({'blind_id':row['blind_id']})
    return s.getvalue()

def score_completed_blind(csv_text:str,private_key:dict,plan:dict)->dict:
    """Only independently filled blinded CSV may be used for arm comparisons.

    Model's own 'physics_status' or 'prior_art_status' cannot become the quality label.
    """
    rows=list(csv.DictReader(io.StringIO(csv_text.lstrip('\ufeff'))))
    if len(rows)!=len(private_key) or len({r.get('blind_id') for r in rows})!=len(rows):
        raise ValueError('Missing/duplicate blind IDs; complete all blinded reviews')
    scoring=['scientific_coherence_0_4','strong_null_discrimination_0_4',
       'independent_measurement_0_4','conceptual_change_0_4','prior_art_distinctness_0_4']
    allowed={'REJECTED','NON_IDENTIFIABLE','KNOWN_PRIOR_ART','RESEARCH_WORTHY',
            'SUBSTANTIVE_NOVELTY_CANDIDATE','UNRESOLVED'}
    indexed={}
    scores=[]
    for r in rows:
        k=r['blind_id']
        if k not in private_key:raise ValueError('Unknown blind review ID')
        if r.get('independent_reviewer_disposition') not in allowed:
            raise ValueError('Blank or unsupported independent reviewer disposition')
        if not str(r.get('reviewer_notes') or '').strip():
            raise ValueError('Each review must explain scientific basis')
        vals={}
        for field in scoring:
            try:v=int(r[field])
            except (ValueError,TypeError) as e:raise ValueError('Score missing '+field) from e
            if not 0<=v<=4:raise ValueError('Invalid 0-4 score '+field)
            vals[field]=v
        info=private_key[k]
        item={'blind_id':k,**info,**vals,
            'disposition':r['independent_reviewer_disposition'],
            'quality_total_0_20':sum(vals.values()),
            'scientifically_followup_eligible':(
                r['independent_reviewer_disposition'] in ('RESEARCH_WORTHY','SUBSTANTIVE_NOVELTY_CANDIDATE')
                and vals['scientific_coherence_0_4']>=3
                and vals['strong_null_discrimination_0_4']>=3
                and vals['independent_measurement_0_4']>=2)}
        scores.append(item)
        pair=(info['replicate'],info['generation'],info['arm'])
        indexed.setdefault(pair,[]).append(item)
    pairs=[]
    for rep in range(1,plan['design']['replicates']+1):
        for gen in range(1,plan['design']['generations']+1):
            try:a=indexed[(rep,gen,'FEEDBACK')];b=indexed[(rep,gen,'CONTROL')]
            except KeyError as e:raise ValueError('Incomplete paired round review') from e
            if len(a)!=2 or len(b)!=2:raise ValueError('Each arm should produce exactly 2 lanes')
            mean_a=sum(x['quality_total_0_20'] for x in a)/2
            mean_b=sum(x['quality_total_0_20'] for x in b)/2
            eligible_a=sum(x['scientifically_followup_eligible'] for x in a)
            eligible_b=sum(x['scientifically_followup_eligible'] for x in b)
            pairs.append({'replicate':rep,'generation':gen,
              'feedback_mean_quality_0_20':mean_a,'control_mean_quality_0_20':mean_b,
              'quality_delta_feedback_minus_control':mean_a-mean_b,
              'feedback_eligible':eligible_a,'control_eligible':eligible_b,
              'eligible_delta':eligible_a-eligible_b})
    per_rep=[]
    for rep in range(1,plan['design']['replicates']+1):
        r=[x for x in pairs if x['replicate']==rep]
        per_rep.append({'replicate':rep,'mean_quality_delta':sum(x['quality_delta_feedback_minus_control'] for x in r)/len(r),
                        'mean_eligible_delta':sum(x['eligible_delta'] for x in r)/len(r)})
    return {'status':'M13_BLIND_SCORE_ANALYSIS_NOT_NOVELTY_CERTIFICATION',
      'novelty_certified':False,'feedback_improvement_established':False,
      'paired_rounds':pairs,'per_replicate':per_rep,
      'aggregate_mean_paired_quality_delta':sum(x['mean_quality_delta'] for x in per_rep)/len(per_rep),
      'aggregate_mean_eligible_delta':sum(x['mean_eligible_delta'] for x in per_rep)/len(per_rep),
      'data_quality':'All pre-specified independent blind judgments supplied',
      'interpretation_warning':'Descriptive matched comparisons; distinguish positive mean from robust improvement and full-text novelty verification.'}
