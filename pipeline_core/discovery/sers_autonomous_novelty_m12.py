"""M12 SERS autonomy: frozen provenance, live bibliographic challenges, bounded recursive search.

ResearchIdea imagination is unconstrained by grounding; scientific claims are not.
No LLM or bibliographic result carries authority to certify novelty or physical truth.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable
from zipfile import ZipFile

SCHEMA = 'm12-autonomous-novelty-search-v1'
LANES = ('REPLACE_OR_BRANCH', 'OPEN_SERS_FRONTIER')
STATUSES = frozenset(('DIRECT_PRIOR_ART_SUSPECTED', 'PARTIAL_PRIOR_ART',
 'KNOWN_PHYSICS_REEXPRESSION', 'INVALID_PREDICTION', 'NON_IDENTIFIABLE',
 'UNRESOLVED_TESTABLE', 'UNRESOLVED_NOT_TESTABLE'))


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str)


def sha(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode('utf-8')).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def new(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(value + ('' if value.endswith('\n') else '\n'))


def in_zip(path: Path, name: str) -> Any:
    with ZipFile(path) as z:
        names = [n for n in z.namelist() if n.rsplit('/', 1)[-1] == name]
        if len(names) != 1:
            raise ValueError('Missing or ambiguous frozen source: ' + name)
        return json.loads(z.read(names[0]))


def prepare(m11_packet: Path, m11_review: Path) -> dict:
    source = in_zip(m11_packet, 'M11_NEW_RESEARCH_IDEAS_UNREVIEWED.json')
    verdict = in_zip(m11_review, 'M11_REVIEW_DECISION.json')
    assert source['novelty_certified'] is False and verdict['novelty_certified'] is False
    assert source['source_experiment_id']
    seed = []
    for idea in source['new_ideas']:
        if not idea.get('task_id') or not idea.get('mechanistic_model'):
            raise ValueError('Source idea missing essential identity or mechanism')
        seed.append({'task_id': idea['task_id'], 'idea_title':idea.get('idea_title'),
          'conceptual_change':idea.get('conceptual_change'),
          'differential_prediction':idea.get('differential_prediction'),
          'strongest_competing_null':idea.get('strongest_competing_null')})
    if len(seed) != 2 or len(verdict.get('results', [])) != 2:
        raise ValueError('Frozen M11 source must contain exactly two reviewed generated ideas')
    review = {r['task_id']: {'disposition': r['disposition'],
              'scores':r['scores']} for r in verdict['results']}
    plan = {
        'schema_version':SCHEMA,
        'research_task':'SERS Au/Ag molecular orientation, dynamic band response and distinct testable adjacent SERS physics',
        'source':{'m11_packet_sha256':file_sha(m11_packet), 'm11_review_sha256':file_sha(m11_review),
                  'm11_experiment_id':source['source_experiment_id']},
        'seed_search_only':seed,
        'seed_independent_review':review,
        'known_invalid_claims':[ 
          'Polarization waveform reversal does not universally reverse dissipative hysteresis sign',
          'Rank-2 orientational correlation is not generally governed by rank-1 rotational diffusion rate',
          'Raw probability or Raman-band difference-in-differences does not identify logit interaction',
          'Constant V*x mixed coefficient does not flip sign when V reverses',
          'Hidden hotspot mixtures can generate nonzero observed field-state interactions without microscopic coupling',
          'No-article-found does NOT entail novel physics',
        ],
        'run_policy':{'generations':2,'lanes_per_generation':list(LANES),
           'model_calls_per_generation':2,'max_model_calls_total':4,
           'max_search_queries_per_candidate':2,'max_results_per_query_per_provider':4,
           'bibliography_providers':['openalex','europepmc'],
           'production_graph_mutated':False,'official_sis_scheduler_executed':False,
           'novelty_certified':False,'scientific_truth_authority':False,
           'prior_art_absence_is_not_evidence':True},
    }
    plan['plan_id']='m12:'+sha(plan)[:20]
    return plan


def checked_plan(obj: dict) -> dict:
    if obj.get('schema_version') != SCHEMA or obj.get('plan_id') != 'm12:'+sha({k:v for k,v in obj.items() if k!='plan_id'})[:20]:
        raise ValueError('Invalid or altered M12 plan')
    p=obj['run_policy']
    if (p.get('generations'),p.get('max_model_calls_total'),p.get('lanes_per_generation')) != (2,4,list(LANES)):
        raise ValueError('M12 model call budget/lanes drift')
    if p.get('novelty_certified') is not False or p.get('production_graph_mutated') is not False:
        raise ValueError('Unauthorized claim or production mutation')
    return obj


def generation_prompt(plan:dict, generation:int, history:list[dict]) -> dict:
    checked_plan(plan)
    if generation not in (1,2) or len(history) != generation-1:
        raise ValueError('Generation does not match verified feedback history')
    system = '''You are an adversarial SERS ResearchIdea generator. Produce speculative scientific hypotheses, NOT validated discoveries. Do not rescue an invalid claim by adding post-hoc conditions: any revised idea is a NEW claim. For each generation propose two separate lanes: REPLACE_OR_BRANCH based on previous scientific failures, and OPEN_SERS_FRONTIER independent of previous hypothesis families. You may abstain. Search aggressively but never fabricate empirical data, references, novelty certification, or mathematical deductions. A clear risky differential prediction against a strong physical null and an independently measurable quantity matter more than word novelty. Return one JSON object and nothing else.'''
    payload={
      'generation':generation,'task':'SERS-only autonomous exploratory search',
      'parent_M11_seed_not_evidence':plan['seed_search_only'] if generation==1 else [],
      'M11_review_search_feedback_not_evidence':plan['seed_independent_review'] if generation==1 else {},
      'known_invalid_claims':plan['known_invalid_claims'],
      'previous_round_scientific_search_feedback_only':history,
      'lanes':list(LANES),
      'strict_output':{'generation':generation,'candidates':[
          {'lane':LANES[0],'title':'','novelty_atom':'','mechanistic_model':'',
           'model_equations':[''], 'strong_null':'', 'differential_prediction':'',
           'independent_measurement':'','falsification_condition':'',
           'prior_art_queries':['',''],'known_physics_risk':''},
          {'lane':LANES[1],'title':'','novelty_atom':'','mechanistic_model':'',
           'model_equations':[''],'strong_null':'','differential_prediction':'',
           'independent_measurement':'','falsification_condition':'',
           'prior_art_queries':['',''],'known_physics_risk':''}],
           'abstention_reason':None,'novelty_certified':False},
      'instructions':['Exactly one nonduplicative candidate per lane; or ABSTAIN with empty candidates and explanation.',
         'Do not merely restate supplied counterexamples as proposed discoveries.',
         'Queries should contain distinctive scientific terms rather than entire thesis text.',
         'Each candidate must have a clearly stated hard null and an intervention with distinct outcomes.',
         'A retrieval gap is not proof of novelty.'],
    }
    return {'system':system,'user':canonical(payload)}


def _txt(value:Any) -> str:
    if isinstance(value,str):return value.strip()
    if isinstance(value,(dict,list)) and value:return canonical(value)
    return ''


def parse_generation(raw:str,g:int)->dict:
    d=json.loads(_unfence(raw))
    if not isinstance(d,dict) or d.get('generation') != g or d.get('novelty_certified') is not False:
        raise ValueError('Generation mismatch or forbidden novelty certification')
    cs=d.get('candidates')
    if not isinstance(cs,list) or len(cs)>2:raise ValueError('Wrong number of candidates')
    if not cs:
        if not _txt(d.get('abstention_reason')):raise ValueError('Abstention requires explanation')
        return {'generation':g,'candidates':[],'abstention_reason':d['abstention_reason'],'novelty_certified':False}
    if len(cs)!=2 or {c.get('lane') for c in cs if isinstance(c,dict)}!=set(LANES):
        raise ValueError('Must keep both distinct lanes in an active generation')
    fixed=[]
    for c in cs:
        keys=('title','novelty_atom','mechanistic_model','model_equations','strong_null','differential_prediction',
          'independent_measurement','falsification_condition','known_physics_risk')
        if any(not _txt(c.get(k)) for k in keys):raise ValueError('Missing scientific content in '+str(c.get('lane')))
        queries=c.get('prior_art_queries')
        if not isinstance(queries,list) or len(queries)<2 or any(not isinstance(q,str) or len(q.strip())<12 for q in queries[:2]):
            raise ValueError('At least two actual prior-art queries needed')
        c=dict(c);c['prior_art_queries']=[q.strip() for q in queries[:2]]
        c['candidate_id']='m12_idea:'+sha({'gen':g,'lane':c['lane'],'candidate':c})[:20]
        fixed.append(c)
    return {'generation':g,'candidates':fixed,'novelty_certified':False}


def _unfence(raw:str)->str:
    s=raw.strip()
    if s.startswith('```'):
        s='\n'.join(s.splitlines()[1:])
        if s.rstrip().endswith('```'):s=s.rstrip()[:-3]
    return s.strip()


def critic_prompt(generation:dict, works:dict)->dict:
    system='''You are a skeptical scientific challenger, NOT a novelty authority. Review 2 speculative SERS claims using the attached independently retrieved bibliographic metadata/abstracts and physical null reasoning. Cite ONLY work_ids actually in the supplied lookup. Titles/abstracts do not establish full paper identity and absent matches do not establish novelty. You may mark claims as preliminarily suspected direct/partial prior art; these require full text human confirmation. Reject mathematically invalid discriminators without rescuing the original claim. Every review produces SEARCH FEEDBACK ONLY, not truth and not novel claim certification. Return JSON only.'''
    payload={'candidates':generation['candidates'],
       'retrieved_bibliographic_evidence_search_context_only':works,
       'allowed_statuses':sorted(STATUSES),
       'required_output':{'reviews':[
          {'candidate_id':'use exact input ID','status':'UNRESOLVED_TESTABLE',
           'cited_work_ids':['from retrieval only'],
           'physical_null_counterexample':'','scientific_reason':'',
           'what_is_already_known':'','unresolved_conditional_prediction':'',
           'next_generation_instruction':''}],
        'novelty_certified':False},
       'rules':['One review per candidate, include explicit strongest competitor.',
         'No article found means insufficient search, not novelty.',
         'Direct prior art requires a retrieved paper, and should remain SUSPECTED until full-text comparison.',
         'The OPEN_SERS_FRONTIER branch must remain free to propose unrelated SERS physics next round.']}
    return {'system':system,'user':canonical(payload)}


def parse_critic(raw:str,candidates:list[dict],works:dict)->dict:
    d=json.loads(_unfence(raw))
    ids={c['candidate_id'] for c in candidates}
    rows=d.get('reviews') if isinstance(d,dict) else None
    if not isinstance(rows,list) or len(rows)!=len(ids) or d.get('novelty_certified') is not False:
        raise ValueError('Critic review mismatch or novelty escalation')
    seen=set()
    for row in rows:
        cid=row.get('candidate_id');status=row.get('status')
        if cid not in ids or cid in seen or status not in STATUSES:raise ValueError('Critic ID/status mismatch')
        seen.add(cid)
        cited=row.get('cited_work_ids',[])
        if not isinstance(cited,list) or any(w not in works['work_ids'] for w in cited):
            raise ValueError('Invented citation/work ID rejected')
        if status in ('DIRECT_PRIOR_ART_SUSPECTED','PARTIAL_PRIOR_ART') and not cited:
            raise ValueError('Prior art disposition without retrieved candidate work')
        for key in ('scientific_reason','physical_null_counterexample','next_generation_instruction'):
            if not _txt(row.get(key)):raise ValueError('Critic missing concrete reasoning')
    return {'reviews':rows,'novelty_certified':False,
            'all_dispositions_model_assisted_unverified':True}


def _http_get(url:str, *, timeout:float=20.0)->dict:
    req=urllib.request.Request(url, headers={'User-Agent':'ForAllKG-M12-Research/1.0 (scholarly literature metadata; non-commercial)','Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as f:
        return json.load(f)


def _oa_abstract(inv: Any)->str:
    if not isinstance(inv,dict):return ''
    pairs=[]
    for word,positions in inv.items():
        if isinstance(positions,list):
            for pos in positions:
                if isinstance(pos,int) and pos>=0 and pos<600:pairs.append((pos,word))
    return ' '.join(w for _,w in sorted(pairs))[:1700]


def lookup(provider:str,query:str,*,get:Callable[...,dict]=_http_get,key:str='')->dict:
    if provider=='openalex':
        params={'search':query,'per_page':4}
        if key:params['api_key']=key
        url='https://api.openalex.org/works?'+urllib.parse.urlencode(params)
        raw=get(url)
        result=[]
        for x in raw.get('results',[]):
            workid=str(x.get('id') or '')
            if not workid.startswith('https://openalex.org/'):continue
            result.append({'work_id':'OA:'+workid.rsplit('/',1)[-1],
              'title':x.get('title') or '','year':x.get('publication_year'),
              'doi':x.get('doi') or '', 'url':workid,
              'abstract':_oa_abstract(x.get('abstract_inverted_index'))})
        return {'provider':provider,'query':query,'works':result, 'success':True}
    if provider=='europepmc':
        params={'query':query,'format':'json','resultType':'core','pageSize':'4'}
        url='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urllib.parse.urlencode(params)
        raw=get(url);result=[]
        for x in raw.get('resultList',{}).get('result',[]):
            eid=str(x.get('id') or '');source=str(x.get('source') or '')
            if not eid or not source:continue
            abstract=re.sub('<[^>]*>',' ',str(x.get('abstractText') or ''))
            result.append({'work_id':'EPMC:'+source+':'+eid,'title':x.get('title') or '',
               'year':x.get('pubYear'),'doi':x.get('doi') or '',
               'url':f'https://europepmc.org/article/{source}/{eid}',
               'abstract':re.sub(r'\s+',' ',abstract).strip()[:1700]})
        return {'provider':provider,'query':query,'works':result,'success':True}
    raise ValueError('Unknown provider')


def unique_works(results:list[dict])->dict:
    bykey={};source=[]
    for r in results:
        source.append({'provider':r['provider'],'query':r['query'],'count':len(r['works']),
                       'success':r['success'],'error':r.get('error')})
        for w in r['works']:
            key=('doi:'+w['doi'].lower()) if w.get('doi') else w['work_id']
            if key not in bykey:bykey[key]=w
    # The reviewer can only cite exact included IDs. Avoid letting synonym work IDs bypass verification.
    works=list(bykey.values())
    return {'works':works,'work_ids':[w['work_id'] for w in works],
       'query_receipts':source,'retrieval_failed':any(not r['success'] for r in results),
       'retrieved_count':len(works),'novelty_certified':False}


def feedback(previous_generation:dict, criticism:dict,search:dict)->dict:
    review={r['candidate_id']:r for r in criticism['reviews']}
    return {'generation':previous_generation['generation'],
       'novelty_certified':False,'science_search_feedback_only':True,
       'bibliographic_coverage_note':'Bibliographic metadata and available abstracts only; full-text identity is not certified.',
       'candidates':[
         {'lane':c['lane'],'prior_idea_id':c['candidate_id'],'original_atomic_hypothesis':c['novelty_atom'],
          'original_prediction':c['differential_prediction'],
          'review_status':review[c['candidate_id']]['status'],
          'physical_null_counterexample':review[c['candidate_id']]['physical_null_counterexample'],
          'scientific_reason':review[c['candidate_id']]['scientific_reason'],
          'next_generation_instruction':review[c['candidate_id']]['next_generation_instruction'],
          'evidence_work_ids':review[c['candidate_id']]['cited_work_ids']}
         for c in previous_generation['candidates']],
       'retrieval_failed':search['retrieval_failed']}


def export_summary(plan:dict, rounds:list[dict])->dict:
    rows=[]
    for r in rounds:
        gm=r['generation'];cr={x['candidate_id']:x for x in r['critic']['reviews']}
        for c in gm['candidates']:
            rr=cr[c['candidate_id']]
            rows.append({'generation':gm['generation'],'lane':c['lane'],
               'candidate_id':c['candidate_id'],'title':c['title'],
               'critique_status':rr['status'], 'cited_works':len(rr['cited_work_ids']),
               'novelty_certified':False})
    return {'status':'M12_BOUNDED_MULTI_GENERATION_SEARCH_COMPLETED_UNREVIEWED',
        'plan_id':plan['plan_id'],'rounds_completed':len(rounds),
        'generated_candidates':len(rows),'literature_and_null_model_reviewed':len(rows),
        'candidate_statuses':rows,'novelty_certified':False,
        'empirical_validation_done':False, 'external_independent_expert_review_done':False,
        'production_graph_mutated':False,'official_sis_scheduler_executed':False,
        'warning':'Literature API data and LLM critique do not constitute academic novelty certification.'}


def review_csv(rounds:list[dict])->str:
    s=io.StringIO();cols=['generation','lane','candidate_id','science_coherence_0_3','strongest_null_overcome_0_3',
       'independent_identifiability_0_3','substantive_novelty_vs_prior_art_0_3',
       'evidence_or_DOI_checked','human_disposition','important_flaw','reviewer_notes']
    w=csv.DictWriter(s,fieldnames=cols);w.writeheader()
    for r in rounds:
        for c in r['generation']['candidates']:
            w.writerow({'generation':r['generation']['generation'],'lane':c['lane'],
             'candidate_id':c['candidate_id']})
    return s.getvalue()
