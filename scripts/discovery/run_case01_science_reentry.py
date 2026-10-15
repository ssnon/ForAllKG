#!/usr/bin/env python3
"""Case01 opt-in native P0 scientific self-review -> next research action.

No baseline SIS files are changed. prepare/verify/freeze-action/route are offline.
Actual model requests require explicit two-stage authorization. Not a mechanism
producer, native root search executor, or scientific novelty certification.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

SCHEMA = 'case01-native-science-reentry-v1'
FIELDS = ('p0.execution.json', 'p0.seed_report.json', 'direct_seed.summary.json', 'hypothesis.context.json')
SCIENCE_FILES = ('Ye_main_text_by_page.json', 'hypothesis_axis_open_world.selected_works.json',
                 'Ye_2025_PRB_source_4yg7-1f85.pdf', *(f'Ye_page_{n:02}.png' for n in (3,4,5,6)))
# Frozen before model calls, not chosen after observing previous B01..B06 outcomes.
PARENTS = (
    'research_idea:01d557431e4bc2d67518',  # molecular residence vs hotspot proxy
    'research_idea:6a072466a86f889ec7a4',  # changing optical field vs surface state
)
DIAGNOSES = {'UNRESOLVED_EVIDENCE', 'NULL_EQUIVALENT', 'FORMAL_ERROR',
             'POTENTIALLY_DISTINCT', 'INDETERMINATE'}
ACTIONS = {'CONTINUE_LINEAGE','DIVERSIFY_FROM_PARENT','REOPEN_PROBLEM_ROOT','ACQUIRE_EVIDENCE','ABSTAIN'}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def hfile(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canon(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def sha(obj: Any) -> str:
    return hashlib.sha256(canon(obj).encode('utf-8')).hexdigest()

def write_new(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')

def nonblank(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())

def strict_json(txt: str) -> dict:
    def unique_pairs(pairs):
        v = {}
        for k, val in pairs:
            if k in v: raise ValueError('DUPLICATE_JSON_KEY:'+k)
            v[k] = val
        return v
    v = json.loads(txt, object_pairs_hook=unique_pairs,
                   parse_constant=lambda s: (_ for _ in ()).throw(ValueError('NONFINITE:'+s)))
    if not isinstance(v, dict): raise ValueError('JSON_OBJECT_REQUIRED')
    return v

def preflight(native_dir: Path, science_dir: Path, freeze_dir: Path, repo_dir: Path | None = None) -> dict:
    pins = load(freeze_dir / 'NATIVE_SHA256.json')
    native = {}
    for fname in FIELDS:
        file = native_dir / fname
        if not file.is_file() or hfile(file) != pins[fname]:
            raise ValueError('NATIVE_SOURCE_MISSING_OR_CHANGED:'+fname)
        native[fname] = load(file)
    s_pins = load(science_dir / 'SOURCE_SHA256.json')
    if s_pins != load(freeze_dir / 'SCIENCE_SHA256.json'):
        raise ValueError('FROZEN_SCIENCE_MANIFEST_CHANGED')
    for fname in SCIENCE_FILES:
        file = science_dir / fname
        if not file.is_file() or hfile(file) != s_pins.get(fname):
            raise ValueError('SCIENCE_SOURCE_MISSING_OR_CHANGED:'+fname)
    pages = load(science_dir / SCIENCE_FILES[0]); works = load(science_dir / SCIENCE_FILES[1])
    if len(pages) != 8 or [row['page'] for row in pages] != list(range(1,9)):
        raise ValueError('YE_PAGES_INVALID')
    if len(works) != 12 or sum((w.get('work',{}).get('doi') or '').lower() == '10.1103/4yg7-1f85' for w in works) != 1:
        raise ValueError('SOURCE_12_ABSTRACTS_INVALID')
    exe = native[FIELDS[0]]; seed = native[FIELDS[1]]; summ = native[FIELDS[2]]; ctx = native[FIELDS[3]]
    nodes = exe['g4_population_nodes']; ids = [n['idea_id'] for n in nodes]
    if (exe['generation_index'] != 2 or exe['g4_population_count'] != 8
            or len(nodes) != 8 or len(set(ids)) != 8 or set(PARENTS)-set(ids)
            or exe.get('llm_call_count') != 0 or exe.get('raw_offspring_count') != 0):
        raise ValueError('NATIVE_P0_CONTRACT_INVALID')
    if (seed['selected_research_idea_ids'] != ids or seed['selected_candidate_count'] != 8
            or seed['source_context_sha256'] != ctx['context_sha256']
            or summ.get('status') != 'DIRECT_P0_READY'
            or not summ.get('old_selection_parity_checked')
            or summ.get('portfolio_evaluation_llm_calls') != 0):
        raise ValueError('NATIVE_SEED_IDENTITY_OR_PARITY_MISMATCH')
    if (any(n['source_context_id'] != ctx['context_id'] or
            n['source_context_sha256'] != ctx['context_sha256'] for n in nodes)
            or 'blinking' not in ctx['question'].lower()
            or any(n.get('epistemic_status') != 'INSPIRATION_ONLY' for n in nodes)):
        raise ValueError('CONTEXT_OR_EPISTEMIC_BOUNDARY_MISMATCH')
    if not ({'FRONTIER','EVOLUTION'} <= {n['origin_kind'] for n in nodes}):
        raise ValueError('P0_ORIGIN_COHORT_INVALID')
    if repo_dir is not None:
        import subprocess
        rev = subprocess.run(['git','-C',str(repo_dir),'rev-parse','HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        if rev != '453492ab81eb2cc122f65f2cce1cd479ea2c47cf':
            raise ValueError('LOCAL_REPO_COMMIT_MISMATCH:'+rev)
        # The real FAKG Pydantic contracts are authoritative in WSL.
        sys.path.insert(0, str(repo_dir.resolve()))
        from pipeline_core.discovery.research_idea_epistemic_generational_evolution import EpistemicG4ExecutionReport
        from pipeline_core.discovery.research_idea_e2e_integration import ResearchIdeaE2ESeedReport
        from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
        EpistemicG4ExecutionReport.model_validate(exe)
        ResearchIdeaE2ESeedReport.model_validate(seed)
        HypothesisContext.model_validate(ctx)
    return {'status':'NATIVE_P0_SCIENCE_READY', 'native': native, 'pages': pages,
            'works': works, 'native_sha256': pins,
            'science_sha256': {f:s_pins[f] for f in SCIENCE_FILES},
            'selected_parent_idea_ids': list(PARENTS)}

def idea_view(node: dict) -> dict:
    return {k:node.get(k) for k in ('idea_id','origin_kind','source_object_id','kernel',
       'differential_prediction','discriminating_observation','falsification_condition',
       'source_parent_object_ids','source_artifact_refs','epistemic_status')}

REVIEW_SYSTEM = '''You are an experimental, FAKG-integrated scientific criticism agent.
Review the exact two frozen native ResearchIdeas, not an assistant-suggested solution.
Treat the primary published Ye paper as source, not proof of the authors' mechanisms;
other retrieved papers are ABSTRACT LEVEL and not exhaustive. Proposed native ideas are
INSPIRATION_ONLY, not established premises. The frozen context's SERS500 evidence
is background across systems, not proof for Ye's specimen. Distinguish observations,
authors' hypotheses, speculative known-physics nulls, and model inferences.
For each idea, identify physical state and causal dependency (or missing specification),
check a concrete physical/mathematical consequence, construct the strongest conventional
countermodel and whether it mimics the prediction, describe independent controls,
and state which observation is unavailable. Be free to diagnose evidence gaps, physical
errors, or indistinguishability; do not invent data or cite papers beyond input.
Do not propose an answer to the open problem as a prompt seed; provide critiques.
Neither failure nor novelty is experimentally certified. You may abstain.
Return exactly a JSON object with key reviews, an array containing exactly 2 objects.
Each object must contain exactly keys: idea_id, proposed_causal_state,
math_or_physics_check, strongest_null_and_mimicry, independent_test,
missing_evidence, diagnosis, diagnosis_reason, evidence_ids, reviewer_abstained.
All fields except evidence_ids (array of IDs) and reviewer_abstained (boolean)
are nonempty strings. diagnosis is one of UNRESOLVED_EVIDENCE, NULL_EQUIVALENT,
FORMAL_ERROR, POTENTIALLY_DISTINCT, INDETERMINATE.
No additional keys.''' 

DECISION_SYSTEM = '''You are the FAKG opt-in scientific research-action planner, not a scientist
claiming novelty or a generic rewriter. Use ONLY the original open question, exact
frozen ResearchIdea identities and the FAKG self-review just produced. That critique
is non-authoritative; it is not an experimental refutation. Do not consult earlier
ChatGPT reviews or B01..B06; do not invent a scientific solution or falsification.
Select ONE justified next scientific research action (not five suggestions):
CONTINUE_LINEAGE, DIVERSIFY_FROM_PARENT, REOPEN_PROBLEM_ROOT, ACQUIRE_EVIDENCE,
ABSTAIN. Do not automatically choose root or continue. If an unresolved question
needs more evidence, ACQUIRE_EVIDENCE can be best. If the parent mechanism is
unidentifiable or known-physics-saturated, parent-free exploration may be needed.
Return exactly one JSON object with keys: action, target_idea_id, diagnostic_idea_ids,
rationale, abandoned_assumption, research_question_for_next_step, evidence_requests,
uncertainty. action must be one of the given tokens; target_idea_id is either one
of the two reviewed idea IDs or null, diagnostic_idea_ids is a nonempty list of
reviewed idea IDs, evidence_requests is a list of strings, all other fields are
nonempty strings. The next research question is a question, NOT a new proposed
physical mechanism. Null target for REOPEN_PROBLEM_ROOT/ACQUIRE_EVIDENCE/ABSTAIN;
otherwise target an actually reviewed parent. Do not claim the model's choice is
scientifically validated.''' 

def review_prompt(pre: dict) -> dict:
    c = pre['native']['hypothesis.context.json']
    ns = {n['idea_id']:n for n in pre['native']['p0.execution.json']['g4_population_nodes']}
    evidence = {'open_scientific_problem':c['question'], 'native_context':{
        'context_id':c['context_id'], 'evidence_statements':c['evidence_statements'],
        'research_gaps':c['research_gaps']},
        'native_research_ideas':[idea_view(ns[id]) for id in PARENTS],
        'ye_original_text':[{'page':r['page'],'text':r['text']} for r in pre['pages']],
        'ye_original_pdf_images_pages':[3,4,5,6],
        'native_retrieved_abstracts':[{'work_id':w['work'].get('work_id'),
            'doi':w['work'].get('doi'),'title':w['work'].get('title'),
            'abstract':w['work'].get('abstract'),'year':w['work'].get('year')}
            for w in pre['works']],
        'not_supplied':['prior assistant science review','B01-B06 sidecar candidates',
                        'previous ABSTAIN verdict','hypothetical answer seeds']}
    return {'schema_version':SCHEMA,'type':'FAKG_NATIVE_P0_SELF_REVIEW_EXPERIMENTAL',
        'messages':[{'role':'system','content':REVIEW_SYSTEM},
                    {'role':'user','content':canon(evidence)}],
        'original_figure_pages':[3,4,5,6]}

def prepare(native_dir: Path, science_dir: Path, freeze_dir: Path, out: Path,
            model: str, temp: float, tokens: int, repo_dir: Path | None = None) -> dict:
    if out.exists(): raise FileExistsError('OUTPUT_EXISTS_NO_OVERWRITE')
    if not nonblank(model) or not 0 <= temp <= 1 or not 1500 <= tokens <= 12000:
        raise ValueError('INVALID_MODEL_PARAMETERS')
    pre = preflight(native_dir,science_dir,freeze_dir,repo_dir)
    rp = review_prompt(pre)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'REVIEW_PROMPT.json',rp)
    plan={'schema_version':SCHEMA,'status':'FROZEN_PREPARED_NO_CALLS',
         'native_dir':str(native_dir.resolve()),'science_dir':str(science_dir.resolve()),
         'freeze_dir':str(freeze_dir.resolve()),'repo_dir':str(repo_dir.resolve()) if repo_dir else None,
         'native_sha256':pre['native_sha256'],'science_sha256':pre['science_sha256'],
         'selected_parent_idea_ids':list(PARENTS),'review_prompt_sha256':hfile(out/'REVIEW_PROMPT.json'),
         'model':model,'temperature':temp,'max_tokens':tokens,
         'planned_paid_calls':2,'completed_paid_calls':0,
         'external_sidecar_ingested':False,'native_review_completed':False,
         'scientific_novelty_certified':False,'native_root_generation_executed':False}
    write_new(out/'PLAN.json',plan)
    return {'status':'FROZEN_PREPARED_NO_CALLS','out':str(out),'parents':list(PARENTS),
            'prompt_characters':len(rp['messages'][1]['content']),'paid_calls':0}

def verify(out: Path) -> tuple[dict,dict]:
    plan = load(out/'PLAN.json')
    pre = preflight(Path(plan['native_dir']),Path(plan['science_dir']),Path(plan['freeze_dir']),
                    Path(plan['repo_dir']) if plan['repo_dir'] else None)
    if (plan['native_sha256']!=pre['native_sha256'] or
        plan['science_sha256']!=pre['science_sha256'] or
        plan['selected_parent_idea_ids']!=list(PARENTS) or
        hfile(out/'REVIEW_PROMPT.json')!=plan['review_prompt_sha256'] or
        load(out/'REVIEW_PROMPT.json')!=review_prompt(pre) or
        plan['external_sidecar_ingested'] is not False):
        raise ValueError('FROZEN_SOURCES_OR_PROMPT_DRIFT')
    return plan,pre

def parse_review(raw: str) -> dict:
    j = strict_json(raw)
    if set(j)!={'reviews'} or not isinstance(j['reviews'],list) or len(j['reviews'])!=2:
        raise ValueError('REVIEW_SCHEMA_WRONG_COUNT')
    keys={'idea_id','proposed_causal_state','math_or_physics_check','strongest_null_and_mimicry',
          'independent_test','missing_evidence','diagnosis','diagnosis_reason','evidence_ids','reviewer_abstained'}
    for row in j['reviews']:
        if not isinstance(row,dict) or set(row)!=keys or row['diagnosis'] not in DIAGNOSES:
            raise ValueError('REVIEW_ROW_SCHEMA')
        if any(not nonblank(row[k]) for k in keys-{'evidence_ids','reviewer_abstained'}):
            raise ValueError('REVIEW_EMPTY_REQUIRED_STRING')
        if not isinstance(row['reviewer_abstained'],bool) or not isinstance(row['evidence_ids'],list) or any(not nonblank(x) for x in row['evidence_ids']):
            raise ValueError('REVIEW_TYPES')
        if row['reviewer_abstained'] and row['diagnosis']!='INDETERMINATE':
            raise ValueError('ABSTENTION_MUST_BE_INDETERMINATE')
    if [r['idea_id'] for r in j['reviews']] != list(PARENTS):
        raise ValueError('REVIEW_PARENT_IDENTITY_MISMATCH')
    return j

def action_prompt(pre:dict, reviewed:dict, review_receipt:dict) -> dict:
    ctx = pre['native']['hypothesis.context.json']
    return {'schema_version':SCHEMA,'type':'FAKG_NATIVE_SCIENCE_ACTION_EXPERIMENTAL',
      'review_prompt_sha256':review_receipt['prompt_sha256'],
      'review_response_sha256':sha(reviewed),
      'messages':[{'role':'system','content':DECISION_SYSTEM},
                  {'role':'user','content':canon({'question':ctx['question'],
                      'reviewed_native_idea_ids':list(PARENTS),
                      'native_research_idea_reviews_unverified':reviewed['reviews'],
                      'prohibition':'No previous assistant model critiques, external sidecar failures or answer seeds.'})}]}

def parse_action(raw: str) -> dict:
    j=strict_json(raw)
    keys={'action','target_idea_id','diagnostic_idea_ids','rationale','abandoned_assumption',
          'research_question_for_next_step','evidence_requests','uncertainty'}
    if set(j)!=keys or j['action'] not in ACTIONS:
        raise ValueError('ACTION_SCHEMA')
    if any(not nonblank(j[k]) for k in ('rationale','abandoned_assumption','research_question_for_next_step','uncertainty')):
        raise ValueError('ACTION_BLANK_REASON')
    if (not isinstance(j['diagnostic_idea_ids'],list) or not j['diagnostic_idea_ids'] or
            any(x not in PARENTS for x in j['diagnostic_idea_ids']) or
            len(set(j['diagnostic_idea_ids']))!=len(j['diagnostic_idea_ids'])):
        raise ValueError('ACTION_DIAGNOSTIC_IDENTITY')
    if not isinstance(j['evidence_requests'],list) or any(not nonblank(x) for x in j['evidence_requests']):
        raise ValueError('ACTION_REQUEST_TYPES')
    if j['action'] in {'CONTINUE_LINEAGE','DIVERSIFY_FROM_PARENT'}:
        if j['target_idea_id'] not in PARENTS: raise ValueError('ACTION_PARENT_REQUIRED')
    elif j['target_idea_id'] is not None:
        raise ValueError('ROOT_EVIDENCE_OR_STOP_CANNOT_CARRY_PARENT')
    if j['action']=='ACQUIRE_EVIDENCE' and not j['evidence_requests']:
        raise ValueError('EVIDENCE_ACTION_NEEDS_REQUEST')
    if '?' not in j['research_question_for_next_step']:
        raise ValueError('RESEARCH_QUESTION_NOT_INTERROGATIVE')
    return j

def attempt_marker(path: Path, phase: str, prompt_hash: str, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as f:
        json.dump({'phase':phase,'status':'ATTEMPT_STARTED_NO_AUTO_RETRY',
                   'model':model,'prompt_sha256':prompt_hash,
                   'utc':datetime.now(timezone.utc).isoformat()},f,sort_keys=True)

def provider_call(messages: list[dict], model: str, temp: float, tokens: int, science_dir: Path | None = None):
    from openai import OpenAI
    msgs=[dict(m) for m in messages]
    if science_dir is not None:
        parts=[{'type':'text','text':msgs[-1]['content']}]
        for n in (3,4,5,6):
            parts.append({'type':'text','text':f'Ye original source PDF page {n} with figures and captions'})
            parts.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+
                     base64.b64encode((science_dir/f'Ye_page_{n:02}.png').read_bytes()).decode('ascii')}})
        msgs[-1]['content']=parts
    client=OpenAI(api_key=os.environ['OPENROUTER_API_KEY'],base_url='https://openrouter.ai/api/v1',
                  timeout=300,max_retries=0)
    r=client.chat.completions.create(model=model,messages=msgs,temperature=temp,max_tokens=tokens,
                                    response_format={'type':'json_object'})
    return {'raw':r.choices[0].message.content or '',
            'finish_reason':r.choices[0].finish_reason,'provider_response_id':r.id,
            'model':model,'usage':r.usage.model_dump(mode='json') if r.usage else None}

def paid_review(out:Path) -> dict:
    plan,pre=verify(out)
    if (out/'attempts/review.json').exists() or (out/'responses/review.json').exists():
        raise FileExistsError('REVIEW_ATTEMPT_ALREADY_EXISTS_NO_RETRY')
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise ValueError('OPENROUTER_API_KEY_NOT_SET')
    prompt=load(out/'REVIEW_PROMPT.json')
    attempt_marker(out/'attempts/review.json','REVIEW',plan['review_prompt_sha256'],plan['model'])
    result=provider_call(prompt['messages'],plan['model'],plan['temperature'],plan['max_tokens'],Path(plan['science_dir']))
    receipt={**result,'phase':'REVIEW','prompt_sha256':plan['review_prompt_sha256'],
             'source_kind':'PROVIDER_FAKG_NATIVE_P0_SELF_REVIEW_EXPERIMENTAL',
             'scientific_truth_authority':False,'empirical_refutation_certified':False}
    write_new(out/'responses/review.json',receipt)
    if result['finish_reason']!='stop':return {'status':'RAW_INCOMPLETE_NO_RETRY'}
    try: parsed=parse_review(result['raw'])
    except Exception as e:
        return {'status':'RAW_SAVED_INVALID_NO_RETRY','error':type(e).__name__+':'+str(e)}
    write_new(out/'parsed/review.json',parsed)
    return {'status':'NATIVE_P0_SELF_REVIEW_GENERATED_UNVERIFIED','parsed':str(out/'parsed/review.json'),
            'provider_model':plan['model'],'usage':result['usage']}

def freeze_action(out:Path) -> dict:
    plan,pre=verify(out)
    if (out/'ACTION_PROMPT.json').exists(): raise FileExistsError('ACTION_ALREADY_FROZEN')
    receipt=load(out/'responses/review.json')
    if receipt.get('source_kind')!='PROVIDER_FAKG_NATIVE_P0_SELF_REVIEW_EXPERIMENTAL' or receipt.get('finish_reason')!='stop' or receipt.get('prompt_sha256')!=plan['review_prompt_sha256']:
        raise ValueError('NON_NATIVE_OR_UNVERIFIED_REVIEW_RECEIPT')
    rv=parse_review(receipt['raw'])
    if load(out/'parsed/review.json')!=rv:raise ValueError('PARSED_REVIEW_MISMATCH')
    ap=action_prompt(pre,rv,receipt)
    write_new(out/'ACTION_PROMPT.json',ap)
    write_new(out/'ACTION_FREEZE.json',{'schema_version':SCHEMA,
        'action_prompt_sha256':hfile(out/'ACTION_PROMPT.json'),
        'review_receipt_sha256':hfile(out/'responses/review.json'),
        'parsed_review_sha256':hfile(out/'parsed/review.json'),
        'native_sidecar_injected':False,'paid_calls_this_step':0})
    return {'status':'ACTION_PROMPT_FROZEN_NO_CALL','action_prompt_sha256':hfile(out/'ACTION_PROMPT.json')}

def verify_action_prompt(out:Path) -> tuple[dict,dict]:
    plan,pre=verify(out)
    freeze=load(out/'ACTION_FREEZE.json'); receipt=load(out/'responses/review.json')
    parsed=load(out/'parsed/review.json'); parse_review(receipt['raw'])
    if (receipt.get('source_kind')!='PROVIDER_FAKG_NATIVE_P0_SELF_REVIEW_EXPERIMENTAL' or
        receipt.get('prompt_sha256')!=plan['review_prompt_sha256'] or
        hfile(out/'responses/review.json')!=freeze['review_receipt_sha256'] or
        hfile(out/'parsed/review.json')!=freeze['parsed_review_sha256'] or
        parsed!=parse_review(receipt['raw']) or
        hfile(out/'ACTION_PROMPT.json')!=freeze['action_prompt_sha256'] or
        load(out/'ACTION_PROMPT.json')!=action_prompt(pre,parsed,receipt)):
        raise ValueError('FROZEN_REVIEW_OR_ACTION_PROMPT_CHANGED')
    return plan,pre

def paid_action(out:Path)->dict:
    plan,_=verify_action_prompt(out)
    if (out/'attempts/action.json').exists() or (out/'responses/action.json').exists():
        raise FileExistsError('ACTION_ATTEMPT_ALREADY_EXISTS_NO_RETRY')
    if not os.environ.get('OPENROUTER_API_KEY'):raise ValueError('OPENROUTER_API_KEY_NOT_SET')
    prompt=load(out/'ACTION_PROMPT.json'); digest=hfile(out/'ACTION_PROMPT.json')
    attempt_marker(out/'attempts/action.json','ACTION',digest,plan['model'])
    response=provider_call(prompt['messages'],plan['model'],plan['temperature'],min(3000,plan['max_tokens']))
    receipt={**response,'phase':'ACTION','prompt_sha256':digest,
             'source_kind':'PROVIDER_FAKG_NATIVE_REENTRY_ACTION_EXPERIMENTAL',
             'scientific_truth_authority':False,'next_generation_executed':False}
    write_new(out/'responses/action.json',receipt)
    if response['finish_reason']!='stop':return {'status':'RAW_INCOMPLETE_NO_RETRY'}
    try: parsed=parse_action(response['raw'])
    except Exception as e:return {'status':'RAW_SAVED_INVALID_NO_RETRY','error':type(e).__name__+':'+str(e)}
    write_new(out/'parsed/action.json',parsed)
    return {'status':'NATIVE_P0_ACTION_SELECTED_UNVERIFIED','action':parsed['action'],
            'model':plan['model'],'usage':response['usage']}

def route(out:Path)->dict:
    plan,_=verify_action_prompt(out)
    receipt=load(out/'responses/action.json'); selected=load(out/'parsed/action.json')
    freeze=load(out/'ACTION_FREEZE.json')
    if (receipt.get('source_kind')!='PROVIDER_FAKG_NATIVE_REENTRY_ACTION_EXPERIMENTAL'
        or receipt.get('finish_reason')!='stop' or receipt.get('prompt_sha256')!=freeze['action_prompt_sha256']
        or selected!=parse_action(receipt['raw'])):
        raise ValueError('ACTION_RECEIPT_NOT_VERIFIED')
    action=selected['action']
    status={'REOPEN_PROBLEM_ROOT':'NEEDS_NATIVE_ROOT_ENTRYPOINT_NOT_EXECUTED',
            'CONTINUE_LINEAGE':'NEEDS_NATIVE_SIS_SCIENCE_REVIEW_ADAPTER_NOT_EXECUTED',
            'DIVERSIFY_FROM_PARENT':'NEEDS_NATIVE_SIS_SCIENCE_REVIEW_ADAPTER_NOT_EXECUTED',
            'ACQUIRE_EVIDENCE':'EVIDENCE_RETRIEVAL_REQUESTED_NOT_EXECUTED',
            'ABSTAIN':'ABSTAINED_STOP'}[action]
    result={'schema_version':SCHEMA,'status':status,'decision':selected,
            'native_p0_sha256':plan['native_sha256']['p0.execution.json'],
            'review_receipt_sha256':hfile(out/'responses/review.json'),
            'action_receipt_sha256':hfile(out/'responses/action.json'),
            'scientific_novelty_certified':False,'autonomous_discovery_certified':False,
            'FAKG_native_production_pipeline_modified':False,
            'new_mechanism_generated':False,'sidecar_feedback_used':False,
            'science_action_generated_by_model':True}
    write_new(out/'REENTRY_ROUTING_DECISION.json',result)
    return {'status':status,'action':action,'output':str(out/'REENTRY_ROUTING_DECISION.json')}

def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    for name in ('preflight','prepare'):
        c=sub.add_parser(name)
        c.add_argument('--native-dir',type=Path,required=True)
        c.add_argument('--science-dir',type=Path,required=True)
        c.add_argument('--freeze-dir',type=Path,required=True)
        c.add_argument('--repo-dir',type=Path)
        if name=='prepare':
            c.add_argument('--out',type=Path,required=True)
            c.add_argument('--model',default='openai/gpt-5.6-luna')
            c.add_argument('--temperature',type=float,default=0.25)
            c.add_argument('--max-tokens',type=int,default=6500)
    for name in ('verify','freeze-action','route','run-paid-review','run-paid-action'):
        c=sub.add_parser(name);c.add_argument('--out',type=Path,required=True)
        if name.startswith('run-paid'):
            c.add_argument('--allow-paid',action='store_true')
            c.add_argument('--i-authorize-one-model-call',action='store_true')
    a=p.parse_args(argv)
    try:
        if a.cmd=='preflight':
            x=preflight(a.native_dir,a.science_dir,a.freeze_dir,a.repo_dir)
            val={'status':x['status'],'parents':x['selected_parent_idea_ids'],'native_sha256':x['native_sha256'],'paid_calls':0}
        elif a.cmd=='prepare':
            val=prepare(a.native_dir,a.science_dir,a.freeze_dir,a.out,a.model,a.temperature,a.max_tokens,a.repo_dir)
        elif a.cmd=='verify':
            plan,_=verify(a.out)
            val={'status':'FROZEN_NATIVE_P0_AND_SCIENCE_PROMPT_PASS','selected_idea_ids':plan['selected_parent_idea_ids'],
                 'paid_calls_performed_by_verify':0}
        elif a.cmd=='freeze-action':val=freeze_action(a.out)
        elif a.cmd=='route':val=route(a.out)
        else:
            if not (a.allow_paid and a.i_authorize_one_model_call and os.environ.get('CASE01_REENTRY_ALLOW_PAID')=='YES'):
                raise PermissionError('PAID_MODEL_CALL_BLOCKED_EXPLICIT_AUTH_REQUIRED')
            val=paid_review(a.out) if a.cmd=='run-paid-review' else paid_action(a.out)
        print(json.dumps(val,ensure_ascii=False,indent=2,default=str))
        return 0
    except Exception as exc:
        print(type(exc).__name__+':'+str(exc),file=sys.stderr)
        return 2
if __name__=='__main__':raise SystemExit(main())
