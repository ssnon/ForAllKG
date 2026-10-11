#!/usr/bin/env python3
"""M14 opt-in scientific problem discovery + matched adaptive-vs-fixed exploration.

No models are called without --allow-paid --i-authorize-model-calls.
Literature retrieval uses live OpenAlex/Europe PMC public metadata APIs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

from pipeline_core.discovery.sers_novelty_search_m14 import (
    ARMS, LANES, SCHEMA, make_plan, frozen, sha, canon, unique_works,
    problem_prompt, parse_problems, candidate_prompt, parse_candidates,
    feedback, blind_rows, computational_null_probes)
from pipeline_core.discovery.sers_quality_first_m13 import (
    physics_prompt, literature_prompt, parse_physics, parse_literature_with_quarantine,
    compress_evidence)


def load(path:Path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def create(path:Path,object_):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=object_ if isinstance(object_,str) else json.dumps(object_,indent=2,sort_keys=True,ensure_ascii=False)
    with path.open('x',encoding='utf-8') as f:f.write(raw if raw.endswith('\n') else raw+'\n')

def freeze(path:Path,value):
    if path.exists():
        if load(path)!=value:raise RuntimeError('Frozen artifact mismatch: '+str(path))
    else:create(path,value)

def source_call(args):
    from openai import OpenAI
    key=os.environ.get(args.api_key_env)
    if not key:raise RuntimeError('Missing API key environment variable '+args.api_key_env)
    client=OpenAI(api_key=key,base_url=args.base_url,timeout=300,max_retries=0)
    def call(model,prompt):
        res=client.chat.completions.create(model=model,
            messages=[{'role':'system','content':prompt['system']},{'role':'user','content':prompt['user']}],
            temperature=args.temperature,max_tokens=args.max_output_tokens)
        return {'text':res.choices[0].message.content or '',
                'finish_reason':res.choices[0].finish_reason,
                'usage':res.usage.model_dump(mode='json') if res.usage else None}
    return call

def source_lookup(provider,query):
    from pipeline_core.discovery.sers_autonomous_novelty_m12 import lookup
    return lookup(provider,query,key=os.getenv('OPENALEX_API_KEY',''))

def saved_model(folder:Path,kind:str,prompt:dict,model:str,call):
    receipt=folder/f'{kind}_RAW_RECEIPT.json'
    intent=folder/f'{kind}_INTENT.json'
    ph=sha(prompt)
    if receipt.exists():
        r=load(receipt)
        if r.get('prompt_sha256')!=ph or r.get('model')!=model:
            raise RuntimeError('Model/prompt changed for saved paid receipt: '+str(receipt))
        if r['finish_reason']!='stop':raise RuntimeError('Truncated paid receipt; manual review required')
        return r['raw'],False
    if intent.exists():raise RuntimeError('Indeterminate previous paid call; no automatic duplicate: '+str(intent))
    create(intent,{'prompt_sha256':ph,'model':model,
                   'status':'PROVIDER_ATTEMPT_STARTED_NO_AUTO_RETRY'})
    try:r=call(model,prompt)
    except Exception as exc:raise RuntimeError('Model provider failed; indeterminate intent retained: '+str(exc)) from exc
    create(receipt,{'prompt_sha256':ph,'model':model,'raw':r['text'],
                    'finish_reason':r['finish_reason'],'usage':r.get('usage'),
                    'novelty_certified':False})
    if r['finish_reason']!='stop':raise RuntimeError('Provider response incomplete; raw saved, no automatic retry')
    return r['text'],True

def retrieve(folder:Path,provider:str,q:str,search):
    name=f'{provider}_{sha({"query":q,"provider":provider})[:18]}_SUCCESS.json'
    target=folder/name
    if target.exists():
        r=load(target)
        if r.get('provider')!=provider or r.get('query')!=q or r.get('success') is not True:
            raise RuntimeError('Saved retrieval receipt corrupted')
        return r
    try:r=search(provider,q)
    except Exception as exc:
        failure=folder/f'{provider}_{sha(q)[:12]}_FAIL_{len(list(folder.glob("*_FAIL_*.json")))+1:03}.json'
        create(failure,{'provider':provider,'query':q,'error':str(exc),'scientific_gap_proven':False})
        raise RuntimeError('Literature retrieval failed; never infer novelty from a failed query: '+str(failure)) from exc
    if r.get('provider')!=provider or r.get('query')!=q or r.get('success') is not True:
        raise RuntimeError('Literature provider response invalid; cannot infer gap')
    create(target,r)
    return r

def prior_evidence(folder:Path,candidates:list[dict],search):
    receipts=[]
    for c in candidates:
        for q in c['prior_art_queries']:
            for provider in ('openalex','europepmc'):
                receipts.append(retrieve(folder/'literature',provider,q,search))
    return compress_evidence(receipts)

def _strict_json(raw:str):
    return json.loads(raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())

def _repair_invalid_json_escapes(raw:str):
    """Escape ONLY illegal JSON escapes inside strings; never alter accepted escapes.

    Result must be audited and then validated. Does not repair truncated/malformed JSON.
    This alone cannot rescue semantic corruption from already-valid escapes such as \\t.
    """
    escaped=[];i=0;quoted=False;edits=[]
    while i<len(raw):
        c=raw[i]
        if c=='"':
            quoted=not quoted;escaped.append(c);i+=1;continue
        if quoted and c=='\\':
            if i+1>=len(raw):raise ValueError('Truncated JSON escape')
            nxt=raw[i+1]
            if nxt in ('"','\\','/','b','f','n','r','t'):
                escaped.append(raw[i:i+2]);i+=2;continue
            if nxt=='u':
                if len(raw[i+2:i+6])==4 and all(x in '0123456789abcdefABCDEF' for x in raw[i+2:i+6]):
                    escaped.append(raw[i:i+6]);i+=6;continue
            escaped.append('\\\\'+nxt);edits.append({'offset':i,'next_char':nxt});i+=2;continue
        escaped.append(c);i+=1
    if quoted:raise ValueError('Unterminated JSON string')
    return ''.join(escaped),edits

def _has_silent_control(value):
    if isinstance(value,str):return any(c in value for c in ('\t','\b','\f','\r'))
    if isinstance(value,list):return any(_has_silent_control(x) for x in value)
    if isinstance(value,dict):return any(_has_silent_control(x) for x in value.values())
    return False


def generation_validated(raw:str,plan:dict,problems:dict,rep:int,g:int,arm:str,history:list[dict]):
    """Only recover deterministic serialization/label errors; never invent missing science."""
    from pipeline_core.discovery.sers_novelty_search_m14 import parse_candidates, nonempty
    edits={}
    original=raw.strip()
    try:d=_strict_json(original)
    except json.JSONDecodeError:
        # The single M13 observed JSON stray ']' after strongest_null. Exact shape only.
        pattern=re.compile(r'(\"strongest_null\"\s*:\s*\"(?:\\.|[^\"\\])*\")\s*\]\s*,\s*(\"title\"\s*:)')
        hits=list(pattern.finditer(original))
        if len(hits)!=1:raise
        fixed=pattern.sub(lambda m:m.group(1)+', '+m.group(2),original,count=1)
        d=_strict_json(fixed)
        edits['single_stray_array_bracket_after_strongest_null']=True
    if _has_silent_control(d):raise ValueError('Model response contains ambiguous JSON control escapes')
    if isinstance(d,dict) and isinstance(d.get('candidates'),list):
        restored=[]
        for ix,c in enumerate(d['candidates']):
            if not isinstance(c,dict):continue
            if c.get('title') is None or (isinstance(c.get('title'),str) and not c['title'].strip()):
                science_fields=('claim_atom','mechanism','equations','strongest_null','null_equations',
                    'conditional_predictions','independent_observables','interventions','falsification','known_physics_overlap')
                if all(nonempty(c.get(x)) for x in science_fields):
                    c['title']='UNTITLED_METADATA_ONLY: '+c['claim_atom'].strip()[:100]
                    restored.append(ix)
        if restored:edits['title_label_only_derived_from_existing_claim_indices']=restored
    parsed=parse_candidates(canon(d),plan,problems,rep,g,arm,history)
    return parsed,(edits or None)


def physics_validated(raw:str,candidates:list[dict])->tuple[dict,dict|None]:
    """Original raw immutable; narrow deterministic corrections flagged as provenance."""
    corrections={}
    try:d=_strict_json(raw)
    except json.JSONDecodeError:
        repaired,edits=_repair_invalid_json_escapes(raw)
        d=_strict_json(repaired)
        corrections['syntax_invalid_escape_edits']=edits
    if _has_silent_control(d):raise ValueError('Ambiguous JSON control escapes in physics review')
    valid={c['candidate_id'] for c in candidates}
    if not isinstance(d.get('reviews'),list) or {x.get('candidate_id') for x in d['reviews']}!=valid:
        raise ValueError('Reviewer candidate identities not matched')
    changes=[]
    for x in d['reviews']:
        if x.get('physical_status')=='KNOWN_PHYSICS_PLAUSIBLE_BUT_NON_IDENTIFIABLE':
            changes.append({'candidate_id':x['candidate_id'],
                'raw_status':x['physical_status'],'normalized_status':'NON_IDENTIFIABLE',
                'retained_interpretation':'Existing physics plausibly explains the observation and model is unidentifiable'})
            x['physical_status']='NON_IDENTIFIABLE'
    if changes:corrections['exact_composite_status_mappings']=changes
    # A correction does not certify scientific truth.
    parsed=parse_physics(canon(d),candidates)
    for r in parsed['reviews']:
        content=canon(r)
        if any(ord(c)<32 for c in content):
            raise ValueError('Physics response includes unescaped control characters')
    return parsed,(corrections or None)

def _audit_error(folder:Path,stage:str,exc:Exception,new_calls:int):
    receipt=folder/f'{stage}_RAW_RECEIPT.json'
    status={'status':'M14_SAVED_MODEL_OUTPUT_INVALID_NO_AUTORECALL',
            'stage':stage,'error_type':type(exc).__name__, 'detail':str(exc),
            'raw_receipt_sha256':digest(receipt) if receipt.exists() else None,
            'novelty_certified':False}
    freeze(folder/f'M14_{stage.upper()}_INVALID.json',status)
    return {**status,'new_model_calls':new_calls}

def discover(plan:dict,root:Path,model:str,call,search):
    frozen(plan)
    folder=root/'common_discovery'
    receipts=[retrieve(folder/'literature',pr,q,search)
              for q in plan['seed_queries'] for pr in ('openalex','europepmc')]
    source=unique_works(receipts,limit=40)
    if len(source['works'])<8:raise RuntimeError('Too few grounded scholarly works; cannot present gap candidates')
    freeze(folder/'M14_SOURCE_SNAPSHOT.json',source)
    prompt=problem_prompt(plan,source)
    raw,new=saved_model(folder,'problem_discovery',prompt,model,call)
    try:parsed=parse_problems(raw,source)
    except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
        return _audit_error(folder,'problem_discovery',exc,int(new))
    freeze(root/'M14_PROVISIONAL_PROBLEMS.json',parsed)
    freeze(root/'M14_COMPUTATIONAL_NULL_PROBES.json',computational_null_probes())
    return {'status':'M14_COMMON_PROVISIONAL_PROBLEMS_DISCOVERED',
      'problems':len(parsed['questions']),'new_model_calls':int(new),
      'source_works':len(source['works']),'novelty_certified':False}

def run(plan:dict,root:Path,model_gen:str,model_phy:str,model_lit:str,
        call,search,max_new_calls:int|None):
    frozen(plan)
    if load(root/'M14_PLAN.json')!=plan:raise RuntimeError('Plan artifact changed')
    if max_new_calls is not None and max_new_calls<0:raise ValueError('Negative model call limit')
    problems=load(root/'M14_PROVISIONAL_PROBLEMS.json')
    if problems.get('novelty_certified') is not False or len(problems['questions'])!=4:
        raise RuntimeError('Invalid provisional problem discovery')
    calls=0; rounds=[]
    for rep in range(1,plan['design']['replicates']+1):
        for g in range(1,plan['design']['generations']+1):
            current={}
            order=ARMS if g%2 else tuple(reversed(ARMS))
            for arm in order:
                folder=root/f'r{rep:02d}'/f'g{g:02d}'/arm
                history=[]
                for prev in range(1,g):
                    h=root/f'r{rep:02d}'/f'g{prev:02d}'/arm/'M14_GENERATION_RESULT.json'
                    if not h.exists():raise RuntimeError('Missing prior generation history')
                    history.append(load(h))
                prompt=candidate_prompt(plan,problems,rep,g,arm,history)
                if not (folder/'generation_RAW_RECEIPT.json').exists() and max_new_calls is not None and calls>=max_new_calls:
                    return {'status':'M14_CHECKPOINTED','new_model_calls':calls,'replicate':rep,'generation':g,'arm':arm,'stage':'generation'}
                raw,new=saved_model(folder,'generation',prompt,model_gen,call);calls+=int(new)
                try:gen,gen_correction=generation_validated(raw,plan,problems,rep,g,arm,history)
                except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
                    return _audit_error(folder,'generation',exc,calls)
                if gen_correction:
                    freeze(folder/'M14_GENERATION_PARSE_CORRECTIONS.json',{
                        'raw_receipt_sha256':digest(folder/'generation_RAW_RECEIPT.json'),
                        'corrections':gen_correction,'no_scientific_claim_added':True})
                freeze(folder/'M14_GENERATION_PARSED.json',gen)
                ev=prior_evidence(folder,gen['candidates'],search)
                if not ev['work_ids']:raise RuntimeError('Zero science retrieval; no novelty inference')
                phy_prompt=physics_prompt(gen['candidates'])
                phy_prompt['system']+=' Write all equations in plain ASCII; do not emit LaTex backslash commands.'
                if not (folder/'physics_RAW_RECEIPT.json').exists() and max_new_calls is not None and calls>=max_new_calls:
                    return {'status':'M14_CHECKPOINTED','new_model_calls':calls,'replicate':rep,'generation':g,'arm':arm,'stage':'physics'}
                raw,new=saved_model(folder,'physics',phy_prompt,model_phy,call);calls+=int(new)
                try:
                    # Audited identity-only replay of the ORIGINAL saved physics receipt.
                    # Never assume review order or guess which scientific critique belongs
                    # to which claim. A missing or invalid audit must fail closed.
                    identity_audit=folder/'M14_PHYSICS_IDENTITY_RECOVERY.json'
                    if identity_audit.is_file():
                        from scripts.discovery.recover_sers_novelty_search_m14_physics_ids import verified_replay
                        raw,identity_provenance=verified_replay(folder,raw,gen['candidates'])
                    else:
                        identity_provenance=None
                    phy,correction=physics_validated(raw,gen['candidates'])
                    if identity_provenance:
                        correction={**(correction or {}), 'identity_only_replay':identity_provenance}
                except (ValueError,TypeError,KeyError,json.JSONDecodeError,RuntimeError) as exc:
                    # Keep original stage failure audit and original paid receipt intact.
                    return _audit_error(folder,'physics',exc,calls)
                if correction:
                    freeze(folder/'M14_PHYSICS_PARSE_CORRECTIONS.json',{
                        'raw_receipt_sha256':digest(folder/'physics_RAW_RECEIPT.json'),
                        'corrections':correction,'not_a_scientific_claim_revision':True})
                freeze(folder/'M14_PHYSICS_CRITIQUE.json',phy)
                lp=literature_prompt(gen['candidates'],ev)
                if not (folder/'literature_RAW_RECEIPT.json').exists() and max_new_calls is not None and calls>=max_new_calls:
                    return {'status':'M14_CHECKPOINTED','new_model_calls':calls,'replicate':rep,'generation':g,'arm':arm,'stage':'literature'}
                raw,new=saved_model(folder,'literature',lp,model_lit,call);calls+=int(new)
                try:lit,quarantine=parse_literature_with_quarantine(raw,gen['candidates'],ev)
                except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
                    return _audit_error(folder,'literature',exc,calls)
                if quarantine:
                    freeze(folder/'M14_LITERATURE_QUARANTINE.json',{
                         'quarantine':quarantine,'raw_receipt_sha256':digest(folder/'literature_RAW_RECEIPT.json'),
                         'metadata_is_not_fulltext':True})
                freeze(folder/'M14_LITERATURE_CRITIQUE.json',lit)
                fb=feedback(gen['candidates'],phy,lit,ev)
                summary={'generation':g,'replicate':rep,'candidates':gen['candidates'],
                         'candidate_reviews':fb,'policy':gen['policy'],
                         'novelty_certified':False}
                freeze(folder/'M14_GENERATION_RESULT.json',summary)
                current[arm]={'number':g,'replicate':rep,'arm':arm,'generation':gen}
            if set(current)!=set(ARMS):raise RuntimeError('Incomplete paired round')
            rounds.extend([current[x] for x in ARMS])
    blind,key=blind_rows(rounds,plan)
    freeze(root/'M14_BLIND_CANDIDATES.json',{'candidates':blind,'arm_assignment_exposed':False,
                                            'novelty_certified':False})
    freeze(root/'M14_PRIVATE_ARM_KEY.json',key)
    calls_by_arm={arm:len(list(root.glob(f'r*/g*/{arm}/*_RAW_RECEIPT.json'))) for arm in ARMS}
    expected=plan['design']['replicates']*plan['design']['generations']*3
    if any(n!=expected for n in calls_by_arm.values()):raise RuntimeError('Realized model call imbalance detected')
    # Extra explicit provider retry may have duplicated an already-billed original call.
    # Do not equate surviving raw receipts with the actual paid call budget.
    ambiguous_original_attempts_by_arm={
        arm:len(list(root.glob(f'r*/g*/{arm}/generation_RETRY_INTENT.json')))
        for arm in ARMS}
    possible_duplicate_charges=sum(ambiguous_original_attempts_by_arm.values())
    strict_call_parity_certifiable=(calls_by_arm['ADAPTIVE']==calls_by_arm['FIXED'] and not possible_duplicate_charges)
    summary={'status':('M14_MATCHED_ARMS_COMPLETE_INDEPENDENT_BLIND_REVIEW_REQUIRED'
                       if strict_call_parity_certifiable
                       else 'M14_ARMS_COMPLETE_POSSIBLE_DUPLICATE_CHARGE_DESCRIPTIVE_ONLY'),
       'replicates':plan['design']['replicates'],'generations':plan['design']['generations'],
       'candidates':len(blind),'calls_by_arm':calls_by_arm,
       'matched_generation_and_review_calls':strict_call_parity_certifiable,
       'equal_saved_generation_and_review_receipts':calls_by_arm['ADAPTIVE']==calls_by_arm['FIXED'],
       'actual_paid_budget_parity_certifiable':strict_call_parity_certifiable,
       'ambiguous_original_provider_attempts_by_arm':ambiguous_original_attempts_by_arm,
       'actual_possible_duplicate_charge_count_unknown':bool(possible_duplicate_charges),
       'novelty_certified':False,'adaptive_better_than_fixed':None,
       'metadata_is_not_fulltext':True,'simulation_or_empirical_validation_performed':False,
       'external_independent_review_required':True,
       'warning':'Completing generation does not establish original physics, causal search superiority, or identified new mechanisms.'}
    freeze(root/'M14_RUN_SUMMARY.json',summary)
    return {**summary,'new_model_calls_this_invocation':calls}

def _authorization(args):
    if not (args.allow_paid and args.i_authorize_model_calls):
        raise RuntimeError('Explicit --allow-paid and --i-authorize-model-calls required')

def cmd_prepare(args):
    root=args.out_dir.expanduser().resolve()
    source=''
    try:source=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):pass
    plan=make_plan(generations=args.generations,replicates=args.replicates,
          queries_per_candidate=args.queries_per_candidate,source_commit=source)
    freeze(root/'M14_PLAN.json',plan)
    print(json.dumps({'status':'M14_PREPARED_FREE','plan':str(root/'M14_PLAN.json'),
          'expected_paid_model_calls':plan['design']['fixed_budget_model_calls'],
          'source_commit':source,'novelty_certified':False},indent=2))

def cmd_preflight(args):
    p=frozen(load(args.plan));q=p['seed_queries'][0]
    out=[]
    for provider in ('openalex','europepmc'):
        result=source_lookup(provider,q)
        if result.get('success') is not True or result.get('provider')!=provider:
            raise RuntimeError('Scholarly source failed: '+provider)
        out.append({'provider':provider,'returned_papers':len(result.get('works',[])),
                    'response_received':True})
    print(json.dumps({'status':'M14_LITERATURE_PREFLIGHT_NO_LLM_CALLS',
          'results':out,'missing_works_prove_novelty':False},indent=2))


def cmd_discover(args):
    _authorization(args)
    p=frozen(load(args.plan));result=discover(p,args.plan.parent,args.problem_model,source_call(args),source_lookup)
    print(json.dumps(result,indent=2,ensure_ascii=False))

def cmd_run(args):
    _authorization(args)
    p=frozen(load(args.plan));out=run(p,args.plan.parent,args.generator_model,args.physics_model,
        args.literature_model,source_call(args),source_lookup,args.max_new_calls)
    print(json.dumps(out,indent=2,ensure_ascii=False))

def cmd_status(args):
    p=frozen(load(args.plan));root=args.plan.parent;count={a:{'generation':0,'physics':0,'literature':0} for a in ARMS}
    for a in ARMS:
        for stage in count[a]:count[a][stage]=len(list(root.glob(f'r*/g*/{a}/{stage}_RAW_RECEIPT.json')))
    print(json.dumps({'status':'M14_STATUS_FREE','common_problems_discovered':(root/'M14_PROVISIONAL_PROBLEMS.json').exists(),
         'planned_rounds_per_arm':p['design']['generations']*p['design']['replicates'],
         'receipts_by_arm_and_stage':count,'complete':(root/'M14_RUN_SUMMARY.json').exists(),
         'novelty_certified':False},indent=2))

def cmd_export(args):
    p=frozen(load(args.plan));root=args.plan.parent
    if not (root/'M14_RUN_SUMMARY.json').is_file():raise RuntimeError('Full matched M14 not complete')
    if args.blind_zip.exists() or args.audit_zip.exists():raise FileExistsError('Export target exists')
    rows=load(root/'M14_BLIND_CANDIDATES.json')['candidates']
    # Four M13-comparable dimensions; genuine parent/child change is not scoreable blind.
    columns=['blind_id','scientific_coherence_0_4','strong_null_discrimination_0_4',
       'independent_measurement_0_4','prior_art_distinctness_0_4',
       'critical_flaw','independent_reviewer_disposition','reviewer_notes']
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=columns);writer.writeheader()
    for r in rows:writer.writerow({'blind_id':r['blind_id']})
    template=buffer.getvalue()
    with zipfile.ZipFile(args.blind_zip,'x',zipfile.ZIP_DEFLATED) as z:
        z.write(root/'M14_BLIND_CANDIDATES.json','M14_BLIND_CANDIDATES.json')
        z.writestr('M14_INDEPENDENT_REVIEW_TO_FILL.csv',template)
        z.writestr('M14_REVIEW_LIMITATIONS.txt','Conceptual change versus parent is UNSCORABLE from this packet; do not assign numeric generational-change score.\n')
    with zipfile.ZipFile(args.audit_zip,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob('*')):
            if f.is_file() and f not in (args.audit_zip,args.blind_zip) and '_INTENT' not in f.name:
                z.write(f,f.relative_to(root).as_posix())
    print(json.dumps({'status':'M14_PACKETS_EXPORTED','blind_zip':str(args.blind_zip),
      'audit_zip':str(args.audit_zip),'blind_review_before_audit':True,'novelty_certified':False},indent=2))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--out-dir',required=True,type=Path)
    q.add_argument('--generations',type=int,default=6);q.add_argument('--replicates',type=int,default=4)
    q.add_argument('--queries-per-candidate',type=int,default=5);q.set_defaults(func=cmd_prepare)
    q=sub.add_parser('preflight-literature');q.add_argument('--plan',required=True,type=Path);q.set_defaults(func=cmd_preflight)
    for name in ('discover-paid','run-paid'):
        q=sub.add_parser(name);q.add_argument('--plan',required=True,type=Path)
        q.add_argument('--api-key-env',default='OPENROUTER_API_KEY');q.add_argument('--base-url',default='https://openrouter.ai/api/v1')
        q.add_argument('--temperature',type=float,default=0.7);q.add_argument('--max-output-tokens',type=int,default=9000)
        q.add_argument('--allow-paid',action='store_true');q.add_argument('--i-authorize-model-calls',action='store_true')
        if name=='discover-paid':
            q.add_argument('--problem-model',default='openai/gpt-5.6-luna');q.set_defaults(func=cmd_discover)
        else:
            q.add_argument('--generator-model',default='openai/gpt-5.6-luna')
            q.add_argument('--physics-model',default='openai/gpt-5.6-luna')
            q.add_argument('--literature-model',default='openai/gpt-5.6-luna')
            q.add_argument('--max-new-calls',type=int,default=12)
            q.set_defaults(func=cmd_run)
    q=sub.add_parser('status');q.add_argument('--plan',required=True,type=Path);q.set_defaults(func=cmd_status)
    q=sub.add_parser('export');q.add_argument('--plan',required=True,type=Path)
    q.add_argument('--blind-zip',type=Path,required=True);q.add_argument('--audit-zip',type=Path,required=True)
    q.set_defaults(func=cmd_export)
    a=p.parse_args(argv)
    return a.func(a)

if __name__=='__main__':main()
