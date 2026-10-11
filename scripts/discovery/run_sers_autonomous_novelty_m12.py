#!/usr/bin/env python3
"""M12 bounded autonomous SERS hypothesis → literature → physics → new hypothesis loop.

The run-paid command spends an explicit bounded number of OpenRouter model calls;
Europe PMC and OpenAlex are queried live, with independent evidence receipts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from pipeline_core.discovery.sers_autonomous_novelty_m12 import (
  canonical, sha, file_sha, new, read, checked_plan, prepare, generation_prompt, parse_generation,
  critic_prompt, parse_critic, lookup, unique_works, feedback, export_summary, review_csv,
)


def _one_model(root:Path,g:int,kind:str,prompt:dict,model_call):
    folder=root/f'g{g:02d}'
    intent=folder/f'{kind}_INTENT.json'
    receipt=folder/f'{kind}_RAW_RECEIPT.json'
    parsed=folder/f'{kind}_PARSED.json'
    prompt_hash=sha(prompt)
    if parsed.exists() and not receipt.exists():raise RuntimeError('Parsed result without raw receipt')
    if receipt.exists():
        r=read(receipt)
        if r.get('prompt_sha256')!=prompt_hash:raise RuntimeError('Stored receipt prompt drift')
        if r.get('finish_reason')!='stop':raise RuntimeError('Saved truncated response; do not recall')
        return r['raw'],False,parsed
    if intent.exists():raise RuntimeError('Indeterminate model attempt: never silently call again; '+str(intent))
    new(intent,{'status':'PROVIDER_ATTEMPTED_NO_AUTORETRY','prompt_sha256':prompt_hash,
       'generation':g,'role':kind,'novelty_certified':False})
    try:
        reply=model_call(prompt['system'],prompt['user'])
    except Exception as exc:
        raise RuntimeError(f'M12 provider failure; intent preserved; no auto retry: {type(exc).__name__}: {exc}') from exc
    new(receipt,{'generation':g,'kind':kind,'prompt_sha256':prompt_hash,
       'model':reply['model'],'finish_reason':reply['finish_reason'],
       'raw':reply['raw'],'usage':reply.get('usage'),
       'novelty_certified':False,'positive_premise_authority':False})
    if reply['finish_reason']!='stop':raise RuntimeError('Saved incomplete raw receipt; no recall')
    return reply['raw'],True,parsed


def _parse_existing(path:Path,object_:dict):
    if path.exists():
        if read(path)!=object_:raise RuntimeError('Parsed receipt was modified')
    else:new(path,object_)


def _retrieve(root:Path,g:int,gen:dict,search_fn):
    folder=root/f'g{g:02d}'/'literature'
    results=[]
    for candidate in gen['candidates']:
        for index, query in enumerate(candidate['prior_art_queries'],start=1):
            for provider in ('openalex','europepmc'):
                key=sha({'candidate_id':candidate['candidate_id'], 'query':query,'provider':provider})[:12]
                cache=folder/f'{index}_{candidate["lane"]}_{provider}_{key}.json'
                if cache.exists():res=read(cache)
                else:
                    try:res=search_fn(provider,query)
                    except Exception as exc:
                        res={'provider':provider,'query':query,'works':[],
                              'success':False,'error':f'{type(exc).__name__}: {exc}'}
                    new(cache,res)
                if res.get('provider')!=provider or res.get('query')!=query or not isinstance(res.get('works'),list):
                    raise RuntimeError('Literature receipt corrupted or wrong query')
                results.append(res)
    works=unique_works(results)
    manifest=folder/'M12_LITERATURE_EVIDENCE.json'
    _parse_existing(manifest,works)
    if works['retrieval_failed'] or not works['retrieved_count']:
        raise RuntimeError('External literature retrieval incomplete, do NOT automatically claim novelty or continue to next generation. See '+str(manifest))
    return works


def drive(plan:dict,root:Path,max_new_calls:int,model_call,search_fn)->dict:
    """Persist each paid response before validating; stop cleanly when budget reached.

    A resumed command never repeats an attempted model call (successful or uncertain).
    """
    checked_plan(plan)
    if not 0<=max_new_calls<=4:raise ValueError('new call budget outside [0,4]')
    original=read(root/'M12_PLAN.json')
    if original!=plan:raise RuntimeError('Plan differs from frozen experiment')
    new_calls=0;history=[];rounds=[]
    for g in (1,2):
        prompt=generation_prompt(plan,g,history)
        intent=root/f'g{g:02d}'/'generation_INTENT.json'
        receipt=root/f'g{g:02d}'/'generation_RAW_RECEIPT.json'
        if not receipt.exists() and new_calls>=max_new_calls:
            break
        raw,was_new,parsed_path=_one_model(root,g,'generation',prompt,model_call)
        new_calls+=int(was_new)
        generation=parse_generation(raw,g)
        _parse_existing(parsed_path,generation)
        if not generation['candidates']:
            info={'generation':g,'abstained':True,'reason':generation['abstention_reason']}
            new(root/f'g{g:02d}'/'M12_ABSTENTION.json',info)
            return {'status':'M12_ABSTAINED_WITHOUT_UNSUPPORTED_CLAIMS',
              'generation':g,'new_provider_calls':new_calls}
        evidence=_retrieve(root,g,generation,search_fn)
        critique_prompt=critic_prompt(generation,evidence)
        creceipt=root/f'g{g:02d}'/'challenge_RAW_RECEIPT.json'
        if not creceipt.exists() and new_calls>=max_new_calls:break
        craw,cnew,cpath=_one_model(root,g,'challenge',critique_prompt,model_call)
        new_calls+=int(cnew)
        criticism=parse_critic(craw,generation['candidates'],evidence)
        _parse_existing(cpath,criticism)
        fb=feedback(generation,criticism,evidence)
        _parse_existing(root/f'g{g:02d}'/'M12_FEEDBACK_TO_NEXT_GENERATION.json',fb)
        rounds.append({'generation':generation,'critic':criticism,
           'bibliographic_evidence':evidence,'feedback':fb})
        history.append(fb)
    if len(rounds)==2:
        summary=export_summary(plan,rounds)
        _parse_existing(root/'M12_AUTONOMOUS_LOOP_SUMMARY.json',summary)
        _parse_existing(root/'M12_ALL_ROUNDS.json',{'plan_id':plan['plan_id'],
          'rounds':rounds,'novelty_certified':False})
        out=root/'M12_INDEPENDENT_HUMAN_REVIEW_TO_FILL.csv'
        if out.exists() and out.read_bytes()!=review_csv(rounds).encode('utf-8'):
            raise RuntimeError('Human review worksheet already modified; no overwrite')
        if not out.exists():new(out,review_csv(rounds))
        return {'status':'M12_TWO_GENERATION_LOOP_COMPLETED_UNREVIEWED',
          'new_provider_calls':new_calls,'output':str(root),'model_calls_total_planned':4,
          'novelty_certified':False}
    return {'status':'M12_CHECKPOINTED_INCOMPLETE_WAITING_FOR_EXPLICIT_MODEL_BUDGET',
        'completed_generations':len(rounds),'new_provider_calls':new_calls,
        'novelty_certified':False}


def _api_call_factory(args):
    from openai import OpenAI
    client=OpenAI(api_key=os.environ[args.api_key_env],base_url=args.base_url,
                  timeout=180.0,max_retries=0)
    def call(system,user):
        result=client.chat.completions.create(model=args.model,
           messages=[{'role':'system','content':system},{'role':'user','content':user}],
           temperature=0,max_tokens=args.max_output_tokens)
        return {'model':args.model,'raw':result.choices[0].message.content or '',
         'finish_reason':result.choices[0].finish_reason,
         'usage':result.usage.model_dump(mode='json') if result.usage else None}
    return call


def cmd_prepare(args):
    if args.out_dir.exists() and any(args.out_dir.iterdir()):raise FileExistsError('Nonempty output dir')
    plan=prepare(args.m11_packet,args.m11_review)
    new(args.out_dir/'M12_PLAN.json',plan)
    print(json.dumps({'status':'M12_PREPARED_NO_PROVIDER_CALLS',
       'plan_id':plan['plan_id'],'out':str(args.out_dir),
       'max_paid_model_calls':4,'external_queries_per_generation':8,
       'novelty_certified':False},ensure_ascii=False,indent=2))


def cmd_preview(args):
    plan=checked_plan(read(args.plan));prompt=generation_prompt(plan,1,[])
    print(json.dumps({'status':'M12_PREVIEW_NO_PROVIDER_CALLS',
       'generation_1_prompt_sha256':sha(prompt),
       'generation_count':2,'lanes':['REPLACE_OR_BRANCH','OPEN_SERS_FRONTIER'],
       'max_provider_calls':4,'live_literature_providers':['OpenAlex','Europe PMC'],
       'no_novelty_certification':True},indent=2))


def cmd_preflight_literature(args):
    plan=checked_plan(read(args.plan));root=args.plan.parent
    checks=[]
    for provider in ('openalex','europepmc'):
        try:
            ans=lookup(provider,'surface enhanced Raman spectroscopy',key=os.getenv('OPENALEX_API_KEY',''))
            checks.append({'provider':provider,'success':True,'result_count':len(ans['works']),
                           'sample_titles':[w['title'] for w in ans['works'][:2]]})
        except Exception as exc:
            checks.append({'provider':provider,'success':False,'result_count':0,
                           'error':f'{type(exc).__name__}: {exc}'})
    ok=all(x['success'] and x['result_count']>0 for x in checks)
    snapshot={'status':'M12_LIVE_LITERATURE_PREFLIGHT_PASS_NO_MODEL_CALLS' if ok else 'M12_LITERATURE_PREFLIGHT_FAILED_NO_MODEL_CALLS',
          'plan_id':plan['plan_id'],'providers':checks,'novelty_certified':False,
          'new_model_calls':0}
    path=root/'M12_LITERATURE_PREFLIGHT.json'
    if not ok:
        print(json.dumps(snapshot,indent=2,ensure_ascii=False))
        raise RuntimeError('Live scholarly retrieval unavailable: do not spend model calls')
    if path.exists():
        if read(path).get('status') != 'M12_LIVE_LITERATURE_PREFLIGHT_PASS_NO_MODEL_CALLS' or read(path).get('plan_id')!=plan['plan_id']:
            raise RuntimeError('Prior live literature preflight invalid')
        print('M12_LITERATURE_PREFLIGHT_ALREADY_SUCCEEDED='+str(path))
        return
    new(path,snapshot)
    print(json.dumps(snapshot,indent=2,ensure_ascii=False))


def cmd_paid(args):
    if not (args.allow_paid and args.i_authorize_new_llm_calls):
        raise ValueError('Paid generation requires both explicit consent flags')
    if not os.getenv(args.api_key_env):raise ValueError('Missing OpenRouter API key')
    plan=checked_plan(read(args.plan));root=args.plan.parent
    gate=root/'M12_LITERATURE_PREFLIGHT.json'
    if not gate.exists() or read(gate).get('status')!='M12_LIVE_LITERATURE_PREFLIGHT_PASS_NO_MODEL_CALLS':
        raise RuntimeError('Run free preflight-literature and confirm two providers before any paid generation')
    model_lock={'model':args.model,'base_url':args.base_url,
                'max_output_tokens':args.max_output_tokens,'temperature':0}
    lock=root/'M12_MODEL_LOCK.json'
    if lock.exists():
        if read(lock)!=model_lock:raise RuntimeError('Model changed across matched run')
    else:new(lock,model_lock)
    api=_api_call_factory(args)
    key=os.getenv('OPENALEX_API_KEY','')
    result=drive(plan,root,args.max_new_calls,api,
                 lambda provider,query:lookup(provider,query,key=key))
    print(json.dumps(result,indent=2,ensure_ascii=False))


def cmd_status(args):
    root=args.plan.parent;checked_plan(read(args.plan))
    rows=[]
    for g in (1,2):
        folder=root/f'g{g:02d}'
        rows.append({'generation':g,
           'generation_receipt':(folder/'generation_RAW_RECEIPT.json').exists(),
           'scientific_critic_receipt':(folder/'challenge_RAW_RECEIPT.json').exists(),
           'feedback_saved':(folder/'M12_FEEDBACK_TO_NEXT_GENERATION.json').exists()})
    print(json.dumps({'status':'M12_STATUS_NO_PROVIDER_CALLS',
       'results':rows,'novelty_certified':False},indent=2))


def cmd_export(args):
    root=args.plan.parent
    if not (root/'M12_AUTONOMOUS_LOOP_SUMMARY.json').is_file():
        raise RuntimeError('Full two-generation loop not yet completed')
    if args.output.exists():raise FileExistsError('Refuse to overwrite science packet')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.output,'x',ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p!=args.output and 'INTENT' not in p.name:
                z.write(p,p.relative_to(root).as_posix())
    print('M12_REVIEW_PACKET='+str(args.output))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--m11-packet',required=True,type=Path)
    q.add_argument('--m11-review',required=True,type=Path)
    q.add_argument('--out-dir',required=True,type=Path);q.set_defaults(func=cmd_prepare)
    q=sub.add_parser('preview');q.add_argument('--plan',required=True,type=Path);q.set_defaults(func=cmd_preview)
    q=sub.add_parser('preflight-literature');q.add_argument('--plan',required=True,type=Path);q.set_defaults(func=cmd_preflight_literature)
    q=sub.add_parser('run-paid');q.add_argument('--plan',required=True,type=Path)
    q.add_argument('--model',default='openai/gpt-5.6-luna')
    q.add_argument('--api-key-env',default='OPENROUTER_API_KEY')
    q.add_argument('--base-url',default='https://openrouter.ai/api/v1')
    q.add_argument('--max-new-calls',type=int,default=2)
    q.add_argument('--max-output-tokens',type=int,default=6000)
    q.add_argument('--allow-paid',action='store_true')
    q.add_argument('--i-authorize-new-llm-calls',action='store_true')
    q.set_defaults(func=cmd_paid)
    q=sub.add_parser('status');q.add_argument('--plan',required=True,type=Path);q.set_defaults(func=cmd_status)
    q=sub.add_parser('export');q.add_argument('--plan',required=True,type=Path)
    q.add_argument('--output',required=True,type=Path);q.set_defaults(func=cmd_export)
    args=p.parse_args();args.func(args)

if __name__=='__main__':main()
