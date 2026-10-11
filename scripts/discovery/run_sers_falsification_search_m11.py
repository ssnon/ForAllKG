#!/usr/bin/env python3
"""M11 SERS adversarial falsification → independent ResearchIdea search (opt-in)."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from pipeline_core.discovery.sers_falsification_search_m11 import (
    SCHEMA, audit_generated, build_frozen_experiment, claim_csv, digest,
    file_sha, make_prompt, render_report, write_new,
)


def _read(p:Path)->dict:
    return json.loads(p.read_text(encoding='utf-8'))


def _cfg(plan: Path)->dict:
    cfg=_read(plan)
    if cfg.get('schema_version')!=SCHEMA or cfg.get('science_policy',{}).get('novelty_certified') is not False:
        raise ValueError('Plan does not authorize scientific claims')
    if not isinstance(cfg.get('search_tasks'),list) or len(cfg['search_tasks'])!=3:
        raise ValueError('Expected exactly three independent bounded search tasks')
    if cfg.get('experiment_id') != 'm11:'+digest({k:v for k,v in cfg.items() if k!='experiment_id'})[:20]:
        raise ValueError('Plan tampered: experiment_id mismatch')
    return cfg


def prepare(args):
    for p in (args.m10_packet,args.m10_review):
        if not p.is_file():raise FileNotFoundError(p)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('Refuse to overwrite existing experiment directory')
    cfg=build_frozen_experiment(args.m10_packet,args.m10_review)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    write_new(args.out_dir/'M11_FROZEN_EXPERIMENT.json',cfg)
    write_new(args.out_dir/'M11_CLAIM_DISPOSITIONS.csv',claim_csv(cfg))
    write_new(args.out_dir/'M11_ADVERSARIAL_SCIENCE_REPORT.md',render_report(cfg))
    write_new(args.out_dir/'M11_CALCULATED_COUNTEREXAMPLES.json',cfg['calculations'])
    prompts=[]
    for t in cfg['search_tasks']:
        p=make_prompt(cfg,t)
        prompts.append({'task_id':t['task_id'],'sha256':digest(p),'prompt_chars':len(p['user'])})
    write_new(args.out_dir/'M11_PREVIEW_MANIFEST.json',{'status':'FALSIFICATION_CHALLENGES_COMPLETE_NO_GENERATION',
      'plan_id':cfg['experiment_id'],'source_checksums':cfg['source_sha256'],
      'prompts':prompts,'max_future_provider_calls':3,
      'source_idea_count':3,'new_generated_ideas':0,
      'novelty_certified':False,'production_population_modified':False})
    print(json.dumps({'status':'M11_FROZEN_SCIENCE_CHALLENGE_PREPARED_NO_API_CALLS',
      'claim_decisions':len(cfg['claim_dispositions']), 'counterexample_families':len(cfg['calculations']),
      'independent_search_tasks':len(cfg['search_tasks']), 'out':str(args.out_dir)},indent=2))


def preview(args):
    cfg=_cfg(args.plan)
    for task in cfg['search_tasks']:
        p=make_prompt(cfg,task)
        print(f"{task['task_id']} role={task['role']} prompt_sha256={digest(p)} words={len(p['user'].split())}")
    print('M11_SCIENCE_SEARCH_PREVIEW_NO_PROVIDER_CALLS')


def _paths(root:Path,task_id:str)->tuple[Path,Path,Path]:
    return (root/'intents'/f'{task_id}.json',root/'receipts'/f'{task_id}.json',root/'content_checks'/f'{task_id}.json')


def paid(args):
    if not (args.allow_paid and args.i_authorize_new_llm_calls):
        raise ValueError('Explicit paid authorization is required')
    if args.max_new_calls not in (1,2,3):raise ValueError('max-new-calls must be in 1..3')
    cfg=_cfg(args.plan);root=args.plan.parent
    key=os.getenv(args.api_key_env)
    if not key: raise RuntimeError(f'{args.api_key_env} not set')
    model_lock={'model':args.model,'base_url':args.base_url,
                'max_output_tokens':args.max_output_tokens,'temperature':0}
    lockfile=root/'M11_MODEL_LOCK.json'
    if lockfile.is_file():
        if _read(lockfile)!=model_lock: raise ValueError('Model parameters mismatch after first attempt')
    else:
        write_new(lockfile,model_lock)
    eligible=[]
    for task in cfg['search_tasks']:
        intent,receipt,content=_paths(root,task['task_id'])
        if receipt.is_file():
            print(f"M11_RECEIPT_EXISTS_NO_RECALL={task['task_id']}")
            continue
        if intent.is_file():
            print(f"M11_INDETERMINATE_INTENT_NO_RECALL={task['task_id']}")
            continue
        eligible.append(task)
    if not eligible:
        print('M11_NO_ELIGIBLE_PAID_CALLS');return
    from openai import OpenAI
    client=OpenAI(api_key=key,base_url=args.base_url,max_retries=0,timeout=180.0)
    used=0
    for task in eligible[:args.max_new_calls]:
        prompt=make_prompt(cfg,task)
        intent,receipt,check=_paths(root,task['task_id'])
        write_new(intent,{'status':'PROVIDER_ATTEMPT_STARTED_NEVER_AUTOMATIC_RECALL',
           'plan_id':cfg['experiment_id'],'task_id':task['task_id'],
           'model':args.model,'prompt_sha256':digest(prompt)})
        try:
            reply=client.chat.completions.create(
                model=args.model,
                messages=[{'role':'system','content':prompt['system']},
                          {'role':'user','content':prompt['user']}],
                temperature=0,max_tokens=args.max_output_tokens)
            used+=1
        except Exception as e:
            print(f"M11_PROVIDER_ERROR_INTENT_PRESERVED_{task['task_id']}: {type(e).__name__}: {e}")
            raise
        text=reply.choices[0].message.content or ''
        reason=reply.choices[0].finish_reason
        usage=reply.usage.model_dump(mode='json') if reply.usage is not None else None
        write_new(receipt,{'task_id':task['task_id'], 'status':'RAW_RECEIPT_SAVED',
          'plan_id':cfg['experiment_id'],'prompt_sha256':digest(prompt),
          'model':args.model,'finish_reason':reason,'usage':usage,'raw':text,
          'scientific_truth_authority':False,'novelty_certified':False})
        assessment=audit_generated(text,task['task_id'])
        assessment.pop('candidate',None) # keep generated raw in immutable receipt; materialization happens offline
        if reason!='stop':assessment['finish_reason_warning']=reason
        write_new(check,assessment)
        print(json.dumps({'task_id':task['task_id'],'receipt_saved':True,
           'science_content_status':assessment['status'],'finish_reason':reason},ensure_ascii=False))
    print(json.dumps({'status':'M11_PAID_SEARCH_RECEIPTS_SAVED_NO_AUTORETRY',
                      'calls_in_this_command':used,'novelty_certified':False},indent=2))


def assemble(args):
    cfg=_cfg(args.plan);root=args.plan.parent
    out=args.out_dir
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refusing to overwrite review output')
    out.mkdir(parents=True,exist_ok=True)
    rows=[]; audits=[]
    for task in cfg['search_tasks']:
        _,receipt,_=_paths(root,task['task_id'])
        if not receipt.is_file():
            audits.append({'task_id':task['task_id'],'status':'NOT_GENERATED'});continue
        r=_read(receipt)
        prompt=make_prompt(cfg,task)
        if r.get('prompt_sha256')!=digest(prompt) or r.get('plan_id')!=cfg['experiment_id']:
            raise ValueError('Saved receipt lineage/prompt mismatch')
        result=audit_generated(r.get('raw',''),task['task_id'])
        if r.get('finish_reason')!='stop':
            result={'task_id':task['task_id'],'status':'INCOMPLETE_FINISH_RAW_SAVED_NOT_SCIENCE_REVIEWED'}
        row={k:v for k,v in result.items() if k!='candidate'}
        row['receipt_sha256']=file_sha(receipt)
        audits.append(row)
        if 'candidate' in result:
            candidate=dict(result['candidate'])
            candidate['review_id']='M11_SEARCH_'+task['task_id']
            candidate['scientific_review_status']='UNREVIEWED_NOT_NOVELTY_CERTIFIED'
            candidate['source_failed_claim_ids']=task['starting_failures']
            candidate['novelty_certified']=False
            rows.append(candidate)
    write_new(out/'M11_NEW_RESEARCH_IDEAS_UNREVIEWED.json',{'schema_version':'m11-search-results-v1',
      'source_experiment_id':cfg['experiment_id'],'new_ideas':rows,
      'receipt_statuses':audits,'novelty_certified':False,'production_graph_mutated':False})
    write_new(out/'M11_RECEIPT_STATUS.json',audits)
    columns=['review_id','task_id','physically_coherent_0_3','strong_null_overcome_0_3',
             'differential_prediction_0_3','independent_identifiability_0_3',
             'prior_art_relation','critical_counterexample','overall_disposition','reviewer_notes']
    s=io.StringIO();w=csv.DictWriter(s,fieldnames=columns);w.writeheader()
    for row in rows:
        w.writerow({'review_id':row['review_id'],'task_id':row['task_id']})
    write_new(out/'M11_INDEPENDENT_SCIENCE_REVIEW_TO_FILL.csv',s.getvalue())
    write_new(out/'M11_FROZEN_CHALLENGES.json',{
      'experiment_id':cfg['experiment_id'], 'claim_dispositions':cfg['claim_dispositions'],
      'counterexample_sha256':digest(cfg['calculations'])})
    summary={'status':'M11_CANDIDATES_SEARCH_ONLY_NOT_NOVELTY_CERTIFIED',
      'valid_generated_count':len(rows),'receipts_saved':len([r for r in audits if 'receipt_sha256' in r]),
      'science_review_automatically_passed':0,'novelty_certified':False,
      'candidate_review_states':audits}
    write_new(out/'M11_ASSEMBLE_SUMMARY.json',summary)
    zip_path=out/'M11_SCIENCE_REVIEW_PACKET.zip'
    with ZipFile(zip_path,'w',ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.is_file() and f!=zip_path:z.write(f,arcname=f.name)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('M11_REVIEW_PACKET='+str(zip_path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--m10-packet',type=Path,required=True)
    p.add_argument('--m10-review',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.set_defaults(func=prepare)
    p=sub.add_parser('preview');p.add_argument('--plan',type=Path,required=True);p.set_defaults(func=preview)
    p=sub.add_parser('generate-paid');p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--model',default='openai/gpt-5.6-luna')
    p.add_argument('--base-url',default='https://openrouter.ai/api/v1')
    p.add_argument('--api-key-env',default='OPENROUTER_API_KEY')
    p.add_argument('--max-output-tokens',type=int,default=4800)
    p.add_argument('--max-new-calls',type=int,default=2)
    p.add_argument('--allow-paid',action='store_true')
    p.add_argument('--i-authorize-new-llm-calls',action='store_true')
    p.set_defaults(func=paid)
    p=sub.add_parser('assemble');p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.set_defaults(func=assemble)
    args=parser.parse_args();args.func(args)

if __name__=='__main__':main()
