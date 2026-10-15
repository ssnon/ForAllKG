"""Opt-in CLI for W06: REASON -> Seed -> targeted REASON -> Development."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import research_work_v16_seed_runner as work
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as base


def main():
    ap=argparse.ArgumentParser(description='W06 Mechanism Construction & Exploratory Seed v1.6')
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('seed-fresh');p.add_argument('--reference',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    for name in ('decision','execute'):
        p=sub.add_parser(name);p.add_argument('--out',type=Path,required=True)
        p.add_argument('--allow-paid',action='store_true');p.add_argument('--max-new-calls',default=1,type=int)
    for name in ('verify','inspect','recover','preview-decision','preview-work'):
        p=sub.add_parser(name);p.add_argument('--out',type=Path,required=True)
        if name.startswith('preview'):p.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.cmd=='seed-fresh':r=work.seed_fresh(reference=a.reference,out=a.out)
    elif a.cmd=='decision':r=work.decision(a.out,allow_paid=a.allow_paid,max_new_calls=a.max_new_calls)
    elif a.cmd=='execute':r=work.execute(a.out,allow_paid=a.allow_paid,max_new_calls=a.max_new_calls)
    elif a.cmd=='recover':r=work.recover(a.out)
    elif a.cmd=='inspect':r=work.inspect(a.out)
    elif a.cmd=='verify':
        c=work.verify(a.out)
        r={'status':'VERIFIED_NO_CALL','decision_count':c['state']['decision_count'],
           'event_count':c['state']['event_count'],'state_sha256':base.sha(c['state']),
           'seed_count':len(c['seed_state']['registry']),
           'seed_mutations':len(c['seed_state']['history'])}
    else:
        c=work.verify(a.out)
        payload=(work.decision_payload(c) if a.cmd=='preview-decision' else base.work_payload(c))
        base.write_new(a.output,payload)
        r={'status':'PROMPT_PAYLOAD_EXPORTED_NO_CALL','file':str(a.output),
           'payload_sha256':base.sha(payload)}
    print(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=='__main__':main()
