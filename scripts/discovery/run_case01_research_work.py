"""Opt-in Research Work CLI. This never opens/mutates R01/R02/GAP/MEM episodes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import research_work_runner as work


def main():
    ap=argparse.ArgumentParser(description='FAKG Research Work: decision, execute, interpret next decision')
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--ye-pages',required=True,type=Path)
    p.add_argument('--selected-works',required=True,type=Path)
    p.add_argument('--native-context',required=True,type=Path)
    p.add_argument('--model',default='openai/gpt-5.6-luna')
    p.add_argument('--max-decisions',type=int,default=12)
    p.add_argument('--temperature',type=float,default=0.5)
    p.add_argument('--max-tokens',type=int,default=4500)
    for name in ('decision','execute'):
        q=sub.add_parser(name)
        q.add_argument('--out',required=True,type=Path)
        q.add_argument('--allow-paid',action='store_true')
        q.add_argument('--max-new-calls',type=int,default=1)
    for name in ('inspect','verify','recover','preview-decision','preview-work','block-work','fail-work'):
        q=sub.add_parser(name)
        q.add_argument('--out',required=True,type=Path)
        if name in ('block-work','fail-work'):
            q.add_argument('--reason',required=True)
        if name.startswith('preview'):
            q.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    if a.cmd=='prepare':
        result=work.prepare(out=a.out,ye_pages=a.ye_pages,selected_works=a.selected_works,
                native_context=a.native_context,model=a.model,max_decisions=a.max_decisions,
                temperature=a.temperature,max_tokens=a.max_tokens)
    elif a.cmd=='decision':
        result=work.decision(a.out,allow_paid=a.allow_paid,max_new_calls=a.max_new_calls)
    elif a.cmd=='execute':
        result=work.execute(a.out,allow_paid=a.allow_paid,max_new_calls=a.max_new_calls)
    elif a.cmd=='recover':
        result=work.recover(a.out)
    elif a.cmd in ('block-work','fail-work'):
        result=work.record_work_failure(a.out,execution_status='BLOCKED' if a.cmd=='block-work' else 'FAILED',reason=a.reason)
    elif a.cmd=='inspect':
        result=work.inspect(a.out)
    elif a.cmd=='verify':
        checked=work.verify(a.out)
        result={'status':'VERIFIED_NO_CALL','event_count':checked['state']['event_count'],
                'decision_count':checked['state']['decision_count'],
                'state_sha256':work.sha(checked['state'])}
    else:
        checked=work.verify(a.out)
        payload=(work.decision_payload(checked) if a.cmd=='preview-decision' else work.work_payload(checked))
        work.write_new(a.output,payload)
        result={'status':'PROMPT_PAYLOAD_EXPORTED_NO_CALL','file':str(a.output),
                'payload_sha256':work.sha(payload)}
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=='__main__':
    main()
