"""W07 Scientific Continuation (no new scientific claims or automatic paid retries)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import research_work_v17_continuation_runner as r
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as b

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--reference',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    for name in ('decision','execute'):
        p=sub.add_parser(name);p.add_argument('--out',required=True,type=Path)
        p.add_argument('--allow-paid',action='store_true');p.add_argument('--max-new-calls',type=int,default=1)
    for name in ('verify','inspect','recover','preview-decision','preview-work'):
        p=sub.add_parser(name);p.add_argument('--out',required=True,type=Path)
        if name.startswith('preview'):p.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    if a.cmd=='prepare':res=r.prepare(reference=a.reference,out=a.out)
    elif a.cmd=='decision':res=r.decision(a.out,allow_paid=a.allow_paid,max_new_calls=a.max_new_calls)
    elif a.cmd=='execute':res=r.execute(a.out,allow_paid=a.allow_paid,max_new_calls=a.max_new_calls)
    elif a.cmd=='recover':res=r.recover(a.out)
    elif a.cmd=='inspect':res=r.inspect(a.out)
    elif a.cmd=='verify':
        c=r.verify(a.out)
        res={'status':'VERIFIED_NO_CALL','decision_count':c['state']['decision_count'],
            'event_count':c['state']['event_count'],'state_sha256':b.sha(c['state']),
            'focus_sha256':b.sha(c['focus_state']),
            'seed_count':len(c['seed_state']['registry'])}
    else:
        c=r.verify(a.out)
        payload=r.decision_payload(c) if a.cmd=='preview-decision' else b.work_payload(c)
        b.write_new(a.output,payload)
        res={'status':'PROMPT_PREVIEW_EXPORTED_NO_CALL','file':str(a.output),'sha256':b.sha(payload)}
    print(json.dumps(res,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=='__main__':main()
