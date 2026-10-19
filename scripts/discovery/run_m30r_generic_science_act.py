"""M30-R Phase 2B: generic native v1.2 Decision/Work + sealed optional scientific instrument.

NOT the full W08 v1.8 Seed/Focus Controller. Use a fresh run. All scientific claims unverified.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m30r_generic_science_act as m

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--packet',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--tool-task',choices=sorted(m.ALLOWED_TASKS))
    p.add_argument('--model',default='openai/gpt-5.6-luna');p.add_argument('--max-decisions',default=8,type=int)
    for n in ['verify','preview','next-decision','prepare-work','next-work','recover-work','finish-work','commit-work','recover-native']:
        q=sub.add_parser(n);q.add_argument('--out',required=True,type=Path)
        if n in ['next-decision','next-work']:q.add_argument('--allow-paid',action='store_true')
    a=ap.parse_args()
    if a.cmd=='prepare':r=m.prepare(a.packet,a.out,tool_task=a.tool_task,model=a.model,max_decisions=a.max_decisions)
    elif a.cmd=='verify':r=m.verify(a.out)
    elif a.cmd=='preview':r=m.preview(a.out)
    elif a.cmd=='next-decision':r=m.next_decision(a.out,allow_paid=a.allow_paid)
    elif a.cmd=='prepare-work':r=m.prepare_work(a.out)
    elif a.cmd=='next-work':r=m.next_work(a.out,allow_paid=a.allow_paid)
    elif a.cmd=='recover-work':r=m.recover_work(a.out)
    elif a.cmd=='finish-work':r=m.finish_work(a.out)
    elif a.cmd=='commit-work':r=m.commit_work(a.out)
    elif a.cmd=='recover-native':r=m.native.recover(m.native_path(a.out))
    else:raise RuntimeError('UNKNOWN_COMMAND')
    print(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True))
if __name__=='__main__':main()
