"""E2 semantic fork: reuse completed first response, compare no feedback vs exact CTMC."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_e2_semantic_fork as e

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    for name in ('prepare','preview','next','recover','verify','inspect','finalize'):
        p=sub.add_parser(name);p.add_argument('--out',type=Path,required=True)
        if name=='prepare':p.add_argument('--source',type=Path,required=True)
        if name in ('preview','next'):p.add_argument('--phase',choices=e.PHASES,required=True)
        if name=='next':
            p.add_argument('--allow-paid',action='store_true')
            p.add_argument('--max-new-paid-attempts',default=2,type=int)
    a=ap.parse_args()
    if a.cmd=='prepare':result=e.prepare(a.out,a.source)
    elif a.cmd=='preview':result=e.preview(a.out,a.phase)
    elif a.cmd=='next':result=e.next_step(a.out,a.phase,allow_paid=a.allow_paid,
                 max_new_paid_attempts=a.max_new_paid_attempts)
    elif a.cmd=='recover':result=e.recover(a.out)
    elif a.cmd=='finalize':result=e.finalize(a.out)
    else:result=e.verify(a.out)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=='__main__':main()
