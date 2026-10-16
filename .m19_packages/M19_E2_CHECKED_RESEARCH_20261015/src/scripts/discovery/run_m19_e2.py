"""M19-E2 paired conditional science probe. Paid only with double authorization."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_e2_paired_probe as e

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    for name in ('prepare','preview','next','recover','verify','inspect','calculate','finalize'):
        p=sub.add_parser(name);p.add_argument('--out',type=Path,required=True)
        if name in ('preview','next'):p.add_argument('--phase',choices=e.PHASES,required=True)
        if name=='next':
            p.add_argument('--allow-paid',action='store_true')
            p.add_argument('--max-total-paid-attempts',type=int,default=3)
        if name=='prepare':p.add_argument('--model',default=e.MODEL)
    a=ap.parse_args()
    if a.cmd=='prepare':result=e.prepare(a.out,model=a.model)
    elif a.cmd=='preview':result=e.preview(a.out,a.phase)
    elif a.cmd=='next':result=e.next_step(a.out,a.phase,allow_paid=a.allow_paid,
              max_total_paid_attempts=a.max_total_paid_attempts)
    elif a.cmd=='calculate':result=e.calculate(a.out)
    elif a.cmd=='recover':result=e.recover(a.out)
    elif a.cmd=='finalize':result=e.finalize(a.out)
    else:result=e.verify(a.out)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))
if __name__=='__main__':main()
