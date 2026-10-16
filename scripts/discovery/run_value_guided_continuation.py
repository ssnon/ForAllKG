"""Value-guided FAKG scientific continuation; no arbitrary paid-call budget."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import value_guided_continuation as research

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','verify','preview','next','recover','finish','trajectory'))
    ap.add_argument('--source',type=Path)
    ap.add_argument('--arm',choices=research.KINDS,default='literature_enabled')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--allow-paid',action='store_true')
    args=ap.parse_args()
    if args.command=='prepare':
        if not args.source:ap.error('--source REQUIRED for prepare')
        result=research.prepare(args.source,args.arm,args.out)
    elif args.command=='verify':result=research.verify(args.out)
    elif args.command=='preview':result=research.preview(args.out)
    elif args.command=='next':result=research.next_step(args.out,allow_paid=args.allow_paid)
    elif args.command=='recover':result=research.recover(args.out)
    elif args.command=='finish':result=research.finish(args.out)
    else:result=research.trajectory(args.out)
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=='__main__':main()
