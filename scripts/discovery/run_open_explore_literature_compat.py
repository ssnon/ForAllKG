"""Compatibility CLI for saved EXPLORE + FAKG literature tool research step."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import open_explore_literature_compat as compat
from pipeline_core.discovery.scientific_research_control import open_literature_research as open_work

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('inspect','verify','recover','preview','next','finish'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=open_work.ARM_NAMES,default='literature_enabled')
    p.add_argument('--allow-paid',action='store_true')
    args=p.parse_args()
    if args.command=='inspect':result=compat.inspect(args.out,args.arm)
    elif args.command=='verify':result=compat.verify(args.out)
    elif args.command=='recover':result=compat.recover(args.out,args.arm)
    elif args.command=='preview':result=compat.preview(args.out,args.arm)
    elif args.command=='next':result=compat.next_step(args.out,args.arm,allow_paid=args.allow_paid)
    else:result=compat.finish(args.out,args.arm)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=='__main__':main()
