"""Inspect, recover and continue previously paid open-literature research safely."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import open_dual_json_literature_recovery as recover
from pipeline_core.discovery.scientific_research_control import open_literature_research as engine

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('inspect','verify','recover','preview','next','finish'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=engine.ARM_NAMES,default='literature_enabled')
    p.add_argument('--allow-paid',action='store_true')
    args=p.parse_args()
    if args.command=='inspect': result=recover.inspect(args.out,args.arm)
    elif args.command=='verify': result=recover.verify(args.out)
    elif args.command=='recover': result=recover.recover(args.out,args.arm)
    elif args.command=='preview': result=recover.preview(args.out,args.arm)
    elif args.command=='next': result=recover.next_step(args.out,args.arm,allow_paid=args.allow_paid)
    else: result=recover.finish(args.out,args.arm)
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=='__main__':main()
