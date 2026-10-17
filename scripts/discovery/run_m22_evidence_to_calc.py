"""M22 independent evidence-to-calculation experiment (never modifies prior science)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m22_evidence_to_calc as m

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','preview','plan','recover','fetch','fork-fetch','analyze','report'))
    parser.add_argument('--m21',type=Path)
    parser.add_argument('--work',type=Path)
    parser.add_argument('--source',type=Path,help='Existing M22 run for fork-fetch')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase',choices=('plan','analyze'))
    parser.add_argument('--allow-paid',action='store_true')
    parser.add_argument('--pdf-file',type=Path,help='Optional user-supplied fulltext PDF for fetch')
    parser.add_argument('--paper-id',help='M21 paper ID corresponding to --pdf-file')
    args=parser.parse_args()
    if args.command=='prepare':
        if args.m21 is None or args.work is None:parser.error('prepare needs --m21 and --work')
        result=m.prepare(args.m21,args.work,args.out)
    elif args.command=='verify':result=m.verify(args.out)
    elif args.command=='preview':
        if args.phase is None:parser.error('preview needs --phase')
        result=m.preview(args.out,args.phase)
    elif args.command in ('plan','analyze'):
        result=m.paid(args.out,args.command,allow_paid=args.allow_paid)
    elif args.command=='recover':
        if args.phase is None:parser.error('recover needs --phase')
        result=m.recover(args.out,args.phase)
    elif args.command=='fetch':result=m.fetch(args.out,local_pdf=args.pdf_file,paper_id=args.paper_id)
    elif args.command=='fork-fetch':
        if args.source is None:parser.error('fork-fetch requires --source')
        result=m.fork_after_failed_fetch(args.source,args.out)
    elif args.command=='report':result=m.report(args.out)
    else:raise AssertionError(args.command)
    print(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
