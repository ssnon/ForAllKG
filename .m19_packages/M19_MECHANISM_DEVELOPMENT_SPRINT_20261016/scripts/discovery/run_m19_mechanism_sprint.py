"""CLI for source-frozen M19 mechanistic research A/B comparison."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_mechanism_sprint as sprint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    x=sub.add_parser('prepare');x.add_argument('--out',type=Path,required=True)
    x.add_argument('--model',default=sprint.DEFAULT_MODEL)
    x.add_argument('--temperature',type=float,default=0.5)
    x.add_argument('--max-tokens',type=int,default=4500)
    for name in ('verify','inspect','finalize'):
        z=sub.add_parser(name);z.add_argument('--out',required=True,type=Path)
    for name in ('preview','next','recover'):
        z=sub.add_parser(name);z.add_argument('--out',required=True,type=Path);z.add_argument('--arm',choices=['plain','work'],required=True)
        if name=='next':
            z.add_argument('--allow-paid',action='store_true')
            z.add_argument('--max-new-paid-attempts',type=int,default=2)
    args=p.parse_args()
    if args.cmd=='prepare':r=sprint.prepare(out=args.out,model=args.model,temperature=args.temperature,max_tokens=args.max_tokens)
    elif args.cmd=='verify':
        r=sprint.inspect(out=args.out)
        r['verified']=True
    elif args.cmd=='inspect':r=sprint.inspect(out=args.out)
    elif args.cmd=='finalize':r=sprint.finalize(out=args.out)
    elif args.cmd=='preview':r=sprint.preview(out=args.out,arm=args.arm)
    elif args.cmd=='recover':r=sprint.recover(out=args.out,arm=args.arm)
    elif args.cmd=='next':r=sprint.next_step(out=args.out,arm=args.arm,allow_paid=args.allow_paid,max_new_paid_attempts=args.max_new_paid_attempts)
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
