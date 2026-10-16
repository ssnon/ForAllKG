"""M19-E4 offline independent audit + isolated self/evidence review of E3 Return."""
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_e4_independent_audit as e

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare');a.add_argument('--e3-work',required=True,type=Path);a.add_argument('--out',required=True,type=Path)
    for op in ('verify','audit','inspect','finalize','preview','next','recover'):
        a=sub.add_parser(op);a.add_argument('--out',required=True,type=Path)
        if op in ('preview','next','recover'):a.add_argument('--arm',required=True,choices=e.ARMS)
        if op=='next':
            a.add_argument('--allow-paid',action='store_true')
            a.add_argument('--max-new-paid-attempts',type=int,default=2)
    args=p.parse_args()
    if args.cmd=='prepare':r=e.prepare(e3_work=args.e3_work,out=args.out)
    elif args.cmd=='verify':r=e.verify(args.out)
    elif args.cmd=='audit':r=e.audit(args.out)
    elif args.cmd=='inspect':r=e.inspect(args.out)
    elif args.cmd=='preview':r=e.preview(args.out,args.arm)
    elif args.cmd=='next':r=e.next_step(args.out,args.arm,allow_paid=args.allow_paid,max_new_paid_attempts=args.max_new_paid_attempts)
    elif args.cmd=='recover':r=e.recover(args.out,args.arm)
    else:r=e.finalize(args.out)
    print(json.dumps(r,indent=2,ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
