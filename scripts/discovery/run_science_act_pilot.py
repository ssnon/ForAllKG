"""CLI for standalone, opt-in scientific investigation behavior pilot."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import science_act_pilot as p

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    x=sub.add_parser('prepare');x.add_argument('--out',required=True,type=Path)
    x.add_argument('--model',default=p.MODEL);x.add_argument('--temperature',type=float,default=0.5)
    x.add_argument('--max-tokens',type=int,default=4500)
    for cmd in ('verify','inspect','finalize'):
        x=sub.add_parser(cmd);x.add_argument('--out',required=True,type=Path)
    for cmd in ('preview','next','recover'):
        x=sub.add_parser(cmd);x.add_argument('--out',required=True,type=Path)
        x.add_argument('--arm',required=True,choices=['plain','instrument'])
        if cmd=='next':
            x.add_argument('--allow-paid',action='store_true');x.add_argument('--max-total-paid-attempts',type=int,default=2)
    a=ap.parse_args()
    if a.cmd=='prepare':o=p.prepare(a.out,model=a.model,temperature=a.temperature,max_tokens=a.max_tokens)
    elif a.cmd=='verify':
        v=p.verify(a.out)
        o={k:({arm:{kk:vv for kk,vv in st.items() if kk!='records'} for arm,st in val.items()} if k=='arms' else val) for k,val in v.items()}
    elif a.cmd=='inspect':o=p.inspect(a.out)
    elif a.cmd=='finalize':o=p.finalize(a.out)
    elif a.cmd=='preview':o=p.preview(a.out,a.arm)
    elif a.cmd=='recover':o=p.recover(a.out,a.arm)
    elif a.cmd=='next':o=p.next_step(a.out,a.arm,allow_paid=a.allow_paid,max_total_paid_attempts=a.max_total_paid_attempts)
    else:raise RuntimeError('unknown action')
    print(json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=='__main__':main()
