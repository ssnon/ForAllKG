"""M19-E3 isolated W08 Decision4: native research choice, no evaluator solution."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_e3_autonomy as e


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='cmd',required=True)
    prepare=subs.add_parser('prepare')
    prepare.add_argument('--w08',type=Path,required=True)
    prepare.add_argument('--out',type=Path,required=True)
    for name in ('verify','inspect','preview','decision','recover'):
        p=subs.add_parser(name);p.add_argument('--out',type=Path,required=True)
        if name=='decision':p.add_argument('--allow-paid',action='store_true')
    a=parser.parse_args()
    if a.cmd=='prepare':res=e.prepare(source=a.w08,out=a.out)
    elif a.cmd=='verify' or a.cmd=='inspect':res=e.verify(a.out)
    elif a.cmd=='preview':res=e.preview(a.out)
    elif a.cmd=='decision':res=e.decision(a.out,allow_paid=a.allow_paid)
    else:res=e.recover(a.out)
    print(json.dumps(res,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=='__main__':main()
