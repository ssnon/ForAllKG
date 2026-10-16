"""Fresh autonomous Ye research opening with paired live/no-new-literature Work arms."""
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import open_literature_research as a

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    x=sub.add_parser('prepare');x.add_argument('--reference-native',type=Path,required=True);x.add_argument('--out',type=Path,required=True)
    for command in ('verify','preview-opening','choose','recover-opening','prepare-arms'):
        x=sub.add_parser(command);x.add_argument('--out',type=Path,required=True)
        if command=='choose':x.add_argument('--allow-paid',action='store_true')
    for command in ('preview','next','recover','finish'):
        x=sub.add_parser(command);x.add_argument('--out',type=Path,required=True)
        x.add_argument('--arm',required=True,choices=a.ARM_NAMES)
        if command=='next':x.add_argument('--allow-paid',action='store_true')
    x=p.parse_args()
    if x.cmd=='prepare':r=a.prepare(x.reference_native,x.out)
    elif x.cmd=='verify':r=a.verify(x.out)
    elif x.cmd=='preview-opening':r=a.preview_opening(x.out)
    elif x.cmd=='choose':r=a.choose(x.out,allow_paid=x.allow_paid)
    elif x.cmd=='recover-opening':r=a.recover_opening(x.out)
    elif x.cmd=='prepare-arms':r=a.prepare_arms(x.out)
    elif x.cmd=='preview':r=a.preview(x.out,x.arm)
    elif x.cmd=='next':r=a.next_step(x.out,x.arm,allow_paid=x.allow_paid)
    elif x.cmd=='recover':r=a.recover(x.out,x.arm)
    elif x.cmd=='finish':r=a.finish(x.out,x.arm)
    print(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True))
if __name__=='__main__':main()
