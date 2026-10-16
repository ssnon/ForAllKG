"""FAKG scientific execution: existing autonomous opening, optional real CTMC/literature Work."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import scientific_execution_work as a

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','preview','next','recover','finish','trajectory'))
    p.add_argument('--source',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-paid',action='store_true')
    x=p.parse_args()
    if x.command=='prepare':
        if x.source is None:p.error('--source required for prepare')
        r=a.prepare(x.source,x.out)
    elif x.command=='verify':r=a.verify(x.out)
    elif x.command=='preview':r=a.preview(x.out)
    elif x.command=='next':r=a.next_step(x.out,allow_paid=x.allow_paid)
    elif x.command=='recover':r=a.recover(x.out)
    elif x.command=='finish':r=a.finish(x.out)
    else:r=a.trajectory(x.out)
    print(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True))
if __name__=='__main__':main()
