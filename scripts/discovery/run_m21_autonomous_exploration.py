"""M21: diverge science -> real independent literature searches -> select -> M20 execution."""
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m21_autonomous_exploration as m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','preview','diverge','recover','literature','select','export-work'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--open-source',type=Path)
    p.add_argument('--m20',type=Path)
    p.add_argument('--work-out',type=Path)
    p.add_argument('--phase',choices=('diverge','select'))
    p.add_argument('--allow-paid',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':
        if a.open_source is None or a.m20 is None:p.error('--open-source and --m20 required')
        result=m.prepare(a.open_source,a.m20,a.out)
    elif a.command=='verify':result=m.verify(a.out)
    elif a.command=='preview':
        if not a.phase:p.error('preview requires --phase')
        result=m.preview(a.out,a.phase)
    elif a.command=='diverge':result=m.paid(a.out,'diverge',allow_paid=a.allow_paid)
    elif a.command=='select':result=m.paid(a.out,'select',allow_paid=a.allow_paid)
    elif a.command=='recover':
        if not a.phase:p.error('recover requires --phase')
        result=m.recover(a.out,a.phase)
    elif a.command=='literature':result=m.search(a.out)
    else:
        if a.work_out is None:p.error('export-work requires --work-out')
        result=m.export_work(a.out,a.work_out)
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False))

if __name__=='__main__':main()
