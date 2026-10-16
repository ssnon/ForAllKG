"""M19 E1 additive reference quarantine overlay; raw-preserving, no automatic retry."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_e1_reference_recovery as r
from pipeline_core.discovery.scientific_research_control import m19_deep_work_e1 as m


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    for command in ('audit','quarantine'):
        p=sub.add_parser(command);p.add_argument('--out',type=Path,required=True)
        p.add_argument('--arm',choices=('B','C'),required=True)
        p.add_argument('--step',type=int,required=True)
        if command=='quarantine':p.add_argument('--accept-invalid-refs',action='store_true')
    for command in ('verify','inspect','recover'):
        p=sub.add_parser(command);p.add_argument('--out',type=Path,required=True)
    for command in ('preview','next','finalize'):
        p=sub.add_parser(command);p.add_argument('--out',type=Path,required=True)
        p.add_argument('--arm',choices=('B','C'),required=True)
        if command=='next':
            p.add_argument('--allow-paid',action='store_true')
            p.add_argument('--max-total-paid-attempts',type=int,default=6)
    a=ap.parse_args()
    if a.cmd=='audit':z=r.audit(a.out,a.arm,a.step)
    elif a.cmd=='quarantine':z=r.quarantine(a.out,a.arm,a.step,accept_invalid_refs=a.accept_invalid_refs)
    elif a.cmd=='verify':z={k:v for k,v in r.verify(a.out).items() if k not in ('steps','frozen')}
    elif a.cmd=='inspect':z=r.inspect(a.out)
    elif a.cmd=='recover':z=r.recover(a.out)
    elif a.cmd=='finalize':z=r.finalize(a.out,a.arm)
    elif a.cmd=='next':z=r.next_step(a.out,a.arm,allow_paid=a.allow_paid,max_total_paid_attempts=a.max_total_paid_attempts)
    else:
        pv=r.preview(a.out,a.arm)
        z={'status':'PREVIEW_NO_CALL','arm':a.arm,'step':pv['next_step'],
           'model':pv['request']['model'],'prompt_sha256':m.base.sha(pv['request']),
           'continuation':a.arm=='C' and pv['next_step']>1,'new_paid_calls':0}
    print(json.dumps(z,ensure_ascii=False,indent=2,sort_keys=True))
if __name__=='__main__':main()
