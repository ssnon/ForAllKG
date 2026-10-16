"""M19 E1: read-only W08 Work3 reconstruction and B/C scientific Work comparison."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_deep_work_e1 as r


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    subs=ap.add_subparsers(dest='cmd',required=True)
    p=subs.add_parser('prepare');p.add_argument('--w08',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    for name in ('verify','inspect','recover'):
        p=subs.add_parser(name);p.add_argument('--out',type=Path,required=True)
    for name in ('preview','finalize','next'):
        p=subs.add_parser(name);p.add_argument('--out',type=Path,required=True)
        p.add_argument('--arm',required=True,choices=('B','C'))
        if name=='next':
            p.add_argument('--allow-paid',action='store_true')
            p.add_argument('--max-total-paid-attempts',type=int,required=True)
    a=ap.parse_args()
    if a.cmd=='prepare':result=r.prepare(w08=a.w08,out=a.out)
    elif a.cmd=='verify':
        z=r.verify(a.out);result={k:v for k,v in z.items() if k not in ('steps','frozen')}
    elif a.cmd=='inspect':result=r.inspect(a.out)
    elif a.cmd=='recover':result=r.recover(a.out)
    elif a.cmd=='preview':
        p=r.preview(a.out,a.arm)
        # Safe to show model input from original frozen W08. Do not print source corpus by default.
        result={'status':'PREVIEW_NO_CALL','arm':a.arm,'step':p['next_step'],
            'model':p['request']['model'],'prompt_sha256':r.base.sha(p['request']),
            'continuation':a.arm=='C' and p['next_step']>1,'new_paid_calls':0}
    elif a.cmd=='next':result=r.next_step(a.out,a.arm,allow_paid=a.allow_paid,
                                   max_total_paid_attempts=a.max_total_paid_attempts)
    else:result=r.finalize(a.out,a.arm)
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))
if __name__=='__main__':main()
