"""M19-E3 autonomous native D4 WorkRequest -> isolated bounded Deep Work."""
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m19_e3_deep_work as w

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare');a.add_argument('--e3',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    for name in ('verify','inspect','preview','next','recover','finalize'):
        a=sub.add_parser(name);a.add_argument('--out',type=Path,required=True)
        if name=='next':
            a.add_argument('--allow-paid',action='store_true')
            a.add_argument('--max-total-paid-attempts',type=int,default=3)
    a=p.parse_args()
    if a.cmd=='prepare':r=w.prepare(e3=a.e3,out=a.out)
    elif a.cmd=='verify':r=w.verify(a.out);r={k:v for k,v in r.items() if k!='steps'}
    elif a.cmd=='inspect':r=w.inspect(a.out)
    elif a.cmd=='preview':r=w.preview(a.out)
    elif a.cmd=='next':r=w.next_step(a.out,allow_paid=a.allow_paid,max_total_paid_attempts=a.max_total_paid_attempts)
    elif a.cmd=='recover':r=w.recover(a.out)
    else:r=w.finalize(a.out)
    print(json.dumps(r,ensure_ascii=False,sort_keys=True,indent=2))
if __name__=='__main__':main()
