"""M30R Phase 2A: independent fresh native ResearchDecision/WorkProduct state root."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import m30r_generic_v12 as native

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    s=ap.add_subparsers(dest='cmd',required=True)
    prep=s.add_parser('prepare');prep.add_argument('--packet',required=True,type=Path);prep.add_argument('--out',required=True,type=Path)
    prep.add_argument('--model',default='openai/gpt-5.6-luna');prep.add_argument('--max-decisions',type=int,default=12)
    prep.add_argument('--temperature',type=float,default=0.5);prep.add_argument('--max-tokens',type=int,default=4500)
    for cmd in ('verify','inspect','preview','decision','execute','recover'):
        q=s.add_parser(cmd);q.add_argument('--out',type=Path,required=True)
        if cmd in ('decision','execute'):q.add_argument('--allow-paid',action='store_true')
    a=ap.parse_args()
    if a.cmd=='prepare':result=native.prepare_generic(packet_file=a.packet,out=a.out,model=a.model,max_decisions=a.max_decisions,temperature=a.temperature,max_tokens=a.max_tokens)
    elif a.cmd=='verify':
        v=native.verify(a.out);result={'status':'GENERIC_NATIVE_V12_VERIFIED_READ_ONLY','decisions':v['state']['decision_count'],'works':len(v['state']['work_history']),'science_certified':False}
    elif a.cmd=='inspect':result=native.inspect(a.out)
    elif a.cmd=='preview':
        v=native.verify(a.out);p=v['state'].get('pending_work')
        result={'status':'PREVIEW_NO_CALL','next':('execute' if p and p['status']=='REQUESTED' else 'decision'),
                'ended':v['state']['ended'],'decision_count':v['state']['decision_count'],'pending_work':p}
    elif a.cmd=='decision':result=native.decision(a.out,allow_paid=a.allow_paid,max_new_calls=1)
    elif a.cmd=='execute':result=native.execute(a.out,allow_paid=a.allow_paid,max_new_calls=1)
    else:result=native.recover(a.out)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
