from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.scientific_research_control.runner import (
    prepare, verify, run, recover, export_native, repair_state
)


def main() -> None:
    p = argparse.ArgumentParser(description='FAKG native autonomous Case 01 research controller')
    sub = p.add_subparsers(dest='cmd',required=True)
    prep = sub.add_parser('prepare',help='freeze genuine Ye paper, FAKG source search, native context; no paid call')
    prep.add_argument('--ye-pages',type=Path,required=True)
    prep.add_argument('--native-context',type=Path,required=True)
    prep.add_argument('--selected-works',type=Path,required=True)
    prep.add_argument('--out',type=Path,required=True)
    prep.add_argument('--model',default='openai/gpt-5.6-luna')
    prep.add_argument('--max-steps',type=int,default=12)
    prep.add_argument('--temperature',type=float,default=0.65)
    prep.add_argument('--max-tokens',type=int,default=7000)
    prep.add_argument('--research-profile',choices=['legacy','mechanistic'],default='legacy')
    prep.add_argument('--prior-episode',type=Path,default=None)
    for name in ('inspect','run','recover','repair-state','export'):
        q=sub.add_parser(name)
        q.add_argument('--out',type=Path,required=True)
        if name=='run':
            q.add_argument('--allow-paid',action='store_true')
            q.add_argument('--max-new-calls',type=int,default=1)
    args=p.parse_args()
    if args.cmd=='prepare':
        result=prepare(out=args.out,pages_file=args.ye_pages,native_context_file=args.native_context,
                      works_file=args.selected_works,model=args.model,max_steps=args.max_steps,
                      temperature=args.temperature,max_tokens=args.max_tokens,
                      research_profile=args.research_profile,prior_episode=args.prior_episode)
    elif args.cmd=='inspect':
        x=verify(args.out)
        s=x['state']
        result={'research_step':s['step'],'max_steps':x['manifest']['max_steps'],
                'ideas':len(s['ideas']),'challenges':len(s['challenges']),
                'pivots':len(s['pivots']),'read_evidence':s['read_ids'],
                'ended':s['ended'],'final_disposition':s['final_disposition'],
                'trajectory':s['trajectory']}
        if 'observation_register' in s:
            result['observations_recorded'] = len(s['observation_register'])
            result['mechanistic_accounts'] = len(s['mechanistic_accounts'])
            result['prior_research_sha256'] = s['prior_research_sha256']
            if s.get('final_record'):
                result['final_scientific_gaps'] = s['final_record'].get('unexplained_observation_ids', [])
                result['level_ii_certified'] = False
    elif args.cmd=='run':
        result=run(args.out,allow_paid=args.allow_paid,max_new_calls=args.max_new_calls)
    elif args.cmd=='recover':result=recover(args.out)
    elif args.cmd=='repair-state':result=repair_state(args.out)
    else:result=export_native(args.out)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
