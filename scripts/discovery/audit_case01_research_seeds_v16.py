"""Read-only factual audit of W06 seed lineage. Never certifies science."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import research_work_v16_seed_runner as work
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as base

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--episode',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    c=work.verify(a.episode);s=c['state'];seed=c['seed_state']
    rows=[{'seed_id':k,'status':v['status'],'versions':len(v['versions']),
           'native_idea_id':v['native_idea_id'],
           'origin_work_id':v['versions'][0]['supporting_work_id']}
          for k,v in sorted(seed['registry'].items())]
    result={'policy_id':work.POLICY_ID,'outcome':'ENDED' if s['ended'] else 'ONGOING',
       'decision_count':s['decision_count'],'event_count':s['event_count'],
       'native_idea_count':len(s['idea_registry']),'seed_count':len(rows),
       'seed_mutation_count':len(seed['history']),'seed_inventory':rows,
       'seed_history':seed['history'],'work_kinds':{kind:sum(1 for w in s['work_history'] if w['work_kind']==kind)
              for kind in ('REASON','READ_FROZEN_SOURCE')},
       'science_certified':False,
       'note':'Seed records and model claims are unverified. Descriptive episode analysis only; blind physical review still required.'}
    base.write_new(a.output,result)
    print(json.dumps({'status':'W06_DESCRIPTIVE_AUDIT_WRITTEN_NO_CALL',
                      'output':str(a.output),'seed_count':len(rows)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
