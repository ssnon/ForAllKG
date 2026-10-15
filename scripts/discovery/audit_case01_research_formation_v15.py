"""Read-only W04 -> W05 comparison; no semantic novelty certification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as stable
from pipeline_core.discovery.scientific_research_control import research_work_v15_formation_runner as current
from scripts.discovery.audit_case01_research_space_v14 import summarize


def summarize_formation(s):
    ans=summarize(s)
    # This is an EVENT-BASED linkage, not a classification of scientific quality.
    interpreted={x['work_id'] for x in s['interpretations']}
    reason_ids={w['work_id'] for w in s['work_history'] if w['work_kind']=='REASON'}
    mutations=[{'decision_number':x['decision_number'],'kind':x['kind'],
        'preceding_work_id':x['preceding_work_id'],
        'preceding_work_was_interpreted_reason':x['preceding_work_id'] in (reason_ids & interpreted),
        'target_idea_id':x['target_idea_id'],
        'created_idea_id':x.get('created_idea_id'),
        'new_version':x.get('new_version')} for x in s['idea_history']]
    ans['formation_handoffs']=mutations
    ans['candidate_certification']=False
    ans['unverified_idea_versions']=sum(len(v['versions']) for v in s['idea_registry'].values())
    return ans


def audit(*,baseline_trace: Path, episode:Path, output:Path):
    if output.exists():raise FileExistsError(f'audit output already exists: {output}')
    prior=json.loads(baseline_trace.read_text(encoding='utf-8'))
    c=current.verify(episode)
    # A completed W04 baseline is mandatory; no migration of its results.
    if prior.get('policy',{}).get('policy_id')!='research-space-exploration-v1.4.0':
        raise ValueError('baseline trace is not the v1.4 policy episode')
    if prior.get('outcome')!='ENDED' or prior.get('decision_count')!=12:
        raise ValueError('W04 baseline not finalized')
    if prior.get('policy',{}).get('source_snapshot_sha256')!=stable.sha(c['sources']):
        raise ValueError('W04/W05 source snapshots differ')
    if c['manifest']['model']!=prior['policy']['model']:
        raise ValueError('W04/W05 model settings differ')
    state={k:prior[k] for k in ('decisions','work_history','knowledge_history','interpretations',
      'idea_registry','idea_history','read_ids','ended','final_disposition','research_end_reason','decision_count')}
    report={'not_a_novelty_or_level_ii_certificate':True,
      'comparison_scope':'One fresh research run per policy. Descriptive, not causal proof.',
      'baseline_trace':str(baseline_trace),'episode':str(episode),
      'frozen_sources_sha256':stable.sha(c['sources']),
      'model':c['manifest']['model'],
      'policy_id':current.POLICY_ID,
      'baseline':summarize_formation(state),'current':summarize_formation(c['state']),
      'manual_review_rule':('Judge genuine causal difference, physical coupling, independent '
           'strong null, conditional nontrivial falsification, and assumption transparency. '
           'Neither PROPOSE count nor model self-review certifies discovery.'),
      'new_paid_calls_this_audit':0}
    stable.write_new(output,report)
    return {'status':'READ_ONLY_FORMATION_AUDIT_WRITTEN', 'path':str(output),
      'baseline_ideas':len(state['idea_registry']),
      'w05_ideas':len(c['state']['idea_registry']),'new_paid_calls':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-trace',type=Path,required=True)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit(baseline_trace=a.baseline_trace,episode=a.episode,output=a.output),
      ensure_ascii=False,indent=2))

if __name__=='__main__':main()
