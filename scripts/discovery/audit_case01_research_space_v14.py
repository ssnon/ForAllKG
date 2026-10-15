"""Read-only descriptive baseline comparison. NOT scientific novelty evaluation."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import research_work_v14_space_runner as new
from pipeline_core.discovery.scientific_research_control import research_work_v12_stable_runner as stable


def summarize(s):
    acts=Counter(x['action'] for x in s['decisions'])
    kinds=Counter(w['work_kind'] for w in s['work_history'])
    deltas=Counter(x['delta']['kind'] for x in s['knowledge_history'])
    drafts=[]
    for idea in s['idea_registry'].values():
        v=idea['versions'][-1]['draft']
        drafts.append({'id':idea['idea_id'],'status':idea['status'],
                       'causal_family':v['causal_family'],
                       'title':v['title'],'versions':len(idea['versions']),
                       'strong_null':v['strong_null'],
                       'differentiating_prediction':v['differentiating_prediction'],
                       'unresolved_assumptions':v['unresolved_assumptions']})
    return {'decision_count':s['decision_count'],'ended':s['ended'],
            'final_disposition':s['final_disposition'],'end_reason':s['research_end_reason'],
            'decision_actions':dict(acts),'work_kinds':dict(kinds),
            'knowledge_delta_kinds':dict(deltas),
            'read_evidence_count':len(s['read_ids']),
            'interpretation_count':len(s['interpretations']),
            'registered_ideas':len(s['idea_registry']),
            'idea_mutation_count':len(s['idea_history']),
            'idea_mutations':[x['kind'] for x in s['idea_history']],
            'ideas_unverified':drafts,
            'research_question_trajectory':[d['question'] for d in s['decisions']]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline-trace',type=Path,required=True)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    base=json.loads(a.baseline_trace.read_text(encoding='utf-8'))
    current=new.verify(a.episode)
    src=base.get('read_ids',[])
    manifest=current['manifest']
    baseline_s={k:base[k] for k in ('decisions','work_history','knowledge_history','interpretations',
           'idea_registry','idea_history','read_ids','ended','final_disposition','research_end_reason','decision_count')}
    summary=summarize(current['state'])
    baseline_summary=summarize(baseline_s)
    note={
      'baseline_trace':str(a.baseline_trace), 'policy_episode':str(a.episode),
      'frozen_source_snapshot_sha256':manifest['source_snapshot_sha256'],
      'policy_model':manifest['model'],
      'policy_id':new.POLICY_ID,
      'baseline':baseline_summary,'policy':summary,
      'evaluation_boundary':('Descriptive only: counts, labels, and model-generated drafts cannot establish '
                             'physical novelty or Level II. Compare conceptual causal differences and '
                             'discriminating predictions against strong conventional nulls by blind review.'),
      'baseline_not_identical_protocol':('W03F was stabilized after Decision 6, whereas this new episode '
                                         'uses exploration policy from Decision 1. Source inventory and '
                                         'settings are copied, but policy prompts, prior random samples '
                                         'and research histories differ.'),
      'not_a_science_certificate':True,
      'new_paid_calls_this_command':0}
    if a.output.exists():
        raise FileExistsError('audit output exists: use a new filename, never overwrite')
    stable.write_new(a.output,note)
    print(json.dumps({'status':'READ_ONLY_DESCRIPTIVE_AUDIT_WRITTEN','path':str(a.output),
                      'baseline_ideas':baseline_summary['registered_ideas'],
                      'policy_ideas':summary['registered_ideas'],
                      'new_paid_calls':0},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
