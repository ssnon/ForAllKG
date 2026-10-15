"""Offline audit of prior scientific attempts visible in the next LLM request.

Produces a local JSON report, NOT a research claim. No provider calls.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pipeline_core.discovery.scientific_research_control import runner as r


def audit(episode: Path) -> dict:
    checked=r.verify(episode)
    s=checked['state']; m=checked['manifest']; src=checked['source_index']
    payload=r.prompt_payload(s,src,m,episode_dir=episode)
    memory=payload['scientific_research_memory']
    ctx=payload.get('research_decision_context') or {}
    projection=ctx.get('verified_prior_action_learning')
    rows=r.verified_step_insights(s,episode)
    expected=set(row['step'] for row in rows)
    projected=set(row['step'] for row in projection['entries']) if projection else set()
    result={
        'episode':str(episode),
        'research_step':s['step'],
        'next_step':s['step']+1,
        'memory_mode':m.get('research_memory_projection_mode','OFF'),
        'prompt_payload_sha256':r.digest(payload),
        'full_prior_ideas_in_memory':len(memory.get('ideas',{})),
        'challenge_records_in_memory':len(memory.get('challenges',[])),
        'deepen_records_in_memory':len(memory.get('deepening',[])),
        'pivot_records_in_memory':len(memory.get('pivots',[])),
        'observation_records_in_memory':len(memory.get('observation_register',{})),
        'mechanistic_accounts_in_memory':len(memory.get('mechanistic_accounts',[])),
        'agent_insight_actions_available_from_steps':len(rows),
        'insight_step_numbers_available':sorted(expected),
        'agent_insight_actions_explicitly_in_next_prompt':len(projected),
        'insight_step_numbers_explicitly_in_next_prompt':sorted(projected),
        'missing_insight_step_numbers':sorted(expected-projected),
        'latest_completed_action_insights':next((row for row in reversed(rows) if row['step']==s['step']),None),
        'warning':'Explicit projection is not external validation; other narrative may already appear in ideas, challenges and mechanistic accounts.',
    }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    report=audit(a.episode)
    if a.out:
        if a.out.exists():
            raise FileExistsError('refusing to overwrite prior audit report')
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
