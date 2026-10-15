"""Epistemically neutral *options*, not a discovery mandate or science oracle.

No existing scientific hypotheses or prior-case solutions are imported here.
Research breadth metadata is computed ONLY from the live episode's own events.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

POLICY_ID = 'research-space-exploration-v1.4.0'

# Small enough to coexist with full source and memory payload, without requiring
# higher LLM budgets. No new physical hypothesis, no auto-registration of ideas.
POLICY_SYSTEM_APPEND = '''\nRESEARCH-SPACE CHOICE (options, NOT mandatory outcomes):
You are seeking a discovery-worthy NEW causal account, not merely selecting among mechanisms named in the supplied paper. Do not assume that the paper's list exhausts the explanation space. Scientific novelty must be evaluated, never asserted by naming a candidate.
At each decision compare: (A) deepen or falsify the present causal family; (B) expand the causal space by questioning a shared assumption, introducing a physically motivated state variable/coupling/boundary case, or finding a qualitatively different causal decomposition; (C) gather evidence needed to choose. Choose by expected information gain and the remaining budget, not by any compulsory action quota. A new family may also be a conservative null; test it fairly.
An INCONCLUSIVE result or repeated NO_PROGRESS is not proof that the research question is exhausted. Reconsider the explored space when the same assumptions are being recycled. Conversely, do NOT force a new idea, pivot, or reasoning work if no concrete defensible proposal exists.
When you have a concrete, physically specified causal conjecture with a discriminating prediction, strong conventional null, and explicit missing assumptions, use the OPTIONAL idea_update PROPOSE. If it already exists, CHALLENGE/REVISE/PIVOT only with a valid registered ID and suitable action. A newly PROPOSED idea has no registered ID until that decision commits: keep work_request.target_idea_id null in that same decision and refer to its registered ID on subsequent decisions. A WorkProduct can motivate an idea but is NOT itself verified evidence; do not silently certify it. An EXPLORE action may be a work request without a proposal if genuine exploration is still needed.
For each REASON work, pick a falsifiable research question rather than a generic survey; request an explicit causal construction, strongest compatible conventional null, nontrivial distinguishing consequence or obstruction, and failure condition where appropriate. Do not prejudge which hypothesis wins. Preserve the mandatory WorkInterpretation contract above and scientific honesty.''' 


def digest(v: Any) -> str:
    return hashlib.sha256(json.dumps(v, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _latest_reason_status(state: dict) -> list[dict]:
    result = []
    assessments = {x['work_id']: x for x in state['interpretations']}
    for w in state['work_history']:
        if w['work_kind'] != 'REASON':
            continue
        interp = assessments.get(w['work_id'])
        result.append({'work_id':w['work_id'], 'work_conclusion':w['work_conclusion'],
                       'assessment':interp['assessment'] if interp else 'PENDING_INTERPRETATION',
                       'scientific_delta_kind': ((interp or {}).get('scientific_delta') or {}).get('kind')})
    return result[-5:]


def space_choice_context(state: dict, max_decisions: int) -> dict:
    """Non-authoritative exploration checkpoint; computed without semantic judging."""
    history = state['decisions']
    nonread = [d for d in history if d['action'] not in ('READ_EVIDENCE','END')]
    statuses = _latest_reason_status(state)
    no_progress = 0
    for item in reversed([x for x in statuses if x['assessment'] != 'PENDING_INTERPRETATION']):
        if (item['assessment'] in ('NO_PROGRESS','INSUFFICIENT_EVIDENCE') or
            item['scientific_delta_kind'] in ('NO_PROGRESS','UNRESOLVED')):
            no_progress += 1
        else:
            break
    return {
        'status':'ADVISORY_NOT_A_HARD_GATE',
        'remaining_decisions':max_decisions-state['decision_count'],
        'registered_idea_count':len(state['idea_registry']),
        'registered_idea_ids':sorted(state['idea_registry']),
        'non_read_action_count':len(nonread),
        'recent_reason_results':statuses,
        'consecutive_unresolved_reason_interpretations':no_progress,
        'available_choices':{
            'DEPTH':'Continue, specify or falsify a promising causal account, with a precise new test.',
            'BREADTH':'Examine a different causal decomposition or revisit an assumption shared by current explanations.',
            'EVIDENCE':'Seek a specific missing source-grounded constraint if available and relevant.'
        },
        'decision_principle':('Choose freely by expected learning and causal novelty, not by a forced action count. '
                              'If no ideas are registered after extensive investigation, explicitly consider '
                              'whether the current explanation space is too narrow; registration remains optional.'),
        'idea_registration_rule':('PROPOSE only when there is an actual mechanism, new discriminating '
                                  'prediction, strong null and acknowledged physical assumptions. '
                                  'No unsupported priority, novelty or validation claims.'),
        'science_boundary':'Never treat a WorkProduct or model self-review as independent empirical validation.'
    }


def decorate_decision_payload(payload: dict, state: dict, max_decisions: int) -> dict:
    result = dict(payload)
    result['research_space_choice'] = space_choice_context(state,max_decisions)
    return result
