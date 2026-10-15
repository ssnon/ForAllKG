"""W05: unverified hypothesis formation and development, not Level II certification.

Policy only: no physics-specific proposals, no automatic idea insertion, no
mechanical semantic judgments, no changes to ResearchIdea or replay contracts.
"""
from __future__ import annotations
from typing import Any

POLICY_ID = 'hypothesis-formation-development-v1.5.0'

POLICY_SYSTEM_APPEND = '''\nHYPOTHESIS FORMATION AND DEVELOPMENT (scientific exploration, not certification):
You may PRESERVE a physically motivated, meaningful, conceptually different causal conjecture as an UNVERIFIED ResearchIdea even when its mechanism is not established and strong alternative accounts have not yet been excluded. Do not confuse 'not demonstrated' with 'disproved'; likewise, lack of a known counterexample does not prove viability or novelty. The underlying physical coupling or transition pathway must be specified enough to research, not merely named.
AFTER interpreting each completed REASON work, consciously compare four voluntary paths:
(1) PRESERVE: if a specific physically coherent and potentially testable explanation was developed, select idea_update PROPOSE with the EXISTING IdeaDraft schema (causal process; state variables/coupling; a clearly CONDITIONAL differentiating prediction; a genuine strongest compatible conventional null; unknown assumptions). Make clear what null can mimic, which assumptions differ, and what would defeat the proposal. A conditional prediction is allowed, but a generic trend shared by the null is not a differentiating prediction. No proof of superiority or empirical validation is required for *unverified* preservation. The explanation must not contradict established physics or known source controls without addressing the contradiction.
(2) DEVELOP FIRST: if the idea is promising but lacks a credible physical transition mechanism or conditional discriminator, request a sharply targeted REASON work on the missing coupling, explicit dynamical model, derived prediction, or strongest null. It need not be discarded while it is in a completed WorkProduct. Do not fabricate missing results just to create an idea.
(3) DECLINE OR CHANGE DIRECTION: if a construction has a known fatal counterexample, no meaningful causal content, is merely a relabeling, or has no plausible way to become testable, document the reason and select another research action or END. Absence of independent data alone is not a fatal counterexample.
(4) DEVELOP AN EXISTING IDEA: when a registered idea exists, preferentially consider a targeted null CHALLENGE, a substantial REVISE, or a genuinely new-family PIVOT. An idea surviving a challenge remains UNVERIFIED; do not claim science certification.
No forced PROPOSE, no automatic work->idea transformation, no quotas, and no automatic rejection on INCONCLUSIVE. A known, concrete physical contradiction *is* a reason not to preserve a specific invalid formulation; uncertainty about feasibility is a reason to spell out the necessary test.
IMPORTANT identity rule: the ID for a new PROPOSE is created only upon committing its Decision. Keep work_request.target_idea_id null in that very decision, and use the registered ID starting with the subsequent Decision. Even after PROPOSE, choose future work by scientific value, not by a mandatory CHALLENGE schedule. Preserve all preceding WorkInterpretations and existing v1.4 breadth/depth/evidence choice. This policy is not an empirical novelty oracle.'''


def formation_checkpoint(state: dict, max_decisions: int) -> dict[str, Any]:
    """Deterministic, non-semantic policy metadata from only the current episode."""
    p = state['pending_work']
    last = next((w for w in reversed(state['work_history'])
                 if p and w['work_id'] == p['work_id']), None)
    reason = bool(last and p and p['status'] == 'EXECUTED' and
                  last['work_kind'] == 'REASON')
    mutations = state['idea_history']
    return {
        'status':'ADVISORY_NOT_A_HARD_GATE',
        'completed_reason_awaiting_interpretation':reason,
        'preceding_completed_reason_artifact': (
            {'work_id':last['work_id'], 'artifact_ref': last['work_id']+':product',
             'product_sha256':last['product_sha256'],
             'work_conclusion':last['work_conclusion']} if reason else None),
        'registered_idea_count':len(state['idea_registry']),
        'registered_idea_ids':sorted(state['idea_registry']),
        'idea_mutations_so_far':len(mutations),
        'recent_idea_actions':[x['kind'] for x in mutations[-5:]],
        'remaining_decisions':max_decisions-state['decision_count'],
        'admission_question': ('If the preceding REASON constructed a scientifically meaningful, '
              'physically plausible, conceptually different causal possibility with NO KNOWN fatal '
              'contradiction, is it sufficiently specified for an UNVERIFIED ResearchIdea now, '
              'or what focused work is required to make it testable?'),
        'options':{
            'PRESERVE_UNVERIFIED':('Optional PROPOSE: concrete physical process and states, '
              'conditional distinguishing prediction, strongest compatible null, and explicit assumptions.'),
            'DEVELOP_FIRST':('Optional focused REASON to specify mechanism, check physical feasibility, '
              'derive nontrivial prediction, or challenge the strongest null; no immediate proposal required.'),
            'DECLINE_OR_REDIRECT':('If merely a label, already refuted, physically incoherent, '
              'or hopelessly unfalsifiable, say why and choose other research.'),
            'CHALLENGE_OR_REVISE_EXISTING':('If ideas exist, test a strong counterexample '
              'and improve, challenge, or pivot with exact registered idea IDs.')},
        'provisional_is_not_level_ii': True,
        'automatic_idea_mutation':False,
        'strong_null_rule':('Challenge not only a weak stationary baseline but the strongest '
              'physically compatible independent, reversible, heterogeneous, or correlated '
              'alternative as applicable; do not assert uniqueness before comparison.'),
        'guardrails':('No invented work, no invented source support, no invented idea IDs; '
              'unknown physics stays an explicit unresolved assumption; no forced output.')
    }


def decorate_decision_payload(payload: dict, state: dict, max_decisions: int) -> dict:
    result=dict(payload)
    result['hypothesis_formation_checkpoint']=formation_checkpoint(state,max_decisions)
    return result
