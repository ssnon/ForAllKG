"""M14-B: auditable, offline-only adaptive scientific search policy.

This is a NEW policy contract, not a retroactive edit of M14's frozen trial.
External scientific evidence MUST be separately verified before it can be scored
as independent information. Decisions/structural checks never certify novelty.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Any

SCHEMA = 'm14b-adaptive-search-reconstruction-v1'
LANES = ('MECHANISM_SEARCH', 'OPEN_FRONTIER')
ACTIONS = ('DISCOVER', 'REFINE', 'BRANCH', 'ESCAPE', 'SUSPEND')
PHYSICS = frozenset(('PHYSICALLY_INVALID', 'NON_IDENTIFIABLE',
    'KNOWN_PHYSICS_PLAUSIBLE', 'UNRESOLVED_TESTABLE', 'UNRESOLVED_UNTESTABLE'))
PRIOR = frozenset(('PRIOR_ART_SUSPECTED', 'KNOWN_PHYSICS_PLAUSIBLE',
    'UNRESOLVED_TESTABLE', 'UNRESOLVED_UNTESTABLE'))
INDEPENDENT_KINDS = frozenset(('EXECUTED_NUMERICAL_MODEL', 'INDEPENDENT_DATA_ANALYSIS',
                               'EMPIRICAL_OBSERVATION'))
SCIENTIFIC_SIGNATURE_FIELDS = ('mechanism_family_id', 'causal_driver', 'core_assumption',
    'predicted_observable', 'independent_intervention', 'strongest_null', 'falsification')


def normalize(x: Any) -> str:
    """Comparable representation; not a scientific ontology or synonym detector."""
    if not isinstance(x, str):
        return ''
    x = unicodedata.normalize('NFKC', x).casefold()
    return re.sub(r'\s+', ' ', x).strip()


def canonical_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(x: Any) -> str:
    return hashlib.sha256(canonical_json(x).encode('utf-8')).hexdigest()


def independent_evidence(evidence: list[dict] | None, *, candidate_id: str) -> list[str]:
    """Only previously verified, candidate-bound external results qualify.

    `verified` is supplied by a separately audited verifier, NOT by the LLM.
    This function does not verify source files or certify experiment quality.
    """
    eligible = []
    for e in evidence or []:
        if not isinstance(e, dict) or e.get('candidate_id') != candidate_id:
            continue
        if (e.get('kind') in INDEPENDENT_KINDS
                and e.get('verified') is True
                and e.get('independent_of_generation') is True
                and isinstance(e.get('artifact_sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', e['artifact_sha256'])
                and isinstance(e.get('artifact_id'), str) and e['artifact_id'].strip()):
            eligible.append(e['artifact_id'])
    return sorted(set(eligible))


def _lane_item(row: dict, lane: str, name: str) -> dict:
    hits = [x for x in row.get(name, []) if x.get('lane') == lane]
    if len(hits) != 1:
        raise ValueError('Expected exactly one '+name+' for lane '+lane)
    return hits[0]


def choose(history: list[dict], lane: str, *, evidence: list[dict] | None = None,
           problem_cards: list[dict] | None = None) -> dict:
    """Offline policy. history contains only PREVIOUS generations of one arm/rep.

    Frozen-history replay is counterfactual diagnostics, NOT a simulated future.
    """
    if lane not in LANES:
        raise ValueError('Unknown lane')
    if not history:
        return {'schema': SCHEMA, 'lane': lane, 'action': 'DISCOVER',
                'reasons': ['FIRST_GENERATION'], 'novelty_certified': False,
                'external_scientific_evidence_certified': False,
                'problem_expansion_required': False,
                'structural_transition_only': True}
    generations = [h.get('generation') for h in history]
    if generations != list(range(1, len(history) + 1)):
        raise ValueError('History must be consecutive previous generations, not future leakage')
    prev = history[-1]
    c = _lane_item(prev, lane, 'candidates')
    r = _lane_item(prev, lane, 'candidate_reviews')
    if r.get('candidate_id') != c.get('candidate_id'):
        raise ValueError('Science review / candidate mismatch')
    ps, ls = r.get('physical_status'), r.get('prior_art_status')
    if ps not in PHYSICS or ls not in PRIOR:
        raise ValueError('Unrecognized scientific review status')
    fam = normalize(c.get('family'))
    if not fam or not c.get('problem_id'):
        raise ValueError('Missing candidate lineage')
    past = [a for h in history for a in h['candidates']]
    family_visits = sum(normalize(a.get('family')) == fam for a in past)
    # Same-lane nonidentifiability streak; a cross-family branch may interrupt it.
    streak = 0
    for h in reversed(history):
        hc = _lane_item(h, lane, 'candidates')
        hr = _lane_item(h, lane, 'candidate_reviews')
        if normalize(hc.get('family')) == fam and hr['physical_status'] == 'NON_IDENTIFIABLE':
            streak += 1
        else:
            break
    sources = independent_evidence(evidence, candidate_id=c['candidate_id'])
    reasons = [f'PHYSICS:{ps}', f'PRIOR_ART:{ls}', f'NORMALIZED_FAMILY_VISITS:{family_visits}',
               f'SAME_LANE_NONIDENTIFIABLE_STREAK:{streak}']
    if r.get('retrieval_missing') is True:
        action = 'SUSPEND'
        reasons.append('MISSING_RETRIEVAL_IS_NOT_A_GAP')
    elif ps == 'PHYSICALLY_INVALID':
        action = 'ESCAPE'
        reasons.append('PHYSICALLY_INVALID_MODEL')
    elif ps == 'UNRESOLVED_UNTESTABLE' and ls == 'UNRESOLVED_UNTESTABLE':
        action = 'SUSPEND'
        reasons.append('INSUFFICIENT_DISCRIMINATING_OBSERVABLE')
    elif (ps == 'KNOWN_PHYSICS_PLAUSIBLE'
            and ls in ('PRIOR_ART_SUSPECTED', 'KNOWN_PHYSICS_PLAUSIBLE')):
        action = 'ESCAPE'
        reasons.append('EXPLANATION_OR_PRIOR_ART_OVERLAP')
    elif ls == 'PRIOR_ART_SUSPECTED':
        action = 'ESCAPE'
        reasons.append('POTENTIAL_PRIOR_ART_OVERLAP_NOT_VERIFIED_KNOWNNESS')
    elif family_visits >= 2 and not sources:
        action = 'ESCAPE'
        reasons.append('STAGNANT_FAMILY_WITHOUT_INDEPENDENT_EVIDENCE')
    elif streak >= 2 and not sources:
        action = 'ESCAPE'
        reasons.append('REPEATED_NONIDENTIFIABILITY')
    elif ps == 'NON_IDENTIFIABLE':
        action = 'BRANCH'
        reasons.append('ONE_CHANCE_TO_BRANCH_FROM_NONIDENTIFIABILITY')
    elif ps == 'UNRESOLVED_TESTABLE' and ls == 'UNRESOLVED_TESTABLE':
        action = 'REFINE'
        reasons.append('PROMISING_BUT_NOVELTY_UNCERTIFIED')
    else:
        action = 'BRANCH'
        reasons.append('EXPLORE_COMPETING_MECHANISM')
    if sources:
        reasons.append('EXTERNAL_EVIDENCE_REFERENCES_REQUIRE_SEPARATE_VERIFICATION')
    # Do not claim ESCAPE if there is no different provisional question/family.
    # The next milestone may add a new problem card rather than picking a recycled one.
    expansion = False
    alternatives = []
    if action == 'ESCAPE' and problem_cards is not None:
        alternatives = [p for p in problem_cards if p.get('problem_id') != c['problem_id']
                        and normalize(p.get('domain_family')) != fam]
        if not alternatives:
            action = 'SUSPEND'
            expansion = True
            reasons.append('NO_DIFFERENT_ANCHORED_PROBLEM_CARD_REQUEST_GAP_DISCOVERY')
    return {'schema': SCHEMA, 'lane': lane, 'action': action, 'reasons': reasons,
            'previous_candidate_id': c['candidate_id'], 'previous_problem_id': c['problem_id'],
            'previous_family': c['family'], 'normalized_family_visits': family_visits,
            'same_lane_nonidentifiability_streak': streak,
            'candidate_bound_independent_evidence_ids': sources,
            'problem_expansion_required': expansion,
            'escape_alternative_problem_ids': sorted(p['problem_id'] for p in alternatives),
            'novelty_certified': False, 'external_scientific_evidence_certified': False,
            'structural_transition_only': True}


def policy_for_next(history: list[dict], *, evidence: list[dict] | None = None,
                    problem_cards: list[dict] | None = None) -> dict:
    lanes = {lane: choose(history, lane, evidence=evidence, problem_cards=problem_cards)
             for lane in LANES}
    return {'schema': SCHEMA, 'next_generation': len(history) + 1,
            'lanes': lanes, 'novelty_certified': False,
            'policy_does_not_claim_discovery': True,
            'evidence_production_pending': not bool(evidence)}


def validate_transition(previous: dict | None, successor: dict | None, decision: dict) -> dict:
    """Strict forward-only M14-C candidate structural gate; no semantic certification.

    New hypotheses must carry explicit mechanism signatures and independently
    discriminating observations. Renaming a mechanism-family label is insufficient.
    """
    action = decision.get('action')
    if action not in ACTIONS or decision.get('schema') != SCHEMA:
        raise ValueError('Unknown M14-B policy decision')
    if action == 'SUSPEND':
        if successor is not None:
            raise ValueError('SUSPEND must not silently generate a successor for the suspended lane')
        return {'accepted': True, 'action': action, 'claims_novelty': False,
                'scientific_semantic_identity_verified': False}
    if successor is None:
        raise ValueError('Missing proposed successor')
    if successor.get('novelty_certified') is not False:
        raise ValueError('Generator cannot certify novelty')
    sig = successor.get('mechanism_signature')
    if not isinstance(sig, dict) or any(not isinstance(sig.get(k), str) or not sig[k].strip()
                                       for k in SCIENTIFIC_SIGNATURE_FIELDS):
        raise ValueError('Incomplete causal mechanism / falsification signature')
    if not successor.get('problem_id') or not successor.get('family'):
        raise ValueError('Missing research problem lineage')
    if action not in ('DISCOVER',) and previous is None:
        raise ValueError('Missing previous hypothesis')
    if action == 'ESCAPE' and 'escape_alternative_problem_ids' in decision:
        if successor['problem_id'] not in decision['escape_alternative_problem_ids']:
            raise ValueError('ESCAPE must choose an independently provided alternate problem card')
    if action in ('BRANCH', 'ESCAPE', 'REFINE'):
        old = previous.get('mechanism_signature')
        if not isinstance(old, dict) or any(not normalize(old.get(k)) for k in SCIENTIFIC_SIGNATURE_FIELDS):
            raise ValueError('Cannot certify a structural transition without a prior structured signature')
        old_driver = normalize(old['causal_driver'])
        new_driver = normalize(sig['causal_driver'])
        old_assumption = normalize(old['core_assumption'])
        new_assumption = normalize(sig['core_assumption'])
        if action == 'REFINE':
            if previous.get('problem_id') != successor['problem_id'] or old_driver != new_driver or old_assumption != new_assumption:
                raise ValueError('REFINE must preserve problem and core causal premise')
            if all(normalize(old.get(k)) == normalize(sig.get(k)) for k in ('predicted_observable', 'independent_intervention', 'falsification')):
                raise ValueError('REFINE must improve at least one measurable prediction, intervention, or falsifier')
        elif action == 'BRANCH':
            if old_driver == new_driver and old_assumption == new_assumption:
                raise ValueError('BRANCH requires a causal driver or assumption change')
        elif action == 'ESCAPE':
            if previous.get('problem_id') == successor['problem_id']:
                raise ValueError('ESCAPE must change scientific problem, not just wording')
            if old_driver == new_driver or old_assumption == new_assumption:
                raise ValueError('ESCAPE must change both causal driver and generating assumption')
            if normalize(previous.get('family')) == normalize(successor['family']):
                raise ValueError('ESCAPE must change nominal family too')
            if normalize(old.get('mechanism_family_id')) == normalize(sig['mechanism_family_id']):
                raise ValueError('ESCAPE must change mechanistic family ID')
            if decision.get('problem_expansion_required'):
                raise ValueError('ESCAPE cannot bypass the missing anchored problem-card requirement')
    return {'accepted': True, 'action': action,
            'transition_content_digest': digest({'previous': previous, 'successor': successor, 'decision': decision}),
            'claims_novelty': False,
            'scientific_semantic_identity_verified': False,
            'external_scientific_evidence_certified': False}


def proposed_next_step(history: list[dict], problem_cards: list[dict],
                       evidence: list[dict] | None = None) -> dict:
    policy = policy_for_next(history, evidence=evidence, problem_cards=problem_cards)
    return {'schema': SCHEMA, 'policy': policy, 'provisional_problem_cards': problem_cards,
            'successor_required_fields': ['lane', 'problem_id', 'family', 'mechanism_signature',
                'conditional_predictions', 'independent_observables', 'novelty_certified'],
            'scientific_signature_fields': list(SCIENTIFIC_SIGNATURE_FIELDS),
            'instructions_by_action': {
                'REFINE': 'Keep problem/causal assumption; change a prediction or discriminating measurement. No novelty claim.',
                'BRANCH': 'Propose a different causal driver or generating assumption; explicitly contrast strong null.',
                'ESCAPE': 'Choose a DIFFERENT anchored problem AND causal mechanism AND premise; require an independent discriminator.',
                'SUSPEND': 'Do NOT propose a successor in this lane; request more evidence/new problem discovery.',
                'DISCOVER': 'Propose a physically grounded problem with alternative explanation and falsifiable observable.'},
            'caveat': 'Offline plan ONLY. LLM-generated mechanism signatures require independent scientific review.',
            'no_model_calls': True, 'novelty_certified': False}
