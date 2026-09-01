from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
)
from pipeline_core.discovery.novelty_nonobviousness import (
    NonObviousnessEvidenceClosure,
)


ClosureSlot = Literal[
    "BASE_RELATION",
    "DISTINGUISHING_FACTOR_EFFECT",
    "BRIDGE_RELATION",
    "FULL_RELATION",
]


_POSITIVE_RELATION_STATUSES = {
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
}


@dataclass(frozen=True)
class ClosureSlotAssessment:
    slot: ClosureSlot
    evidence_state: Literal[
        "ESTABLISHED",
        "NOT_FOUND",
        "UNASSESSED",
    ]
    review_statuses: tuple[str, ...]
    positive_work_ids: tuple[str, ...]


def compile_closure_slot(
    slot: ClosureSlot,
    reviews: Sequence[ClaimPriorArtReview],
) -> ClosureSlotAssessment:
    """Compile one closure slot from atomic targeted reviews.

    NOT_FOUND means only:
      no positive relation-backed evidence was established in the
      supplied reviewed search results.

    It is NOT a literature-wide absence claim.
    """

    rows = list(reviews)

    if not rows:
        return ClosureSlotAssessment(
            slot=slot,
            evidence_state="UNASSESSED",
            review_statuses=(),
            positive_work_ids=(),
        )

    positive_work_ids: set[str] = set()

    for review in rows:
        for match in review.matches:
            if match.relationship in {
                "DIRECT_PRIOR_ART",
                "PARTIAL_PRIOR_ART",
            }:
                positive_work_ids.add(
                    match.work_id
                )

    positive_status = any(
        review.status
        in _POSITIVE_RELATION_STATUSES
        for review in rows
    )

    state = (
        "ESTABLISHED"
        if positive_status
        else "NOT_FOUND"
    )

    return ClosureSlotAssessment(
        slot=slot,
        evidence_state=state,
        review_statuses=tuple(
            review.status
            for review in rows
        ),
        positive_work_ids=tuple(
            sorted(positive_work_ids)
        ),
    )


@dataclass(frozen=True)
class CompiledEvidenceClosure:
    closure: NonObviousnessEvidenceClosure

    base: ClosureSlotAssessment
    factor: ClosureSlotAssessment
    bridge: ClosureSlotAssessment
    full: ClosureSlotAssessment


def compile_nonobviousness_evidence_closure(
    *,
    base_reviews: Sequence[ClaimPriorArtReview],
    factor_reviews: Sequence[ClaimPriorArtReview],
    bridge_reviews: Sequence[ClaimPriorArtReview],
    full_reviews: Sequence[ClaimPriorArtReview],
    bridge_kind: str = "NONE",
    scope_compatible: bool = True,
) -> CompiledEvidenceClosure:
    base = compile_closure_slot(
        "BASE_RELATION",
        base_reviews,
    )

    factor = compile_closure_slot(
        "DISTINGUISHING_FACTOR_EFFECT",
        factor_reviews,
    )

    bridge = compile_closure_slot(
        "BRIDGE_RELATION",
        bridge_reviews,
    )

    full = compile_closure_slot(
        "FULL_RELATION",
        full_reviews,
    )

    closure = NonObviousnessEvidenceClosure(
        base_relation=base.evidence_state,
        distinguishing_factor_effect=(
            factor.evidence_state
        ),
        bridge_relation=bridge.evidence_state,
        full_relation=full.evidence_state,
        bridge_kind=bridge_kind,
        scope_compatible=scope_compatible,
        base_work_ids=base.positive_work_ids,
        factor_work_ids=factor.positive_work_ids,
        bridge_work_ids=bridge.positive_work_ids,
        full_relation_work_ids=(
            full.positive_work_ids
        ),
    )

    return CompiledEvidenceClosure(
        closure=closure,
        base=base,
        factor=factor,
        bridge=bridge,
        full=full,
    )
