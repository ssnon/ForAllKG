from __future__ import annotations

import re

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    PriorArtPacket,
)
from pipeline_core.discovery.novelty_closure_execution import (
    ClosureLiteratureQueryPlan,
    ExecutableClosureTarget,
)


ClosureEvidenceRelationship = Literal[
    "ESTABLISHES_SLOT",
    "PARTIAL_SLOT_RELATION",
    "COMPONENT_ONLY",
    "TITLE_ONLY_NEIGHBOR",
    "UNRELATED",
    "INSUFFICIENT_METADATA",
]

ClosureEvidenceState = Literal[
    "ESTABLISHED",
    "NOT_FOUND",
    "UNASSESSED",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClosureEvidenceMatchDraft(StrictModel):
    work_id: str
    relationship: ClosureEvidenceRelationship
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    rationale: str = Field(
        min_length=1,
    )


class ClosureSlotReviewDraft(StrictModel):
    matches: list[
        ClosureEvidenceMatchDraft
    ] = Field(default_factory=list)

    interpretation: str = Field(
        min_length=1,
    )


class ClosureEvidenceMatch(StrictModel):
    work_id: str
    relationship: ClosureEvidenceRelationship
    confidence: float
    rationale: str

    relevance_score: float
    semantic_similarity: float
    lexical_coverage: float

    title: str
    year: int | None = None
    doi: str | None = None
    abstract_available: bool


class ClosureSlotEvidenceReview(StrictModel):
    target_id: str
    slot: str
    source_claim_id: str
    evidence_state: ClosureEvidenceState

    query_count: int
    successful_query_count: int

    provider_execution_count: int
    successful_provider_execution_count: int

    candidate_work_count: int
    abstract_candidate_count: int
    material_abstract_review_count: int
    negative_eligible_material_abstract_review_count: int
    negative_coverage_sufficient: bool

    matches: list[
        ClosureEvidenceMatch
    ] = Field(default_factory=list)

    positive_work_ids: list[str] = Field(
        default_factory=list
    )

    reason_codes: list[str] = Field(
        default_factory=list
    )

    reviewer_unknown_work_ids: list[str] = Field(
        default_factory=list
    )

    interpretation: str


# A closure slot is ESTABLISHED only when the bounded
# title/abstract evidence explicitly establishes that slot.
#
# PARTIAL_SLOT_RELATION is retained as useful neighboring evidence,
# but it cannot close the slot. This is especially important for
# FULL_RELATION: partial overlap must never collapse the residual
# claim to DIRECTLY_KNOWN.
def _normalize_identity_text(
    value: str,
) -> str:
    text = str(value or "").casefold()

    text = re.sub(
        r"[‐-‒–—−-]+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    return " ".join(
        text.split()
    )


def _abstract_contains_identity_anchor(
    *,
    abstract: str,
    anchors: tuple[str, ...],
) -> bool:
    normalized_abstract = (
        _normalize_identity_text(
            abstract
        )
    )

    for anchor in anchors:
        normalized_anchor = (
            _normalize_identity_text(
                anchor
            )
        )

        if (
            normalized_anchor
            and normalized_anchor
            in normalized_abstract
        ):
            return True

    return False


_POSITIVE_RELATIONSHIPS = {
    "ESTABLISHES_SLOT",
}


def compile_closure_slot_review(
    *,
    target: ExecutableClosureTarget,
    draft: ClosureSlotReviewDraft,
    candidates: ClaimPriorArtCandidateSet,
    packet: PriorArtPacket,
    plan: ClosureLiteratureQueryPlan,
    min_positive_confidence: float = 0.65,
    min_material_abstract_reviews_for_negative: int = 3,
) -> ClosureSlotEvidenceReview:
    if (
        packet.source_query_plan_id
        != plan.plan_id
    ):
        raise ValueError(
            "Closure packet/query-plan provenance mismatch."
        )

    if (
        candidates.claim_id
        != target.target_id
    ):
        raise ValueError(
            "Closure target/candidate-set mismatch."
        )

    works = {
        row.work_id: row
        for row in packet.works
    }

    ranking = {
        row.work_id: row
        for row in candidates.ranked_works
    }

    allowed = set(ranking)

    unknown = sorted(
        {
            row.work_id
            for row in draft.matches
        }
        - allowed
    )

    reason_codes: list[str] = []

    if unknown:
        reason_codes.append(
            "reviewer_unknown_work_id_dropped"
        )

    matches: list[
        ClosureEvidenceMatch
    ] = []

    for row in draft.matches:
        if row.work_id not in allowed:
            continue

        work = works.get(
            row.work_id
        )

        ranked = ranking[
            row.work_id
        ]

        if work is None:
            continue

        relationship = row.relationship

        # Title-only metadata cannot establish a scientific
        # relation in closure.
        if (
            not work.abstract
            and relationship
            in _POSITIVE_RELATIONSHIPS
        ):
            relationship = (
                "TITLE_ONLY_NEIGHBOR"
            )

            reason_codes.append(
                "positive_slot_match_downgraded_without_abstract"
            )

        matches.append(
            ClosureEvidenceMatch(
                work_id=work.work_id,
                relationship=relationship,
                confidence=row.confidence,
                rationale=row.rationale,
                relevance_score=(
                    ranked.relevance_score
                ),
                semantic_similarity=(
                    ranked.semantic_similarity
                ),
                lexical_coverage=(
                    ranked.lexical_coverage
                ),
                title=work.title,
                year=work.year,
                doi=work.doi,
                abstract_available=bool(
                    work.abstract
                ),
            )
        )

    positive = [
        row
        for row in matches
        if (
            row.relationship
            in _POSITIVE_RELATIONSHIPS
            and row.confidence
            >= min_positive_confidence
            and row.abstract_available
        )
    ]

    query_ids = {
        row.query_id
        for row in plan.queries
        if (
            row.claim_id
            == target.target_id
        )
    }

    executions = [
        row
        for row in packet.executions
        if row.query_id in query_ids
    ]

    successful_executions = [
        row
        for row in executions
        if row.success
    ]

    successful_query_ids = {
        row.query_id
        for row in successful_executions
    }

    candidate_work_ids = set(
        ranking
    )

    abstract_candidate_count = sum(
        1
        for work_id in candidate_work_ids
        if (
            works.get(work_id)
            and works[work_id].abstract
        )
    )

    # Negative closure must be based on abstract-backed records
    # that the reviewer judged materially related to this target.
    # Unrelated abstracts do not count merely because they exist.
    material_relationships = {
        "ESTABLISHES_SLOT",
        "PARTIAL_SLOT_RELATION",
        "COMPONENT_ONLY",
    }

    material_abstract_work_ids = {
        row.work_id
        for row in matches
        if (
            row.abstract_available
            and row.relationship
            in material_relationships
        )
    }

    material_abstract_review_count = len(
        material_abstract_work_ids
    )

    if target.slot == "BASE_RELATION":
        # BASE has no novelty-bearing branch identity.
        negative_eligible_work_ids = set(
            material_abstract_work_ids
        )
    else:
        # For FACTOR / BRIDGE / FULL, neighboring component
        # evidence contributes to bounded negative closure only
        # when the abstract actually contains the atomic branch
        # identity. Otherwise nearby literature can manufacture
        # false absence coverage.
        negative_eligible_work_ids = {
            work_id
            for work_id in material_abstract_work_ids
            if (
                works.get(work_id)
                and _abstract_contains_identity_anchor(
                    abstract=(
                        works[work_id].abstract
                        or ""
                    ),
                    anchors=(
                        target.identity_anchor_terms
                    ),
                )
            )
        }

    negative_eligible_material_abstract_review_count = len(
        negative_eligible_work_ids
    )

    negative_coverage_sufficient = (
        bool(successful_query_ids)
        and negative_eligible_material_abstract_review_count
        >= int(
            min_material_abstract_reviews_for_negative
        )
    )

    if not successful_query_ids:
        evidence_state: ClosureEvidenceState = (
            "UNASSESSED"
        )

        reason_codes.append(
            "no_successful_query"
        )

    elif positive:
        # Positive evidence is asymmetric: one explicit,
        # abstract-backed relation can establish the slot
        # even when search breadth is otherwise limited.
        evidence_state = "ESTABLISHED"

        reason_codes.append(
            "abstract_backed_slot_relation_established"
        )

    elif not negative_coverage_sufficient:
        evidence_state = "UNASSESSED"

        if target.slot == "BASE_RELATION":
            reason_codes.append(
                "insufficient_abstract_coverage_for_negative_closure"
            )
        elif not target.identity_anchor_terms:
            reason_codes.append(
                "missing_identity_anchor_for_negative_closure"
            )
        else:
            reason_codes.append(
                "insufficient_identity_anchored_material_coverage_for_negative_closure"
            )

    else:
        evidence_state = "NOT_FOUND"

        reason_codes.append(
            "no_positive_relation_in_reviewed_bounded_results"
        )

    return ClosureSlotEvidenceReview(
        target_id=target.target_id,
        slot=target.slot,
        source_claim_id=target.source_claim_id,
        evidence_state=evidence_state,
        query_count=len(query_ids),
        successful_query_count=len(
            successful_query_ids
        ),

        provider_execution_count=len(
            executions
        ),
        successful_provider_execution_count=len(
            successful_executions
        ),

        candidate_work_count=len(
            candidate_work_ids
        ),
        abstract_candidate_count=(
            abstract_candidate_count
        ),
        material_abstract_review_count=(
            material_abstract_review_count
        ),
        negative_eligible_material_abstract_review_count=(
            negative_eligible_material_abstract_review_count
        ),
        negative_coverage_sufficient=(
            negative_coverage_sufficient
        ),
        matches=matches,
        positive_work_ids=sorted(
            {
                row.work_id
                for row in positive
            }
        ),
        reason_codes=reason_codes,
        reviewer_unknown_work_ids=unknown,
        interpretation=draft.interpretation,
    )
