from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.novelty_closure_review import (
    _abstract_contains_identity_anchor,
    _identity_content_tokens,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AblationMode = Literal[
    "CURRENT_LOCAL_IDENTITY",
    "DISPERSED_CONSTITUENT_IDENTITY",
    "SEARCH_COVERAGE_ONLY_CONTROL",
]


class SlotNegativeClosureAblation(StrictModel):
    slot: str
    strict_state: str
    positive_work_ids: list[str]
    successful_query_count: int = Field(ge=0)
    material_abstract_work_count: int = Field(ge=0)
    identity_anchor_terms: list[str]

    strict_negative_eligible_count: int = Field(ge=0)
    dispersed_constituent_negative_eligible_count: int = Field(ge=0)
    search_coverage_only_negative_eligible_count: int = Field(ge=0)

    strict_negative_eligible_work_ids: list[str]
    dispersed_constituent_negative_eligible_work_ids: list[str]
    search_coverage_only_negative_eligible_work_ids: list[str]

    strict_shadow_state: Literal["ESTABLISHED", "NOT_FOUND", "UNASSESSED"]
    dispersed_constituent_shadow_state: Literal[
        "ESTABLISHED",
        "NOT_FOUND",
        "UNASSESSED",
    ]
    search_coverage_only_shadow_state: Literal[
        "ESTABLISHED",
        "NOT_FOUND",
        "UNASSESSED",
    ]

    local_window_is_only_difference_for_dispersed_mode: Literal[True] = True
    synonym_expansion_used: Literal[False] = False
    embedding_matching_used: Literal[False] = False
    scientific_semantic_inference_used: Literal[False] = False


class ClaimNegativeClosureAblation(StrictModel):
    claim_id: str
    slots: list[SlotNegativeClosureAblation]
    strict_state_counts: dict[str, int]
    dispersed_constituent_state_counts: dict[str, int]
    search_coverage_only_state_counts: dict[str, int]
    strict_full_relation_state: str | None = None
    dispersed_constituent_full_relation_state: str | None = None
    search_coverage_only_full_relation_state: str | None = None


class N10NegativeClosureIdentityAblationReport(StrictModel):
    schema_version: Literal[
        "n10-negative-closure-identity-ablation-v1"
    ] = "n10-negative-closure-identity-ablation-v1"

    claim_count: int = Field(ge=0)
    slot_count: int = Field(ge=0)
    claims: list[ClaimNegativeClosureAblation]

    strict_unassessed_nonbase_slot_count: int = Field(ge=0)
    dispersed_constituent_not_found_nonbase_slot_count: int = Field(ge=0)
    search_coverage_only_not_found_nonbase_slot_count: int = Field(ge=0)

    strict_full_relation_not_found_count: int = Field(ge=0)
    dispersed_constituent_full_relation_not_found_count: int = Field(ge=0)
    search_coverage_only_full_relation_not_found_count: int = Field(ge=0)

    diagnostic_only: Literal[True] = True
    production_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    positive_evidence_semantics_changed: Literal[False] = False
    negative_closure_only: Literal[True] = True

    dispersed_mode_description: Literal[
        "same_canonical_identity_tokens_and_same_75_percent_threshold_without_local_window"
    ] = (
        "same_canonical_identity_tokens_and_same_75_percent_threshold_without_local_window"
    )
    search_coverage_control_description: Literal[
        "all_material_abstracts_count_for_negative_coverage_regardless_of_identity"
    ] = (
        "all_material_abstracts_count_for_negative_coverage_regardless_of_identity"
    )


_MATERIAL_RELATIONSHIPS = {
    "ESTABLISHES_SLOT",
    "PARTIAL_SLOT_RELATION",
    "COMPONENT_ONLY",
}


def _distinct(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _global_constituent_identity_match(
    *,
    abstract: str,
    anchors: tuple[str, ...],
) -> bool:
    """Ablation-only identity match.

    This intentionally changes exactly one dimension relative to the
    production long-anchor matcher: local lexical co-occurrence is removed.
    It keeps the same canonical content tokens and the same >=75% / >=3
    matched-token requirement. Short anchors remain on production behavior.
    """

    abstract_tokens = set(_identity_content_tokens(abstract))

    for anchor in anchors:
        anchor_tokens = _distinct(_identity_content_tokens(anchor))
        anchor_count = len(anchor_tokens)

        if anchor_count <= 2:
            if _abstract_contains_identity_anchor(
                abstract=abstract,
                anchors=(anchor,),
            ):
                return True
            continue

        required_match_count = max(
            3,
            (3 * anchor_count + 3) // 4,
        )
        matched = sum(
            token in abstract_tokens
            for token in anchor_tokens
        )
        if matched >= required_match_count:
            return True

    return False


def _material_work_ids(
    *,
    slot_review: Mapping[str, Any],
) -> set[str]:
    result: set[str] = set()
    for row in slot_review.get("matches") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("abstract_available") is not True:
            continue
        if str(row.get("relationship") or "") not in _MATERIAL_RELATIONSHIPS:
            continue
        work_id = str(row.get("work_id") or "").strip()
        if work_id:
            result.add(work_id)
    return result


def _shadow_state(
    *,
    successful_query_count: int,
    positive_work_ids: list[str],
    eligible_count: int,
    minimum_negative_abstracts: int,
) -> Literal["ESTABLISHED", "NOT_FOUND", "UNASSESSED"]:
    if successful_query_count <= 0:
        return "UNASSESSED"
    if positive_work_ids:
        return "ESTABLISHED"
    if eligible_count >= minimum_negative_abstracts:
        return "NOT_FOUND"
    return "UNASSESSED"


def analyze_slot(
    *,
    target: Mapping[str, Any],
    slot_review: Mapping[str, Any],
    works_by_id: Mapping[str, Mapping[str, Any]],
    minimum_negative_abstracts: int = 3,
) -> SlotNegativeClosureAblation:
    slot = str(slot_review.get("slot") or "")
    anchors = tuple(
        str(value)
        for value in (target.get("identity_anchor_terms") or [])
        if str(value).strip()
    )
    successful_query_count = int(
        slot_review.get("successful_query_count") or 0
    )
    positive_work_ids = sorted(
        str(value)
        for value in (slot_review.get("positive_work_ids") or [])
        if str(value).strip()
    )

    material = _material_work_ids(slot_review=slot_review)

    if slot == "BASE_RELATION":
        strict_eligible = set(material)
        dispersed_eligible = set(material)
    else:
        strict_eligible = {
            work_id
            for work_id in material
            if (
                work_id in works_by_id
                and _abstract_contains_identity_anchor(
                    abstract=str(
                        works_by_id[work_id].get("abstract") or ""
                    ),
                    anchors=anchors,
                )
            )
        }
        dispersed_eligible = {
            work_id
            for work_id in material
            if (
                work_id in works_by_id
                and _global_constituent_identity_match(
                    abstract=str(
                        works_by_id[work_id].get("abstract") or ""
                    ),
                    anchors=anchors,
                )
            )
        }

    search_only_eligible = set(material)

    strict_shadow_state = _shadow_state(
        successful_query_count=successful_query_count,
        positive_work_ids=positive_work_ids,
        eligible_count=len(strict_eligible),
        minimum_negative_abstracts=minimum_negative_abstracts,
    )
    dispersed_state = _shadow_state(
        successful_query_count=successful_query_count,
        positive_work_ids=positive_work_ids,
        eligible_count=len(dispersed_eligible),
        minimum_negative_abstracts=minimum_negative_abstracts,
    )
    search_state = _shadow_state(
        successful_query_count=successful_query_count,
        positive_work_ids=positive_work_ids,
        eligible_count=len(search_only_eligible),
        minimum_negative_abstracts=minimum_negative_abstracts,
    )

    return SlotNegativeClosureAblation(
        slot=slot,
        strict_state=str(slot_review.get("evidence_state") or ""),
        positive_work_ids=positive_work_ids,
        successful_query_count=successful_query_count,
        material_abstract_work_count=len(material),
        identity_anchor_terms=list(anchors),
        strict_negative_eligible_count=len(strict_eligible),
        dispersed_constituent_negative_eligible_count=len(dispersed_eligible),
        search_coverage_only_negative_eligible_count=len(search_only_eligible),
        strict_negative_eligible_work_ids=sorted(strict_eligible),
        dispersed_constituent_negative_eligible_work_ids=sorted(
            dispersed_eligible
        ),
        search_coverage_only_negative_eligible_work_ids=sorted(
            search_only_eligible
        ),
        strict_shadow_state=strict_shadow_state,
        dispersed_constituent_shadow_state=dispersed_state,
        search_coverage_only_shadow_state=search_state,
    )


def analyze_claim_detail(
    *,
    closure_plan: Mapping[str, Any],
    slot_reviews: list[Mapping[str, Any]],
    prior_art: Mapping[str, Any],
    minimum_negative_abstracts: int = 3,
) -> ClaimNegativeClosureAblation:
    targets = {
        str(row.get("slot") or ""): row
        for row in closure_plan.get("targets") or []
        if isinstance(row, Mapping)
    }
    works_by_id = {
        str(row.get("work_id") or ""): row
        for row in prior_art.get("works") or []
        if isinstance(row, Mapping)
        and str(row.get("work_id") or "").strip()
    }

    rows: list[SlotNegativeClosureAblation] = []
    for review in slot_reviews:
        slot = str(review.get("slot") or "")
        target = targets.get(slot)
        if target is None:
            raise ValueError(f"closure plan missing target slot: {slot}")
        rows.append(
            analyze_slot(
                target=target,
                slot_review=review,
                works_by_id=works_by_id,
                minimum_negative_abstracts=minimum_negative_abstracts,
            )
        )

    strict_counts = Counter(row.strict_shadow_state for row in rows)
    dispersed_counts = Counter(
        row.dispersed_constituent_shadow_state for row in rows
    )
    search_counts = Counter(
        row.search_coverage_only_shadow_state for row in rows
    )

    by_slot = {row.slot: row for row in rows}
    full = by_slot.get("FULL_RELATION")

    return ClaimNegativeClosureAblation(
        claim_id=str(closure_plan.get("claim_id") or ""),
        slots=rows,
        strict_state_counts=dict(sorted(strict_counts.items())),
        dispersed_constituent_state_counts=dict(
            sorted(dispersed_counts.items())
        ),
        search_coverage_only_state_counts=dict(sorted(search_counts.items())),
        strict_full_relation_state=(
            full.strict_shadow_state if full is not None else None
        ),
        dispersed_constituent_full_relation_state=(
            full.dispersed_constituent_shadow_state
            if full is not None
            else None
        ),
        search_coverage_only_full_relation_state=(
            full.search_coverage_only_shadow_state
            if full is not None
            else None
        ),
    )


def build_report(
    claims: list[ClaimNegativeClosureAblation],
) -> N10NegativeClosureIdentityAblationReport:
    slots = [slot for claim in claims for slot in claim.slots]
    nonbase = [slot for slot in slots if slot.slot != "BASE_RELATION"]

    return N10NegativeClosureIdentityAblationReport(
        claim_count=len(claims),
        slot_count=len(slots),
        claims=claims,
        strict_unassessed_nonbase_slot_count=sum(
            slot.strict_shadow_state == "UNASSESSED"
            for slot in nonbase
        ),
        dispersed_constituent_not_found_nonbase_slot_count=sum(
            slot.dispersed_constituent_shadow_state == "NOT_FOUND"
            for slot in nonbase
        ),
        search_coverage_only_not_found_nonbase_slot_count=sum(
            slot.search_coverage_only_shadow_state == "NOT_FOUND"
            for slot in nonbase
        ),
        strict_full_relation_not_found_count=sum(
            claim.strict_full_relation_state == "NOT_FOUND"
            for claim in claims
        ),
        dispersed_constituent_full_relation_not_found_count=sum(
            claim.dispersed_constituent_full_relation_state == "NOT_FOUND"
            for claim in claims
        ),
        search_coverage_only_full_relation_not_found_count=sum(
            claim.search_coverage_only_full_relation_state == "NOT_FOUND"
            for claim in claims
        ),
    )


__all__ = [
    "ClaimNegativeClosureAblation",
    "N10NegativeClosureIdentityAblationReport",
    "SlotNegativeClosureAblation",
    "_global_constituent_identity_match",
    "analyze_claim_detail",
    "analyze_slot",
    "build_report",
]
