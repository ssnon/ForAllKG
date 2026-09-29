from __future__ import annotations

from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
)


def select_pre_review_resolution_targets(
    *,
    plan: LiteratureQueryPlan,
    packet: PriorArtPacket,
    ranker: Any,
) -> dict[str, Any]:
    """
    Deterministically select metadata-resolution targets before LLM review.

    Selection authority:
    - core novelty claims only;
    - existing PriorArtRanker top-k only;
    - missing-abstract works only.

    No review outcome, novelty status, or post-hoc TITLE_ONLY label is used.
    """
    works = {
        row.work_id: row
        for row in packet.works
    }

    selected_ids: set[str] = set()
    per_claim: list[dict[str, Any]] = []

    claims = [
        claim
        for group in plan.claims
        for claim in group.claims
        if str(claim.importance) == "core"
    ]

    for claim in claims:
        ranked = ranker.rank(
            claim,
            packet,
            plan,
        )
        unresolved = [
            row.work_id
            for row in ranked.ranked_works
            if (
                row.work_id in works
                and not works[row.work_id].abstract
            )
        ]
        selected_ids.update(unresolved)

        per_claim.append(
            {
                "hypothesis_id": claim.hypothesis_id,
                "claim_id": claim.claim_id,
                "ranked_work_count":
                    len(ranked.ranked_works),
                "missing_abstract_ranked_work_ids":
                    unresolved,
            }
        )

    return {
        "selector_version":
            "pre-review-resolution-target-selector-v1",
        "core_claim_count": len(claims),
        "selected_work_ids": sorted(selected_ids),
        "selected_work_count": len(selected_ids),
        "per_claim": per_claim,
        "review_outcome_consumed": False,
        "novelty_status_consumed": False,
        "production_selection_authority": False,
    }


__all__ = [
    "select_pre_review_resolution_targets",
]
