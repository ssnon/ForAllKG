from __future__ import annotations

from collections import defaultdict
from typing import Any

from pipeline_core.discovery.scientific_novelty_action_policy import (
    ScientificNoveltyActionPolicy,
)
from pipeline_core.discovery.semantic_distinctiveness_contracts import (
    SemanticDistinctivenessReview,
)


_BATCH_SCHEMA = "scientific-novelty-action-shadow-batch-v1"


def build_scientific_novelty_action_batch(
    *,
    external_payload: dict[str, Any],
    semantic_reviews: list[SemanticDistinctivenessReview],
) -> dict[str, Any]:
    """Compile frozen novelty signals into deterministic N1 shadow decisions.

    This function is intentionally non-mutating. It does not alter portfolios,
    hypotheses, external assessments, or Alpha6 selection.
    """

    cards = external_payload.get("cards")

    if not isinstance(cards, list):
        raise ValueError(
            "external novelty report cards must be a list"
        )

    external_by_id: dict[str, str] = {}

    for card in cards:
        if not isinstance(card, dict):
            raise ValueError(
                "external novelty report card must be an object"
            )

        hypothesis_id = str(
            card.get("hypothesis_id") or ""
        ).strip()

        status = str(
            card.get("status") or ""
        ).strip()

        if not hypothesis_id:
            raise ValueError(
                "external novelty card is missing hypothesis_id"
            )

        if not status:
            raise ValueError(
                f"external novelty card {hypothesis_id} is missing status"
            )

        if hypothesis_id in external_by_id:
            raise ValueError(
                f"duplicate external novelty hypothesis_id: {hypothesis_id}"
            )

        external_by_id[hypothesis_id] = status

    semantic_by_id: dict[
        str,
        dict[int, SemanticDistinctivenessReview],
    ] = defaultdict(dict)

    for review in semantic_reviews:
        hypothesis_id = review.hypothesis_id
        pass_index = review.review_pass_index

        if pass_index not in {1, 2}:
            raise ValueError(
                "scientific novelty action batch requires only "
                f"semantic pass 1/2, got pass={pass_index} "
                f"for {hypothesis_id}"
            )

        if pass_index in semantic_by_id[hypothesis_id]:
            raise ValueError(
                "duplicate semantic review pass for "
                f"{hypothesis_id}: pass={pass_index}"
            )

        semantic_by_id[hypothesis_id][pass_index] = review

    external_ids = set(external_by_id)
    semantic_ids = set(semantic_by_id)

    if external_ids != semantic_ids:
        raise ValueError(
            "external/semantic hypothesis sets do not match: "
            f"external_only={sorted(external_ids - semantic_ids)}, "
            f"semantic_only={sorted(semantic_ids - external_ids)}"
        )

    policy = ScientificNoveltyActionPolicy()
    decisions = []

    for hypothesis_id in external_by_id:
        reviews = semantic_by_id[hypothesis_id]

        if set(reviews) != {1, 2}:
            raise ValueError(
                f"semantic pass pair incomplete for {hypothesis_id}: "
                f"passes={sorted(reviews)}"
            )

        pass_1 = reviews[1]
        pass_2 = reviews[2]

        decision = policy.evaluate(
            external_status=external_by_id[hypothesis_id],
            semantic_pass_1=pass_1.overall_tier,
            semantic_pass_2=pass_2.overall_tier,
        )

        decisions.append(
            {
                "hypothesis_id": hypothesis_id,
                "semantic_review_ids": [
                    pass_1.review_id,
                    pass_2.review_id,
                ],
                "semantic_review_passes": [
                    pass_1.review_pass_index,
                    pass_2.review_pass_index,
                ],
                "decision": decision.model_dump(
                    mode="json"
                ),
            }
        )

    return {
        "schema_version": _BATCH_SCHEMA,
        "source_external_report_id": external_payload.get(
            "report_id"
        ),
        "decision_count": len(decisions),
        "decisions": decisions,
        "shadow_only": True,
        "action_policy_applied": False,
        "scientific_selection_changed": False,
    }
