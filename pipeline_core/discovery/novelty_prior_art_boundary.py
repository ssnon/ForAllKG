from __future__ import annotations

import json

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyCard,
)


def render_higher_order_relational_gap_boundary(
    card: ExternalNoveltyCard,
) -> str:
    """Render lower-order prior art as exclusion/boundary context only.

    This renderer is intentionally non-decisional:
    - it does not alter ExternalNoveltyStatus;
    - it does not trigger re-axis or rejection;
    - it does not promote external literature to a positive premise.

    Only lower-order evidence attached to recorded higher-order relational-gap
    core claims is included. This avoids depending on generic match ordering
    such as review.matches[:3].
    """

    if (
        card.relational_gap_kind
        != "HIGHER_ORDER_RELATIONAL_GAP"
    ):
        return ""

    gap_claim_ids = set(
        card.higher_order_relational_gap_claim_ids
    )

    rows: list[dict[str, object]] = []

    for review in card.claim_reviews:
        if (
            review.importance != "core"
            or review.claim_id not in gap_claim_ids
        ):
            continue

        matches = [
            {
                "work_id": match.work_id,
                "relationship": match.relationship,
                "title": match.title,
                "rationale": match.rationale,
            }
            for match in review.matches
            if (
                match.relationship
                == "LOWER_ORDER_RELATION_PRIOR_ART"
            )
        ]

        if not matches:
            continue

        rows.append(
            {
                "claim_id": review.claim_id,
                "claim_text": review.claim_text,
                "full_claim_status": review.status,
                "lower_order_prior_art": matches,
            }
        )

    # Fail closed if the card annotation and claim provenance somehow drift.
    if not rows:
        return ""

    payload = json.dumps(
        rows,
        ensure_ascii=False,
        indent=2,
    )

    return "\n".join(
        [
            "HIGHER-ORDER RELATIONAL-GAP NEGATIVE BOUNDARY",
            "=============================================",
            (
                "relational_gap_kind: "
                "HIGHER_ORDER_RELATIONAL_GAP"
            ),
            (
                "The bounded prior-art review found explicit "
                "LOWER-ORDER subrelations around one or more "
                "unresolved higher-order core claims."
            ),
            "",
            payload,
            "",
            "BOUNDARY RULES",
            "==============",
            (
                "1. The lower-order records above are "
                "EXCLUSION/BOUNDARY information only."
            ),
            (
                "2. They MUST NOT become positive scientific "
                "premises or premise_statement_ids."
            ),
            (
                "3. Do NOT reinterpret them as DIRECT or PARTIAL "
                "prior art for the full higher-order claim."
            ),
            (
                "4. Avoid merely restating, recombining, or adding "
                "cosmetic qualifiers to these already represented "
                "lower-order relations."
            ),
            (
                "5. Preserve the unresolved higher-order distinction, "
                "or move to a genuinely different grounded relation "
                "axis when the surrounding task permits it."
            ),
            (
                "6. Any new moderator, mechanism, regime, observable, "
                "or relation nucleus must come from the supplied "
                "grounded HypothesisContext, not from external literature."
            ),
        ]
    )
