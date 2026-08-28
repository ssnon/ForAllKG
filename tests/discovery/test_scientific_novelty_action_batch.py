from __future__ import annotations

from pipeline_core.discovery.scientific_novelty_action_batch import (
    build_scientific_novelty_action_batch,
)
from pipeline_core.discovery.semantic_distinctiveness_contracts import (
    SemanticDistinctivenessReview,
)


def _review(
    *,
    hypothesis_id: str,
    pass_index: int,
    tier: str,
) -> SemanticDistinctivenessReview:
    # Reuse the real contract through model_construct because this test
    # targets only batch pairing/policy semantics, not the full semantic
    # review compiler contract.
    return SemanticDistinctivenessReview.model_construct(
        review_id=(
            f"review:{hypothesis_id}:{pass_index}"
        ),
        hypothesis_id=hypothesis_id,
        review_pass_index=pass_index,
        overall_tier=tier,
    )


def test_stable_low_is_ineligible_reaxis_required():
    payload = {
        "report_id": "external:r1",
        "cards": [
            {
                "hypothesis_id": "hypothesis:h1",
                "status":
                    "LITERATURE_SUPPORTED_EXTENSION",
            }
        ],
    }

    result = build_scientific_novelty_action_batch(
        external_payload=payload,
        semantic_reviews=[
            _review(
                hypothesis_id="hypothesis:h1",
                pass_index=1,
                tier="LOW",
            ),
            _review(
                hypothesis_id="hypothesis:h1",
                pass_index=2,
                tier="LOW",
            ),
        ],
    )

    assert result[
        "scientific_selection_changed"
    ] is False

    assert result[
        "action_policy_applied"
    ] is False

    assert result[
        "decision_count"
    ] == 1

    decision = result[
        "decisions"
    ][0]["decision"]

    assert decision[
        "action"
    ] == "REAXIS_REQUIRED"

    assert decision[
        "selection_class"
    ] == "INELIGIBLE"

    assert (
        "STABLE_SEMANTIC_LOW"
        in decision[
            "reason_codes"
        ]
    )


def test_unstable_semantic_pair_is_unresolved_ineligible():
    payload = {
        "report_id": "external:r2",
        "cards": [
            {
                "hypothesis_id": "hypothesis:h2",
                "status":
                    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
            }
        ],
    }

    result = build_scientific_novelty_action_batch(
        external_payload=payload,
        semantic_reviews=[
            _review(
                hypothesis_id="hypothesis:h2",
                pass_index=1,
                tier="MODERATE",
            ),
            _review(
                hypothesis_id="hypothesis:h2",
                pass_index=2,
                tier="HIGH",
            ),
        ],
    )

    decision = result[
        "decisions"
    ][0]["decision"]

    assert decision[
        "action"
    ] == "UNRESOLVED"

    assert decision[
        "selection_class"
    ] == "INELIGIBLE"


def test_every_external_card_requires_exactly_two_semantic_passes():
    payload = {
        "cards": [
            {
                "hypothesis_id": "hypothesis:h3",
                "status":
                    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
            }
        ]
    }

    try:
        build_scientific_novelty_action_batch(
            external_payload=payload,
            semantic_reviews=[
                _review(
                    hypothesis_id="hypothesis:h3",
                    pass_index=1,
                    tier="MODERATE",
                )
            ],
        )
    except ValueError as exc:
        assert (
            "semantic pass pair incomplete"
            in str(exc)
        )
    else:
        raise AssertionError(
            "missing semantic pass 2 must fail closed"
        )
