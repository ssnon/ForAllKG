from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.diagnostic_prior_art_review import (
    DiagnosticPriorArtMatchDraft,
    DiagnosticPriorArtReviewDraft,
    compile_diagnostic_prior_art_review,
)
from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    NoveltyClaim,
    PriorArtPacket,
    PriorArtWork,
    RankedPriorArtWork,
)


def _claim(
    *,
    kind: str = "LOWER_ORDER_RELATION",
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id=(
            "external_novelty_claim:test"
        ),
        hypothesis_id=(
            "hypothesis:test"
        ),
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        text=(
            "Metal identity moderates "
            "the X to Y relation."
        ),
        rationale="test",
        diagnostic_query_kind=kind,
        diagnostic_search_query=(
            "X Y dependence"
        ),
        diagnostic_execution_query=(
            "X Y relationship dependence"
        ),
        diagnostic_relation_terms=[
            "X",
            "Y",
            "dependence",
        ],
    )


def _packet(
    *,
    abstract: str | None = (
        "X is correlated with Y across "
        "the tested systems."
    ),
) -> PriorArtPacket:
    return PriorArtPacket(
        packet_id="prior_art_packet:test",
        packet_sha256="sha",
        source_portfolio_id=(
            "hypothesis_portfolio:test"
        ),
        source_query_plan_id=(
            "literature_query_plan:test"
        ),
        searched_at_utc=(
            "2026-09-04T00:00:00+00:00"
        ),
        works=[
            PriorArtWork(
                work_id="work:test",
                title="X and Y relationship",
                abstract=abstract,
            )
        ],
    )


def _candidates():
    return ClaimPriorArtCandidateSet(
        hypothesis_id="hypothesis:test",
        claim_id=(
            "external_novelty_claim:test"
        ),
        ranked_works=[
            RankedPriorArtWork(
                work_id="work:test",
                relevance_score=0.9,
                semantic_similarity=0.9,
                lexical_coverage=0.8,
                reaction_domain_relevance=0.9,
                catalyst_scope_relevance=0.9,
                abstract_available=True,
            )
        ],
    )


def test_diagnostic_schema_forbids_partial_prior_art() -> None:
    with pytest.raises(
        ValidationError
    ):
        DiagnosticPriorArtMatchDraft(
            work_id="work:test",
            relationship=(
                "PARTIAL_PRIOR_ART"
            ),
            confidence=0.9,
            rationale="forbidden",
        )


def test_lower_order_signal_is_eligible() -> None:
    review = (
        compile_diagnostic_prior_art_review(
            claim=_claim(),
            candidates=_candidates(),
            draft=(
                DiagnosticPriorArtReviewDraft(
                    matches=[
                        DiagnosticPriorArtMatchDraft(
                            work_id="work:test",
                            relationship=(
                                "LOWER_ORDER_RELATION_PRIOR_ART"
                            ),
                            confidence=0.95,
                            rationale=(
                                "Abstract establishes "
                                "the lower-order X-Y relation."
                            ),
                        )
                    ],
                    interpretation="test",
                )
            ),
            packet=_packet(),
        )
    )

    assert review.signal_work_ids == [
        "work:test"
    ]

    assert (
        review.matches[0].relationship
        == "LOWER_ORDER_RELATION_PRIOR_ART"
    )


def test_wrong_diagnostic_signal_kind_is_dropped() -> None:
    review = (
        compile_diagnostic_prior_art_review(
            claim=_claim(),
            candidates=_candidates(),
            draft=(
                DiagnosticPriorArtReviewDraft(
                    matches=[
                        DiagnosticPriorArtMatchDraft(
                            work_id="work:test",
                            relationship=(
                                "DIRECTIONAL_COUNTEREVIDENCE"
                            ),
                            confidence=0.95,
                            rationale="wrong lane",
                        )
                    ],
                    interpretation="test",
                )
            ),
            packet=_packet(),
        )
    )

    assert review.signal_work_ids == []
    assert review.matches == []

    assert (
        "diagnostic_signal_kind_mismatch_dropped"
        in review.reason_codes
    )


def test_strong_diagnostic_signal_requires_abstract() -> None:
    review = (
        compile_diagnostic_prior_art_review(
            claim=_claim(),
            candidates=_candidates(),
            draft=(
                DiagnosticPriorArtReviewDraft(
                    matches=[
                        DiagnosticPriorArtMatchDraft(
                            work_id="work:test",
                            relationship=(
                                "LOWER_ORDER_RELATION_PRIOR_ART"
                            ),
                            confidence=0.95,
                            rationale="title only",
                        )
                    ],
                    interpretation="test",
                )
            ),
            packet=_packet(
                abstract=None
            ),
        )
    )

    assert review.signal_work_ids == []

    assert (
        review.matches[0].relationship
        == "TITLE_ONLY_NEIGHBOR"
    )


def test_diagnostic_prompt_targets_diagnostic_relation_not_full_status() -> None:
    from pipeline_core.discovery import (
        external_novelty_llm,
    )

    prompt = (
        external_novelty_llm
        ._DIAGNOSTIC_REVIEW_SYSTEM
    )

    lower = prompt.lower()

    assert (
        "not the ordinary full-claim "
        "novelty review"
        in lower
    )

    assert (
        "do not decide whether the full "
        "claim is direct_prior_art or "
        "partial_prior_art"
        in lower
    )

    assert (
        "lower_order_relation_prior_art"
        in lower
    )
