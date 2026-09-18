from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisSearchCoverage,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    RefinementAttempt,
)
from pipeline_core.discovery.novelty_refinement_runtime import (
    TargetedNoveltyRefinementRuntime,
)


def _card(status: str, sufficient: bool):
    return SimpleNamespace(
        status=status,
        coverage=SimpleNamespace(
            sufficient_for_absence_based_novelty=sufficient,
        ),
    )


def test_s26b_insufficient_search_never_earns_reaxis_replacement():
    assert (
        TargetedNoveltyRefinementRuntime
        ._reaxis_replacement_evidence_ready(
            _card("INSUFFICIENT_SEARCH_EVIDENCE", True)
        )
        is False
    )


def test_s26b_gap_like_reaxis_requires_absence_coverage():
    assert (
        TargetedNoveltyRefinementRuntime
        ._reaxis_replacement_evidence_ready(
            _card("NEW_COMBINATION_OF_KNOWN_EFFECTS", False)
        )
        is False
    )
    assert (
        TargetedNoveltyRefinementRuntime
        ._reaxis_replacement_evidence_ready(
            _card("NEW_COMBINATION_OF_KNOWN_EFFECTS", True)
        )
        is True
    )


def test_s26b_refinement_attempt_records_hold_and_fresh_coverage():
    coverage = HypothesisSearchCoverage(
        hypothesis_id="h:new",
        query_count=2,
        successful_query_count=2,
        provider_success_count=1,
        unique_work_count=4,
        abstract_work_count=2,
        core_claim_count=1,
        core_claims_with_minimum_abstract_coverage=0,
        sufficient_for_absence_based_novelty=False,
    )

    attempt = RefinementAttempt(
        original_hypothesis_id="h:old",
        candidate_hypothesis_id="h:new",
        gap_id="gap:1",
        action="targeted_search_then_refine",
        decision="held_for_evidence",
        original_external_status="LITERATURE_SUPPORTED_EXTENSION",
        targeted_external_status="LITERATURE_SUPPORTED_EXTENSION",
        final_external_status="INSUFFICIENT_SEARCH_EVIDENCE",
        reaxis_search_coverage=coverage,
        grounding_preserved=False,
        refinement_generated=True,
        generation_mode="fresh_context_reaxis",
        context_grounding_valid=True,
        reason_codes=["fresh_reaxis_held_for_evidence"],
        interpretation="Held pending adequate fresh external evidence.",
    )

    assert attempt.decision == "held_for_evidence"
    assert attempt.reaxis_search_coverage is not None
    assert (
        attempt.reaxis_search_coverage
        .sufficient_for_absence_based_novelty
        is False
    )
