from __future__ import annotations

import copy
import inspect
from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.novelty_atomic_semantic_fidelity import (
    assess_atomic_semantic_fidelity,
    compile_ordered_marker_source_channel_shadow,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    NoveltyClaimDecomposer,
)


_ORDERED_REASON_PAIR = [
    "atomic_claim_ordered_language_not_in_source_basis",
    "atomic_claim_ordered_language_not_in_source_context",
]


def _hypothesis(
    *,
    statement: str,
    prediction_rationale: str,
):
    return SimpleNamespace(
        hypothesis_statement=statement,
        inferential_bridge=statement,
        assumptions=[],
        predicted_observations=[
            SimpleNamespace(
                observation_id="prediction:p1",
                observable=(
                    "p99 latency during leader handoff for comparable workloads"
                ),
                expected_direction="qualitative_change",
                rationale=prediction_rationale,
            )
        ],
        falsification_criteria=[
            SimpleNamespace(
                criterion_id="falsifier:f1",
                observable=(
                    "p99 latency during leader handoff for comparable workloads"
                ),
                falsifying_outcome="No qualitative difference is observed.",
            )
        ],
    )


def _claim(
    *,
    text: str,
    basis: str,
    predicted: str,
) -> NoveltyClaimDraft:
    return NoveltyClaimDraft(
        local_id="claim_1",
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="TESTING_PREDICTION",
        text=text,
        rationale="diagnostic test",
        search_concepts=["request-load distribution", "p99 latency"],
        search_queries=["request-load distribution p99 latency"],
        prior_art_identity_terms=["request-load distribution"],
        relation_nucleus_terms=["request-load distribution", "p99 latency"],
        required_bridge=basis,
        predicted_observation=predicted,
        falsification_condition="No qualitative difference is observed.",
        semantic_fidelity_binding=NoveltyClaimSemanticFidelityBindingDraft(
            proposition_basis=basis,
            relation_endpoint_anchors=[
                "request-load distribution",
                "p99 latency",
            ],
            scope_qualifier_spans=[
                "Under the condition of comparable aggregate request rate"
            ],
            prediction_observation_id="prediction:p1",
            falsification_criterion_id="falsifier:f1",
        ),
    )


def test_prediction_only_exact_markers_are_channel_supported_without_authority():
    source = (
        "Under the condition of comparable aggregate request rate, "
        "a more even request-load distribution reduces p99 latency "
        "during leader handoff."
    )
    hypothesis = _hypothesis(
        statement=source,
        prediction_rationale=(
            "The more even distribution is expected to produce the "
            "smaller p99 latency increase."
        ),
    )
    claim = _claim(
        text=source,
        basis=source,
        predicted=(
            "The more even distribution is expected to produce a "
            "smaller p99 latency increase."
        ),
    )

    review = assess_atomic_semantic_fidelity(hypothesis, claim)
    assert review["reason_codes"] == _ORDERED_REASON_PAIR

    review_before = copy.deepcopy(review)
    claim_before = claim.model_dump(mode="json")

    shadow = compile_ordered_marker_source_channel_shadow(
        hypothesis,
        claim,
        semantic_fidelity_shadow=review,
    )

    assert shadow["diagnostic_only"] is True
    assert shadow["production_authority"] is False
    assert shadow["production_recompile_enabled"] is False
    assert shadow["recompile_performed"] is False
    assert shadow["reason_codes_modified"] is False
    assert shadow["semantic_synonymy_allowed"] is False
    assert shadow["unsupported_markers_by_required_channel"] == []
    assert shadow["channel_aware_exact_status"] == (
        "NO_UNSUPPORTED_ORDERED_MARKERS"
    )
    assert (
        shadow["would_clear_current_ordered_reason_pair_if_authoritative"]
        is True
    )
    assert review == review_before
    assert claim.model_dump(mode="json") == claim_before
    assert "ordered_marker_source_channel_shadow" not in claim_before


def test_prediction_near_synonym_does_not_count_as_exact_support():
    source = (
        "Under the condition of comparable aggregate request rate, "
        "a more even request-load distribution reduces p99 latency "
        "during leader handoff."
    )
    hypothesis = _hypothesis(
        statement=source,
        prediction_rationale=(
            "The more even distribution is expected to be more compatible."
        ),
    )
    claim = _claim(
        text=source,
        basis=source,
        predicted="The greater compatibility is expected.",
    )

    review = assess_atomic_semantic_fidelity(hypothesis, claim)
    shadow = compile_ordered_marker_source_channel_shadow(
        hypothesis,
        claim,
        semantic_fidelity_shadow=review,
    )

    assert "greater" in shadow[
        "unsupported_markers_by_required_channel"
    ]
    assert shadow["channel_aware_exact_status"] == "REVIEW_REQUIRED"
    assert (
        shadow["would_clear_current_ordered_reason_pair_if_authoritative"]
        is False
    )


def test_prediction_source_cannot_launder_claim_text_marker():
    source = (
        "Under the condition of comparable aggregate request rate, "
        "a more even request-load distribution reduces p99 latency "
        "during leader handoff."
    )
    hypothesis = _hypothesis(
        statement=source,
        prediction_rationale=(
            "The smaller latency increase is expected."
        ),
    )
    claim = _claim(
        text=(
            "Under the condition of comparable aggregate request rate, "
            "a smaller request-load distribution reduces p99 latency "
            "during leader handoff."
        ),
        basis=source,
        predicted="The smaller latency increase is expected.",
    )

    review = {
        "proposition_basis": source,
        "proposition_basis_source_contexts": [
            {
                "source_path": "hypothesis_statement",
                "sentence": source,
            }
        ],
        "prediction_observation_id": "prediction:p1",
        "reason_codes": list(_ORDERED_REASON_PAIR),
    }

    shadow = compile_ordered_marker_source_channel_shadow(
        hypothesis,
        claim,
        semantic_fidelity_shadow=review,
    )

    smaller = next(
        row
        for row in shadow["marker_provenance"]
        if row["marker"] == "smaller"
    )
    assert smaller["in_claim_text"] is True
    assert smaller["selected_prediction_exact_supported"] is True
    assert smaller["basis_or_context_exact_supported"] is False
    assert smaller["required_channel"] == (
        "CLAIM_TEXT_BASIS_OR_CONTEXT"
    )
    assert smaller["channel_supported"] is False
    assert "smaller" in shadow[
        "unsupported_markers_by_required_channel"
    ]


def test_missing_selected_prediction_fails_closed_for_prediction_only_marker():
    source = (
        "Under the condition of comparable aggregate request rate, "
        "a more even request-load distribution reduces p99 latency."
    )
    hypothesis = _hypothesis(
        statement=source,
        prediction_rationale="A smaller latency increase is expected.",
    )
    claim = _claim(
        text=source,
        basis=source,
        predicted="A smaller latency increase is expected.",
    )

    review = {
        "proposition_basis": source,
        "proposition_basis_source_contexts": [
            {
                "source_path": "hypothesis_statement",
                "sentence": source,
            }
        ],
        "prediction_observation_id": "prediction:missing",
        "reason_codes": list(_ORDERED_REASON_PAIR),
    }

    shadow = compile_ordered_marker_source_channel_shadow(
        hypothesis,
        claim,
        semantic_fidelity_shadow=review,
    )

    assert shadow["selected_prediction_source_found"] is False
    assert "smaller" in shadow[
        "unsupported_markers_by_required_channel"
    ]
    assert shadow["channel_aware_exact_status"] == "REVIEW_REQUIRED"


def test_decomposer_records_channel_shadow_only_in_diagnostic_record():
    source = inspect.getsource(NoveltyClaimDecomposer.decompose)

    assert "compile_ordered_marker_source_channel_shadow" in source
    assert '"ordered_marker_source_channel_shadow"' in source
