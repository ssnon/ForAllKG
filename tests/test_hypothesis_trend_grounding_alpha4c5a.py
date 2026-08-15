from __future__ import annotations

import pytest

from dac_her.hypothesis_trend_grounding import (
    HYPOTHESIS_TREND_GROUNDING_CONTRACT_SEMANTICS_ID,
    HypothesisTrendRelationGrounding,
    capabilities_for_status,
)


def test_status_capabilities_are_non_majoritarian_and_fail_closed():
    repeated = capabilities_for_status(
        "repeated",
        directions=["positive"],
    )
    assert repeated["cross_context_replicated_premise_allowed"] is True
    assert repeated["directional_cross_paper_premise_allowed"] is True
    assert repeated["replication_gap_signal_allowed"] is False

    context = capabilities_for_status(
        "context_specific",
        directions=["positive", "non_monotonic"],
    )
    assert context["cross_context_replicated_premise_allowed"] is False
    assert context["context_dependency_premise_allowed"] is True
    assert context["directional_cross_paper_premise_allowed"] is False

    reversed_ = capabilities_for_status(
        "reversed",
        directions=["positive", "negative"],
    )
    assert reversed_["reversal_counterevidence_required"] is True
    assert reversed_["cross_context_replicated_premise_allowed"] is False
    assert reversed_["directional_cross_paper_premise_allowed"] is False

    insufficient = capabilities_for_status(
        "insufficient",
        directions=["positive"],
    )
    assert insufficient["cross_context_replicated_premise_allowed"] is False
    assert insufficient["replication_gap_signal_allowed"] is True
    assert insufficient["requires_verification"] is True


def test_repeated_mixed_direction_cannot_be_directional_premise():
    values = capabilities_for_status(
        "repeated",
        directions=["positive", "negative"],
    )
    assert values["cross_context_replicated_premise_allowed"] is True
    assert values["directional_cross_paper_premise_allowed"] is False


def _base(status: str):
    caps = capabilities_for_status(status, directions=["positive"])
    return dict(
        grounding_id="g",
        contract_semantics_id=
            HYPOTHESIS_TREND_GROUNDING_CONTRACT_SEMANTICS_ID,
        grounding_semantics_id="sers_test",
        domain_profile_id="sers_au_ag",
        relation_id="r",
        independent_variable_key="particle_size",
        dependent_observable_key="sers_performance",
        control_family="structural",
        observable_semantics="qualitative_sers_performance",
        local_result_ids=["l1"],
        paper_ids=["p1"],
        member_trend_ids=["t1"],
        directions=["positive"],
        shapes=["monotonic"],
        evidence_kinds=["reported_claim"],
        evidence_bases=["reported_directional_claim"],
        source_claim_ids=["c1"],
        source_node_ids=["c1"],
        cross_context_assessment_id="a1",
        cross_context_status=status,
        pairwise_contrast_ids=[],
        repeated_pair_ids=[],
        reversal_pair_ids=[],
        context_specific_pair_ids=[],
        unresolved_pair_ids=[],
        differentiating_dimensions=[],
        unresolved_dimensions=[],
        cross_context_reason_codes=["test"],
        **caps,
    )


def test_insufficient_cannot_be_promoted_to_replicated_support():
    row = _base("insufficient")
    row["cross_context_replicated_premise_allowed"] = True
    with pytest.raises(ValueError):
        HypothesisTrendRelationGrounding(**row)


def test_trend_grounding_never_authorizes_causal_or_universal_claim():
    row = _base("insufficient")
    row["causal_claim_allowed"] = True
    with pytest.raises(ValueError):
        HypothesisTrendRelationGrounding(**row)
