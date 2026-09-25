from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline_core.discovery.external_novelty_contracts import NoveltyClaim
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    PreVerifierContractGateV2Row,
    _count_exact_observation_bindings,
    classify_router_hint,
)


def _claim(
    *,
    kind: str = "moderator_interaction",
    prediction: str = "signal decreases",
    falsifier: str = "signal does not decrease",
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="claim:1",
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        kind=kind,
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="Increasing laser power decreases signal.",
        rationale="test",
        search_concepts=[],
        search_queries=[],
        prior_art_identity_terms=["gold substrate"],
        relation_nucleus_terms=["laser power", "signal"],
        required_bridge="Increasing laser power decreases signal.",
        predicted_observation=prediction,
        falsification_condition=falsifier,
    )


def _card(
    *,
    observable: str = "signal decreases",
    falsifying_outcome: str = "signal does not decrease",
    falsifier_observable: str = "signal decreases",
):
    return SimpleNamespace(
        predicted_observations=[
            SimpleNamespace(observable=observable)
        ],
        falsification_criteria=[
            SimpleNamespace(
                observable=falsifier_observable,
                falsifying_outcome=falsifying_outcome,
            )
        ],
    )


def test_router_prefers_decompose_for_unsupported_atomic_kind() -> None:
    assert classify_router_hint(
        binding_reason_codes=[],
        source_reason_codes=[
            "unsupported_atomic_claim_kind:composite",
            "prediction_exact_source_binding_cardinality:0",
        ],
    ) == "DECOMPOSE_OR_REGENERATE_REVIEW"


def test_router_uses_source_alignment_for_exact_binding_failure() -> None:
    assert classify_router_hint(
        binding_reason_codes=[],
        source_reason_codes=[
            "prediction_exact_source_binding_cardinality:0",
            "falsifier_exact_source_binding_cardinality:0",
        ],
    ) == "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"


def test_router_uses_specification_repair_for_0084_only_failure() -> None:
    assert classify_router_hint(
        binding_reason_codes=["missing_required_bridge"],
        source_reason_codes=[],
    ) == "SPECIFICATION_REPAIR_REVIEW"


def test_router_proceeds_only_when_both_contracts_are_clean() -> None:
    assert classify_router_hint(
        binding_reason_codes=[],
        source_reason_codes=[],
    ) == "PROCEED_TO_LITERAL_ENDPOINT_BINDING"


def test_router_explicitly_routes_missing_novelty_role_to_specification_repair() -> None:
    assert classify_router_hint(
        binding_reason_codes=[],
        source_reason_codes=["missing_novelty_selection_role"],
    ) == "SPECIFICATION_REPAIR_REVIEW"


def test_router_rejects_unknown_source_reason_instead_of_proceeding() -> None:
    with pytest.raises(
        ValueError,
        match="unclassified source-contract reason codes",
    ):
        classify_router_hint(
            binding_reason_codes=[],
            source_reason_codes=["future_source_contract_reason"],
        )


def test_router_rejects_unknown_binding_reason_instead_of_repairing() -> None:
    with pytest.raises(
        ValueError,
        match="unclassified binding-contract reason codes",
    ):
        classify_router_hint(
            binding_reason_codes=["future_binding_contract_reason"],
            source_reason_codes=[],
        )


def test_router_rejects_mixed_known_and_unknown_source_reasons() -> None:
    with pytest.raises(
        ValueError,
        match="unclassified source-contract reason codes",
    ):
        classify_router_hint(
            binding_reason_codes=[],
            source_reason_codes=[
                "prediction_exact_source_binding_cardinality:0",
                "future_source_contract_reason",
            ],
        )


def test_not_ready_row_cannot_carry_proceed_route() -> None:
    with pytest.raises(
        ValueError,
        match="router may proceed iff claim is contract-ready",
    ):
        PreVerifierContractGateV2Row(
            candidate_hypothesis_id="hypothesis:candidate",
            final_hypothesis_id="hypothesis:final",
            claim_id="claim:1",
            novelty_selection_role=None,
            claim_kind="mechanistic_link",
            binding_plan_status="INELIGIBLE_INCOMPLETE_ATOMIC_SPECIFICATION",
            binding_contract_reason_codes=[],
            source_contract_reason_codes=["missing_novelty_selection_role"],
            gate_status="NOT_READY_FOR_LITERAL_ENDPOINT_BINDING",
            router_hint="PROCEED_TO_LITERAL_ENDPOINT_BINDING",
            atomic_kind_supported=True,
            prediction_exact_source_binding_count=1,
            falsifier_exact_source_binding_count=1,
            shared_observable_identity_satisfied=True,
        )


def test_exact_observation_binding_counts_and_shared_identity() -> None:
    p, f, shared = _count_exact_observation_bindings(
        candidate_card=_card(),
        claim=_claim(),
    )
    assert (p, f, shared) == (1, 1, True)


def test_observable_identity_mismatch_is_visible_before_endpoint() -> None:
    p, f, shared = _count_exact_observation_bindings(
        candidate_card=_card(
            falsifier_observable="different observable"
        ),
        claim=_claim(),
    )
    assert (p, f, shared) == (1, 1, False)


def test_missing_prediction_binding_count_is_zero() -> None:
    p, f, shared = _count_exact_observation_bindings(
        candidate_card=_card(observable="different prediction"),
        claim=_claim(),
    )
    assert p == 0
    assert f == 1
    assert shared is None
