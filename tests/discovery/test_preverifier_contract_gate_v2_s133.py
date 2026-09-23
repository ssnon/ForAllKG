from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import NoveltyClaim
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
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
