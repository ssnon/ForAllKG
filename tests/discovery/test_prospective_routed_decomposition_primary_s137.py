from __future__ import annotations

from pipeline_core.discovery.external_novelty_contracts import NoveltyClaim
from pipeline_core.discovery.prospective_routed_decomposition_primary import (
    assess_deterministic_decomposition,
)


def _claim(
    claim_id: str,
    *,
    kind: str,
    components: list[str] | None = None,
    basis: list[str] | None = None,
    topology_reasons: list[str] | None = None,
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        kind=kind,
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="claim text",
        rationale="rationale",
        prior_art_identity_terms=["identity"],
        relation_nucleus_terms=["left", "right"],
        required_bridge="bridge",
        predicted_observation="prediction",
        falsification_condition="falsifier",
        higher_order_component_claim_ids=list(components or []),
        higher_order_relation_basis=list(basis or []),
        higher_order_relation_reason_codes=list(topology_reasons or []),
    )


def test_two_explicit_atomic_components_are_deterministically_available() -> None:
    a = _claim("claim:a", kind="mechanistic_link")
    b = _claim("claim:b", kind="moderator_interaction")
    composite = _claim(
        "claim:c",
        kind="composite",
        components=["claim:a", "claim:b"],
        basis=["Explicit higher-order relation."],
    )
    result = assess_deterministic_decomposition(
        composite_claim=composite,
        source_claims_by_id={
            "claim:a": a,
            "claim:b": b,
            "claim:c": composite,
        },
    )
    assert result.status == "DETERMINISTIC_DECOMPOSITION_AVAILABLE"
    assert result.component_claim_ids == ["claim:a", "claim:b"]
    assert result.reason_codes == []


def test_single_component_is_not_sufficient_decomposition() -> None:
    a = _claim("claim:a", kind="context_condition")
    composite = _claim(
        "claim:c",
        kind="composite",
        components=["claim:a"],
        basis=["Explicit higher-order relation."],
    )
    result = assess_deterministic_decomposition(
        composite_claim=composite,
        source_claims_by_id={
            "claim:a": a,
            "claim:c": composite,
        },
    )
    assert result.status == "DETERMINISTIC_DECOMPOSITION_UNAVAILABLE"
    assert "insufficient_explicit_component_cardinality:1" in result.reason_codes


def test_missing_component_is_fail_closed() -> None:
    a = _claim("claim:a", kind="mechanistic_link")
    composite = _claim(
        "claim:c",
        kind="composite",
        components=["claim:a", "claim:missing"],
        basis=["Explicit higher-order relation."],
    )
    result = assess_deterministic_decomposition(
        composite_claim=composite,
        source_claims_by_id={
            "claim:a": a,
            "claim:c": composite,
        },
    )
    assert result.status == "DETERMINISTIC_DECOMPOSITION_UNAVAILABLE"
    assert "missing_component_claim:claim:missing" in result.reason_codes


def test_non_atomic_component_is_fail_closed() -> None:
    a = _claim("claim:a", kind="mechanistic_link")
    b = _claim("claim:b", kind="composite")
    composite = _claim(
        "claim:c",
        kind="composite",
        components=["claim:a", "claim:b"],
        basis=["Explicit higher-order relation."],
    )
    result = assess_deterministic_decomposition(
        composite_claim=composite,
        source_claims_by_id={
            "claim:a": a,
            "claim:b": b,
            "claim:c": composite,
        },
    )
    assert result.status == "DETERMINISTIC_DECOMPOSITION_UNAVAILABLE"
    assert (
        "unsupported_component_atomic_kind:claim:b:composite"
        in result.reason_codes
    )


def test_missing_higher_order_basis_is_fail_closed() -> None:
    a = _claim("claim:a", kind="mechanistic_link")
    b = _claim("claim:b", kind="moderator_interaction")
    composite = _claim(
        "claim:c",
        kind="composite",
        components=["claim:a", "claim:b"],
        basis=[],
    )
    result = assess_deterministic_decomposition(
        composite_claim=composite,
        source_claims_by_id={
            "claim:a": a,
            "claim:b": b,
            "claim:c": composite,
        },
    )
    assert result.status == "DETERMINISTIC_DECOMPOSITION_UNAVAILABLE"
    assert "missing_higher_order_relation_basis" in result.reason_codes


def test_topology_reason_codes_block_deterministic_decomposition() -> None:
    a = _claim("claim:a", kind="mechanistic_link")
    b = _claim("claim:b", kind="moderator_interaction")
    composite = _claim(
        "claim:c",
        kind="composite",
        components=["claim:a", "claim:b"],
        basis=["Explicit higher-order relation."],
        topology_reasons=["ambiguous_component_binding"],
    )
    result = assess_deterministic_decomposition(
        composite_claim=composite,
        source_claims_by_id={
            "claim:a": a,
            "claim:b": b,
            "claim:c": composite,
        },
    )
    assert result.status == "DETERMINISTIC_DECOMPOSITION_UNAVAILABLE"
    assert "higher_order_relation_provenance_not_clean" in result.reason_codes
