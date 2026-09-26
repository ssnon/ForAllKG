
from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.novelty_atomic_semantic_fidelity import (
    assess_atomic_semantic_fidelity,
    compile_atomic_semantic_fidelity_taxonomy_shadow,
)


def _hypothesis(*, statement: str, bridge: str = ""):
    return SimpleNamespace(
        hypothesis_statement=statement,
        inferential_bridge=bridge,
        assumptions=[],
        predicted_observations=[],
        falsification_criteria=[],
    )


def _claim(*, text: str, basis: str, bridge: str = ""):
    return NoveltyClaimDraft(
        local_id="c1",
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=text,
        rationale="fixture",
        search_concepts=["factor", "outcome"],
        search_queries=["factor outcome"],
        prior_art_identity_terms=["factor"],
        relation_nucleus_terms=["modulates"],
        required_bridge=bridge,
        predicted_observation="",
        falsification_condition="",
        semantic_fidelity_binding=NoveltyClaimSemanticFidelityBindingDraft(
            proposition_basis=basis,
            relation_endpoint_anchors=["Factor M", "outcome O"],
        ),
    )


def test_later_coordinated_branch_condition_is_diagnostic_not_blocking():
    basis = "Factor M modulates outcome O"
    source = (
        "Factor M modulates outcome O, and a second response changes "
        "under condition C."
    )
    claim = _claim(
        text=basis + ".",
        basis=basis,
        bridge=basis + ".",
    )

    review = assess_atomic_semantic_fidelity(
        _hypothesis(statement=source, bridge=basis + "."),
        claim,
    )

    assert review["source_context_condition_markers"] == ["under"]
    assert review["source_context_bound_condition_markers"] == []
    assert review["source_context_ambiguous_condition_markers"] == ["under"]
    assert "atomic_claim_context_condition_scope_may_be_dropped" not in review["reason_codes"]
    assert "required_bridge_condition_scope_may_be_dropped" not in review["reason_codes"]

    taxonomy = compile_atomic_semantic_fidelity_taxonomy_shadow(
        claim,
        semantic_fidelity_shadow=review,
        sanitized_required_bridge=claim.required_bridge,
        sanitized_predicted_observation="",
        sanitized_falsification_condition="",
    )
    assert taxonomy["binding_contract"]["status"] == "VALID"
    assert taxonomy["claim_fidelity"]["status"] == "NO_FLAG"


def test_leading_same_branch_condition_remains_blocking():
    basis = "Factor M modulates outcome O"
    source = "Under condition C, Factor M modulates outcome O."
    claim = _claim(
        text=basis + ".",
        basis=basis,
        bridge=basis + ".",
    )

    review = assess_atomic_semantic_fidelity(
        _hypothesis(statement=source, bridge=basis + "."),
        claim,
    )

    assert review["source_context_bound_condition_markers"] == ["Under"]
    assert review["source_context_ambiguous_condition_markers"] == []
    assert "atomic_claim_context_condition_scope_may_be_dropped" in review["reason_codes"]
    assert "required_bridge_condition_scope_may_be_dropped" in review["reason_codes"]


def test_trailing_same_branch_condition_remains_blocking():
    basis = "Factor M modulates outcome O"
    source = "Factor M modulates outcome O under condition C."
    claim = _claim(
        text=basis + ".",
        basis=basis,
        bridge=basis + ".",
    )

    review = assess_atomic_semantic_fidelity(
        _hypothesis(statement=source, bridge=basis + "."),
        claim,
    )

    assert review["source_context_bound_condition_markers"] == ["under"]
    assert review["source_context_ambiguous_condition_markers"] == []
    assert "atomic_claim_context_condition_scope_may_be_dropped" in review["reason_codes"]


def test_p40_shape_does_not_promote_second_conjunct_condition():
    basis = (
        "Substrate thermal conductivity moderates the "
        "laser-power-dependent local temperature rise at plasmonic hotspots"
    )
    source = (
        basis
        + ", and this thermal response is associated with "
        "laser-power-dependent variation in SERS spectra under otherwise "
        "comparable hotspot conditions."
    )

    claim = NoveltyClaimDraft(
        local_id="claim_1",
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=basis + ".",
        rationale="fixture",
        search_concepts=["thermal conductivity"],
        search_queries=["thermal conductivity hotspot temperature"],
        prior_art_identity_terms=["substrate thermal conductivity"],
        relation_nucleus_terms=["laser power", "local temperature rise", "moderates"],
        required_bridge=(
            "The proposed inference is that "
            + basis[0].lower()
            + basis[1:]
            + ", with the direction of that moderation left open."
        ),
        predicted_observation="",
        falsification_condition="",
        semantic_fidelity_binding=NoveltyClaimSemanticFidelityBindingDraft(
            proposition_basis=basis,
            relation_endpoint_anchors=[
                "substrate thermal conductivity",
                "laser-power-dependent local temperature rise",
                "plasmonic hotspots",
            ],
        ),
    )

    review = assess_atomic_semantic_fidelity(
        _hypothesis(statement=source, bridge=claim.required_bridge),
        claim,
    )

    assert review["source_context_condition_markers"] == ["under"]
    assert review["source_context_bound_condition_markers"] == []
    assert review["source_context_ambiguous_condition_markers"] == ["under"]
    assert "atomic_claim_context_condition_scope_may_be_dropped" not in review["reason_codes"]
    assert "required_bridge_condition_scope_may_be_dropped" not in review["reason_codes"]
