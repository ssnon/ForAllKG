from __future__ import annotations

from pipeline_core.discovery.atomic_scientific_novelty_projection import (
    project_compiled_atomic_specification_to_novelty_claim,
)
from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaim,
    NoveltyClaimScientificStructure,
)


def _spec() -> CompiledAtomicSpecification:
    return CompiledAtomicSpecification(
        local_id="atomic:1",
        claim_id="external_novelty_claim:1",
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=(
            "Factor M moderates the relation between descriptor D "
            "and outcome O."
        ),
        rationale="Atomic scientific relation.",
        source_candidate_ids=["candidate:1"],
        premise_statement_ids=["statement:1"],
        gap_statement_ids=["statement:gap"],
        prior_art_identity_terms=["Factor M"],
        relation_endpoint_anchors=["descriptor D", "outcome O"],
        scope_qualifier_spans=["under condition C"],
        directional_qualifier_spans=["increases"],
        relation_nucleus_terms=[
            "descriptor D",
            "outcome O",
            "increases",
        ],
        distinguishing_terms=["Factor M", "condition C"],
        required_bridge=(
            "Under condition C, Factor M moderates the relation between "
            "descriptor D and outcome O."
        ),
        observable="outcome O under condition C",
        predicted_observation=(
            "Outcome O increases with descriptor D when Factor M is high."
        ),
        falsification_condition=(
            "Outcome O does not change with descriptor D when Factor M "
            "is high."
        ),
        prediction_observation_id="prediction:1",
        falsification_criterion_id="falsifier:1",
        search_concepts=["Factor M", "descriptor D", "outcome O"],
        search_queries=["Factor M descriptor D outcome O"],
        scientific_structure=NoveltyClaimScientificStructure(),
        scientific_structure_reason_codes=[],
    )


def test_compiled_specification_is_projected_deterministically_to_novelty_view():
    spec = _spec()

    claim = project_compiled_atomic_specification_to_novelty_claim(
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        specification=spec,
    )

    assert claim.claim_id == spec.claim_id
    assert claim.hypothesis_id == "hypothesis:1"
    assert claim.claim_rank == 1
    assert claim.kind == spec.kind
    assert claim.importance == spec.importance
    assert claim.novelty_selection_role == spec.novelty_selection_role
    assert claim.text == spec.text
    assert claim.rationale == spec.rationale
    assert claim.search_concepts == spec.search_concepts
    assert claim.search_queries == spec.search_queries
    assert claim.distinguishing_terms == spec.distinguishing_terms
    assert claim.prior_art_identity_terms == spec.prior_art_identity_terms
    assert claim.relation_nucleus_terms == spec.relation_nucleus_terms
    assert claim.required_bridge == spec.required_bridge
    assert claim.predicted_observation == spec.predicted_observation
    assert claim.falsification_condition == spec.falsification_condition
    assert (
        claim.scientific_structure.model_dump(mode="json")
        == spec.scientific_structure.model_dump(mode="json")
    )
    assert (
        claim.scientific_structure_reason_codes
        == spec.scientific_structure_reason_codes
    )


def test_source_identity_remains_owned_by_compiled_specification():
    spec = _spec()
    claim = project_compiled_atomic_specification_to_novelty_claim(
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        specification=spec,
    )

    assert spec.prediction_observation_id == "prediction:1"
    assert spec.falsification_criterion_id == "falsifier:1"
    assert spec.relation_endpoint_anchors == [
        "descriptor D",
        "outcome O",
    ]
    assert spec.scope_qualifier_spans == ["under condition C"]
    assert spec.source_candidate_ids == ["candidate:1"]

    novelty_fields = set(NoveltyClaim.model_fields)
    assert "prediction_observation_id" not in novelty_fields
    assert "falsification_criterion_id" not in novelty_fields
    assert "relation_endpoint_anchors" not in novelty_fields
    assert "scope_qualifier_spans" not in novelty_fields
    assert "source_candidate_ids" not in novelty_fields

    assert not hasattr(claim, "prediction_observation_id")
    assert not hasattr(claim, "falsification_criterion_id")


def test_projection_uses_atomic_defaults_for_novelty_only_metadata():
    claim = project_compiled_atomic_specification_to_novelty_claim(
        hypothesis_id="hypothesis:1",
        claim_rank=2,
        specification=_spec(),
    )

    assert claim.higher_order_relation_basis == []
    assert claim.higher_order_component_claim_ids == []
    assert claim.inference_provenance is None
    assert claim.diagnostic_query_kind == "NONE"
    assert claim.diagnostic_search_query is None
    assert claim.diagnostic_execution_query is None
    assert claim.diagnostic_structural_terms == []
    assert claim.diagnostic_relation_terms == []
    assert claim.specification_sanitization_reason_codes == []


def test_projection_rejects_missing_hypothesis_or_invalid_rank():
    spec = _spec()

    try:
        project_compiled_atomic_specification_to_novelty_claim(
            hypothesis_id="",
            claim_rank=1,
            specification=spec,
        )
    except ValueError as exc:
        assert "hypothesis_id" in str(exc)
    else:
        raise AssertionError("expected missing hypothesis_id rejection")

    try:
        project_compiled_atomic_specification_to_novelty_claim(
            hypothesis_id="hypothesis:1",
            claim_rank=0,
            specification=spec,
        )
    except ValueError as exc:
        assert "claim_rank" in str(exc)
    else:
        raise AssertionError("expected invalid claim_rank rejection")
