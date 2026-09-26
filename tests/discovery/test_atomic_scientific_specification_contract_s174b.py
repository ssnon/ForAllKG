from __future__ import annotations

from typing import get_args

from pipeline_core.discovery.atomic_scientific_specification import (
    AtomicClaimKind,
    AtomicSelectionRole,
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicClaimKind as CompatibilityAtomicClaimKind,
    AtomicSelectionRole as CompatibilityAtomicSelectionRole,
    CompiledAtomicSpecification as CompatibilityCompiledAtomicSpecification,
)


EXPECTED_ATOMIC_KINDS = (
    "mediator",
    "moderator_interaction",
    "context_condition",
    "pathway_competition",
    "descriptor_interaction",
    "distinctive_prediction",
    "mechanistic_link",
)

EXPECTED_SELECTION_ROLES = (
    "NOVELTY_BEARING",
    "REQUIRED_ENABLING_RELATION",
    "TESTING_PREDICTION",
    "AUXILIARY",
)


def _payload() -> dict:
    return {
        "local_id": "atomic:1",
        "claim_id": "external_novelty_claim:1",
        "kind": "moderator_interaction",
        "importance": "core",
        "novelty_selection_role": "NOVELTY_BEARING",
        "text": (
            "Factor M moderates the relation between descriptor D "
            "and outcome O."
        ),
        "rationale": "Atomic scientific relation.",
        "source_candidate_ids": ["candidate:1"],
        "premise_statement_ids": ["statement:1"],
        "gap_statement_ids": ["statement:gap"],
        "prior_art_identity_terms": ["Factor M"],
        "relation_endpoint_anchors": ["descriptor D", "outcome O"],
        "scope_qualifier_spans": ["under condition C"],
        "directional_qualifier_spans": ["increases"],
        "relation_nucleus_terms": [
            "descriptor D",
            "outcome O",
            "increases",
        ],
        "distinguishing_terms": ["Factor M", "condition C"],
        "required_bridge": (
            "Under condition C, Factor M moderates the relation between "
            "descriptor D and outcome O."
        ),
        "observable": "outcome O under condition C",
        "predicted_observation": (
            "Outcome O increases with descriptor D when Factor M is high."
        ),
        "falsification_condition": (
            "Outcome O does not change with descriptor D when Factor M "
            "is high."
        ),
        "prediction_observation_id": "prediction:1",
        "falsification_criterion_id": "falsifier:1",
        "search_concepts": ["Factor M", "descriptor D", "outcome O"],
        "search_queries": ["Factor M descriptor D outcome O"],
        "scientific_structure": (
            NoveltyClaimScientificStructure().model_dump(mode="json")
        ),
        "scientific_structure_reason_codes": [],
    }


def test_neutral_contract_owns_exact_existing_atomic_literals():
    assert get_args(AtomicClaimKind) == EXPECTED_ATOMIC_KINDS
    assert get_args(AtomicSelectionRole) == EXPECTED_SELECTION_ROLES
    assert CompatibilityAtomicClaimKind == AtomicClaimKind
    assert CompatibilityAtomicSelectionRole == AtomicSelectionRole


def test_legacy_reframing_import_path_reexports_same_compiled_class():
    assert (
        CompatibilityCompiledAtomicSpecification
        is CompiledAtomicSpecification
    )
    assert CompiledAtomicSpecification.__module__ == (
        "pipeline_core.discovery.atomic_scientific_specification"
    )


def test_compiled_atomic_specification_payload_and_schema_are_stable():
    payload = _payload()

    neutral = CompiledAtomicSpecification.model_validate(payload)
    compatibility = (
        CompatibilityCompiledAtomicSpecification.model_validate(payload)
    )

    assert neutral.model_dump(mode="json") == payload
    assert compatibility.model_dump(mode="json") == payload
    assert neutral.model_dump(mode="json") == compatibility.model_dump(
        mode="json"
    )
    assert (
        CompiledAtomicSpecification.model_json_schema()
        == CompatibilityCompiledAtomicSpecification.model_json_schema()
    )


def test_neutral_contract_does_not_add_authority_fields():
    fields = set(CompiledAtomicSpecification.model_fields)

    assert "production_authority" not in fields
    assert "novelty_authority" not in fields
    assert "semantic_fidelity_status" not in fields
    assert "source_reference_status" not in fields
