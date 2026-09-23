from __future__ import annotations

import pytest

from domains.registry import get_domain_profile
from domains.sers.relation_ir_semantics import (
    SERS_RELATION_TYPING_ADAPTER,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.scientific_relation_ir import (
    compile_atomic_specification_relation_ir,
)


def _spec(
    *,
    claim_id: str,
    text: str,
    endpoints: list[str],
    identity: list[str],
    bridge: str,
    observable: str,
) -> CompiledAtomicSpecification:
    return CompiledAtomicSpecification(
        local_id="RELATIONAL_ATOMIC_PROJECTION_1",
        claim_id=claim_id,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=text,
        rationale="frozen source claim",
        source_candidate_ids=["hypothesis:candidate"],
        premise_statement_ids=[],
        gap_statement_ids=[],
        prior_art_identity_terms=identity,
        relation_endpoint_anchors=endpoints,
        scope_qualifier_spans=[],
        directional_qualifier_spans=[],
        relation_nucleus_terms=endpoints,
        distinguishing_terms=[],
        required_bridge=bridge,
        observable=observable,
        predicted_observation=observable,
        falsification_condition=observable,
        prediction_observation_id="prediction:1",
        falsification_criterion_id="falsifier:1",
        search_concepts=[],
        search_queries=["source bounded query"],
        scientific_structure=NoveltyClaimScientificStructure(),
        scientific_structure_reason_codes=[],
    )


def test_enhancement_factor_requires_explicit_sers_family_context() -> None:
    adapter = SERS_RELATION_TYPING_ADAPTER

    assert adapter.type_labels(
        surface_text="enhancement factor",
        relation_context="generic analytical response",
    ) == ()

    assert adapter.type_labels(
        surface_text="enhancement factor",
        relation_context=(
            "SERS enhancement factor changes with nanostructure geometry."
        ),
    ) == ("sers_enhancement_metric",)


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        (
            "measurement repeatability",
            "measurement_reproducibility_metric",
        ),
        (
            "measurement reproducibility",
            "measurement_reproducibility_metric",
        ),
        (
            "relative standard deviation",
            "measurement_reproducibility_metric",
        ),
        ("RSD", "measurement_reproducibility_metric"),
        (
            "nanostructure-design modification",
            "nanostructure_design_variable",
        ),
        (
            "nanostructure spacing",
            "nanostructure_design_variable",
        ),
        (
            "substrate composition",
            "material_composition_variable",
        ),
        (
            "composition of the substrate",
            "material_composition_variable",
        ),
    ],
)
def test_general_relation_concept_families_are_surface_typed(
    surface: str,
    expected: str,
) -> None:
    labels = SERS_RELATION_TYPING_ADAPTER.type_labels(
        surface_text=surface,
        relation_context="unrelated neighboring relation text",
    )
    assert expected in labels


def test_p02_enhancement_relation_becomes_ready_without_identity_leakage() -> None:
    profile = get_domain_profile("sers_au_ag")
    text = (
        "Substrate composition changes the sensitivity of SERS enhancement "
        "factor to a defined nanostructure-design modification."
    )
    bridge = (
        "The proposed relation is that substrate composition changes the "
        "sensitivity of SERS enhancement factor to a defined "
        "nanostructure-design modification"
    )
    observable = (
        "The change in enhancement factor produced by a defined "
        "nanostructure-design modification differs between substrate "
        "compositions."
    )

    relation = compile_atomic_specification_relation_ir(
        hypothesis_id="hypothesis:p02",
        spec=_spec(
            claim_id="claim:p02-ef",
            text=text,
            endpoints=[
                "SERS enhancement factor",
                "defined nanostructure-design modification",
            ],
            identity=["substrate composition"],
            bridge=bridge,
            observable=observable,
        ),
        domain_profile=profile,
        typing_adapter=SERS_RELATION_TYPING_ADAPTER,
    )

    assert relation.typing_status == "READY"
    assert set(relation.relation_type_labels) >= {
        "sers_enhancement_metric",
        "nanostructure_design_variable",
        "material_composition_variable",
    }
    assert [
        row.surface_text for row in relation.endpoint_concepts
    ] == [
        "SERS enhancement factor",
        "defined nanostructure-design modification",
    ]
    assert [
        row.surface_text for row in relation.identity_concepts
    ] == ["substrate composition"]


def test_p02_repeatability_relation_becomes_ready() -> None:
    profile = get_domain_profile("sers_au_ag")
    text = (
        "Substrate composition changes the sensitivity of measurement "
        "repeatability to the same defined nanostructure-design modification."
    )
    bridge = (
        "The proposed relation is also that substrate composition changes "
        "the sensitivity of measurement repeatability to the same "
        "nanostructure-design modification"
    )
    observable = (
        "The change in measurement reproducibility produced by a defined "
        "nanostructure-design modification differs between substrate "
        "compositions."
    )

    relation = compile_atomic_specification_relation_ir(
        hypothesis_id="hypothesis:p02",
        spec=_spec(
            claim_id="claim:p02-repeatability",
            text=text,
            endpoints=[
                "measurement repeatability",
                "nanostructure-design modification",
            ],
            identity=["substrate composition"],
            bridge=bridge,
            observable=observable,
        ),
        domain_profile=profile,
        typing_adapter=SERS_RELATION_TYPING_ADAPTER,
    )

    assert relation.typing_status == "READY"
    assert set(relation.relation_type_labels) >= {
        "measurement_reproducibility_metric",
        "nanostructure_design_variable",
        "material_composition_variable",
    }


def test_existing_bare_ris_ambiguity_contract_is_unchanged() -> None:
    adapter = SERS_RELATION_TYPING_ADAPTER
    types = adapter.type_labels(
        surface_text="RIS",
        relation_context="RIS changes with morphology.",
    )
    ambiguity = adapter.ambiguity_labels(
        surface_text="RIS",
        relation_context="RIS changes with morphology.",
        type_labels=types,
    )
    assert types == ()
    assert "RIS_expansion_not_source_explicit" in ambiguity
    assert "bare_RIS_without_refractive_index_identity" in ambiguity


def test_existing_hotspot_cross_domain_collision_is_unchanged() -> None:
    adapter = SERS_RELATION_TYPING_ADAPTER
    context = (
        "Plasmonic SERS hotspot behavior is compared with shock-induced "
        "HMX energetic-material hotspot behavior."
    )
    types = adapter.type_labels(
        surface_text="hotspot",
        relation_context=context,
    )
    ambiguity = adapter.ambiguity_labels(
        surface_text="hotspot",
        relation_context=context,
        type_labels=types,
    )
    assert {
        "electromagnetic_plasmonic_hotspot",
        "energetic_material_hotspot",
    }.issubset(types)
    assert "hotspot_cross_domain_identity_collision" in ambiguity
