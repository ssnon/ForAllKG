from __future__ import annotations

from domains.registry import get_domain_profile
from pipeline_core.discovery.projection_relation_adjudication import (
    _candidate_compatibility,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificConceptIR,
    ScientificRelationIR,
)


def _concept(
    concept_id: str,
    role: str,
    text: str,
    index: int,
    type_labels: list[str],
) -> ScientificConceptIR:
    return ScientificConceptIR(
        concept_id=concept_id,
        role=role,
        source_field=role.lower(),
        source_index=index,
        surface_text=text,
        normalized_text=text.casefold(),
        lexical_tokens=text.casefold().replace("-", " ").split(),
        type_labels=type_labels,
        ambiguity_labels=[],
        literal_in_claim_text=True,
        literal_in_required_bridge=(
            None if role == "OBSERVABLE" else True
        ),
    )


def _p02_repeatability_relation() -> ScientificRelationIR:
    return ScientificRelationIR(
        relation_ir_id="relation:p02-repeatability",
        hypothesis_id="hypothesis:p02",
        claim_id="claim:p02-repeatability",
        claim_kind="moderator_interaction",
        novelty_selection_role="NOVELTY_BEARING",
        claim_text=(
            "Substrate composition changes the sensitivity of measurement "
            "repeatability to the same defined nanostructure-design "
            "modification."
        ),
        endpoint_concepts=[
            _concept(
                "e1",
                "RELATION_ENDPOINT",
                "measurement repeatability",
                0,
                ["measurement_reproducibility_metric"],
            ),
            _concept(
                "e2",
                "RELATION_ENDPOINT",
                "nanostructure-design modification",
                1,
                ["nanostructure_design_variable"],
            ),
        ],
        identity_concepts=[
            _concept(
                "i1",
                "BRANCH_IDENTITY",
                "substrate composition",
                0,
                ["material_composition_variable"],
            ),
        ],
        observable_concept=_concept(
            "o1",
            "OBSERVABLE",
            (
                "The change in measurement reproducibility produced by a "
                "defined nanostructure-design modification differs between "
                "substrate compositions."
            ),
            0,
            [
                "measurement_reproducibility_metric",
                "nanostructure_design_variable",
                "material_composition_variable",
            ],
        ),
        relation_type_labels=[
            "measurement_reproducibility_metric",
            "nanostructure_design_variable",
            "material_composition_variable",
        ],
        relation_domain_labels=[],
        relation_scope_features=[],
        typing_status="READY",
        reason_codes=[],
        source_contract="relational-atomic-projection-report-v1",
    )


def test_generic_type_overlap_without_sers_context_is_neighboring() -> None:
    state, relation_types, document_types, _, document_domains, reasons = (
        _candidate_compatibility(
            relation=_p02_repeatability_relation(),
            document=(
                "Simulation design and measurement of welding robot "
                "repeatability utilizing a contact measurement method."
            ),
            domain_profile=get_domain_profile("sers_au_ag"),
        )
    )

    assert "measurement_reproducibility_metric" in relation_types
    assert "measurement_reproducibility_metric" in document_types
    assert document_domains == []
    assert state == "NEIGHBORING_SCOPE"
    assert reasons == [
        "typed_concept_overlap_without_explicit_domain_compatibility"
    ]


def test_generic_type_overlap_with_explicit_sers_context_is_typed() -> None:
    state, relation_types, document_types, _, document_domains, reasons = (
        _candidate_compatibility(
            relation=_p02_repeatability_relation(),
            document=(
                "SERS measurement repeatability was compared across "
                "nanostructure-design modifications in plasmonic substrates."
            ),
            domain_profile=get_domain_profile("sers_au_ag"),
        )
    )

    assert "measurement_reproducibility_metric" in relation_types
    assert "measurement_reproducibility_metric" in document_types
    assert "SERS" in document_domains
    assert state == "TYPED_COMPATIBLE"
    assert reasons == []
