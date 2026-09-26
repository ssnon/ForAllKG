from __future__ import annotations

from domains.registry import get_domain_profile
from domains.sers.relation_ir_semantics import (
    SERS_RELATION_TYPING_ADAPTER,
)
from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)
from pipeline_core.discovery.scientific_relation_ir import (
    assess_relation_document_compatibility,
    compile_atomic_specification_relation_ir,
)


def _spec(
    *,
    claim_id: str = "claim:test",
    text: str,
    endpoints: list[str],
    identity: list[str],
    bridge: str,
    observable: str,
) -> CompiledAtomicSpecification:
    return CompiledAtomicSpecification(
        local_id="ATOM_01",
        claim_id=claim_id,
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text=text,
        rationale="test",
        source_candidate_ids=["candidate:1"],
        premise_statement_ids=["p1"],
        gap_statement_ids=["g1"],
        prior_art_identity_terms=identity,
        relation_endpoint_anchors=endpoints,
        scope_qualifier_spans=[],
        directional_qualifier_spans=[],
        relation_nucleus_terms=endpoints,
        distinguishing_terms=[],
        required_bridge=bridge,
        observable=observable,
        predicted_observation=bridge,
        falsification_condition=bridge,
        prediction_observation_id="pred:1",
        falsification_criterion_id="false:1",
        search_concepts=[],
        search_queries=["test"],
        scientific_structure=NoveltyClaimScientificStructure(),
        scientific_structure_reason_codes=[],
    )


def test_sers_relation_ir_types_plasmonic_hotspot_in_context():
    profile = get_domain_profile("sers_au_ag")
    spec = _spec(
        text=(
            "Under plasmonic hotspot accessibility, hotspot population "
            "stability is associated with SERS calibration transfer error."
        ),
        endpoints=[
            "hotspot population stability",
            "SERS calibration transfer error",
        ],
        identity=["plasmonic hotspot accessibility"],
        bridge=(
            "Under plasmonic hotspot accessibility, hotspot population "
            "stability should determine SERS calibration transfer error."
        ),
        observable=(
            "hotspot population stability and SERS calibration transfer error"
        ),
    )

    relation = compile_atomic_specification_relation_ir(
        hypothesis_id="hypothesis:1",
        spec=spec,
        domain_profile=profile,
        typing_adapter=SERS_RELATION_TYPING_ADAPTER,
    )

    assert relation.typing_status == "READY"
    assert (
        "electromagnetic_plasmonic_hotspot"
        in relation.relation_type_labels
    )
    assert relation.relation_domain_labels == ["SERS"]


def test_typed_relation_flags_explosive_hotspot_as_identity_conflict():
    profile = get_domain_profile("sers_au_ag")
    spec = _spec(
        text=(
            "Under plasmonic hotspot accessibility, hotspot population "
            "stability is associated with SERS calibration transfer error."
        ),
        endpoints=[
            "hotspot population stability",
            "SERS calibration transfer error",
        ],
        identity=["plasmonic hotspot accessibility"],
        bridge=(
            "Under plasmonic hotspot accessibility, hotspot population "
            "stability should determine SERS calibration transfer error."
        ),
        observable=(
            "hotspot population stability and SERS calibration transfer error"
        ),
    )
    relation = compile_atomic_specification_relation_ir(
        hypothesis_id="hypothesis:1",
        spec=spec,
        domain_profile=profile,
        typing_adapter=SERS_RELATION_TYPING_ADAPTER,
    )

    diagnostic = assess_relation_document_compatibility(
        relation=relation,
        document_text=(
            "Shock-induced hotspot formation in HMX energetic materials "
            "controls detonation initiation."
        ),
        domain_profile=profile,
        typing_adapter=SERS_RELATION_TYPING_ADAPTER,
    )

    assert diagnostic.state == "TYPE_IDENTITY_CONFLICT"
    assert diagnostic.conflicting_type_pairs == [
        [
            "electromagnetic_plasmonic_hotspot",
            "energetic_material_hotspot",
        ]
    ]


def test_bare_ris_is_ambiguous_but_expanded_ris_is_typed():
    adapter = SERS_RELATION_TYPING_ADAPTER

    bare_types = adapter.type_labels(
        surface_text="RIS",
        relation_context="RIS changes with morphology.",
    )
    bare_ambiguity = adapter.ambiguity_labels(
        surface_text="RIS",
        relation_context="RIS changes with morphology.",
        type_labels=bare_types,
    )
    assert bare_types == ()
    assert "RIS_expansion_not_source_explicit" in bare_ambiguity

    expanded_types = adapter.type_labels(
        surface_text="RIS",
        relation_context=(
            "Refractive index sensitivity (RIS) changes with morphology."
        ),
    )
    expanded_ambiguity = adapter.ambiguity_labels(
        surface_text="RIS",
        relation_context=(
            "Refractive index sensitivity (RIS) changes with morphology."
        ),
        type_labels=expanded_types,
    )
    assert expanded_types == ("refractive_index_sensitivity",)
    assert "RIS_expansion_not_source_explicit" not in expanded_ambiguity


def test_relation_ir_fails_closed_on_nonliteral_endpoint():
    profile = get_domain_profile("sers_au_ag")
    spec = _spec(
        text="A SERS response changes with nanogap size.",
        endpoints=[
            "nonexistent endpoint",
            "SERS response",
        ],
        identity=["nanogap size"],
        bridge=(
            "Nanogap size should determine SERS response while "
            "nonexistent endpoint is considered."
        ),
        observable="SERS response",
    )

    relation = compile_atomic_specification_relation_ir(
        hypothesis_id="hypothesis:2",
        spec=spec,
        domain_profile=profile,
        typing_adapter=SERS_RELATION_TYPING_ADAPTER,
    )

    assert relation.typing_status == "STRUCTURALLY_INVALID"
    assert any(
        code.startswith("missing_literal:claim_text:")
        for code in relation.reason_codes
    )
