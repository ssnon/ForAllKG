from __future__ import annotations

import pytest

from pipeline_core.discovery.higher_order_hypothesis_context import (
    HigherOrderShadowHypothesisPromptAssembler,
    materialize_higher_order_hypothesis_context,
)
from pipeline_core.discovery.higher_order_synthesis_context import (
    HigherOrderStructuralOpportunityView,
    HigherOrderSynthesisContext,
    HigherOrderSynthesisPremiseView,
)
from pipeline_core.discovery.higher_order_topology_carrier import (
    HigherOrderSynthesisLineageView,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
)
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
)


def _source_context() -> HypothesisContext:
    return HypothesisContext(
        context_id="hypothesis_context:source",
        context_sha256="source-context-sha",
        source_packet_id="packet:test",
        source_packet_sha256="packet-sha",
        source_report_id="report:test",
        source_report_sha256="report-sha",
        task_id="task:test",
        question=(
            "How does interparticle separation affect SERS intensity?"
        ),
        corpus_id="corpus:test",
        domain_profile_id="domain:test",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="stmt:positive:1",
                text=(
                    "Decreasing separation changes plasmon coupling "
                    "and the measured optical response."
                ),
                epistemic_role="reported",
                claim_kind="mechanistic_relation",
                paper_ids=["paper:1"],
                requires_verification=False,
                eligible_as_premise=True,
                eligible_as_gap=False,
            ),
            HypothesisEvidenceStatement(
                statement_id="stmt:restricted:existing",
                text="Existing restricted context.",
                epistemic_role="navigation_note",
                claim_kind="navigation",
                paper_ids=[],
                requires_verification=False,
                eligible_as_premise=False,
                eligible_as_gap=False,
                premise_restrictions=["existing_restriction"],
            ),
        ],
    )


def _higher_order_context(
    *,
    modifier_authority: RelationComponentAuthority = (
        RelationComponentAuthority.CONFIRMED_KNOWN
    ),
) -> HigherOrderSynthesisContext:
    modifier_use = (
        "confirmed_known_component"
        if modifier_authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
        else "candidate_inspiration_component"
    )

    modifier_provenance = (
        "accepted_pattern:modifier"
        if modifier_authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
        else "candidate_unit:modifier"
    )

    lineage = HigherOrderSynthesisLineageView(
        higher_order_topology_id="higher_order_topology:test",
        backbone_topology_id="relational_topology:test",
        source_component_id="component:source",
        target_component_id="component:target",
        modifier_component_id="component:modifier",
        source_authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        target_authority=RelationComponentAuthority.CONFIRMED_KNOWN,
        modifier_authority=modifier_authority,
        source_provenance_source_id="accepted_pattern:source",
        target_provenance_source_id="accepted_pattern:target",
        modifier_provenance_source_id=modifier_provenance,
        source_endpoint_binding_authority="exact",
        target_endpoint_binding_authority="exact",
        source_endpoint_equivalence_witness_id=None,
        target_endpoint_equivalence_witness_id=None,
        modifier_eligibility_witness_id="modifier_witness:test",
        modifier_anchor_role="mediator",
        modifier_anchor_slot="object",
        modifier_slot="subject",
        modifier_anchor_text="electric-field enhancement",
        modifier_text="nanoparticle edge sharpness",
    )

    return HigherOrderSynthesisContext(
        context_id="higher_order_synthesis_context:test",
        carrier_id="higher_order_synthesis_carrier:test",
        higher_order_topology_id="higher_order_topology:test",
        requested_source="interparticle separation",
        requested_target="SERS intensity",
        lineage=lineage,
        premises=[
            HigherOrderSynthesisPremiseView(
                premise_id="ho_premise:source",
                premise_role="source_backbone_relation",
                component_id="component:source",
                subject="interparticle separation",
                relation="VARIES_WITH",
                object="electric-field enhancement",
                authority=RelationComponentAuthority.CONFIRMED_KNOWN,
                provenance_source_id="accepted_pattern:source",
                epistemic_use="confirmed_known_component",
            ),
            HigherOrderSynthesisPremiseView(
                premise_id="ho_premise:target",
                premise_role="target_backbone_relation",
                component_id="component:target",
                subject="SERS intensity",
                relation="VARIES_WITH",
                object="electric-field enhancement",
                authority=RelationComponentAuthority.CONFIRMED_KNOWN,
                provenance_source_id="accepted_pattern:target",
                epistemic_use="confirmed_known_component",
            ),
            HigherOrderSynthesisPremiseView(
                premise_id="ho_premise:modifier",
                premise_role="modifier_relation",
                component_id="component:modifier",
                subject="nanoparticle edge sharpness",
                relation="PROMOTES",
                object="electric-field enhancement",
                authority=modifier_authority,
                provenance_source_id=modifier_provenance,
                epistemic_use=modifier_use,
            ),
        ],
        structural_opportunity=HigherOrderStructuralOpportunityView(
            requested_source="interparticle separation",
            requested_target="SERS intensity",
            source_side_mediator_text="electric-field enhancement",
            target_side_mediator_text="electric-field enhancement",
            modifier_text="nanoparticle edge sharpness",
            modifier_anchor_role="mediator",
            modifier_anchor_text="electric-field enhancement",
        ),
    )


def test_materialization_preserves_positive_premise_authority() -> None:
    source = _source_context()
    higher = _higher_order_context()

    projection = materialize_higher_order_hypothesis_context(
        source_context=source,
        higher_order_context=higher,
    )

    source_positive = [
        row.statement_id
        for row in source.evidence_statements
        if row.eligible_as_premise
    ]
    derived_positive = [
        row.statement_id
        for row in projection.context.evidence_statements
        if row.eligible_as_premise
    ]

    assert source_positive == ["stmt:positive:1"]
    assert derived_positive == source_positive
    assert (
        projection.materialization
        .canonical_positive_premise_statement_ids
        == source_positive
    )

    generated_ids = {
        row.statement_id
        for row in (
            projection.materialization
            .generated_statement_lineage
        )
    }
    generated = [
        row
        for row in projection.context.evidence_statements
        if row.statement_id in generated_ids
    ]

    assert len(generated) == 3
    assert all(not row.eligible_as_premise for row in generated)
    assert all(not row.eligible_as_gap for row in generated)
    assert all(
        row.claim_kind == "higher_order_composition_inspiration"
        for row in generated
    )
    assert all(row.paper_ids == [] for row in generated)
    assert all(
        row.premise_restrictions
        == [
            "higher_order_composition_inspiration_only",
            "confirmed_known_does_not_create_positive_premise_authority",
            "must_not_appear_in_premise_statement_ids",
        ]
        for row in generated
    )

    assert len(source.evidence_statements) == 2


def test_materialization_is_deterministic() -> None:
    source = _source_context()
    higher = _higher_order_context()

    left = materialize_higher_order_hypothesis_context(
        source_context=source,
        higher_order_context=higher,
    )
    right = materialize_higher_order_hypothesis_context(
        source_context=source,
        higher_order_context=higher,
    )

    assert left == right
    assert left.context.context_id == right.context.context_id
    assert left.context.context_sha256 == right.context.context_sha256
    assert (
        left.materialization.materialization_id
        == right.materialization.materialization_id
    )


def test_materialization_preserves_exact_higher_order_lineage() -> None:
    projection = materialize_higher_order_hypothesis_context(
        source_context=_source_context(),
        higher_order_context=_higher_order_context(),
    )

    rows = {
        row.premise_role: row
        for row in (
            projection.materialization
            .generated_statement_lineage
        )
    }

    assert rows["source_backbone_relation"].component_id == "component:source"
    assert (
        rows["source_backbone_relation"].provenance_source_id
        == "accepted_pattern:source"
    )
    assert rows["target_backbone_relation"].component_id == "component:target"
    assert rows["modifier_relation"].component_id == "component:modifier"
    assert (
        rows["modifier_relation"].provenance_source_id
        == "accepted_pattern:modifier"
    )
    assert all(
        row.authority == RelationComponentAuthority.CONFIRMED_KNOWN
        for row in rows.values()
    )


def test_materialization_allows_candidate_modifier_only_as_restricted_nonpremise() -> None:
    projection = materialize_higher_order_hypothesis_context(
        source_context=_source_context(),
        higher_order_context=_higher_order_context(
            modifier_authority=(
                RelationComponentAuthority.CANDIDATE_INSPIRATION
            )
        ),
    )

    lineage = next(
        row
        for row in projection.materialization.generated_statement_lineage
        if row.premise_role == "modifier_relation"
    )
    statement = next(
        row
        for row in projection.context.evidence_statements
        if row.statement_id == lineage.statement_id
    )

    assert (
        lineage.authority
        == RelationComponentAuthority.CANDIDATE_INSPIRATION
    )
    assert statement.eligible_as_premise is False
    assert statement.eligible_as_gap is False
    assert statement.requires_verification is True

    positive = [
        row.statement_id
        for row in projection.context.evidence_statements
        if row.eligible_as_premise
    ]
    assert positive == ["stmt:positive:1"]


def test_materialization_does_not_authorize_llm_or_selection() -> None:
    projection = materialize_higher_order_hypothesis_context(
        source_context=_source_context(),
        higher_order_context=_higher_order_context(),
    )
    audit = projection.materialization

    assert audit.shadow_only is True
    assert audit.llm_call_authorized is False
    assert audit.discovery_axis_materialization_authorized is False
    assert audit.candidate_anchor_synthesis_authorized is False
    assert audit.novelty_authority is False
    assert audit.production_selection_changed is False
    assert audit.downstream_prior_art_review_required is True
    assert audit.downstream_n10_review_required is True


def test_shadow_prompt_focus_keeps_generated_relations_nonpremise() -> None:
    higher = _higher_order_context()
    projection = materialize_higher_order_hypothesis_context(
        source_context=_source_context(),
        higher_order_context=higher,
    )

    assembler = HigherOrderShadowHypothesisPromptAssembler(
        higher_order_context=higher,
        materialization=projection.materialization,
    )
    prompt = assembler.build(projection.context)

    assert "HIGHER-ORDER SHADOW SYNTHESIS FOCUS" in prompt.user_prompt
    assert "nanoparticle edge sharpness" in prompt.user_prompt
    assert "stmt:positive:1" in prompt.user_prompt
    assert (
        "premise_statement_ids may contain ONLY the canonical "
        "positive premise IDs"
        in prompt.user_prompt
    )
    assert (
        "This prompt rendering does not itself authorize an LLM call"
        in prompt.user_prompt
    )

    for row in (
        projection.materialization
        .generated_statement_lineage
    ):
        assert row.statement_id in prompt.user_prompt


def test_prompt_rejects_wrong_derived_context() -> None:
    higher = _higher_order_context()
    projection = materialize_higher_order_hypothesis_context(
        source_context=_source_context(),
        higher_order_context=higher,
    )

    assembler = HigherOrderShadowHypothesisPromptAssembler(
        higher_order_context=higher,
        materialization=projection.materialization,
    )

    with pytest.raises(
        ValueError,
        match="wrong derived HypothesisContext",
    ):
        assembler.build(_source_context())
