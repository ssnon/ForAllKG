from __future__ import annotations

from pipeline_core.discovery.direct_higher_order_hypothesis_runtime import (
    DirectHigherOrderShadowHypothesisPromptAssembler,
    authorize_direct_higher_order_shadow_generation,
    materialize_direct_higher_order_hypothesis_context,
)
from pipeline_core.discovery.direct_higher_order_synthesis_context import (
    DirectHigherOrderStructuralOpportunityView,
    DirectHigherOrderSynthesisContext,
    DirectHigherOrderSynthesisPremiseView,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
)
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
)


def _source_context():
    return HypothesisContext(
        context_id="ctx:source",
        context_sha256="sha:source",
        source_packet_id="packet:1",
        source_packet_sha256="packetsha:1",
        source_report_id="report:1",
        source_report_sha256="reportsha:1",
        task_id="task:1",
        question="How does orientation alter Raman intensity?",
        corpus_id="sers",
        domain_profile_id="sers_au_ag",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="stmt:positive",
                text="Canonical supported premise.",
                epistemic_role="reported",
                claim_kind="reported",
                eligible_as_premise=True,
            )
        ],
    )


def _direct_context():
    return DirectHigherOrderSynthesisContext(
        context_id="directctx:1",
        direct_higher_order_topology_id="topology:1",
        direct_backbone_topology_id="backbone:1",
        requested_source="molecular orientation",
        requested_target="Raman intensity",
        direct_backbone_candidate_id="rp:1",
        modifier_component_id="candidate:1",
        premises=[
            DirectHigherOrderSynthesisPremiseView(
                premise_id="premise:known",
                premise_role="direct_task_relation",
                component_id="known:1",
                subject="vibrational-mode SERS intensity",
                relation="VARIES_WITH",
                object="Raman-tensor orientation",
                authority=RelationComponentAuthority.CONFIRMED_KNOWN,
                provenance_source_id="accepted:1",
                epistemic_use="confirmed_known_component",
            ),
            DirectHigherOrderSynthesisPremiseView(
                premise_id="premise:modifier",
                premise_role="modifier_relation",
                component_id="candidate:1",
                subject="SERS intensity",
                relation="VARIES_WITH",
                object="excitation wavelength",
                authority=RelationComponentAuthority.CANDIDATE_INSPIRATION,
                provenance_source_id="candidate_unit:1",
                epistemic_use="candidate_inspiration_component",
            ),
        ],
        structural_opportunity=DirectHigherOrderStructuralOpportunityView(
            requested_source="molecular orientation",
            requested_target="Raman intensity",
            source_role_text="Raman-tensor orientation",
            target_role_text="vibrational-mode SERS intensity",
            modifier_text="excitation wavelength",
            modifier_anchor_role="target",
            modifier_anchor_text="SERS intensity",
        ),
    )


def test_materialization_preserves_positive_premise_set():
    projection = materialize_direct_higher_order_hypothesis_context(
        source_context=_source_context(),
        direct_context=_direct_context(),
    )

    positive = {
        row.statement_id
        for row in projection.context.evidence_statements
        if row.eligible_as_premise
    }

    assert positive == {"stmt:positive"}

    generated = projection.materialization.generated_statement_lineage
    assert len(generated) == 2

    restricted_ids = {
        row.statement_id
        for row in generated
    }

    for row in projection.context.evidence_statements:
        if row.statement_id in restricted_ids:
            assert row.eligible_as_premise is False
            assert row.eligible_as_gap is False


def test_prompt_keeps_direct_relations_restricted():
    projection = materialize_direct_higher_order_hypothesis_context(
        source_context=_source_context(),
        direct_context=_direct_context(),
    )

    assembler = DirectHigherOrderShadowHypothesisPromptAssembler(
        direct_context=_direct_context(),
        materialization=projection.materialization,
    )

    prompt = assembler.build(
        projection.context
    )

    assert (
        "restricted direct-higher-order IDs MUST NOT appear"
        in prompt.user_prompt
    )
    assert (
        "Treat that C interaction as a NEW HYPOTHESIS"
        in prompt.user_prompt
    )
    assert (
        "excitation wavelength"
        in prompt.user_prompt
    )


def test_authorization_is_separate_from_context_and_materialization():
    direct_context = _direct_context()

    projection = materialize_direct_higher_order_hypothesis_context(
        source_context=_source_context(),
        direct_context=direct_context,
    )

    assert (
        direct_context.guard.llm_call_authorized
        is False
    )
    assert (
        projection.materialization.llm_call_authorized
        is False
    )

    authorization = (
        authorize_direct_higher_order_shadow_generation(
            projection=projection,
            direct_context=direct_context,
        )
    )

    assert authorization.llm_call_authorized is True
    assert authorization.shadow_only is True
    assert authorization.novelty_authority_created is False
    assert authorization.production_selection_changed is False


def test_candidate_modifier_remains_nonpremise_in_derived_context():
    projection = materialize_direct_higher_order_hypothesis_context(
        source_context=_source_context(),
        direct_context=_direct_context(),
    )

    rows = [
        row
        for row in projection.context.evidence_statements
        if row.claim_kind
        == "direct_higher_order_composition_inspiration"
    ]

    assert len(rows) == 2
    assert all(
        row.eligible_as_premise is False
        for row in rows
    )
    assert all(
        row.eligible_as_gap is False
        for row in rows
    )
