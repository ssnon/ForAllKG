from pipeline_core.discovery.direct_higher_order_shadow_bundle import (
    build_direct_higher_order_structural_views,
    finalize_direct_higher_order_shadow_bundle,
    DirectHigherOrderExternalNoveltyDisposition,
    DirectHigherOrderSemanticDisposition,
    DirectHigherOrderShadowArmBundle,
)
from pipeline_core.discovery.direct_higher_order_synthesis_context import (
    DirectHigherOrderStructuralOpportunityView,
    DirectHigherOrderSynthesisContext,
    DirectHigherOrderSynthesisPremiseView,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisPortfolio,
)
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
)


def _context():
    return DirectHigherOrderSynthesisContext(
        context_id="ctx:1",
        direct_higher_order_topology_id="topology:1",
        direct_backbone_topology_id="backbone:1",
        requested_source="molecular orientation",
        requested_target="Raman intensity",
        direct_backbone_candidate_id="candidate:1",
        modifier_component_id="modifier:1",
        premises=[
            DirectHigherOrderSynthesisPremiseView(
                premise_id="premise:base",
                premise_role="direct_task_relation",
                component_id="base:1",
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
                component_id="modifier:1",
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


def _portfolio():
    card = HypothesisCard.model_construct(
        hypothesis_id="hypothesis:1",
        title="Wavelength-dependent orientation response",
        hypothesis_statement=(
            "Excitation wavelength moderates how molecular orientation "
            "changes relative vibrational SERS intensities."
        ),
        rationale="Testable higher-order relation.",
        inferential_bridge=(
            "The wavelength-dependent optical response may condition "
            "orientation-sensitive mode redistribution."
        ),
        testable_predictions=[
            "The orientation-sensitive mode ratio changes across wavelengths."
        ],
        falsification_conditions=[
            "The orientation-sensitive mode ratio is wavelength invariant."
        ],
        premise_statement_ids=["stmt:1"],
        gap_statement_ids=[],
        requires_verification=True,
    )

    return HypothesisPortfolio.model_construct(
        portfolio_id="portfolio:1",
        source_context_id="source_context:1",
        source_context_sha256="a" * 64,
        domain_profile_id="sers_au_ag",
        hypotheses=[card],
        portfolio_sha256="b" * 64,
    )


def test_structural_view_preserves_authority_separation():
    rows = build_direct_higher_order_structural_views(
        direct_context=_context(),
        portfolio=_portfolio(),
    )

    assert len(rows) == 1
    row = rows[0]

    assert row.base_authority == "confirmed_known"
    assert row.modifier_authority == "candidate_inspiration"
    assert row.base_plus_modifier_is_interaction_evidence is False
    assert row.structural_adapter_is_novelty_verdict is False
    assert row.positive_premise_authority_created is False
    assert row.novelty_authority_created is False
    assert row.production_selection_authority is False


def test_bundle_is_shadow_only_and_reserves_conceptual_knownness_slot():
    arm = DirectHigherOrderShadowArmBundle(
        arm_index=1,
        direct_context_id="ctx:1",
        direct_topology_id="topology:1",
        modifier_component_id="modifier:1",
        modifier_text="excitation wavelength",
        generation_status="proposed",
        accepted_portfolio_id="portfolio:1",
        accepted_portfolio_sha256="b" * 64,
        semantic=DirectHigherOrderSemanticDisposition(
            status="ACCEPTED",
            hard_gate_passed=True,
            run_id="semantic:1",
        ),
        external_novelty=DirectHigherOrderExternalNoveltyDisposition(
            status="COMPLETED",
            report_path="/tmp/report.json",
            card_statuses=["KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"],
        ),
        structural_views=list(
            build_direct_higher_order_structural_views(
                direct_context=_context(),
                portfolio=_portfolio(),
            )
        ),
    )

    bundle = finalize_direct_higher_order_shadow_bundle(
        source_generation_report="/tmp/generation.json",
        source_context_id="source_context:1",
        source_context_sha256="a" * 64,
        source_direct_relationpattern_report_id="direct_report:1",
        domain_profile_id="sers_au_ag",
        arms=[arm],
    )

    assert bundle.semantic_attempted_count == 1
    assert bundle.semantic_accepted_count == 1
    assert bundle.external_novelty_completed_count == 1
    assert bundle.structural_view_count == 1
    assert bundle.conceptual_knownness_integrated is False
    assert bundle.candidate_survival_authority is False
    assert bundle.semantic_rejection_authority is False
    assert bundle.production_selection_authority is False
    assert bundle.stage8_input_changed is False

def test_downstream_script_does_not_read_nonexistent_portfolio_sha_field():
    from pathlib import Path

    source = Path(
        "scripts/discovery/run_direct_higher_order_downstream_shadow.py"
    ).read_text(encoding="utf-8")

    assert "portfolio.portfolio_sha256" not in source
    assert "_sha256_json(portfolio)" in source
