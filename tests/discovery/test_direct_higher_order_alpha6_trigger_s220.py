from pipeline_core.discovery.direct_higher_order_alpha6_trigger_shadow import (
    recommend_direct_higher_order_alpha6_triggers,
)
from pipeline_core.discovery.direct_higher_order_shadow_bundle import (
    DirectHigherOrderConceptualKnownnessSlot,
    DirectHigherOrderExternalNoveltyDisposition,
    DirectHigherOrderSemanticDisposition,
    DirectHigherOrderShadowArmBundle,
    finalize_direct_higher_order_shadow_bundle,
)


def _bundle(*, first_gap_level, external_status):
    arm = DirectHigherOrderShadowArmBundle(
        arm_index=1,
        direct_context_id="ctx:1",
        direct_topology_id="topology:1",
        modifier_component_id="modifier:1",
        modifier_text="modifier",
        generation_status="proposed",
        semantic=DirectHigherOrderSemanticDisposition(
            status="ACCEPTED",
        ),
        external_novelty=DirectHigherOrderExternalNoveltyDisposition(
            status="COMPLETED",
            card_statuses=[external_status],
        ),
        conceptual_knownness=DirectHigherOrderConceptualKnownnessSlot(
            status="AVAILABLE",
            first_gap_level=first_gap_level,
        ),
    )

    return finalize_direct_higher_order_shadow_bundle(
        source_generation_report="/tmp/generation.json",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_direct_relationpattern_report_id="direct:1",
        domain_profile_id="sers_au_ag",
        arms=[arm],
    )


def test_l3_only_gap_recommends_reaxis_candidate():
    bundle = _bundle(
        first_gap_level="L3_EXACT",
        external_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
    )

    row = recommend_direct_higher_order_alpha6_triggers(
        bundle
    )[0]

    assert row.recommendation == "REAXIS_CANDIDATE"
    assert row.alpha6_trigger_authority is False
    assert row.production_selection_authority is False


def test_l2_gap_with_relational_gap_recommends_keep():
    bundle = _bundle(
        first_gap_level="L2_INTERMEDIATE",
        external_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
    )

    row = recommend_direct_higher_order_alpha6_triggers(
        bundle
    )[0]

    assert row.recommendation == "KEEP"
    assert row.alpha6_trigger_authority is False
    assert row.production_selection_authority is False
