from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.higher_order_semantic_critic import (
    critique_higher_order_shadow_batch,
)


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _card(statement: str, *, bridge: str = ""):
    return ns(
        title="shadow",
        hypothesis_statement=statement,
        inferential_bridge=bridge,
        predicted_observations=[],
        falsification_criteria=[],
        assumptions=[],
    )


def _arm(
    *,
    context_id: str,
    component_id: str,
    authority: str,
    modifier: str,
    anchor_role: str,
    anchor: str,
    statement: str,
    bridge: str = "",
):
    premise = ns(
        premise_role="modifier_relation",
        authority=ns(value=authority),
    )
    context = ns(
        context_id=context_id,
        requested_source="nanostructure shape",
        requested_target=(
            "electromagnetic hotspot location and intensity"
        ),
        premises=[premise],
        lineage=ns(modifier_component_id=component_id),
        structural_opportunity=ns(
            modifier_text=modifier,
            modifier_anchor_role=anchor_role,
            modifier_anchor_text=anchor,
            source_side_mediator_text=(
                "electromagnetic-field enhancement"
            ),
            target_side_mediator_text=(
                "hotspot intensity and distribution"
            ),
        ),
    )
    portfolio = ns(hypotheses=[_card(statement, bridge=bridge)])
    canonical = ns(accepted_portfolio=portfolio)
    run = ns(canonical_outcome=canonical)
    return ns(higher_order_context=context, run_outcome=run)


def _outcome(*arms):
    return ns(arms=arms)


def _codes(row):
    return {issue.code for issue in row.issues}


def test_candidate_mediator_anchor_omission_is_diagnostic_only():
    arm = _arm(
        context_id="ctx:c2",
        component_id="candidate:air",
        authority="candidate_inspiration",
        modifier="air exposure time",
        anchor_role="mediator",
        anchor="SERS enhancement factor",
        statement=(
            "Air exposure time may moderate how nanostructure shape "
            "determines electromagnetic hotspot location and intensity."
        ),
    )
    critique = critique_higher_order_shadow_batch(_outcome(arm))
    row = critique.arms[0]

    assert "MODIFIER_ANCHOR_BRIDGE_OMITTED" in _codes(row)
    assert "MODIFIER_UTILIZATION_UNRESOLVED" not in _codes(row)
    assert critique.rejection_authority is False
    assert critique.production_selection_authority is False
    assert critique.novelty_authority is False


def test_explicit_anchor_bridge_avoids_anchor_omission_flag():
    arm = _arm(
        context_id="ctx:bridge",
        component_id="candidate:air",
        authority="candidate_inspiration",
        modifier="air exposure time",
        anchor_role="mediator",
        anchor="SERS enhancement factor",
        statement=(
            "Air exposure time may moderate how nanostructure shape "
            "relates to electromagnetic hotspot location and intensity."
        ),
        bridge=(
            "The candidate relation concerns SERS enhancement factor; "
            "whether that observable tracks electromagnetic hotspots "
            "requires verification."
        ),
    )
    critique = critique_higher_order_shadow_batch(_outcome(arm))
    assert (
        "MODIFIER_ANCHOR_BRIDGE_OMITTED"
        not in _codes(critique.arms[0])
    )


def test_short_generic_source_adjacent_modifier_flags_restatement_risk():
    arm = _arm(
        context_id="ctx:c1",
        component_id="candidate:properties",
        authority="candidate_inspiration",
        modifier="plasmonic properties",
        anchor_role="source",
        anchor="nanostructure size shape composition arrangement",
        statement=(
            "Plasmonic properties associated with nanostructure shape "
            "may mediate electromagnetic hotspot location and intensity."
        ),
    )
    critique = critique_higher_order_shadow_batch(_outcome(arm))
    assert "BACKBONE_RESTATEMENT_RISK" in _codes(critique.arms[0])


def test_candidate_known_duplicate_is_flagged_without_novelty_authority():
    known = _arm(
        context_id="ctx:known-spacing",
        component_id="known:spacing",
        authority="confirmed_known",
        modifier="inter-particle distance",
        anchor_role="mediator",
        anchor="local-field enhancement",
        statement=(
            "Inter-particle distance moderates the relationship between "
            "plasmonic nanostructure shape and electromagnetic hotspot "
            "location and intensity, so the shape effect is "
            "coupling-dependent."
        ),
    )
    candidate = _arm(
        context_id="ctx:candidate-spacing",
        component_id="candidate:spacing",
        authority="candidate_inspiration",
        modifier="interparticle spacing",
        anchor_role="mediator",
        anchor="SERS enhancement",
        statement=(
            "Interparticle spacing may moderate the relationship between "
            "plasmonic nanostructure shape and electromagnetic hotspot "
            "behavior, so hotspot intensity and spatial distributions "
            "depend on particle spacing."
        ),
    )
    critique = critique_higher_order_shadow_batch(
        _outcome(known, candidate)
    )
    candidate_row = critique.arms[1]

    assert "KNOWN_CONTEXT_DUPLICATE" in _codes(candidate_row)
    assert critique.known_context_duplicate_pair_count == 1
    assert critique.novelty_authority is False
