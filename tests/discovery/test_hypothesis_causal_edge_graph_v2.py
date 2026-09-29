from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.hypothesis_causal_edge_graph_v2 import (
    build_hypothesis_causal_edge_graph_novelty_depth_v2,
)
from pipeline_core.discovery.novelty_depth_causal_edge_coverage import (
    build_novelty_depth_causal_edge_coverage,
)


def _match(rel, wid):
    return SimpleNamespace(
        relationship=rel,
        work_id=wid,
    )


def _review(
    cid,
    status,
    matches,
    *,
    text,
    importance,
):
    return SimpleNamespace(
        claim_id=cid,
        importance=importance,
        claim_text=text,
        status=status,
        matches=matches,
    )


def _build(reviews, premise_text, conceptual=None):
    context = SimpleNamespace(
        context_id="ctx:1",
        evidence_statements=[
            SimpleNamespace(
                statement_id="s1",
                text=premise_text,
            )
        ],
    )
    portfolio = SimpleNamespace(
        portfolio_id="portfolio:1",
        hypotheses=[
            SimpleNamespace(
                hypothesis_id="h1",
                premise_statement_ids=["s1"],
            )
        ],
    )
    external = SimpleNamespace(
        report_id="external:1",
        cards=[
            SimpleNamespace(
                hypothesis_id="h1",
                title="fixture",
                status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                claim_reviews=reviews,
            )
        ],
    )

    v1 = build_novelty_depth_causal_edge_coverage(
        context=context,
        portfolio=portfolio,
        external_report=external,
    )
    v2 = build_hypothesis_causal_edge_graph_novelty_depth_v2(
        context=context,
        portfolio=portfolio,
        external_report=external,
        v1_report=v1,
        conceptual_knownness=conceptual,
    )
    return v1, v2


def test_cross_claim_backbone_promotes_local_v1_to_higher_order_v2():
    core = _review(
        "core",
        "COMPONENTS_ONLY",
        [_match("LOWER_ORDER_RELATION_PRIOR_ART", "w1")],
        text=(
            "excitation wavelength changes how orientation "
            "affects Raman mode intensity"
        ),
        importance="core",
    )
    supporting = _review(
        "support",
        "PARTIAL_PRIOR_ART",
        [_match("PARTIAL_PRIOR_ART", "w2")],
        text=(
            "excitation wavelength changes Raman mode intensity "
            "for oriented molecules"
        ),
        importance="supporting",
    )

    v1, v2 = _build(
        [core, supporting],
        (
            "orientation changes Raman mode intensity and excitation "
            "wavelength changes Raman mode intensity"
        ),
    )

    assert v1.profiles[0].novelty_depth_class == "SHALLOW_LOCAL_EXTENSION"

    row = v2.profiles[0]
    assert row.novelty_depth_class == "HIGHER_ORDER_INTERACTION_GAP"
    assert row.planner_advisory == "KEEP_TESTABLE_GAP"
    assert row.cross_claim_backbone_used is True
    assert row.known_backbone_claim_ids == ["support"]


def test_component_only_supporting_claim_does_not_rescue_weak_bridge():
    core = _review(
        "core",
        "COMPONENTS_ONLY",
        [_match("COMPONENT_ONLY", "w1")],
        text=(
            "superlattice order determines adsorption orientation "
            "distribution"
        ),
        importance="core",
    )
    supporting = _review(
        "support",
        "COMPONENTS_ONLY",
        [_match("COMPONENT_ONLY", "w2")],
        text="superlattice order changes with building block uniformity",
        importance="supporting",
    )

    _, v2 = _build(
        [core, supporting],
        "building block uniformity changes superlattice order",
    )

    row = v2.profiles[0]
    assert (
        row.novelty_depth_class
        == "UNSUPPORTED_BRIDGE_DEPTH_UNKNOWN"
    )
    assert row.cross_claim_backbone_used is False
    assert row.weak_bridge_claim_ids == ["core"]


def test_one_independent_cross_claim_backbone_is_local_extension():
    core = _review(
        "core",
        "COMPONENTS_ONLY",
        [_match("COMPONENT_ONLY", "w1")],
        text=(
            "composition changes orientation selective Raman mode intensity"
        ),
        importance="core",
    )
    supporting = _review(
        "support",
        "PARTIAL_PRIOR_ART",
        [_match("PARTIAL_PRIOR_ART", "w2")],
        text=(
            "metal composition changes Raman mode intensity "
            "through adsorption orientation"
        ),
        importance="supporting",
    )

    _, v2 = _build(
        [core, supporting],
        "metal composition and adsorption orientation affect Raman intensity",
    )

    row = v2.profiles[0]
    assert row.novelty_depth_class == "SHALLOW_LOCAL_EXTENSION"
    assert row.planner_advisory == "SHARPEN_LOCAL_EXTENSION"


def test_conceptual_knownness_is_overlay_only_not_authority():
    core = _review(
        "core",
        "COMPONENTS_ONLY",
        [_match("LOWER_ORDER_RELATION_PRIOR_ART", "w1")],
        text="wavelength changes how orientation affects Raman intensity",
        importance="core",
    )
    supporting = _review(
        "support",
        "PARTIAL_PRIOR_ART",
        [_match("PARTIAL_PRIOR_ART", "w2")],
        text="wavelength changes Raman intensity for oriented molecules",
        importance="supporting",
    )

    conceptual = {
        "records": [
            {
                "hypothesis_id": "h1",
                "first_gap_level": "L3_EXACT",
                "levels": [
                    {"level": "L1_BROAD", "sufficient_coverage": True},
                    {"level": "L2_INTERMEDIATE", "sufficient_coverage": True},
                    {"level": "L3_EXACT", "sufficient_coverage": True},
                ],
            }
        ]
    }

    _, v2 = _build(
        [core, supporting],
        "orientation and wavelength affect Raman intensity",
        conceptual=conceptual,
    )

    row = v2.profiles[0]
    assert row.conceptual_knownness.available is True
    assert row.conceptual_knownness.first_gap_level == "L3_EXACT"
    assert row.conceptual_knownness.coverage_sufficient is True
    assert row.conceptual_knownness.changes_v2_depth is False
    assert v2.conceptual_knownness_changed_depth_count == 0
