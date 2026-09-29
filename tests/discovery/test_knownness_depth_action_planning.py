from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.knownness_depth_action_planning import (
    build_actionable_refinement_plan,
    build_knownness_depth_fusion,
)


def _conceptual(
    *,
    available=False,
    first_gap=None,
    sufficient=None,
):
    return SimpleNamespace(
        available=available,
        first_gap_level=first_gap,
        disposition=None,
        coverage_sufficient=sufficient,
    )


def _edge_row(
    depth,
    advisory,
    *,
    conceptual=None,
    hypothesis_id="h1",
):
    return SimpleNamespace(
        hypothesis_id=hypothesis_id,
        title="fixture",
        source_external_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        novelty_depth_class=depth,
        planner_advisory=advisory,
        known_backbone_claim_ids=["known:1"],
        novelty_bearing_claim_ids=["gap:1"],
        weak_bridge_claim_ids=(
            ["gap:1"]
            if depth == "UNSUPPORTED_BRIDGE_DEPTH_UNKNOWN"
            else []
        ),
        conceptual_knownness=(
            conceptual
            if conceptual is not None
            else _conceptual()
        ),
    )


def _edge_report(rows):
    return SimpleNamespace(
        report_id="edge:v2",
        source_portfolio_id="portfolio:1",
        source_external_report_id="external:1",
        profiles=rows,
    )


def _context(*, unused=False):
    rows = [
        SimpleNamespace(
            statement_id="A",
            text="orientation changes Raman intensity",
            eligible_as_premise=True,
            premise_restrictions=[],
        ),
    ]
    if unused:
        rows.append(
            SimpleNamespace(
                statement_id="B",
                text="surface geometry changes adsorption configuration",
                eligible_as_premise=True,
                premise_restrictions=[],
            )
        )
    return SimpleNamespace(
        context_id="ctx:1",
        evidence_statements=rows,
    )


def _portfolio():
    return SimpleNamespace(
        portfolio_id="portfolio:1",
        hypotheses=[
            SimpleNamespace(
                hypothesis_id="h1",
                premise_statement_ids=["A"],
            )
        ],
    )


def _gap_plan():
    return SimpleNamespace(
        plan_id="gap-plan:1",
        gaps=[
            SimpleNamespace(
                hypothesis_id="h1",
            )
        ],
    )


def test_higher_order_depth_only_keeps_testable_gap():
    report = build_knownness_depth_fusion(
        edge_graph_report=_edge_report(
            [
                _edge_row(
                    "HIGHER_ORDER_INTERACTION_GAP",
                    "KEEP_TESTABLE_GAP",
                )
            ]
        )
    )
    row = report.rows[0]
    assert row.fusion_state == "HIGHER_ORDER_GAP_DEPTH_ONLY"

    plan = build_actionable_refinement_plan(
        context=_context(),
        portfolio=_portfolio(),
        gap_plan=_gap_plan(),
        fusion_report=report,
        edge_graph_report=_edge_report([]),
    )
    target = plan.targets[0]
    assert target.primary_action == "KEEP_TESTABLE_GAP"
    assert target.operator_candidates == []


def test_shallow_extension_gets_structural_residual_plan():
    report = build_knownness_depth_fusion(
        edge_graph_report=_edge_report(
            [
                _edge_row(
                    "SHALLOW_LOCAL_EXTENSION",
                    "SHARPEN_LOCAL_EXTENSION",
                )
            ]
        )
    )
    row = report.rows[0]
    assert row.fusion_state == "SHALLOW_EXTENSION_DEPTH_ONLY"

    plan = build_actionable_refinement_plan(
        context=_context(),
        portfolio=_portfolio(),
        gap_plan=_gap_plan(),
        fusion_report=report,
        edge_graph_report=_edge_report([]),
    )
    target = plan.targets[0]
    assert target.primary_action == "STRUCTURAL_SHARPEN"
    assert target.preferred_operator == "RESIDUAL"
    assert target.operator_candidates == ["RESIDUAL", "BOUNDARY"]
    assert target.generation_candidate is True


def test_weak_bridge_without_unused_evidence_requests_support():
    report = build_knownness_depth_fusion(
        edge_graph_report=_edge_report(
            [
                _edge_row(
                    "UNSUPPORTED_BRIDGE_DEPTH_UNKNOWN",
                    "REAXIS_OR_ABSTAIN",
                )
            ]
        )
    )
    plan = build_actionable_refinement_plan(
        context=_context(unused=False),
        portfolio=_portfolio(),
        gap_plan=_gap_plan(),
        fusion_report=report,
        edge_graph_report=_edge_report([]),
    )
    target = plan.targets[0]
    assert target.primary_action == "RETRIEVE_MECHANISM_SUPPORT"
    assert target.operator_candidates == []
    assert target.generation_candidate is False


def test_weak_bridge_with_safe_unused_evidence_reaxes():
    report = build_knownness_depth_fusion(
        edge_graph_report=_edge_report(
            [
                _edge_row(
                    "UNSUPPORTED_BRIDGE_DEPTH_UNKNOWN",
                    "REAXIS_OR_ABSTAIN",
                )
            ]
        )
    )
    plan = build_actionable_refinement_plan(
        context=_context(unused=True),
        portfolio=_portfolio(),
        gap_plan=_gap_plan(),
        fusion_report=report,
        edge_graph_report=_edge_report([]),
    )
    target = plan.targets[0]
    assert target.primary_action == "EVIDENCE_REAXIS"
    assert target.evidence_reaxis_capacity_present is True
    assert target.safe_unused_evidence_statement_ids == ["B"]


def test_l3_conceptual_signal_confirms_shallow_exact_gap():
    report = build_knownness_depth_fusion(
        edge_graph_report=_edge_report(
            [
                _edge_row(
                    "SHALLOW_LOCAL_EXTENSION",
                    "SHARPEN_LOCAL_EXTENSION",
                    conceptual=_conceptual(
                        available=True,
                        first_gap="L3_EXACT",
                        sufficient=True,
                    ),
                )
            ]
        )
    )
    row = report.rows[0]
    assert row.conceptual_signal == "L3_EXACT_GAP"
    assert row.fusion_state == "SHALLOW_EXACT_GAP_CONFIRMED"


def test_conceptual_depth_mismatch_holds_unresolved():
    report = build_knownness_depth_fusion(
        edge_graph_report=_edge_report(
            [
                _edge_row(
                    "SHALLOW_LOCAL_EXTENSION",
                    "SHARPEN_LOCAL_EXTENSION",
                    conceptual=_conceptual(
                        available=True,
                        first_gap="L1_BROAD",
                        sufficient=True,
                    ),
                )
            ]
        )
    )
    row = report.rows[0]
    assert row.fusion_state == "DEPTH_CONCEPTUAL_MISMATCH"

    plan = build_actionable_refinement_plan(
        context=_context(),
        portfolio=_portfolio(),
        gap_plan=_gap_plan(),
        fusion_report=report,
        edge_graph_report=_edge_report([]),
    )
    assert plan.targets[0].primary_action == "HOLD_UNRESOLVED"
