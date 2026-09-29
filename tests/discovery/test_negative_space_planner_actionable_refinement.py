from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.negative_space_planner import (
    build_negative_space_planner_plan,
)


def _fixture(action):
    context = SimpleNamespace(
        context_id="ctx:1",
        evidence_statements=[
            SimpleNamespace(
                statement_id="A",
                text="orientation changes Raman intensity",
                eligible_as_premise=True,
                premise_restrictions=[],
            )
        ],
    )
    portfolio = SimpleNamespace(
        portfolio_id="portfolio:1",
        hypotheses=[
            SimpleNamespace(
                hypothesis_id="h1",
                premise_statement_ids=["A"],
            )
        ],
    )
    gap_plan = SimpleNamespace(
        plan_id="gap:plan",
        gaps=[
            SimpleNamespace(
                gap_id="gap:1",
                hypothesis_id="h1",
                source_external_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                action="targeted_search_only",
                differentiator="orientation Raman interaction",
                unresolved_boundary=["relation unresolved"],
                sharpening_operators=[],
            )
        ],
    )
    depth = SimpleNamespace(
        profiles=[
            SimpleNamespace(
                hypothesis_id="h1",
                novelty_depth_class="SHALLOW_LOCAL_EXTENSION",
                planner_advisory="SHARPEN_LOCAL_EXTENSION",
            )
        ]
    )
    actionable = SimpleNamespace(
        targets=[
            SimpleNamespace(
                hypothesis_id="h1",
                primary_action=action,
                operator_candidates=(
                    ["RESIDUAL", "BOUNDARY"]
                    if action == "STRUCTURAL_SHARPEN"
                    else []
                ),
                preferred_operator=(
                    "RESIDUAL"
                    if action == "STRUCTURAL_SHARPEN"
                    else None
                ),
            )
        ]
    )
    return context, portfolio, gap_plan, depth, actionable


def test_actionable_structural_sharpen_overrides_depth_only_abstain():
    context, portfolio, gap_plan, depth, actionable = _fixture(
        "STRUCTURAL_SHARPEN"
    )
    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        novelty_depth_profile=depth,
        actionable_refinement_plan=actionable,
    )
    row = plan.targets[0]
    assert row.if_resolved_candidate == "SAME_PREMISE_SHARPEN"
    assert row.actionable_refinement_action == "STRUCTURAL_SHARPEN"
    assert row.actionable_preferred_operator == "RESIDUAL"


def test_actionable_keep_preserves_keep_resolved():
    context, portfolio, gap_plan, depth, actionable = _fixture(
        "KEEP_TESTABLE_GAP"
    )
    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        novelty_depth_profile=depth,
        actionable_refinement_plan=actionable,
    )
    assert plan.targets[0].if_resolved_candidate == "KEEP_RESOLVED"


def test_retrieve_mechanism_support_is_fail_closed_for_current_runtime():
    context, portfolio, gap_plan, depth, actionable = _fixture(
        "RETRIEVE_MECHANISM_SUPPORT"
    )
    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        novelty_depth_profile=depth,
        actionable_refinement_plan=actionable,
    )
    row = plan.targets[0]
    assert row.if_resolved_candidate == "ABSTAIN"
    assert row.actionable_refinement_action == "RETRIEVE_MECHANISM_SUPPORT"
