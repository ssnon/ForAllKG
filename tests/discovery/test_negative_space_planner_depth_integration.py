from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.negative_space_planner import (
    build_negative_space_planner_plan,
)


def _statement(sid, text):
    return SimpleNamespace(
        statement_id=sid,
        text=text,
        eligible_as_premise=True,
        premise_restrictions=[],
    )


def _base(advisory, *, operators=None, unused=False):
    statements = [
        _statement("A", "orientation changes Raman mode intensity"),
        _statement("B", "wavelength changes Raman mode intensity"),
    ]
    if unused:
        statements.append(
            _statement(
                "C",
                "surface geometry changes adsorption configuration",
            )
        )

    context = SimpleNamespace(
        context_id="ctx:1",
        evidence_statements=statements,
    )
    portfolio = SimpleNamespace(
        portfolio_id="portfolio:1",
        hypotheses=[
            SimpleNamespace(
                hypothesis_id="h1",
                premise_statement_ids=["A", "B"],
            )
        ],
    )
    gap = SimpleNamespace(
        gap_id="gap:1",
        hypothesis_id="h1",
        source_external_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        action="targeted_search_only",
        differentiator="orientation wavelength interaction",
        unresolved_boundary=["interaction remains unresolved"],
        sharpening_operators=list(operators or []),
    )
    gap_plan = SimpleNamespace(plan_id="gap_plan:1", gaps=[gap])

    depth = SimpleNamespace(
        profiles=[
            SimpleNamespace(
                hypothesis_id="h1",
                novelty_depth_class="fixture-depth",
                planner_advisory=advisory,
            )
        ]
    )
    return context, portfolio, gap_plan, depth


def test_keep_testable_gap_preserves_resolved_keep():
    context, portfolio, gap_plan, depth = _base(
        "KEEP_TESTABLE_GAP"
    )
    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        novelty_depth_profile=depth,
    )
    row = plan.targets[0]
    assert row.if_resolved_candidate == "KEEP_RESOLVED"
    assert row.novelty_depth_advisory == "KEEP_TESTABLE_GAP"


def test_weak_bridge_does_not_remain_keep_resolved():
    context, portfolio, gap_plan, depth = _base(
        "REAXIS_OR_ABSTAIN",
        unused=False,
    )
    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        novelty_depth_profile=depth,
    )
    row = plan.targets[0]
    assert row.if_resolved_candidate == "ABSTAIN"


def test_weak_bridge_can_reaxis_only_with_safe_unused_capacity():
    context, portfolio, gap_plan, depth = _base(
        "REAXIS_OR_ABSTAIN",
        unused=True,
    )
    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        novelty_depth_profile=depth,
    )
    row = plan.targets[0]
    assert row.if_resolved_candidate == "EVIDENCE_REAXIS"


def test_local_extension_sharpens_only_with_explicit_operator():
    context, portfolio, gap_plan, depth = _base(
        "SHARPEN_LOCAL_EXTENSION",
        operators=["RESIDUAL"],
    )
    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        novelty_depth_profile=depth,
    )
    row = plan.targets[0]
    assert row.if_resolved_candidate == "SAME_PREMISE_SHARPEN"


def test_known_relation_abstains_from_novelty_routing():
    context, portfolio, gap_plan, depth = _base(
        "KNOWN_RELATION"
    )
    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        novelty_depth_profile=depth,
    )
    row = plan.targets[0]
    assert row.if_resolved_candidate == "ABSTAIN"
    assert row.if_known_or_extension == "ABSTAIN"
