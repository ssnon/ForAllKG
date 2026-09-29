from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.negative_space_planner import (
    build_negative_space_planner_plan,
    evaluate_negative_space_planner_execution,
    planned_action_for_external_status,
)


def _statement(
    sid,
    text,
    *,
    eligible=True,
    restrictions=None,
):
    return SimpleNamespace(
        statement_id=sid,
        text=text,
        eligible_as_premise=eligible,
        premise_restrictions=list(restrictions or []),
    )


def _portfolio(*, include_unused=True):
    context_rows = [
        _statement(
            "A",
            "molecular orientation changes Raman intensity",
        ),
        _statement(
            "B",
            "mode intensity depends on tensor orientation",
        ),
    ]

    if include_unused:
        context_rows.append(
            _statement(
                "C",
                "surface geometry conditions adsorption orientation",
            )
        )

    context = SimpleNamespace(
        context_id="ctx:1",
        evidence_statements=context_rows,
    )

    portfolio = SimpleNamespace(
        portfolio_id="portfolio:1",
        hypotheses=[
            SimpleNamespace(
                hypothesis_id="h1",
                premise_statement_ids=["A", "B"],
            ),
            SimpleNamespace(
                hypothesis_id="h2",
                premise_statement_ids=["A"],
            ),
        ],
    )

    return context, portfolio


def _gap(action, *, operators=None):
    return SimpleNamespace(
        gap_id="gap:1",
        hypothesis_id="h1",
        source_external_status=(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        action=action,
        differentiator=(
            "surface geometry conditions how orientation "
            "modulates Raman intensity"
        ),
        unresolved_boundary=[
            "the boundary condition is unresolved"
        ],
        sharpening_operators=list(operators or []),
    )


def test_targeted_search_only_prefers_keep_after_resolution():
    context, portfolio = _portfolio(include_unused=False)
    gap = _gap("targeted_search_only")
    gap_plan = SimpleNamespace(plan_id="gap_plan:1", gaps=[gap])

    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
    )

    target = plan.targets[0]
    assert target.initial_action == "RETRIEVE_MORE"
    assert target.if_resolved_candidate == "KEEP_RESOLVED"
    assert target.if_known_or_extension == "ABSTAIN"
    assert target.if_insufficient_evidence == "ABSTAIN"


def test_reaxis_is_available_only_with_safe_unused_capacity():
    context, portfolio = _portfolio(include_unused=True)
    gap = _gap("targeted_search_then_refine")
    gap_plan = SimpleNamespace(plan_id="gap_plan:1", gaps=[gap])

    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
    )

    target = plan.targets[0]
    assert target.if_known_or_extension == "EVIDENCE_REAXIS"
    assert target.evidence_allocation.candidate_capacity_present is True
    assert "C" in target.evidence_allocation.globally_unused_safe_statement_ids


def test_gap_sharpen_preserves_explicit_operator_set():
    context, portfolio = _portfolio(include_unused=True)
    gap = _gap(
        "gap_sharpen",
        operators=["MODERATOR", "BOUNDARY"],
    )
    gap_plan = SimpleNamespace(plan_id="gap_plan:1", gaps=[gap])

    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
    )

    target = plan.targets[0]
    assert target.if_resolved_candidate == "SAME_PREMISE_SHARPEN"
    assert target.if_known_or_extension == "SAME_PREMISE_SHARPEN"
    assert target.operator_allocation.allowed_operators == [
        "MODERATOR",
        "BOUNDARY",
    ]
    assert (
        target.operator_allocation.inferred_operator_outside_allowed_set
        is False
    )


def test_insufficient_evidence_never_triggers_novelty_chasing():
    context, portfolio = _portfolio(include_unused=True)
    gap = _gap("targeted_search_then_refine")
    gap_plan = SimpleNamespace(plan_id="gap_plan:1", gaps=[gap])

    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
    )

    target = plan.targets[0]
    assert (
        planned_action_for_external_status(
            target,
            "INSUFFICIENT_SEARCH_EVIDENCE",
        )
        == "ABSTAIN"
    )


def test_execution_matches_fresh_sers_keep_resolved_shape():
    context, portfolio = _portfolio(include_unused=False)
    gap = _gap("targeted_search_only")
    gap_plan = SimpleNamespace(plan_id="gap_plan:1", gaps=[gap])

    plan = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
    )

    attempt = SimpleNamespace(
        gap_id="gap:1",
        decision="kept_original",
        generation_mode="none",
        targeted_external_status=(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        final_external_status=(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        reason_codes=[
            "targeted_search_resolved_candidate_status"
        ],
    )

    report = SimpleNamespace(
        report_id="refinement:1",
        final_portfolio_id="portfolio:final",
        attempts=[attempt],
    )

    execution = evaluate_negative_space_planner_execution(
        plan=plan,
        refinement_report=report,
    )

    assert execution.match_count == 1
    assert execution.diverged_count == 0
    assert (
        execution.rows[0].planned_effective_action
        == "KEEP_RESOLVED"
    )
    assert execution.rows[0].observed_actions == ["KEEP_RESOLVED"]
