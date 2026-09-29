from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.planner_controlled_discovery_routing import (
    build_planner_controlled_routing_execution,
    build_planner_controlled_routing_plan,
    structural_sharpen_gap,
)


class FakeGap:
    def __init__(
        self,
        *,
        hypothesis_id="h1",
        action="targeted_search_only",
        target_claim_ids=None,
        sharpening_operators=None,
        reason_codes=None,
    ):
        self.gap_id = "gap:1"
        self.hypothesis_id = hypothesis_id
        self.action = action
        self.target_claim_ids = list(target_claim_ids or ["c1"])
        self.sharpening_operators = list(sharpening_operators or [])
        self.reason_codes = list(reason_codes or [])

    def model_copy(self, *, update):
        new = FakeGap(
            hypothesis_id=self.hypothesis_id,
            action=update.get("action", self.action),
            target_claim_ids=update.get(
                "target_claim_ids",
                self.target_claim_ids,
            ),
            sharpening_operators=update.get(
                "sharpening_operators",
                self.sharpening_operators,
            ),
            reason_codes=update.get(
                "reason_codes",
                self.reason_codes,
            ),
        )
        new.gap_id = self.gap_id
        return new


def _target(
    action,
    *,
    operators=None,
    preferred=None,
    reaxis_capacity=False,
):
    return SimpleNamespace(
        hypothesis_id="h1",
        primary_action=action,
        operator_candidates=list(operators or []),
        preferred_operator=preferred,
        evidence_reaxis_capacity_present=reaxis_capacity,
    )


def _plan(target, gap, *, enabled=True):
    action_plan = SimpleNamespace(
        plan_id="action:1",
        targets=[target],
    )
    gap_plan = SimpleNamespace(
        plan_id="gap-plan:1",
        source_portfolio_id="portfolio:1",
        gaps=[gap],
    )
    return build_planner_controlled_routing_plan(
        actionable_refinement_plan=action_plan,
        gap_plan=gap_plan,
        enabled=enabled,
    )


def test_higher_order_keep_maps_to_keep_original():
    plan = _plan(
        _target("KEEP_TESTABLE_GAP"),
        FakeGap(),
    )
    assert plan.directives[0].runtime_route == "KEEP_ORIGINAL"


def test_structural_sharpen_forwards_residual_boundary():
    gap = FakeGap()
    plan = _plan(
        _target(
            "STRUCTURAL_SHARPEN",
            operators=["RESIDUAL", "BOUNDARY"],
            preferred="RESIDUAL",
        ),
        gap,
    )
    directive = plan.directives[0]
    assert directive.runtime_route == "GAP_SHARPEN"

    routed = structural_sharpen_gap(
        gap,
        directive,
    )
    assert routed.action == "gap_sharpen"
    assert routed.sharpening_operators == [
        "RESIDUAL",
        "BOUNDARY",
    ]


def test_weak_bridge_retrieve_maps_to_hold():
    plan = _plan(
        _target("RETRIEVE_MECHANISM_SUPPORT"),
        FakeGap(),
    )
    assert (
        plan.directives[0].runtime_route
        == "HOLD_FOR_EVIDENCE"
    )


def test_reaxis_requires_capacity():
    no_capacity = _plan(
        _target(
            "EVIDENCE_REAXIS",
            reaxis_capacity=False,
        ),
        FakeGap(),
    )
    assert (
        no_capacity.directives[0].runtime_route
        == "HOLD_FOR_EVIDENCE"
    )

    with_capacity = _plan(
        _target(
            "EVIDENCE_REAXIS",
            reaxis_capacity=True,
        ),
        FakeGap(),
    )
    assert (
        with_capacity.directives[0].runtime_route
        == "FRESH_CONTEXT_REAXIS"
    )


def test_execution_detects_controlled_routes():
    gap = FakeGap()
    plan = _plan(
        _target(
            "STRUCTURAL_SHARPEN",
            operators=["RESIDUAL", "BOUNDARY"],
            preferred="RESIDUAL",
        ),
        gap,
    )

    report = SimpleNamespace(
        report_id="refinement:1",
        attempts=[
            SimpleNamespace(
                gap_id="gap:1",
                decision="accepted_refinement",
                action="gap_sharpen",
                generation_mode="same_premise_refinement",
                refinement_generated=True,
                reason_codes=[],
            )
        ],
    )

    execution = (
        build_planner_controlled_routing_execution(
            routing_plan=plan,
            refinement_report=report,
            runtime_consumed_plan=True,
        )
    )

    assert execution.match_count == 1
    assert execution.diverged_count == 0


def test_disabled_plan_reports_disabled_not_match():
    plan = _plan(
        _target("KEEP_TESTABLE_GAP"),
        FakeGap(),
        enabled=False,
    )
    report = SimpleNamespace(
        report_id="refinement:1",
        attempts=[],
    )

    execution = (
        build_planner_controlled_routing_execution(
            routing_plan=plan,
            refinement_report=report,
            runtime_consumed_plan=False,
        )
    )

    assert execution.disabled_count == 1
    assert execution.match_count == 0
