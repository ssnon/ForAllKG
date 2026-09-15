from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline_core.discovery.higher_order_external_shadow_plan import (
    build_higher_order_external_shadow_batch_plan,
)


def _portfolio(
    *,
    portfolio_id: str,
    hypothesis_id: str,
    context_id: str,
    context_sha256: str,
):
    return SimpleNamespace(
        portfolio_id=portfolio_id,
        source_context_id=context_id,
        source_context_sha256=context_sha256,
        hypotheses=[
            SimpleNamespace(
                hypothesis_id=hypothesis_id,
            )
        ],
    )


def _arm(
    *,
    index: int,
    status: str = "proposed",
    portfolio=None,
):
    context_id = (
        f"higher_order_synthesis_context:{index:02d}"
    )
    topology_id = (
        f"higher_order_topology:{index:02d}"
    )
    modifier_component_id = (
        f"relation_component:modifier:{index:02d}"
    )
    modifier_text = f"modifier {index}"
    derived_context_id = (
        f"hypothesis_context:higher_order:{index:02d}"
    )
    derived_context_sha = (
        f"derived-context-sha-{index:02d}"
    )

    if portfolio is None and status == "proposed":
        portfolio = _portfolio(
            portfolio_id=(
                f"hypothesis_portfolio:{index:02d}"
            ),
            hypothesis_id=f"hypothesis:{index:02d}",
            context_id=derived_context_id,
            context_sha256=derived_context_sha,
        )

    return SimpleNamespace(
        higher_order_context=SimpleNamespace(
            context_id=context_id,
            higher_order_topology_id=topology_id,
            lineage=SimpleNamespace(
                modifier_component_id=modifier_component_id,
                modifier_text=modifier_text,
            ),
        ),
        projection=SimpleNamespace(
            context=SimpleNamespace(
                context_id=derived_context_id,
                context_sha256=derived_context_sha,
            ),
            materialization=SimpleNamespace(
                materialization_id=(
                    f"higher_order_materialization:{index:02d}"
                ),
            ),
        ),
        authorization=SimpleNamespace(
            authorization_id=(
                f"higher_order_generation_authorization:{index:02d}"
            ),
        ),
        run_outcome=SimpleNamespace(
            status=status,
            portfolio_for_downstream=portfolio,
        ),
    )


def _batch(arms):
    return SimpleNamespace(
        arms=tuple(arms),
        record=SimpleNamespace(
            batch_run_id=(
                "higher_order_shadow_batch_run:test"
            ),
            selected_context_count=len(arms),
        ),
    )


def test_plan_includes_all_and_only_proposed_generation_arms() -> None:
    batch = _batch([
        _arm(index=1, status="proposed"),
        _arm(index=2, status="abstained"),
        _arm(index=3, status="proposed"),
        _arm(
            index=4,
            status="canonical_rejected",
        ),
    ])

    plan = build_higher_order_external_shadow_batch_plan(
        batch
    )

    assert plan.source_selected_context_count == 4
    assert plan.source_proposed_count == 2
    assert plan.planned_external_arm_count == 2
    assert plan.skipped_nonproposed_count == 2

    assert [
        row.arm_index
        for row in plan.arms
    ] == [1, 3]

    assert [
        row.hypothesis_portfolio_id
        for row in plan.arms
    ] == [
        "hypothesis_portfolio:01",
        "hypothesis_portfolio:03",
    ]


def test_plan_requires_fresh_query_and_prior_art_per_proposed_arm() -> None:
    plan = build_higher_order_external_shadow_batch_plan(
        _batch([
            _arm(index=1),
            _arm(index=2),
        ])
    )

    assert plan.fresh_query_plans_required is True
    assert (
        plan.fresh_prior_art_retrieval_required
        is True
    )
    assert (
        plan.shared_query_plan_reuse_authorized
        is False
    )
    assert (
        plan.shared_prior_art_reuse_authorized
        is False
    )

    for arm in plan.arms:
        assert arm.query_plan_mode == "fresh_required"
        assert (
            arm.prior_art_retrieval_mode
            == "fresh_required"
        )
        assert arm.reuse_query_plan_authorized is False
        assert arm.reuse_prior_art_authorized is False


def test_plan_preserves_shadow_and_authority_boundaries() -> None:
    plan = build_higher_order_external_shadow_batch_plan(
        _batch([_arm(index=1)])
    )

    assert plan.external_novelty_shadow_only is True
    assert (
        plan.pre_review_coverage_shadow_required
        is True
    )
    assert (
        plan.downstream_gate_shadow_required
        is True
    )
    assert (
        plan.scientific_quality_ranking_performed
        is False
    )
    assert plan.novelty_authority is False
    assert plan.selection_class_authority is False
    assert plan.production_selection_changed is False
    assert plan.legacy_portfolio_mutated is False

    arm = plan.arms[0]
    assert arm.shadow_only is True
    assert arm.novelty_authority is False
    assert arm.selection_class_authority is False
    assert arm.production_selection_changed is False


def test_plan_id_is_deterministic_and_generation_order_is_preserved() -> None:
    batch = _batch([
        _arm(index=1),
        _arm(index=2),
        _arm(index=3),
    ])

    first = build_higher_order_external_shadow_batch_plan(
        batch
    )
    second = build_higher_order_external_shadow_batch_plan(
        batch
    )

    assert first.plan_id == second.plan_id
    assert [
        row.arm_index
        for row in first.arms
    ] == [1, 2, 3]


def test_proposed_arm_without_downstream_portfolio_fails_closed() -> None:
    arm = _arm(
        index=1,
        status="proposed",
    )
    arm.run_outcome.portfolio_for_downstream = None

    with pytest.raises(
        ValueError,
        match="lacks downstream portfolio",
    ):
        build_higher_order_external_shadow_batch_plan(
            _batch([arm])
        )


def test_proposed_portfolio_projection_context_mismatch_fails_closed() -> None:
    arm = _arm(index=1)
    arm.run_outcome.portfolio_for_downstream = (
        _portfolio(
            portfolio_id="hypothesis_portfolio:wrong",
            hypothesis_id="hypothesis:wrong",
            context_id="hypothesis_context:wrong",
            context_sha256="wrong-sha",
        )
    )

    with pytest.raises(
        ValueError,
        match="portfolio/source projection context mismatch",
    ):
        build_higher_order_external_shadow_batch_plan(
            _batch([arm])
        )


def test_zero_proposed_arms_cannot_create_external_shadow_plan() -> None:
    with pytest.raises(
        ValueError,
        match="zero proposed arms",
    ):
        build_higher_order_external_shadow_batch_plan(
            _batch([
                _arm(
                    index=1,
                    status="abstained",
                ),
                _arm(
                    index=2,
                    status="shadow_contract_rejected",
                ),
            ])
        )


def test_duplicate_portfolio_identity_fails_closed() -> None:
    arm1 = _arm(index=1)
    arm2 = _arm(index=2)

    arm2.run_outcome.portfolio_for_downstream = (
        _portfolio(
            portfolio_id="hypothesis_portfolio:01",
            hypothesis_id="hypothesis:02",
            context_id=(
                arm2.projection.context.context_id
            ),
            context_sha256=(
                arm2.projection.context.context_sha256
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="duplicate hypothesis portfolio",
    ):
        build_higher_order_external_shadow_batch_plan(
            _batch([arm1, arm2])
        )
