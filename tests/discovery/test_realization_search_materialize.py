from __future__ import annotations

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
    DiscoveryAxisPlan,
    DiscoveryAxisPlannerPolicy,
    DiscoveryAxisSynthesisReport,
    DiscoveryHypothesisLineage,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
)
from pipeline_core.discovery.realization_search_cohort import (
    build_axis_realization_cohort,
)
from pipeline_core.discovery.realization_search_materialize import (
    materialize_realization_winners,
)
from pipeline_core.discovery.realization_search_production import (
    select_axis_realization_production_winner,
)
from pipeline_core.discovery.realization_search_shadow import (
    RealizationSearchPolicy,
    RealizationSemanticObservation,
)


AGG = (
    "semantic-distinctiveness-aggregation-v2.1"
)

MODEL = "openai/gpt-5.6-luna"


def axis(
    axis_id: str,
    rank: int,
) -> DiscoveryAxis:
    return DiscoveryAxis(
        axis_id=axis_id,
        axis_rank=rank,
        inspiration_id=(
            f"inspiration:{axis_id}"
        ),
        source_path_id=(
            f"path:{axis_id}"
        ),
        candidate_unit_id=(
            f"unit:{axis_id}"
        ),
        label=axis_id,
        rendered_path=axis_id,
        source_mode="test",
        exploration_score=0.8,
        candidate_unit_score=0.8,
        planner_score=0.8,
        mechanistic_continuity_band="high",
    )


def plan() -> DiscoveryAxisPlan:
    return DiscoveryAxisPlan(
        plan_id="plan:test",
        plan_sha256="p" * 64,
        source_dual_context_id="dual:test",
        source_dual_context_sha256="d" * 64,
        source_bundle_id="bundle:test",
        source_bundle_sha256="b" * 64,
        corpus_id="test",
        axes=[
            axis(
                "axis:a",
                0,
            ),
            axis(
                "axis:b",
                1,
            ),
        ],
        policy=(
            DiscoveryAxisPlannerPolicy()
        ),
    )


def card(
    hid: str,
) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hid,
        domain_profile_id="test-domain",
        source_context_id="context:test",
        source_context_sha256="c" * 64,
        source_report_id="report:test",
        source_report_sha256="r" * 64,
        title=hid,
        hypothesis_statement=hid,
        hypothesis_type=(
            "mechanistic_extension"
        ),
        premise_statement_ids=[
            "statement:1"
        ],
        inferential_bridge="bridge",
        predicted_observations=[
            {
                "observation_id":
                    f"obs:{hid}",
                "observable":
                    "signal",
                "expected_direction":
                    "increase",
                "rationale":
                    "rationale",
            }
        ],
        falsification_criteria=[
            {
                "criterion_id":
                    f"false:{hid}",
                "observable":
                    "signal",
                "falsifying_outcome":
                    "no increase",
            }
        ],
        evidence_profile=(
            HypothesisEvidenceProfile(
                premise_count=1,
                gap_count=0,
                source_paper_count=1,
                candidate_premise_count=1,
                reported_premise_count=1,
                synthesis_premise_count=0,
            )
        ),
    )


def lineage(
    hid: str,
    axis_id: str,
) -> DiscoveryHypothesisLineage:
    return DiscoveryHypothesisLineage(
        hypothesis_id=hid,
        axis_id=axis_id,
        inspiration_id=(
            f"inspiration:{axis_id}"
        ),
        candidate_unit_id=(
            f"unit:{axis_id}"
        ),
        axis_fidelity_status="pass",
        inference_status="pass",
        internal_novelty_status=(
            "corpus_distinct_candidate"
        ),
    )


def portfolio(
    cards,
    *,
    slot: int,
) -> HypothesisPortfolio:
    return HypothesisPortfolio(
        portfolio_id=(
            f"portfolio:slot{slot}"
        ),
        domain_profile_id="test-domain",
        source_context_id="context:test",
        source_context_sha256="c" * 64,
        source_report_id="report:test",
        source_report_sha256="r" * 64,
        hypotheses=list(cards),
        abstention_reason=(
            None
            if cards
            else "slot empty"
        ),
    )


def lineage_report(
    p: DiscoveryAxisPlan,
    rows,
    *,
    slot: int,
) -> DiscoveryAxisSynthesisReport:
    return DiscoveryAxisSynthesisReport(
        report_id=(
            f"lineage:slot{slot}"
        ),
        report_sha256=(
            f"{slot}" * 64
        ),
        source_dual_context_id=(
            p.source_dual_context_id
        ),
        source_dual_context_sha256=(
            p.source_dual_context_sha256
        ),
        axis_plan_id=p.plan_id,
        axis_plan_sha256=(
            p.plan_sha256
        ),
        final_portfolio_id=(
            f"portfolio:slot{slot}"
        ),
        final_portfolio_sha256=(
            "f" * 64
        ),
        attempted_axis_count=2,
        accepted_hypothesis_count=(
            len(rows)
        ),
        lineages=list(rows),
        attempts=[],
        policy_version=(
            "discovery-axis-synthesis-policy-v2"
        ),
    )


def obs(
    slot,
    hid,
    first,
    second,
):
    return RealizationSemanticObservation(
        slot_index=slot,
        hypothesis_id=hid,
        pass_tiers=(
            first,
            second,
        ),
        pass_aggregation_versions=(
            AGG,
            AGG,
        ),
        pass_served_models=(
            MODEL,
            MODEL,
        ),
    )


def build_fixture():
    p = plan()

    a0 = card(
        "hypothesis:a0"
    )

    b0 = card(
        "hypothesis:b0"
    )

    a1 = card(
        "hypothesis:a1"
    )

    b2 = card(
        "hypothesis:b2"
    )

    slot_portfolios = {
        0:
            portfolio(
                [
                    a0,
                    b0,
                ],
                slot=0,
            ),

        1:
            portfolio(
                [
                    a1,
                ],
                slot=1,
            ),

        2:
            portfolio(
                [
                    b2,
                ],
                slot=2,
            ),
    }

    slot_lineages = {
        0:
            lineage_report(
                p,
                [
                    lineage(
                        a0.hypothesis_id,
                        "axis:a",
                    ),
                    lineage(
                        b0.hypothesis_id,
                        "axis:b",
                    ),
                ],
                slot=0,
            ),

        1:
            lineage_report(
                p,
                [
                    lineage(
                        a1.hypothesis_id,
                        "axis:a",
                    ),
                ],
                slot=1,
            ),

        2:
            lineage_report(
                p,
                [
                    lineage(
                        b2.hypothesis_id,
                        "axis:b",
                    ),
                ],
                slot=2,
            ),
    }

    cohort_report = (
        build_axis_realization_cohort(
            axis_ids=[
                "axis:a",
                "axis:b",
            ],
            search_width=3,
            slot_payloads=[
                {
                    "slot_index": 0,
                    "alpha4_empty":
                        False,
                    "hypothesis_by_axis":
                        {
                            "axis:a":
                                a0.hypothesis_id,
                            "axis:b":
                                b0.hypothesis_id,
                        },
                    "semantic_by_hypothesis":
                        {
                            a0.hypothesis_id:
                                obs(
                                    0,
                                    a0.hypothesis_id,
                                    "MODERATE",
                                    "MODERATE",
                                ),
                            b0.hypothesis_id:
                                obs(
                                    0,
                                    b0.hypothesis_id,
                                    "HIGH",
                                    "MODERATE",
                                ),
                        },
                },
                {
                    "slot_index": 1,
                    "alpha4_empty":
                        False,
                    "hypothesis_by_axis":
                        {
                            "axis:a":
                                a1.hypothesis_id,
                        },
                    "semantic_by_hypothesis":
                        {
                            a1.hypothesis_id:
                                obs(
                                    1,
                                    a1.hypothesis_id,
                                    "HIGH",
                                    "HIGH",
                                ),
                        },
                },
                {
                    "slot_index": 2,
                    "alpha4_empty":
                        False,
                    "hypothesis_by_axis":
                        {
                            "axis:b":
                                b2.hypothesis_id,
                        },
                    "semantic_by_hypothesis":
                        {
                            b2.hypothesis_id:
                                obs(
                                    2,
                                    b2.hypothesis_id,
                                    "MODERATE",
                                    "MODERATE",
                                ),
                        },
                },
            ],
        )
    )

    policy = (
        RealizationSearchPolicy()
    )

    selections = {
        row.axis_id:
            (
                select_axis_realization_production_winner(
                    row,
                    policy=policy,
                )
            )
        for row
        in cohort_report.axes
    }

    return (
        p,
        slot_portfolios,
        slot_lineages,
        cohort_report,
        selections,
    )


def test_materializes_per_axis_winners_in_plan_order():
    (
        p,
        slot_portfolios,
        slot_lineages,
        cohort_report,
        selections,
    ) = build_fixture()

    result = (
        materialize_realization_winners(
            plan=p,
            slot_portfolios=(
                slot_portfolios
            ),
            slot_lineage_reports=(
                slot_lineages
            ),
            cohort_report=(
                cohort_report
            ),
            selections_by_axis=(
                selections
            ),
        )
    )

    assert [
        row.hypothesis_id
        for row
        in result.portfolio.hypotheses
    ] == [
        "hypothesis:a1",
        "hypothesis:b2",
    ]

    assert [
        row.axis_id
        for row
        in result.lineage_report.lineages
    ] == [
        "axis:a",
        "axis:b",
    ]

    assert (
        result.report.materialized_winner_count
        == 2
    )

    assert (
        result.report.production_selection_applied
        is True
    )

    assert (
        result.report.production_selection_changed
        is True
    )


def test_materialized_lineage_matches_new_portfolio():
    (
        p,
        slot_portfolios,
        slot_lineages,
        cohort_report,
        selections,
    ) = build_fixture()

    result = (
        materialize_realization_winners(
            plan=p,
            slot_portfolios=(
                slot_portfolios
            ),
            slot_lineage_reports=(
                slot_lineages
            ),
            cohort_report=(
                cohort_report
            ),
            selections_by_axis=(
                selections
            ),
        )
    )

    assert (
        result.lineage_report.final_portfolio_id
        == result.portfolio.portfolio_id
    )

    assert (
        result.lineage_report.accepted_hypothesis_count
        == len(
            result.portfolio.hypotheses
        )
    )

    assert {
        row.hypothesis_id
        for row
        in result.lineage_report.lineages
    } == {
        row.hypothesis_id
        for row
        in result.portfolio.hypotheses
    }


def test_no_eligible_realizations_materializes_fail_closed_empty_portfolio():
    (
        p,
        slot_portfolios,
        slot_lineages,
        cohort_report,
        selections,
    ) = build_fixture()

    selections = {
        axis_id:
            selection.model_copy(
                update={
                    "status":
                        (
                            "NO_STABLE_DETERMINATE_CANDIDATE"
                        ),
                    "winner_slot_index":
                        None,
                    "winner_hypothesis_id":
                        None,
                    "winner_tier":
                        None,
                }
            )
        for axis_id, selection
        in selections.items()
    }

    result = (
        materialize_realization_winners(
            plan=p,
            slot_portfolios=(
                slot_portfolios
            ),
            slot_lineage_reports=(
                slot_lineages
            ),
            cohort_report=(
                cohort_report
            ),
            selections_by_axis=(
                selections
            ),
        )
    )

    assert (
        result.portfolio.hypotheses
        == []
    )

    assert (
        result.portfolio.abstention_reason
        is not None
    )

    assert (
        result.lineage_report.lineages
        == []
    )

    assert (
        result.report.materialized_winner_count
        == 0
    )
