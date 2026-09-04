from copy import deepcopy

import pytest

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
    DiscoveryAxisPlan,
    DiscoveryAxisSynthesisReport,
    DiscoveryHypothesisLineage,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.n10_post_generation_lineage_projection import (
    build_n10_post_generation_lineage_projection,
)


def _axis():
    return DiscoveryAxis(
        axis_id="axis:1",
        axis_rank=1,
        label="axis",
        source_mode="grounded",
        source_path_id="path:1",
        inspiration_id="inspiration:1",
        candidate_unit_id="candidate-unit:1",
        entry_anchor_id="a",
        entry_anchor_label="A",
        exit_anchor_id="b",
        exit_anchor_label="B",
        rendered_path="A -> B",
        proposed_subject="A",
        proposed_relation="relates to",
        proposed_object="B",
        planner_score=1.0,
        exploration_score=1.0,
        candidate_unit_score=1.0,
        grounding_semantic_overlap=1.0,
        registry_hop_fraction=0.0,
        generic_entity_fraction=0.0,
        reaction_domain_switch_penalty=0.0,
        mechanistic_continuity_band="high",
        requires_verification=False,
        reason_codes=[],
    )


def _plan():
    return DiscoveryAxisPlan(
        plan_id="plan:1",
        plan_sha256="plan-sha",
        corpus_id="corpus:1",
        source_bundle_id="bundle:1",
        source_bundle_sha256="bundle-sha",
        source_dual_context_id="dual:1",
        source_dual_context_sha256="dual-sha",
        axes=[
            _axis()
        ],
        excluded_inspiration_ids=[],
        policy={},
    )


def _source_report():
    return DiscoveryAxisSynthesisReport(
        report_id="report:1",
        report_sha256="report-sha",
        source_dual_context_id="dual:1",
        source_dual_context_sha256="dual-sha",
        axis_plan_id="plan:1",
        axis_plan_sha256="plan-sha",
        final_portfolio_id="portfolio:h0",
        final_portfolio_sha256="portfolio-h0-sha",
        attempted_axis_count=1,
        accepted_hypothesis_count=1,
        lineages=[
            DiscoveryHypothesisLineage(
                hypothesis_id="hypothesis:h0",
                axis_id="axis:1",
                inspiration_id="inspiration:1",
                candidate_unit_id="candidate-unit:1",
                axis_fidelity_status="pass",
                inference_status="pass",
                internal_novelty_status=(
                    "corpus_distinct_candidate"
                ),
            )
        ],
        attempts=[],
    )


def _card(
    hypothesis_id="hypothesis:h1",
):
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        domain_profile_id="synthetic",
        source_context_id="context:1",
        source_context_sha256="context-sha",
        source_report_id="report:source",
        source_report_sha256="report-sha",
        title="Projected candidate",
        hypothesis_statement=(
            "A candidate continues under the "
            "same frozen discovery axis."
        ),
        hypothesis_type="mechanistic_extension",
        premise_statement_ids=[
            "statement:1"
        ],
        gap_statement_ids=[],
        inferential_bridge=(
            "The continuation preserves the "
            "source discovery-axis provenance."
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id="observation:1",
                observable="Projected observable",
                expected_direction=(
                    "qualitative_change"
                ),
                rationale="Synthetic test rationale.",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable="Projected observable",
                falsifying_outcome="No projected change.",
            )
        ],
        assumptions=[],
        source_paper_ids=[
            "paper:1"
        ],
        evidence_profile=(
            HypothesisEvidenceProfile(
                premise_count=1,
                gap_count=0,
                source_paper_count=1,
                candidate_premise_count=0,
                reported_premise_count=1,
                synthesis_premise_count=0,
            )
        ),
    )


def _portfolio(
    hypothesis_id="hypothesis:h1",
):
    return HypothesisPortfolio(
        portfolio_id=(
            "portfolio:"
            + hypothesis_id.split(":")[-1]
        ),
        domain_profile_id="synthetic",
        source_context_id="context:1",
        source_context_sha256="context-sha",
        source_report_id="report:source",
        source_report_sha256="report-sha",
        hypotheses=[
            _card(
                hypothesis_id
            )
        ],
    )


def test_identity_projection_preserves_axis_provenance():
    report = _source_report()
    plan = _plan()
    portfolio = _portfolio()

    before_report = deepcopy(
        report
    )
    before_plan = deepcopy(
        plan
    )
    before_portfolio = deepcopy(
        portfolio
    )

    result = (
        build_n10_post_generation_lineage_projection(
            source_lineage_report=report,
            axis_plan=plan,
            source_hypothesis_id="hypothesis:h0",
            projected_portfolio=portfolio,
        )
    )

    assert (
        result.source_hypothesis_id
        == "hypothesis:h0"
    )

    assert (
        result.projected_hypothesis_id
        == "hypothesis:h1"
    )

    assert result.axis_id == "axis:1"

    assert len(
        result.lineages
    ) == 1

    projected = result.lineages[0]

    assert (
        projected.hypothesis_id
        == "hypothesis:h1"
    )

    assert (
        projected.axis_id
        == "axis:1"
    )

    assert (
        projected.inspiration_id
        == "inspiration:1"
    )

    assert (
        projected.candidate_unit_id
        == "candidate-unit:1"
    )

    assert (
        projected.axis_fidelity_status
        == "pass"
    )

    assert (
        projected.inference_status
        == "pass"
    )

    assert (
        projected.internal_novelty_status
        == "corpus_distinct_candidate"
    )

    assert (
        result.identity_projection_only
        is True
    )

    assert (
        result.production_authority
        is False
    )

    assert report == before_report
    assert plan == before_plan
    assert portfolio == before_portfolio


def test_projection_is_deterministic():
    one = (
        build_n10_post_generation_lineage_projection(
            source_lineage_report=_source_report(),
            axis_plan=_plan(),
            source_hypothesis_id="hypothesis:h0",
            projected_portfolio=_portfolio(),
        )
    )

    two = (
        build_n10_post_generation_lineage_projection(
            source_lineage_report=_source_report(),
            axis_plan=_plan(),
            source_hypothesis_id="hypothesis:h0",
            projected_portfolio=_portfolio(),
        )
    )

    assert (
        one.model_dump(
            mode="json"
        )
        == two.model_dump(
            mode="json"
        )
    )


def test_wrong_axis_plan_fails_closed():
    plan = (
        _plan().model_copy(
            update={
                "plan_id":
                    "wrong-plan"
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="axis-plan ID mismatch",
    ):
        build_n10_post_generation_lineage_projection(
            source_lineage_report=_source_report(),
            axis_plan=plan,
            source_hypothesis_id="hypothesis:h0",
            projected_portfolio=_portfolio(),
        )


def test_wrong_axis_plan_sha_fails_closed():
    plan = (
        _plan().model_copy(
            update={
                "plan_sha256":
                    "wrong-sha"
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="axis-plan SHA mismatch",
    ):
        build_n10_post_generation_lineage_projection(
            source_lineage_report=_source_report(),
            axis_plan=plan,
            source_hypothesis_id="hypothesis:h0",
            projected_portfolio=_portfolio(),
        )


def test_missing_source_lineage_fails_closed():
    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        build_n10_post_generation_lineage_projection(
            source_lineage_report=_source_report(),
            axis_plan=_plan(),
            source_hypothesis_id=(
                "hypothesis:missing"
            ),
            projected_portfolio=_portfolio(),
        )


def test_projection_requires_single_hypothesis():
    portfolio = _portfolio().model_copy(
        update={
            "hypotheses": [
                _card("hypothesis:h1"),
                _card("hypothesis:h2"),
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="exactly one hypothesis",
    ):
        build_n10_post_generation_lineage_projection(
            source_lineage_report=_source_report(),
            axis_plan=_plan(),
            source_hypothesis_id="hypothesis:h0",
            projected_portfolio=portfolio,
        )


def test_projection_requires_distinct_identity():
    portfolio = _portfolio(
        "hypothesis:h0"
    )

    with pytest.raises(
        ValueError,
        match="distinct",
    ):
        build_n10_post_generation_lineage_projection(
            source_lineage_report=_source_report(),
            axis_plan=_plan(),
            source_hypothesis_id="hypothesis:h0",
            projected_portfolio=portfolio,
        )
