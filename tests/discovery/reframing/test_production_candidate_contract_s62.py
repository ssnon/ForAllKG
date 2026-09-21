from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline_core.discovery.hypothesis_contracts import HypothesisCard, HypothesisPortfolio
from pipeline_core.discovery.reframing.contradiction_resolution import (
    ContradictionResolutionRunReport,
    ScientificContradictionResolutionCandidate,
)
from pipeline_core.discovery.reframing.cross_lane_portfolio import (
    CrossLaneEntryEnvelope,
    CrossLaneScientificReasoningShadowPortfolio,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionCandidatePrediction,
    build_production_facing_candidate_portfolio,
)
from pipeline_core.discovery.reframing.proxy_challenge import (
    ProxyChallengeRunReport,
    ScientificProxyChallengeCandidate,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframeCandidate,
    ScientificReframingShadowReport,
)


def _model(summary: str):
    return SimpleNamespace(summary=summary, assumptions=["controlled"])


def _contrast_prediction():
    return SimpleNamespace(
        observable="response",
        baseline_expectation="tracks baseline",
        alternative_expectation="diverges under perturbation",
        discriminating_outcome="divergence favors alternative",
    )


def _test():
    return SimpleNamespace(
        test_design="matched perturbation test",
        primary_observables=["response"],
        baseline_favoring_outcome="tracking",
        alternative_favoring_outcome="divergence",
    )


def _relational():
    return HypothesisCard.model_construct(
        schema_version="hypothesis-card-v1",
        hypothesis_id="hypothesis:r1",
        title="Relational candidate",
        hypothesis_statement="X changes with Y.",
        hypothesis_type="context_dependency",
        premise_statement_ids=["p1"],
        gap_statement_ids=["g1"],
        inferential_bridge="Observed context dependence motivates the relation.",
        predicted_observations=[
            SimpleNamespace(observable="response", expected_direction="increase", rationale="relation")
        ],
        falsification_criteria=[
            SimpleNamespace(observable="response", falsifying_outcome="no change")
        ],
        assumptions=["matched conditions"],
    )


def _reframe(cid: str = "scientific_reframe:f1"):
    return ScientificReframeCandidate.model_construct(
        schema_version="scientific-reframe-candidate-v1",
        reframe_id=cid,
        title="Reframe candidate",
        premise_statement_ids=["p1"],
        gap_statement_ids=["g1"],
        baseline_model=_model("one-rule baseline"),
        alternative_model=_model("context-dependent alternative"),
        challenged_assumption="one rule applies everywhere",
        differential_predictions=[_contrast_prediction()],
        falsifiers=[SimpleNamespace(falsifying_outcome="one rule fits all")],
        discriminating_test=_test(),
        unresolved_questions=["boundary variable?"],
    )


def _proxy():
    return ScientificProxyChallengeCandidate.model_construct(
        schema_version="scientific-proxy-challenge-candidate-v1",
        candidate_id="scientific_proxy_challenge:p1",
        title="Proxy candidate",
        premise_statement_ids=["p1"],
        gap_statement_ids=["g1"],
        baseline_model=_model("observable is sufficient"),
        alternative_model=_model("observable under-covers target"),
        challenged_proxy_assumption="observable equals construct",
        challenged_observable="observable",
        target_construct="target construct",
        differential_predictions=[_contrast_prediction()],
        falsifiers=[SimpleNamespace(falsifying_outcome="proxy is always sufficient")],
        discriminating_test=_test(),
        unresolved_questions=[],
    )


def _contradiction():
    return ScientificContradictionResolutionCandidate.model_construct(
        schema_version="scientific-contradiction-resolution-candidate-v1",
        candidate_id="scientific_contradiction_resolution:c1",
        title="Resolution candidate",
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        baseline_model=_model("one global rule"),
        resolution_model=_model("conditional balance"),
        apparent_contradiction="reports disagree",
        resolution_principle="different contexts weight mechanisms differently",
        differential_predictions=[_contrast_prediction()],
        falsifiers=[SimpleNamespace(falsifying_outcome="one rule explains both")],
        discriminating_test=_test(),
        unresolved_questions=[],
    )


def _entry(entry_id: str, lane: str, source_id: str, label: str):
    return CrossLaneEntryEnvelope.model_construct(
        entry_id=entry_id,
        lane_id=lane,
        source_object_id=source_id,
        source_scientific_neighborhood_cluster_id=None,
        source_epistemic_status="hypothesis_only",
        reasoning_label=label,
    )


def _inputs(*, sparse: bool = False):
    relational = _relational()
    reframes = [_reframe("scientific_reframe:f1"), _reframe("scientific_reframe:f2")]
    proxy_candidate = None if sparse else _proxy()
    contradiction_candidate = None if sparse else _contradiction()
    entries = [
        _entry("e:r1", "RELATIONAL_DISCOVERY", relational.hypothesis_id, "context_dependency"),
        _entry("e:f1", "SCIENTIFIC_REFRAMING", reframes[0].reframe_id, "introduces_hidden_construct"),
        _entry("e:f2", "SCIENTIFIC_REFRAMING", reframes[1].reframe_id, "partitions_response_law"),
    ]
    if proxy_candidate is not None:
        entries.append(_entry("e:p1", "SCIENTIFIC_REFRAMING", proxy_candidate.candidate_id, "challenges_measurement_equivalence"))
    if contradiction_candidate is not None:
        entries.append(_entry("e:c1", "SCIENTIFIC_REFRAMING", contradiction_candidate.candidate_id, "reconciles_apparently_incompatible_evidence"))
    cross_lane = CrossLaneScientificReasoningShadowPortfolio.model_construct(
        portfolio_id="cross:1",
        source_task_id="task:1",
        source_context_id="ctx:1",
        source_context_sha256="ctxsha",
        question="How does context alter response?",
        domain_profile_id="sers",
        entries=entries,
    )
    relational_portfolio = HypothesisPortfolio.model_construct(
        source_context_id="ctx:1",
        hypotheses=[relational],
    )
    reframe_shadow = ScientificReframingShadowReport.model_construct(
        source_task_id="task:1",
        source_context_id="ctx:1",
        candidates=reframes,
    )
    proxy_shadow = None if sparse else ProxyChallengeRunReport.model_construct(
        source_task_id="task:1", source_context_id="ctx:1", candidates=[proxy_candidate]
    )
    contradiction_shadow = None if sparse else ContradictionResolutionRunReport.model_construct(
        source_task_id="task:1", source_context_id="ctx:1", candidates=[contradiction_candidate]
    )
    return cross_lane, relational_portfolio, reframe_shadow, proxy_shadow, contradiction_shadow


def _build(*, sparse: bool = False):
    cross_lane, relational, reframe, proxy, contradiction = _inputs(sparse=sparse)
    return build_production_facing_candidate_portfolio(
        cross_lane_portfolio=cross_lane,
        relational_portfolio=relational,
        reframe_shadow=reframe,
        proxy_shadow=proxy,
        contradiction_shadow=contradiction,
        source_cross_lane_portfolio_file_sha256="4" * 64,
    )


def test_full_portfolio_projects_every_cross_lane_candidate_once():
    portfolio = _build()
    assert portfolio.candidate_count == 5
    assert portfolio.relational_candidate_count == 1
    assert portfolio.reframing_candidate_count == 4
    assert set(portfolio.source_kind_counts) == {
        "relational_hypothesis",
        "scientific_reframe",
        "scientific_proxy_challenge",
        "scientific_contradiction_resolution",
    }


def test_sparse_operator_portfolio_requires_no_placeholder_candidates():
    portfolio = _build(sparse=True)
    assert portfolio.candidate_count == 3
    assert portfolio.source_kind_counts == {"relational_hypothesis": 1, "scientific_reframe": 2}


def test_relational_candidate_preserves_relational_prediction_shape():
    candidate = next(c for c in _build().candidates if c.source_lane == "RELATIONAL_DISCOVERY")
    assert candidate.predictions[0].expected_direction == "increase"
    assert candidate.predictions[0].baseline_expectation is None
    assert candidate.discriminating_test is None


def test_reframe_candidate_preserves_model_contrast_and_test():
    candidate = next(c for c in _build().candidates if c.source_object_kind == "scientific_reframe")
    assert candidate.predictions[0].baseline_expectation == "tracks baseline"
    assert candidate.predictions[0].alternative_expectation == "diverges under perturbation"
    assert candidate.discriminating_test is not None


def test_proxy_and_contradiction_keep_distinct_source_kind():
    kinds = {c.source_object_kind for c in _build().candidates}
    assert "scientific_proxy_challenge" in kinds
    assert "scientific_contradiction_resolution" in kinds


def test_projection_is_shadow_only_and_cannot_select_production():
    portfolio = _build()
    assert portfolio.cross_lane_synthesis_performed is False
    assert portfolio.scientific_quality_ranking_performed is False
    assert portfolio.final_hypothesis_selection_performed is False
    assert portfolio.production_selection_changed is False


def test_source_object_ids_match_cross_lane_exactly():
    cross_lane, *_ = _inputs()
    portfolio = _build()
    assert {c.source_object_id for c in portfolio.candidates} == {e.source_object_id for e in cross_lane.entries}


def test_candidate_ids_are_deterministic():
    first = _build()
    second = _build()
    assert [c.candidate_id for c in first.candidates] == [c.candidate_id for c in second.candidates]
    assert first.portfolio_id == second.portfolio_id


def test_prediction_contract_rejects_mixed_shape():
    with pytest.raises(ValueError, match="exactly one"):
        ProductionCandidatePrediction(
            observable="x", expected_direction="increase", rationale="r",
            baseline_expectation="b", alternative_expectation="a", discriminating_outcome="d",
        )


def test_lineage_mismatch_fails_closed():
    cross_lane, relational, reframe, proxy, contradiction = _inputs()
    reframe = reframe.model_copy(update={"source_task_id": "task:wrong"})
    with pytest.raises(ValueError, match="task mismatch"):
        build_production_facing_candidate_portfolio(
            cross_lane_portfolio=cross_lane,
            relational_portfolio=relational,
            reframe_shadow=reframe,
            proxy_shadow=proxy,
            contradiction_shadow=contradiction,
            source_cross_lane_portfolio_file_sha256="4" * 64,
        )
