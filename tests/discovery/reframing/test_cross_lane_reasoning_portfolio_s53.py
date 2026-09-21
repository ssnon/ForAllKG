from __future__ import annotations

import pytest

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisPolicy,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.reframing.contradiction_resolution import (
    ContradictionResolutionRunReport,
    ScientificContradictionResolutionCandidate,
)
from pipeline_core.discovery.reframing.cross_lane_portfolio import (
    build_cross_lane_scientific_reasoning_portfolio,
)
from pipeline_core.discovery.reframing.proxy_challenge import (
    ProxyChallengeRunReport,
    ScientificProxyChallengeCandidate,
)
from pipeline_core.discovery.reframing.reasoning_portfolio import (
    SharedScientificNeighborhoodCluster,
    UnifiedReasoningModeSlot,
    UnifiedReasoningPortfolioEntry,
    UnifiedScientificReasoningShadowPortfolio,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    DifferentialPredictionDraft,
    DiscriminatingTestDraft,
    ReframeFalsifierDraft,
    ReframeOperatorRunRecord,
    ScientificModelDraft,
    ScientificReframeCandidate,
    ScientificReframingShadowReport,
)


CTX_ID = "ctx:1"
CTX_SHA = "c" * 64
TASK_ID = "task:1"
REPORT_ID = "report:1"
REPORT_SHA = "r" * 64


def _context() -> HypothesisContext:
    return HypothesisContext(
        context_id=CTX_ID,
        context_sha256=CTX_SHA,
        source_packet_id="packet:1",
        source_packet_sha256="p" * 64,
        source_report_id=REPORT_ID,
        source_report_sha256=REPORT_SHA,
        task_id=TASK_ID,
        question="How does X relate to Y?",
        corpus_id="c1",
        domain_profile_id="sers_au_ag",
        evidence_statements=[],
        policy=HypothesisPolicy(),
    )


def _relational_card(hypothesis_id: str = "hypothesis:h1") -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        domain_profile_id="sers_au_ag",
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        source_report_id=REPORT_ID,
        source_report_sha256=REPORT_SHA,
        title="Relational hypothesis",
        hypothesis_statement="A relation is proposed.",
        hypothesis_type="context_dependency",
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        inferential_bridge="bridge",
        predicted_observations=[
            PredictedObservation(
                observation_id="o1",
                observable="response",
                expected_direction="shift",
                rationale="rationale",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="f1",
                observable="response",
                falsifying_outcome="no shift",
            )
        ],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=2,
            gap_count=1,
            source_paper_count=2,
            candidate_premise_count=0,
            reported_premise_count=2,
            synthesis_premise_count=0,
        ),
    )


def _relational_portfolio() -> HypothesisPortfolio:
    return HypothesisPortfolio(
        portfolio_id="hypothesis_portfolio:1",
        domain_profile_id="sers_au_ag",
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        source_report_id=REPORT_ID,
        source_report_sha256=REPORT_SHA,
        hypotheses=[_relational_card()],
    )


def _model(summary: str, explained: list[str]) -> ScientificModelDraft:
    return ScientificModelDraft(
        summary=summary,
        explained_statement_ids=explained,
        expected_observations=["observable changes"],
    )


def _prediction(local_id: str = "dp1") -> DifferentialPredictionDraft:
    return DifferentialPredictionDraft(
        local_id=local_id,
        observable="response",
        baseline_expectation="baseline",
        alternative_expectation="alternative",
        discriminating_outcome="different result",
    )


def _test() -> DiscriminatingTestDraft:
    return DiscriminatingTestDraft(
        test_design="measure both",
        primary_observables=["response"],
        baseline_favoring_outcome="baseline result",
        alternative_favoring_outcome="alternative result",
    )


def _latent() -> ScientificReframeCandidate:
    return ScientificReframeCandidate(
        reframe_id="scientific_reframe:latent",
        operator_id="LATENT_VARIABLE",
        source_task_id=TASK_ID,
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        title="latent",
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        baseline_model=_model("baseline", ["p1"]),
        alternative_model=_model("latent alternative", ["p1", "p2"]),
        challenged_assumption="no hidden construct",
        proposed_constructs=["latent construct"],
        latent_constructs=["latent construct"],
        boundary_variables=[],
        regime_change_kind=None,
        differential_predictions=[_prediction()],
        falsifiers=[ReframeFalsifierDraft(local_id="lf1", falsifying_outcome="no latent effect")],
        discriminating_test=_test(),
        unresolved_questions=[],
    )


def _regime() -> ScientificReframeCandidate:
    return ScientificReframeCandidate(
        reframe_id="scientific_reframe:regime",
        operator_id="REGIME_BOUNDARY",
        source_task_id=TASK_ID,
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        title="regime",
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        baseline_model=_model("single law", ["p1"]),
        alternative_model=_model("two regimes", ["p1", "p2"]),
        challenged_assumption="one law",
        proposed_constructs=["boundary"],
        latent_constructs=[],
        boundary_variables=["condition"],
        regime_change_kind="qualitative_response_change",
        differential_predictions=[_prediction("rdp1")],
        falsifiers=[ReframeFalsifierDraft(local_id="rf1", falsifying_outcome="one law remains")],
        discriminating_test=_test(),
        unresolved_questions=[],
    )


def _proxy() -> ScientificProxyChallengeCandidate:
    return ScientificProxyChallengeCandidate(
        candidate_id="scientific_proxy_challenge:1",
        source_task_id=TASK_ID,
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        source_enrichment_report_id="enrich:1",
        source_annotation_sha256="a" * 64,
        title="proxy",
        semantic_seed_annotation_ids=["ann:1"],
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        baseline_model=_model("proxy sufficient", ["p1"]),
        alternative_model=_model("proxy incomplete", ["p1", "p2"]),
        challenged_proxy_assumption="M equals T",
        challenged_observable="M",
        target_construct="T",
        proxy_failure_mode="construct_undercoverage",
        differential_predictions=[_prediction("pdp1")],
        falsifiers=[ReframeFalsifierDraft(local_id="pf1", falsifying_outcome="M fully predicts T")],
        discriminating_test=_test(),
        unresolved_questions=[],
    )


def _contradiction() -> ScientificContradictionResolutionCandidate:
    return ScientificContradictionResolutionCandidate(
        candidate_id="scientific_contradiction_resolution:1",
        source_task_id=TASK_ID,
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        source_tension_report_id="tension_report:1",
        source_tension_report_sha256="t" * 64,
        title="resolution",
        tension_witness_ids=["tw1"],
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        side_a_statement_ids=["p1"],
        side_b_statement_ids=["p2"],
        baseline_model=_model("single rule", ["p1"]),
        resolution_model=_model("conditional resolution", ["p1", "p2"]),
        apparent_contradiction="A versus B",
        resolution_principle="both hold under different conditions",
        resolution_kind="context_partition",
        distinguishing_context_variables=["condition"],
        proposed_resolution_constructs=[],
        differential_predictions=[_prediction("cdp1")],
        falsifiers=[ReframeFalsifierDraft(local_id="cf1", falsifying_outcome="same rule everywhere")],
        discriminating_test=_test(),
        unresolved_questions=[],
    )


def _reframe_shadow() -> ScientificReframingShadowReport:
    candidates = [_latent(), _regime()]
    return ScientificReframingShadowReport(
        report_id="reframe_report:1",
        source_task_id=TASK_ID,
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        backend_name="fake",
        model_name="fake",
        runs=[
            ReframeOperatorRunRecord(
                operator_id="LATENT_VARIABLE",
                readiness_status="ready_now",
                decision="generated",
                candidate_ids=[candidates[0].reframe_id],
            ),
            ReframeOperatorRunRecord(
                operator_id="REGIME_BOUNDARY",
                readiness_status="ready_now",
                decision="generated",
                candidate_ids=[candidates[1].reframe_id],
            ),
        ],
        candidates=candidates,
        llm_calls_performed=2,
    )


def _proxy_shadow() -> ProxyChallengeRunReport:
    candidate = _proxy()
    return ProxyChallengeRunReport(
        report_id="proxy_report:1",
        source_task_id=TASK_ID,
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        source_enrichment_report_id="enrich:1",
        source_annotation_sha256="a" * 64,
        backend_name="fake",
        model_name="fake",
        decision="generated",
        candidate_ids=[candidate.candidate_id],
        candidates=[candidate],
        task_relevant_seed_ids=["ann:1"],
        llm_calls_performed=1,
    )


def _contradiction_shadow() -> ContradictionResolutionRunReport:
    candidate = _contradiction()
    return ContradictionResolutionRunReport(
        report_id="contradiction_report:1",
        source_task_id=TASK_ID,
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        source_tension_report_id="tension_report:1",
        source_tension_report_sha256="t" * 64,
        backend_name="fake",
        model_name="fake",
        decision="generated",
        candidate_ids=[candidate.candidate_id],
        candidates=[candidate],
        eligible_witness_ids=["tw1"],
        llm_calls_performed=1,
    )


def _reframing_portfolio() -> UnifiedScientificReasoningShadowPortfolio:
    ids = [
        "scientific_reframe:latent",
        "scientific_reframe:regime",
        "scientific_proxy_challenge:1",
        "scientific_contradiction_resolution:1",
    ]
    operators = [
        ("LATENT_VARIABLE", "introduces_hidden_construct"),
        ("REGIME_BOUNDARY", "partitions_response_law"),
        ("PROXY_CHALLENGE", "challenges_measurement_equivalence"),
        ("CONTRADICTION_RESOLUTION", "reconciles_apparently_incompatible_evidence"),
    ]
    cluster_id = "scientific_reasoning_neighborhood:1"
    entries = [
        UnifiedReasoningPortfolioEntry(
            candidate_id=candidate_id,
            operator_id=operator_id,
            representation_transform=transform,
            premise_count=2,
            gap_count=1,
            prediction_observable_count=1,
            test_observable_count=1,
            scientific_neighborhood_cluster_id=cluster_id,
        )
        for candidate_id, (operator_id, transform) in zip(ids, operators)
    ]
    slots = [
        UnifiedReasoningModeSlot(
            slot_id=operator_id,
            representation_transform=transform,
            candidate_ids=[candidate_id],
        )
        for candidate_id, (operator_id, transform) in zip(ids, operators)
    ]
    cluster = SharedScientificNeighborhoodCluster(
        cluster_id=cluster_id,
        candidate_ids=ids,
        operator_ids=[row[0] for row in operators],
        representation_transforms=[row[1] for row in operators],
        pair_relations={"distinct_mode_shared_scientific_neighborhood": 6},
    )
    return UnifiedScientificReasoningShadowPortfolio(
        portfolio_id="unified_scientific_reasoning_portfolio:1",
        source_mode_contrast_report_id="mode_contrast:1",
        source_task_id=TASK_ID,
        entries=entries,
        slots=slots,
        scientific_neighborhood_clusters=[cluster],
        candidate_count=4,
        occupied_reasoning_mode_count=4,
        empty_reasoning_mode_count=0,
        scientific_neighborhood_cluster_count=1,
        relation_counts={"distinct_mode_shared_scientific_neighborhood": 6},
    )


def _build(**overrides):
    kwargs = dict(
        context=_context(),
        relational_portfolio=_relational_portfolio(),
        reframing_portfolio=_reframing_portfolio(),
        reframe_shadow=_reframe_shadow(),
        proxy_shadow=_proxy_shadow(),
        contradiction_shadow=_contradiction_shadow(),
        source_context_file_sha256="1" * 64,
        relational_portfolio_file_sha256="2" * 64,
        reframing_portfolio_file_sha256="3" * 64,
    )
    kwargs.update(overrides)
    return build_cross_lane_scientific_reasoning_portfolio(**kwargs)


def test_cross_lane_adapter_preserves_both_source_schemas():
    portfolio = _build()
    assert portfolio.relational_entry_count == 1
    assert portfolio.reframing_entry_count == 4
    assert portfolio.entry_count == 5
    assert portfolio.relational_lane.relational_hypotheses_retyped_as_reframes is False
    assert (
        portfolio.reframing_lane.reframing_candidates_retyped_as_relational_hypotheses
        is False
    )
    assert portfolio.source_lane_schemas_preserved is True


def test_adapter_records_relational_novelty_without_reassessing_it():
    portfolio = _build()
    assert portfolio.relational_lane.novelty_status_counts == {"not_assessed": 1}
    assert portfolio.cross_lane_external_novelty_evaluated is False
    assert portfolio.novelty_authority_created is False


def test_adapter_preserves_reframing_neighborhood_lineage():
    portfolio = _build()
    reframing = [
        row for row in portfolio.entries if row.lane_id == "SCIENTIFIC_REFRAMING"
    ]
    assert len(reframing) == 4
    assert all(row.source_scientific_neighborhood_cluster_id for row in reframing)
    assert portfolio.cross_lane_scientific_neighborhood_assignment_performed is False


def test_adapter_does_not_rank_or_prune_across_lanes():
    portfolio = _build()
    assert portfolio.cross_lane_quality_ranking_performed is False
    assert portfolio.cross_lane_redundancy_pruning_performed is False
    assert portfolio.cross_lane_winner_selected is False
    assert portfolio.integrated_portfolio_is_production_selection is False


def test_relational_context_mismatch_fails_closed():
    relational = _relational_portfolio().model_copy(update={"source_context_id": "ctx:other"})
    with pytest.raises(ValueError, match="source_context_id mismatch"):
        _build(relational_portfolio=relational)


def test_reframing_context_mismatch_fails_closed():
    proxy = _proxy_shadow().model_copy(update={"source_context_id": "ctx:other"})
    with pytest.raises(ValueError, match="reframing lineage source_context_id mismatch"):
        _build(proxy_shadow=proxy)


def test_missing_reframing_candidate_lineage_fails_closed():
    proxy = _proxy_shadow().model_copy(update={"candidate_ids": [], "candidates": []})
    with pytest.raises(ValueError, match="reframing candidate lineage mismatch"):
        _build(proxy_shadow=proxy)


def test_source_file_digests_are_preserved():
    portfolio = _build()
    assert portfolio.source_context_file_sha256 == "1" * 64
    assert portfolio.relational_lane.source_portfolio_file_sha256 == "2" * 64
    assert portfolio.reframing_lane.source_portfolio_file_sha256 == "3" * 64
    assert portfolio.lineage_validation.source_file_digests_recorded is True


def test_cross_lane_entries_do_not_gain_positive_premise_authority():
    portfolio = _build()
    assert portfolio.positive_premise_authority_created is False
    assert portfolio.source_lane_authorities_preserved is True
    assert portfolio.production_selection_changed is False


def test_cross_lane_accepts_empty_proxy_and_contradiction_slots():
    from pipeline_core.discovery.reframing.mode_contrast import (
        build_reasoning_mode_contrast,
    )
    from pipeline_core.discovery.reframing.reasoning_portfolio import (
        build_unified_reasoning_portfolio,
    )

    reframe = _reframe_shadow()
    contrast = build_reasoning_mode_contrast(
        reframe_shadow=reframe,
        proxy_shadow=None,
        contradiction_shadow=None,
    )
    reframing_portfolio = build_unified_reasoning_portfolio(contrast)
    portfolio = build_cross_lane_scientific_reasoning_portfolio(
        context=_context(),
        relational_portfolio=_relational_portfolio(),
        reframing_portfolio=reframing_portfolio,
        reframe_shadow=reframe,
        proxy_shadow=None,
        contradiction_shadow=None,
        source_context_file_sha256="1" * 64,
        relational_portfolio_file_sha256="2" * 64,
        reframing_portfolio_file_sha256="3" * 64,
    )
    assert portfolio.reframing_entry_count == 2
    assert portfolio.entry_count == 3
    assert set(portfolio.reframing_lane.candidate_ids) == {
        "scientific_reframe:latent",
        "scientific_reframe:regime",
    }
