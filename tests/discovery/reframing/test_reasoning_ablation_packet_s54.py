from __future__ import annotations

import json

import pytest

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisEvidenceStatement,
    HypothesisPolicy,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.reframing.ablation_evaluation import (
    ablation_condition_counts,
    build_scientific_reasoning_ablation_packet,
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


def _evidence(
    statement_id: str,
    text: str,
    *,
    premise: bool = False,
    gap: bool = False,
    epistemic_role: str = "reported",
) -> HypothesisEvidenceStatement:
    return HypothesisEvidenceStatement(
        statement_id=statement_id,
        text=text,
        epistemic_role=epistemic_role,
        claim_kind="association" if premise else "scope_limit",
        paper_ids=[f"paper:{statement_id}"],
        eligible_as_premise=premise,
        eligible_as_gap=gap,
    )


def _context() -> HypothesisContext:
    return HypothesisContext(
        context_id=CTX_ID,
        context_sha256=CTX_SHA,
        source_packet_id="packet:1",
        source_packet_sha256="p" * 64,
        source_report_id=REPORT_ID,
        source_report_sha256=REPORT_SHA,
        task_id=TASK_ID,
        question="How does spectral environment alter the measured response?",
        corpus_id="c1",
        domain_profile_id="sers_au_ag",
        evidence_statements=[
            _evidence("p1", "Observation one supports spectral matching.", premise=True),
            _evidence("p2", "Observation two shows an offset response.", premise=True),
            _evidence(
                "g1",
                "The reason for the mismatch remains unresolved.",
                gap=True,
                epistemic_role="unresolved",
            ),
            _evidence("x1", "Unselected background statement."),
        ],
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
        title="Condition-dependent spectral response",
        hypothesis_statement="The response shifts when the local environment changes.",
        hypothesis_type="context_dependency",
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        inferential_bridge="The two observations motivate a conditional relation.",
        predicted_observations=[
            PredictedObservation(
                observation_id="o1",
                observable="spectral response",
                expected_direction="shift",
                rationale="changing the environment changes the response",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="f1",
                observable="spectral response",
                falsifying_outcome="no shift under the controlled perturbation",
            )
        ],
        assumptions=["the perturbation can be controlled independently"],
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
        assumptions=["controlled conditions"],
        explained_statement_ids=explained,
        expected_observations=["observable changes"],
    )


def _prediction(local_id: str = "dp1") -> DifferentialPredictionDraft:
    return DifferentialPredictionDraft(
        local_id=local_id,
        observable="spectral response",
        baseline_expectation="the maxima track",
        alternative_expectation="the maxima can separate",
        discriminating_outcome="systematic separation distinguishes the models",
    )


def _test() -> DiscriminatingTestDraft:
    return DiscriminatingTestDraft(
        test_design="measure the far-field and local response under matched perturbations",
        primary_observables=["far-field spectrum", "local response"],
        baseline_favoring_outcome="the spectra track",
        alternative_favoring_outcome="the spectra reproducibly diverge",
    )


def _latent() -> ScientificReframeCandidate:
    return ScientificReframeCandidate(
        reframe_id="scientific_reframe:latent",
        operator_id="LATENT_VARIABLE",
        source_task_id=TASK_ID,
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        title="A local spectral accessibility state explains the offset",
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        baseline_model=_model("far-field matching is sufficient", ["p1"]),
        alternative_model=_model("a local accessibility state explains both", ["p1", "p2"]),
        challenged_assumption="the measured resonance is sufficient",
        proposed_constructs=["local spectral accessibility"],
        latent_constructs=["local spectral accessibility"],
        boundary_variables=[],
        regime_change_kind=None,
        differential_predictions=[_prediction()],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="lf1",
                falsifying_outcome="the measured resonance predicts every controlled case",
            )
        ],
        discriminating_test=_test(),
        unresolved_questions=["which local measurement best captures accessibility"],
    )


def _regime() -> ScientificReframeCandidate:
    return ScientificReframeCandidate(
        reframe_id="scientific_reframe:regime",
        operator_id="REGIME_BOUNDARY",
        source_task_id=TASK_ID,
        source_context_id=CTX_ID,
        source_context_sha256=CTX_SHA,
        title="Tracking and offset response states",
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        baseline_model=_model("one response law", ["p1"]),
        alternative_model=_model("condition-dependent response states", ["p1", "p2"]),
        challenged_assumption="one law applies in all conditions",
        proposed_constructs=["response-state boundary"],
        latent_constructs=[],
        boundary_variables=["local environment"],
        regime_change_kind="qualitative_response_change",
        differential_predictions=[_prediction("rdp1")],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="rf1",
                falsifying_outcome="one response law fits every condition",
            )
        ],
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
        title="The measured spectral maximum may be an incomplete stand-in",
        semantic_seed_annotation_ids=["ann:1"],
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        baseline_model=_model("the observable is a sufficient stand-in", ["p1"]),
        alternative_model=_model("the observable under-covers the target construct", ["p1", "p2"]),
        challenged_proxy_assumption="the measured maximum is interchangeable with the useful local response",
        challenged_observable="measured maximum",
        target_construct="useful local response",
        proxy_failure_mode="construct_undercoverage",
        differential_predictions=[_prediction("pdp1")],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="pf1",
                falsifying_outcome="the observable fully predicts the target under all controls",
            )
        ],
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
        title="A conditional balance reconciles matching and offset reports",
        tension_witness_ids=["tw1"],
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        side_a_statement_ids=["p1"],
        side_b_statement_ids=["p2"],
        baseline_model=_model("one global matching rule", ["p1"]),
        resolution_model=_model("a conditional balance explains both sides", ["p1", "p2"]),
        apparent_contradiction="one report tracks while another is offset",
        resolution_principle="different optical contributions dominate under different conditions",
        resolution_kind="competing_mechanism_balance",
        distinguishing_context_variables=["local environment"],
        proposed_resolution_constructs=["conditional optical balance"],
        differential_predictions=[_prediction("cdp1")],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="cf1",
                falsifying_outcome="the same global rule explains both sides without context dependence",
            )
        ],
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
            test_observable_count=2,
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


def _cross_lane(context=None, relational=None, reframe=None, proxy=None, contradiction=None):
    context = context or _context()
    relational = relational or _relational_portfolio()
    reframe = reframe or _reframe_shadow()
    proxy = proxy or _proxy_shadow()
    contradiction = contradiction or _contradiction_shadow()
    return build_cross_lane_scientific_reasoning_portfolio(
        context=context,
        relational_portfolio=relational,
        reframing_portfolio=_reframing_portfolio(),
        reframe_shadow=reframe,
        proxy_shadow=proxy,
        contradiction_shadow=contradiction,
        source_context_file_sha256="1" * 64,
        relational_portfolio_file_sha256="2" * 64,
        reframing_portfolio_file_sha256="3" * 64,
    )


def _build(**overrides):
    context = overrides.pop("context", _context())
    relational = overrides.pop("relational_portfolio", _relational_portfolio())
    reframe = overrides.pop("reframe_shadow", _reframe_shadow())
    proxy = overrides.pop("proxy_shadow", _proxy_shadow())
    contradiction = overrides.pop("contradiction_shadow", _contradiction_shadow())
    cross_lane = overrides.pop(
        "cross_lane_portfolio",
        _cross_lane(context, relational, reframe, proxy, contradiction),
    )
    kwargs = dict(
        context=context,
        cross_lane_portfolio=cross_lane,
        relational_portfolio=relational,
        reframe_shadow=reframe,
        proxy_shadow=proxy,
        contradiction_shadow=contradiction,
        source_cross_lane_portfolio_file_sha256="0" * 64,
        source_context_file_sha256="1" * 64,
        source_relational_portfolio_file_sha256="2" * 64,
        source_reframe_shadow_file_sha256="3" * 64,
        source_proxy_shadow_file_sha256="4" * 64,
        source_contradiction_shadow_file_sha256="5" * 64,
    )
    kwargs.update(overrides)
    return build_scientific_reasoning_ablation_packet(**kwargs)


def test_packet_builds_three_blind_pairwise_comparisons():
    packet, key = _build()
    assert packet.comparison_count == 3
    assert {row.comparison_alias for row in packet.comparisons} == {
        "COMPARISON_01",
        "COMPARISON_02",
        "COMPARISON_03",
    }
    assert len(key.comparisons) == 3


def test_condition_counts_preserve_relational_reframing_and_combined_arms():
    _, key = _build()
    assert ablation_condition_counts(key) == {
        "RELATIONAL_ONLY": 1,
        "REFRAMING_ONLY": 4,
        "COMBINED": 5,
    }


def test_blind_packet_does_not_expose_condition_lane_operator_or_source_ids():
    packet, _ = _build()
    text = json.dumps(packet.model_dump(mode="json"), ensure_ascii=False)
    for forbidden in (
        "RELATIONAL_ONLY",
        "REFRAMING_ONLY",
        "COMBINED",
        "RELATIONAL_DISCOVERY",
        "SCIENTIFIC_REFRAMING",
        "LATENT_VARIABLE",
        "REGIME_BOUNDARY",
        "PROXY_CHALLENGE",
        "CONTRADICTION_RESOLUTION",
        "hypothesis:h1",
        "scientific_reframe:latent",
        "scientific_proxy_challenge:1",
        "scientific_contradiction_resolution:1",
    ):
        assert forbidden not in text


def test_blind_key_retains_reversible_condition_and_source_mapping():
    packet, key = _build()
    assert key.packet_id == packet.packet_id
    assert set(key.condition_candidate_source_ids) == {
        "RELATIONAL_ONLY",
        "REFRAMING_ONLY",
        "COMBINED",
    }
    assert key.blind_key_must_not_be_supplied_to_evaluator is True
    assert key.evidence_alias_to_statement_id


def test_all_arms_share_one_identical_task_evidence_view():
    packet, _ = _build()
    assert packet.identical_task_evidence_supplied_to_all_arms is True
    assert {row.text for row in packet.evidence_statements} == {
        "Observation one supports spectral matching.",
        "Observation two shows an offset response.",
        "The reason for the mismatch remains unresolved.",
    }
    assert packet.evidence_statement_count == 3


def test_packet_is_deterministic_for_identical_inputs():
    packet_a, key_a = _build()
    packet_b, key_b = _build()
    assert packet_a == packet_b
    assert key_a == key_b


def test_candidate_count_asymmetry_is_recorded_not_scored():
    packet, _ = _build()
    differences = sorted(row.absolute_candidate_count_difference for row in packet.comparisons)
    assert differences == [1, 3, 4]
    assert packet.candidate_count_asymmetry_recorded is True
    assert packet.candidate_count_is_not_quality_signal is True
    assert all(row.candidate_count_is_not_quality_signal for row in packet.comparisons)


def test_ineligible_positive_premise_fails_closed():
    context = _context()
    rows = []
    for row in context.evidence_statements:
        if row.statement_id == "p2":
            rows.append(row.model_copy(update={"eligible_as_premise": False}))
        else:
            rows.append(row)
    context = context.model_copy(update={"evidence_statements": rows})
    cross_lane = _cross_lane(context=context)
    with pytest.raises(ValueError, match="outside eligible evaluation evidence|not eligible_as_premise"):
        _build(context=context, cross_lane_portfolio=cross_lane)


def test_cross_lane_task_lineage_mismatch_fails_closed():
    cross_lane = _cross_lane().model_copy(update={"source_task_id": "task:other"})
    with pytest.raises(ValueError, match="task_id mismatch"):
        _build(cross_lane_portfolio=cross_lane)


def test_missing_source_candidate_lineage_fails_closed():
    proxy = _proxy_shadow().model_copy(update={"candidate_ids": [], "candidates": []})
    with pytest.raises(ValueError, match="candidate lineage mismatch"):
        _build(proxy_shadow=proxy)


def test_evaluation_dimensions_are_vector_only_with_no_composite_score():
    packet, _ = _build()
    assert len(packet.evaluation_dimensions) == 7
    assert all(row.no_single_composite_score for row in packet.evaluation_dimensions)
    assert packet.overall_score_computed is False
    assert packet.winner_selected is False
    assert packet.scientific_quality_ranking_performed is False


def test_packet_creates_no_scientific_or_production_authority():
    packet, key = _build()
    assert packet.evaluation_judgment_performed is False
    assert packet.scientific_authority_created is False
    assert packet.external_novelty_evaluated is False
    assert packet.production_selection_changed is False
    assert packet.canonical_graph_mutated is False
    assert key.scientific_authority_created is False
    assert key.production_selection_changed is False


def test_ablation_packet_accepts_sparse_optional_operator_reports():
    from pipeline_core.discovery.reframing.mode_contrast import (
        build_reasoning_mode_contrast,
    )
    from pipeline_core.discovery.reframing.reasoning_portfolio import (
        build_unified_reasoning_portfolio,
    )

    context = _context()
    relational = _relational_portfolio()
    reframe = _reframe_shadow()
    contrast = build_reasoning_mode_contrast(
        reframe_shadow=reframe,
        proxy_shadow=None,
        contradiction_shadow=None,
    )
    reframing_portfolio = build_unified_reasoning_portfolio(contrast)
    cross_lane = build_cross_lane_scientific_reasoning_portfolio(
        context=context,
        relational_portfolio=relational,
        reframing_portfolio=reframing_portfolio,
        reframe_shadow=reframe,
        proxy_shadow=None,
        contradiction_shadow=None,
        source_context_file_sha256="1" * 64,
        relational_portfolio_file_sha256="2" * 64,
        reframing_portfolio_file_sha256="3" * 64,
    )
    packet, key = build_scientific_reasoning_ablation_packet(
        context=context,
        cross_lane_portfolio=cross_lane,
        relational_portfolio=relational,
        reframe_shadow=reframe,
        proxy_shadow=None,
        contradiction_shadow=None,
        source_cross_lane_portfolio_file_sha256="0" * 64,
        source_context_file_sha256="1" * 64,
        source_relational_portfolio_file_sha256="2" * 64,
        source_reframe_shadow_file_sha256="3" * 64,
        source_proxy_shadow_file_sha256=None,
        source_contradiction_shadow_file_sha256=None,
    )
    assert packet.source_candidate_count == 3
    assert key.source_proxy_shadow_file_sha256 is None
    assert key.source_contradiction_shadow_file_sha256 is None
    assert key.condition_candidate_source_ids["REFRAMING_ONLY"] == [
        "scientific_reframe:latent",
        "scientific_reframe:regime",
    ]
