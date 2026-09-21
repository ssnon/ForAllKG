from __future__ import annotations

from pipeline_core.discovery.reframing.critic_contracts import (
    CRITIC_DIMENSIONS,
    ReframeCriticDimensionReview,
    ReframeStructuralAudit,
    ScientificReframeCriticReport,
    ScientificReframeCriticReview,
)
from pipeline_core.discovery.reframing.portfolio import (
    build_scientific_reframe_shadow_portfolio,
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


def _candidate(candidate_id: str, operator_id: str) -> ScientificReframeCandidate:
    regime = operator_id == "REGIME_BOUNDARY"
    return ScientificReframeCandidate(
        reframe_id=candidate_id,
        operator_id=operator_id,
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        title=candidate_id,
        premise_statement_ids=["s1", "s2"],
        gap_statement_ids=[],
        baseline_model=ScientificModelDraft(
            summary="baseline",
            explained_statement_ids=["s1"],
            expected_observations=["baseline observation"],
        ),
        alternative_model=ScientificModelDraft(
            summary="alternative",
            explained_statement_ids=["s1", "s2"],
            expected_observations=["alternative observation"],
        ),
        challenged_assumption="assumption",
        proposed_constructs=["construct"],
        latent_constructs=[] if regime else ["latent"],
        boundary_variables=["condition"] if regime else [],
        regime_change_kind="threshold" if regime else None,
        differential_predictions=[
            DifferentialPredictionDraft(
                local_id="dp",
                observable="response",
                baseline_expectation="smooth",
                alternative_expectation="different",
                discriminating_outcome="contrast",
            )
        ],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="f",
                falsifying_outcome="baseline persists",
            )
        ],
        discriminating_test=DiscriminatingTestDraft(
            test_design="measure",
            primary_observables=["response"],
            baseline_favoring_outcome="baseline",
            alternative_favoring_outcome="alternative",
        ),
        unresolved_questions=[],
    )


def _shadow(candidates: list[ScientificReframeCandidate]) -> ScientificReframingShadowReport:
    runs = []
    for candidate in candidates:
        runs.append(
            ReframeOperatorRunRecord(
                operator_id=candidate.operator_id,
                readiness_status="ready_now",
                decision="generated",
                candidate_ids=[candidate.reframe_id],
            )
        )
    return ScientificReframingShadowReport(
        report_id="shadow:r",
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        backend_name="fake",
        model_name="fake",
        runs=runs,
        candidates=candidates,
        llm_calls_performed=len(runs),
    )


def _review(
    candidate: ScientificReframeCandidate,
    ratings: dict[str, int | None],
    *,
    shape_valid: bool = True,
) -> ScientificReframeCriticReview:
    return ScientificReframeCriticReview(
        review_id=f"review:{candidate.reframe_id}",
        candidate_id=candidate.reframe_id,
        operator_id=candidate.operator_id,
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        structural_audit=ReframeStructuralAudit(
            grounded_premise_count=2,
            gap_count=0,
            baseline_explained_count=1,
            alternative_explained_count=2,
            speculative_construct_count=1,
            differential_prediction_count=1,
            falsifier_count=1,
            primary_observable_count=1,
            operator_shape_valid=shape_valid,
            all_premise_ids_grounded=True,
            all_gap_ids_grounded=True,
        ),
        dimensions=[
            ReframeCriticDimensionReview(
                dimension=dimension,
                rating=ratings.get(dimension),
                review_status=(
                    "reviewed" if ratings.get(dimension) is not None else "missing"
                ),
            )
            for dimension in CRITIC_DIMENSIONS
        ],
        llm_review_complete=all(ratings.get(d) is not None for d in CRITIC_DIMENSIONS),
    )


def _critic(reviews: list[ScientificReframeCriticReview]) -> ScientificReframeCriticReport:
    return ScientificReframeCriticReport(
        report_id="critic:r",
        source_shadow_report_id="shadow:r",
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        backend_name="fake",
        model_name="fake",
        reviews=reviews,
        llm_calls_attempted=len(reviews),
        llm_calls_succeeded=len(reviews),
    )


def _ratings(**overrides: int) -> dict[str, int]:
    values = {dimension: 2 for dimension in CRITIC_DIMENSIONS}
    values.update(overrides)
    return values


def test_pareto_dominance_uses_full_vector_without_scalar_ranking():
    latent = _candidate("latent", "LATENT_VARIABLE")
    regime = _candidate("regime", "REGIME_BOUNDARY")
    latent_ratings = _ratings(
        operator_validity=3,
        premise_fidelity=2,
        explanatory_span=3,
        differential_prediction_quality=3,
        falsifiability=3,
        over_specificity=2,
        triviality=3,
        reframe_depth=3,
    )
    regime_ratings = _ratings(
        operator_validity=3,
        premise_fidelity=1,
        explanatory_span=3,
        differential_prediction_quality=2,
        falsifiability=3,
        over_specificity=1,
        triviality=3,
        reframe_depth=3,
    )
    portfolio = build_scientific_reframe_shadow_portfolio(
        shadow=_shadow([latent, regime]),
        critic=_critic([
            _review(latent, latent_ratings),
            _review(regime, regime_ratings),
        ]),
    )
    by_id = {row.candidate_id: row for row in portfolio.assessments}
    assert by_id["latent"].pareto_status == "non_dominated"
    assert by_id["regime"].pareto_status == "dominated"
    assert by_id["regime"].dominated_by_candidate_ids == ["latent"]
    assert portfolio.candidate_ranking_performed is False
    assert portfolio.overall_score_computed is False
    assert portfolio.winner_selected is False


def test_high_risk_dominated_candidate_is_preserved_in_slot_d():
    latent = _candidate("latent", "LATENT_VARIABLE")
    regime = _candidate("regime", "REGIME_BOUNDARY")
    portfolio = build_scientific_reframe_shadow_portfolio(
        shadow=_shadow([latent, regime]),
        critic=_critic([
            _review(latent, _ratings(operator_validity=3, premise_fidelity=3, over_specificity=3, reframe_depth=3)),
            _review(regime, _ratings(operator_validity=3, premise_fidelity=1, over_specificity=1, reframe_depth=3)),
        ]),
    )
    regime_assessment = next(row for row in portfolio.assessments if row.candidate_id == "regime")
    assert regime_assessment.assigned_slot_id == "D_HIGH_RISK_EXPLORATORY"
    assert regime_assessment.route_signals.high_risk_signal is True
    slot_d = portfolio.slots[3]
    assert slot_d.candidate_ids == ["regime"]


def test_grounded_latent_candidate_routes_to_structural_slot_b():
    latent = _candidate("latent", "LATENT_VARIABLE")
    portfolio = build_scientific_reframe_shadow_portfolio(
        shadow=_shadow([latent]),
        critic=_critic([
            _review(latent, _ratings(operator_validity=3, premise_fidelity=2, over_specificity=2, reframe_depth=3)),
        ]),
    )
    assessment = portfolio.assessments[0]
    assert assessment.route_kind == "structural_reframe"
    assert assessment.assigned_slot_id == "B_STRUCTURAL_REFRAME"
    assert assessment.route_signals.structural_followup_signal is True


def test_low_risk_regime_routes_to_measurement_regime_slot_c():
    regime = _candidate("regime", "REGIME_BOUNDARY")
    portfolio = build_scientific_reframe_shadow_portfolio(
        shadow=_shadow([regime]),
        critic=_critic([
            _review(regime, _ratings(operator_validity=3, premise_fidelity=2, over_specificity=2, reframe_depth=3)),
        ]),
    )
    assessment = portfolio.assessments[0]
    assert assessment.route_kind == "measurement_regime_challenge"
    assert assessment.assigned_slot_id == "C_MEASUREMENT_REGIME_CHALLENGE"
    assert assessment.route_signals.measurement_regime_signal is True


def test_incomplete_critic_vector_is_not_imputed_or_pareto_compared():
    latent = _candidate("latent", "LATENT_VARIABLE")
    ratings = _ratings()
    ratings["premise_fidelity"] = None
    portfolio = build_scientific_reframe_shadow_portfolio(
        shadow=_shadow([latent]),
        critic=_critic([_review(latent, ratings)]),
    )
    assessment = portfolio.assessments[0]
    assert assessment.vector_complete is False
    assert assessment.pareto_status == "not_evaluable"
    assert assessment.assigned_slot_id is None
    assert "incomplete_critic_vector" in assessment.risk_flags


def test_incomparable_vectors_both_remain_on_pareto_frontier():
    latent = _candidate("latent", "LATENT_VARIABLE")
    regime = _candidate("regime", "REGIME_BOUNDARY")
    latent_ratings = _ratings(premise_fidelity=3, explanatory_span=1)
    regime_ratings = _ratings(premise_fidelity=1, explanatory_span=3)
    portfolio = build_scientific_reframe_shadow_portfolio(
        shadow=_shadow([latent, regime]),
        critic=_critic([
            _review(latent, latent_ratings),
            _review(regime, regime_ratings),
        ]),
    )
    assert portfolio.non_dominated_candidate_ids == ["latent", "regime"]
    assert portfolio.dominated_candidate_ids == []


def test_relational_slot_is_explicitly_reserved_and_never_force_filled():
    latent = _candidate("latent", "LATENT_VARIABLE")
    portfolio = build_scientific_reframe_shadow_portfolio(
        shadow=_shadow([latent]),
        critic=_critic([_review(latent, _ratings())]),
    )
    slot_a = portfolio.slots[0]
    assert slot_a.slot_id == "A_RELATIONAL_EVIDENCE_NEAR"
    assert slot_a.candidate_ids == []
    assert "deferred" in slot_a.empty_reason.lower()
    assert portfolio.relational_lane_integrated is False


def test_lineage_mismatch_fails_closed():
    latent = _candidate("latent", "LATENT_VARIABLE")
    critic = _critic([_review(latent, _ratings())]).model_copy(
        update={"source_shadow_report_id": "shadow:wrong"}
    )
    try:
        build_scientific_reframe_shadow_portfolio(
            shadow=_shadow([latent]),
            critic=critic,
        )
    except ValueError as exc:
        assert "lineage mismatch" in str(exc)
    else:
        raise AssertionError("expected lineage mismatch to fail closed")
