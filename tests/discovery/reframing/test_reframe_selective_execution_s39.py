from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.reframing.benchmark_audit import (
    BenchmarkTaskReframeAudit,
    summarize_benchmark_task_audits,
)
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
from pipeline_core.discovery.reframing.selective_execution import (
    BenchmarkTaskSelectiveExecution,
    ScientificReframeSelectiveExecutionReport,
    resumable_critic_report,
    resumable_portfolio_report,
    resumable_shadow_report,
    skipped_stage,
    stage_from_critic,
    stage_from_portfolio,
    stage_from_shadow,
    summarize_selective_execution,
    triggered_operator_ids,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    ConditionDiversitySignal,
    ReframeOperatorTriggerAssessment,
    ScientificReframeTriggerReport,
)


def _audit_row(pattern: str = "latent_only") -> BenchmarkTaskReframeAudit:
    latent = pattern in {"latent_only", "latent_and_regime"}
    regime = pattern in {"regime_only", "latent_and_regime"}
    return BenchmarkTaskReframeAudit(
        task_key="K01/replicate_01",
        case_key="K01",
        replicate_key="replicate_01",
        canonical_dir="/bench/K01/replicate_01/canonical",
        status="complete",
        task_id="task:t",
        readiness_status_by_operator={
            "LATENT_VARIABLE": "ready_now",
            "REGIME_BOUNDARY": "ready_with_optional_enrichment",
        },
        trigger_decision_by_operator={
            "LATENT_VARIABLE": "triggered" if latent else "not_triggered",
            "REGIME_BOUNDARY": "triggered" if regime else "not_triggered",
        },
        trigger_signal_by_operator={
            "LATENT_VARIABLE": latent,
            "REGIME_BOUNDARY": regime,
        },
        trigger_pattern=pattern,
    )


def _trigger(latent: bool = True, regime: bool = False) -> ScientificReframeTriggerReport:
    assessments = []
    for operator, signal in (("LATENT_VARIABLE", latent), ("REGIME_BOUNDARY", regime)):
        assessments.append(
            ReframeOperatorTriggerAssessment(
                operator_id=operator,
                decision="triggered" if signal else "not_triggered",
                scientific_trigger_signal=signal,
            )
        )
    return ScientificReframeTriggerReport(
        report_id="trigger:r",
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        source_explorer_report_sha256="explorer",
        source_evidence_sha256="evidence",
        trigger_resolution_mode="evidence_level_refined",
        condition_diversity=ConditionDiversitySignal(
            example_count=0,
            paper_count=0,
            distinct_condition_signature_count=0,
        ),
        assessments=assessments,
    )


def _candidate() -> ScientificReframeCandidate:
    return ScientificReframeCandidate(
        reframe_id="candidate:1",
        operator_id="LATENT_VARIABLE",
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        title="candidate",
        premise_statement_ids=["s1", "s2"],
        gap_statement_ids=[],
        baseline_model=ScientificModelDraft(
            summary="baseline",
            explained_statement_ids=["s1"],
            expected_observations=["baseline"],
        ),
        alternative_model=ScientificModelDraft(
            summary="alternative",
            explained_statement_ids=["s1", "s2"],
            expected_observations=["alternative"],
        ),
        challenged_assumption="assumption",
        proposed_constructs=["construct"],
        latent_constructs=["latent"],
        boundary_variables=[],
        regime_change_kind=None,
        differential_predictions=[
            DifferentialPredictionDraft(
                local_id="dp",
                observable="response",
                baseline_expectation="baseline",
                alternative_expectation="alternative",
                discriminating_outcome="different",
            )
        ],
        falsifiers=[
            ReframeFalsifierDraft(local_id="f", falsifying_outcome="baseline persists")
        ],
        discriminating_test=DiscriminatingTestDraft(
            test_design="measure",
            primary_observables=["response"],
            baseline_favoring_outcome="baseline",
            alternative_favoring_outcome="alternative",
        ),
        unresolved_questions=[],
    )


def _shadow(*, failed: bool = False, trigger_id: str = "trigger:r") -> ScientificReframingShadowReport:
    candidate = _candidate()
    return ScientificReframingShadowReport(
        report_id="shadow:r",
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        backend_name="fake",
        model_name="fake",
        runs=[
            ReframeOperatorRunRecord(
                operator_id="LATENT_VARIABLE",
                readiness_status="ready_now",
                decision="generation_failed" if failed else "generated",
                candidate_ids=[] if failed else [candidate.reframe_id],
                generation_error_type="RuntimeError" if failed else None,
            ),
            ReframeOperatorRunRecord(
                operator_id="REGIME_BOUNDARY",
                readiness_status="ready_with_optional_enrichment",
                decision="skipped_not_triggered",
            ),
        ],
        candidates=[] if failed else [candidate],
        llm_calls_performed=1,
        source_trigger_report_id=trigger_id,
        scientific_trigger_evaluated=True,
    )


def _critic(shadow: ScientificReframingShadowReport, *, complete: bool = True) -> ScientificReframeCriticReport:
    candidate = shadow.candidates[0]
    review = ScientificReframeCriticReview(
        review_id="review:1",
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
            operator_shape_valid=True,
            all_premise_ids_grounded=True,
            all_gap_ids_grounded=True,
        ),
        dimensions=[
            ReframeCriticDimensionReview(
                dimension=dimension,
                rating=2 if complete else None,
                review_status="reviewed" if complete else "missing",
            )
            for dimension in CRITIC_DIMENSIONS
        ],
        llm_review_complete=complete,
        llm_error_type=None if complete else "RuntimeError",
    )
    return ScientificReframeCriticReport(
        report_id="critic:r",
        source_shadow_report_id=shadow.report_id,
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        backend_name="fake",
        model_name="fake",
        reviews=[review],
        llm_calls_attempted=1,
        llm_calls_succeeded=1 if complete else 0,
    )


def _write(path: Path, model) -> None:
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def test_triggered_operator_ids_preserve_canonical_order() -> None:
    assert triggered_operator_ids(_audit_row("latent_and_regime")) == [
        "LATENT_VARIABLE",
        "REGIME_BOUNDARY",
    ]
    assert triggered_operator_ids(_audit_row("none")) == []


def test_shadow_resume_requires_current_trigger_lineage(tmp_path: Path) -> None:
    path = tmp_path / "shadow.json"
    _write(path, _shadow())
    report, reason = resumable_shadow_report(
        path=path,
        trigger=_trigger(),
        expected_task_id="task:t",
    )
    assert report is not None
    assert "current trigger lineage" in reason

    stale, reason = resumable_shadow_report(
        path=path,
        trigger=_trigger().model_copy(update={"report_id": "trigger:new"}),
        expected_task_id="task:t",
    )
    assert stale is None
    assert "stale" in reason


def test_shadow_resume_rejects_generation_failure(tmp_path: Path) -> None:
    path = tmp_path / "shadow.json"
    _write(path, _shadow(failed=True))
    report, reason = resumable_shadow_report(
        path=path,
        trigger=_trigger(),
        expected_task_id="task:t",
    )
    assert report is None
    assert "generation_failed" in reason


def test_critic_resume_requires_complete_reviews_and_shadow_lineage(tmp_path: Path) -> None:
    shadow = _shadow()
    good_path = tmp_path / "critic_good.json"
    _write(good_path, _critic(shadow))
    good, _ = resumable_critic_report(path=good_path, shadow=shadow)
    assert good is not None

    bad_path = tmp_path / "critic_bad.json"
    _write(bad_path, _critic(shadow, complete=False))
    bad, reason = resumable_critic_report(path=bad_path, shadow=shadow)
    assert bad is None
    assert "incomplete" in reason


def test_portfolio_resume_requires_both_lineages(tmp_path: Path) -> None:
    shadow = _shadow()
    critic = _critic(shadow)
    portfolio = build_scientific_reframe_shadow_portfolio(shadow=shadow, critic=critic)
    path = tmp_path / "portfolio.json"
    _write(path, portfolio)
    loaded, _ = resumable_portfolio_report(path=path, shadow=shadow, critic=critic)
    assert loaded is not None

    stale_critic = critic.model_copy(update={"report_id": "critic:new"})
    stale, reason = resumable_portfolio_report(
        path=path,
        shadow=shadow,
        critic=stale_critic,
    )
    assert stale is None
    assert "critic lineage" in reason


def test_summary_counts_calls_without_charging_resumed_stages(tmp_path: Path) -> None:
    audit = summarize_benchmark_task_audits(
        benchmark_root="/bench",
        extraction_roots=["/corpus"],
        task_audits=[_audit_row("latent_only")],
    )
    shadow = _shadow()
    critic = _critic(shadow)
    portfolio = build_scientific_reframe_shadow_portfolio(shadow=shadow, critic=critic)
    task = BenchmarkTaskSelectiveExecution(
        task_key="K01/replicate_01",
        case_key="K01",
        replicate_key="replicate_01",
        canonical_dir="/bench/K01/replicate_01/canonical",
        task_id="task:t",
        trigger_pattern="latent_only",
        triggered_operator_ids=["LATENT_VARIABLE"],
        status="completed",
        generation=stage_from_shadow(report=shadow, path=tmp_path / "s.json", status="executed"),
        critic=stage_from_critic(report=critic, path=tmp_path / "c.json", status="executed"),
        portfolio=stage_from_portfolio(path=tmp_path / "p.json", status="resumed", resume_reason="valid"),
        candidate_count=1,
        critic_review_count=1,
        portfolio_assessment_count=len(portfolio.assessments),
        resume_used=True,
        llm_calls_attempted=2,
        llm_calls_succeeded=2,
    )
    report = summarize_selective_execution(
        audit=audit,
        extraction_roots=["/corpus"],
        resume_enabled=True,
        task_executions=[task],
    )
    assert report.llm_calls_attempted == 2
    assert report.generation_calls_attempted == 1
    assert report.critic_calls_attempted == 1
    assert report.resumed_task_count == 1


def test_skipped_task_invariant_requires_all_stages_skipped() -> None:
    with pytest.raises(ValidationError):
        BenchmarkTaskSelectiveExecution(
            task_key="K01/replicate_01",
            case_key="K01",
            replicate_key="replicate_01",
            canonical_dir="/bench/K01/replicate_01/canonical",
            trigger_pattern="none",
            triggered_operator_ids=[],
            status="skipped_not_triggered",
            generation=skipped_stage("generation"),
            critic=skipped_stage("critic"),
            portfolio=stage_from_portfolio(
                path=Path("/tmp/p.json"),
                status="executed",
            ),
            llm_calls_attempted=0,
            llm_calls_succeeded=0,
        )


def test_report_preserves_shadow_only_no_authority_invariants() -> None:
    audit = summarize_benchmark_task_audits(
        benchmark_root="/bench",
        extraction_roots=["/corpus"],
        task_audits=[_audit_row("none")],
    )
    task = BenchmarkTaskSelectiveExecution(
        task_key="K01/replicate_01",
        case_key="K01",
        replicate_key="replicate_01",
        canonical_dir="/bench/K01/replicate_01/canonical",
        task_id="task:t",
        trigger_pattern="none",
        triggered_operator_ids=[],
        status="skipped_not_triggered",
        generation=skipped_stage("generation"),
        critic=skipped_stage("critic"),
        portfolio=skipped_stage("portfolio"),
        llm_calls_attempted=0,
        llm_calls_succeeded=0,
    )
    report = summarize_selective_execution(
        audit=audit,
        extraction_roots=["/corpus"],
        resume_enabled=True,
        task_executions=[task],
    )
    assert report.shadow_only is True
    assert report.candidate_ranking_performed is False
    assert report.overall_score_computed is False
    assert report.winner_selected is False
    assert report.external_novelty_evaluated is False
    assert report.n10_run is False
    assert report.production_selection_changed is False
