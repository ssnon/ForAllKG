from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_core.discovery.reframing.benchmark_synthesis import (
    build_benchmark_reframe_synthesis,
    lexical_construct_tokens,
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
    SelectiveExecutionStage,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    ConditionDiversitySignal,
    DirectScientificTriggerSignal,
    ReframeOperatorTriggerAssessment,
    ScientificReframeTriggerReport,
)


def _candidate(candidate_id: str, title: str, construct: str) -> ScientificReframeCandidate:
    return ScientificReframeCandidate(
        reframe_id=candidate_id,
        operator_id="LATENT_VARIABLE",
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        title=title,
        premise_statement_ids=["s1", "s2"],
        gap_statement_ids=["g1"],
        baseline_model=ScientificModelDraft(
            summary="baseline response",
            explained_statement_ids=["s1"],
            expected_observations=["baseline observation"],
        ),
        alternative_model=ScientificModelDraft(
            summary="alternative response",
            explained_statement_ids=["s1", "s2"],
            expected_observations=["alternative observation"],
        ),
        challenged_assumption="separable mechanisms",
        proposed_constructs=[construct],
        latent_constructs=[construct],
        boundary_variables=[],
        regime_change_kind=None,
        differential_predictions=[
            DifferentialPredictionDraft(
                local_id="dp",
                observable="signal",
                baseline_expectation="tracks loading",
                alternative_expectation="tracks hotspot occupancy",
                discriminating_outcome="ranking changes",
            )
        ],
        falsifiers=[
            ReframeFalsifierDraft(local_id="f", falsifying_outcome="ranking never changes")
        ],
        discriminating_test=DiscriminatingTestDraft(
            test_design="measure loading and hotspot occupancy",
            primary_observables=["signal"],
            baseline_favoring_outcome="loading explains signal",
            alternative_favoring_outcome="occupancy explains signal",
        ),
        unresolved_questions=[],
    )


def _trigger() -> ScientificReframeTriggerReport:
    return ScientificReframeTriggerReport(
        report_id="trigger:r",
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        source_explorer_report_sha256="explorer",
        source_evidence_sha256="evidence",
        trigger_resolution_mode="evidence_level_refined",
        direct_trigger_signals=[
            DirectScientificTriggerSignal(
                signal_id="signal:m",
                kind="MEDIATION_GAP",
                target_operator_id="LATENT_VARIABLE",
                strength="sufficient",
                gap_statement_ids=["g1"],
            )
        ],
        condition_diversity=ConditionDiversitySignal(
            example_count=1,
            paper_count=1,
            distinct_condition_signature_count=1,
        ),
        assessments=[
            ReframeOperatorTriggerAssessment(
                operator_id="LATENT_VARIABLE",
                decision="triggered",
                direct_signal_ids=["signal:m"],
                scientific_trigger_signal=True,
            ),
            ReframeOperatorTriggerAssessment(
                operator_id="REGIME_BOUNDARY",
                decision="not_triggered",
                scientific_trigger_signal=False,
            ),
        ],
    )


def _shadow(candidate: ScientificReframeCandidate) -> ScientificReframingShadowReport:
    return ScientificReframingShadowReport(
        report_id=f"shadow:{candidate.reframe_id}",
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        backend_name="fake",
        model_name="fake",
        runs=[
            ReframeOperatorRunRecord(
                operator_id="LATENT_VARIABLE",
                readiness_status="ready_now",
                decision="generated",
                candidate_ids=[candidate.reframe_id],
            ),
            ReframeOperatorRunRecord(
                operator_id="REGIME_BOUNDARY",
                readiness_status="ready_with_optional_enrichment",
                decision="skipped_not_triggered",
            ),
        ],
        candidates=[candidate],
        llm_calls_performed=1,
        source_trigger_report_id="trigger:r",
        scientific_trigger_evaluated=True,
    )


def _critic(shadow: ScientificReframingShadowReport) -> ScientificReframeCriticReport:
    candidate = shadow.candidates[0]
    review = ScientificReframeCriticReview(
        review_id=f"review:{candidate.reframe_id}",
        candidate_id=candidate.reframe_id,
        operator_id=candidate.operator_id,
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        structural_audit=ReframeStructuralAudit(
            grounded_premise_count=2,
            gap_count=1,
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
                rating=2,
                review_status="reviewed",
                rationale=f"{dimension} rationale",
            )
            for dimension in CRITIC_DIMENSIONS
        ],
        llm_review_complete=True,
    )
    return ScientificReframeCriticReport(
        report_id=f"critic:{candidate.reframe_id}",
        source_shadow_report_id=shadow.report_id,
        source_task_id="task:t",
        source_context_id="ctx",
        source_context_sha256="sha",
        backend_name="fake",
        model_name="fake",
        reviews=[review],
        llm_calls_attempted=1,
        llm_calls_succeeded=1,
    )


def _write_artifacts(tmp_path: Path, candidate: ScientificReframeCandidate, task_key: str):
    canonical = tmp_path / task_key / "replicate_01" / "canonical"
    canonical.mkdir(parents=True)
    trigger = _trigger()
    shadow = _shadow(candidate)
    critic = _critic(shadow)
    portfolio = build_scientific_reframe_shadow_portfolio(shadow=shadow, critic=critic)
    trigger_path = canonical / "scientific_reframing_triggers.json"
    shadow_path = canonical / "shadow.json"
    critic_path = canonical / "critic.json"
    portfolio_path = canonical / "portfolio.json"
    trigger_path.write_text(trigger.model_dump_json(indent=2), encoding="utf-8")
    shadow_path.write_text(shadow.model_dump_json(indent=2), encoding="utf-8")
    critic_path.write_text(critic.model_dump_json(indent=2), encoding="utf-8")
    portfolio_path.write_text(portfolio.model_dump_json(indent=2), encoding="utf-8")
    task = BenchmarkTaskSelectiveExecution(
        task_key=f"{task_key}/replicate_01",
        case_key=task_key,
        replicate_key="replicate_01",
        canonical_dir=str(canonical),
        task_id="task:t",
        trigger_pattern="latent_only",
        triggered_operator_ids=["LATENT_VARIABLE"],
        status="completed",
        generation=SelectiveExecutionStage(stage="generation", status="executed", artifact_path=str(shadow_path), llm_calls_attempted=1, llm_calls_succeeded=1),
        critic=SelectiveExecutionStage(stage="critic", status="executed", artifact_path=str(critic_path), llm_calls_attempted=1, llm_calls_succeeded=1),
        portfolio=SelectiveExecutionStage(stage="portfolio", status="executed", artifact_path=str(portfolio_path)),
        candidate_count=1,
        critic_review_count=1,
        portfolio_assessment_count=1,
        llm_calls_attempted=2,
        llm_calls_succeeded=2,
    )
    return task


def _execution(tmp_path: Path, tasks: list[BenchmarkTaskSelectiveExecution]):
    skipped = BenchmarkTaskSelectiveExecution(
        task_key="SKIP/replicate_01",
        case_key="SKIP",
        replicate_key="replicate_01",
        canonical_dir=str(tmp_path / "SKIP" / "replicate_01" / "canonical"),
        trigger_pattern="none",
        triggered_operator_ids=[],
        status="skipped_not_triggered",
        generation=SelectiveExecutionStage(stage="generation", status="skipped"),
        critic=SelectiveExecutionStage(stage="critic", status="skipped"),
        portfolio=SelectiveExecutionStage(stage="portfolio", status="skipped"),
    )
    all_tasks = [*tasks, skipped]
    return ScientificReframeSelectiveExecutionReport(
        execution_id="execution:r",
        source_benchmark_audit_id="audit:r",
        benchmark_root=str(tmp_path),
        extraction_roots=["/corpus"],
        resume_enabled=True,
        task_executions=all_tasks,
        task_count=len(all_tasks),
        triggered_task_count=len(tasks),
        skipped_task_count=1,
        completed_task_count=len(tasks),
        completed_with_errors_task_count=0,
        error_task_count=0,
        resumed_task_count=0,
        status_counts={"completed": len(tasks), "skipped_not_triggered": 1},
        llm_calls_attempted=2 * len(tasks),
        llm_calls_succeeded=2 * len(tasks),
        generation_calls_attempted=len(tasks),
        critic_calls_attempted=len(tasks),
    )


def test_lexical_tokenization_removes_generic_words_and_stems() -> None:
    tokens = lexical_construct_tokens([
        "A latent variable for hotspot occupancies and delivery effects",
    ])
    assert "latent" not in tokens
    assert "variable" not in tokens
    assert "hotspot" in tokens
    assert "occupancy" in tokens
    assert "delivery" in tokens


def test_synthesis_collects_candidate_trigger_critic_and_portfolio(tmp_path: Path) -> None:
    task = _write_artifacts(
        tmp_path,
        _candidate("candidate:1", "Hotspot delivery concordance", "hotspot delivery concordance"),
        "K01",
    )
    report = build_benchmark_reframe_synthesis(execution=_execution(tmp_path, [task]))
    assert report.candidate_count == 1
    row = report.candidates[0]
    assert row.trigger_direct_signal_kinds == ["MEDIATION_GAP"]
    assert row.critic_vector[0].rating == 2
    assert row.assigned_slot_id == "B_STRUCTURAL_REFRAME"


def test_synthesis_excludes_not_triggered_tasks(tmp_path: Path) -> None:
    task = _write_artifacts(
        tmp_path,
        _candidate("candidate:1", "Hotspot delivery", "hotspot delivery"),
        "K01",
    )
    report = build_benchmark_reframe_synthesis(execution=_execution(tmp_path, [task]))
    assert report.triggered_task_count == 1
    assert all(row.case_key != "SKIP" for row in report.candidates)


def test_synthesis_fails_closed_on_stale_shadow_trigger_lineage(tmp_path: Path) -> None:
    task = _write_artifacts(
        tmp_path,
        _candidate("candidate:1", "Hotspot delivery", "hotspot delivery"),
        "K01",
    )
    shadow_path = Path(task.generation.artifact_path)
    shadow = ScientificReframingShadowReport.model_validate_json(shadow_path.read_text())
    shadow = shadow.model_copy(update={"source_trigger_report_id": "trigger:stale"})
    shadow_path.write_text(shadow.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="lineage"):
        build_benchmark_reframe_synthesis(execution=_execution(tmp_path, [task]))


def test_lexical_overlap_is_same_operator_diagnostic_not_semantic_collapse(tmp_path: Path) -> None:
    left = _write_artifacts(
        tmp_path,
        _candidate("candidate:1", "Hotspot delivery concordance", "hotspot delivery concordance"),
        "K01",
    )
    right = _write_artifacts(
        tmp_path,
        _candidate("candidate:2", "Hotspot delivery accessibility", "hotspot delivery accessibility"),
        "K02",
    )
    report = build_benchmark_reframe_synthesis(
        execution=_execution(tmp_path, [left, right]),
        lexical_overlap_threshold=0.30,
    )
    assert len(report.lexical_overlap_diagnostics) == 1
    overlap = report.lexical_overlap_diagnostics[0]
    assert overlap.threshold_exceeded is True
    assert set(overlap.shared_tokens) >= {"hotspot", "delivery"}
    assert overlap.semantic_collapse_asserted is False
    assert report.semantic_collapse_evaluated is False


def test_operator_summary_keeps_dimension_distributions_without_scalar_score(tmp_path: Path) -> None:
    task = _write_artifacts(
        tmp_path,
        _candidate("candidate:1", "Hotspot delivery", "hotspot delivery"),
        "K01",
    )
    report = build_benchmark_reframe_synthesis(execution=_execution(tmp_path, [task]))
    latent = next(row for row in report.operator_summaries if row.operator_id == "LATENT_VARIABLE")
    assert latent.dimension_rating_counts["premise_fidelity"] == {"2": 1}
    assert report.aggregate_quality_score_computed is False
    assert report.candidate_ranking_performed is False


def test_exact_construct_signatures_are_reported_as_duplicates_only(tmp_path: Path) -> None:
    left = _write_artifacts(
        tmp_path,
        _candidate("candidate:1", "Hotspot delivery", "hotspot delivery"),
        "K01",
    )
    right = _write_artifacts(
        tmp_path,
        _candidate("candidate:2", "Hotspot delivery", "hotspot delivery"),
        "K02",
    )
    report = build_benchmark_reframe_synthesis(execution=_execution(tmp_path, [left, right]))
    latent = next(row for row in report.operator_summaries if row.operator_id == "LATENT_VARIABLE")
    assert latent.exact_construct_signature_duplicate_groups == [["candidate:1", "candidate:2"]]
    assert report.semantic_equivalence_evaluated is False


def test_lineage_errors_can_be_retained_diagnostically_when_requested(tmp_path: Path) -> None:
    task = _write_artifacts(
        tmp_path,
        _candidate("candidate:1", "Hotspot delivery", "hotspot delivery"),
        "K01",
    )
    critic_path = Path(task.critic.artifact_path)
    critic = ScientificReframeCriticReport.model_validate_json(critic_path.read_text())
    critic = critic.model_copy(update={"source_shadow_report_id": "shadow:stale"})
    critic_path.write_text(critic.model_dump_json(indent=2), encoding="utf-8")
    report = build_benchmark_reframe_synthesis(
        execution=_execution(tmp_path, [task]),
        fail_on_lineage_error=False,
    )
    assert report.candidate_count == 0
    assert any("critic shadow lineage mismatch" in row for row in report.artifact_lineage_errors)
