from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.scientific_verifier_materialization_planner import (
    build_scientific_verifier_materialization_plan,
)


def _canonical(root: Path, case: str) -> Path:
    path = root / case / "replicate_01" / "canonical"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _context(path: Path) -> None:
    _write(
        path / "hypothesis.context.json",
        {
            "schema_version": "hypothesis-context-v1",
            "context_id": "ctx:1",
        },
    )


def _candidate(path: Path) -> None:
    _write(
        path / "scientific_pre_n10_candidate_portfolio.json",
        {
            "schema_version": "production-facing-scientific-candidate-portfolio-v1",
            "portfolio_id": "candidate:1",
            "source_context_id": "ctx:1",
            "source_cross_lane_portfolio_id": "cross:1",
        },
    )


def _atomic(path: Path) -> None:
    _write(
        path / "scientific_atomic_cross_lane_v2.report.json",
        {
            "schema_version": "atomic-cross-lane-scientific-synthesis-report-v1",
            "report_id": "atomic-report:1",
            "source_candidate_portfolio_id": "candidate:1",
            "source_context_id": "ctx:1",
            "hypotheses": [{"hypothesis_id": "hypothesis:1"}],
        },
    )
    _write(
        path / "scientific_atomic_cross_lane_v2.portfolio.json",
        {
            "schema_version": "hypothesis-portfolio-v1",
            "portfolio_id": "atomic-portfolio:1",
            "source_context_id": "ctx:1",
            "hypotheses": [{"hypothesis_id": "hypothesis:1"}],
        },
    )


def _annotation(path: Path) -> None:
    _write(
        path / "scientific_atomic_grounded_identity_v2.annotation.json",
        {
            "schema_version": "grounded-identity-constituent-annotation-report-v1",
            "source_candidate_portfolio_id": "candidate:1",
            "source_atomic_synthesis_report_id": "atomic-report:1",
            "annotations": [],
        },
    )


def _external(path: Path, *, source_portfolio_id: str = "atomic-portfolio:1") -> None:
    _write(
        path / "scientific_atomic_n10_external.report.json",
        {
            "schema_version": "external-novelty-report-v1",
            "report_id": "external:1",
            "source_portfolio_id": source_portfolio_id,
            "cards": [{"hypothesis_id": "hypothesis:1"}],
        },
    )


def test_exact_lineage_is_ready_for_verifier(tmp_path: Path) -> None:
    canonical = _canonical(tmp_path, "G01")
    _context(canonical)
    _candidate(canonical)
    _atomic(canonical)
    _annotation(canonical)
    _external(canonical)

    plan = build_scientific_verifier_materialization_plan(
        benchmark_root=tmp_path,
        case_keys=["G01"],
    )

    assert plan.status_counts == {"READY_FOR_VERIFIER": 1}
    task = plan.tasks[0]
    assert task.verifier_required_artifacts_present is True
    assert all(stage.state == "REUSE_MATCHED" for stage in task.stages)
    assert task.planned_minimum_llm_calls == 0
    assert task.planned_retrieval_runs == 0


def test_mismatched_external_report_is_not_reused(tmp_path: Path) -> None:
    canonical = _canonical(tmp_path, "G02")
    _context(canonical)
    _candidate(canonical)
    _atomic(canonical)
    _annotation(canonical)
    _external(canonical, source_portfolio_id="stale-portfolio")

    plan = build_scientific_verifier_materialization_plan(
        benchmark_root=tmp_path,
        case_keys=["G02"],
    )

    task = plan.tasks[0]
    assert task.status == "MATERIALIZATION_REQUIRED"
    external = next(stage for stage in task.stages if stage.stage_id == "external_novelty_report")
    assert external.state == "NEEDS_RETRIEVAL_AND_REVIEW"
    assert external.expected_retrieval_runs == 1
    assert external.runtime_dependent_llm_calls is True
    assert len(external.ignored_stale_or_mismatched_artifacts) == 1


def test_candidate_can_be_planned_deterministically_from_existing_upstream(tmp_path: Path) -> None:
    canonical = _canonical(tmp_path, "K01")
    _context(canonical)
    _write(
        canonical / "novelty_refinement_a6.portfolio.json",
        {
            "schema_version": "hypothesis-portfolio-v1",
            "portfolio_id": "alpha6:1",
            "source_context_id": "ctx:1",
            "hypotheses": [{"hypothesis_id": "source:1"}],
        },
    )
    _write(
        canonical / "scientific_pre_n10_cross_lane_portfolio.json",
        {
            "schema_version": "cross-lane-scientific-reasoning-shadow-portfolio-v1",
            "portfolio_id": "cross:1",
            "source_context_id": "ctx:1",
        },
    )
    _write(
        canonical / "scientific_pre_n10_reframing_shadow.json",
        {
            "schema_version": "scientific-reframing-shadow-report-v1",
            "report_id": "reframe:1",
            "source_context_id": "ctx:1",
        },
    )

    plan = build_scientific_verifier_materialization_plan(
        benchmark_root=tmp_path,
        case_keys=["K01"],
    )

    task = plan.tasks[0]
    candidate = next(
        stage
        for stage in task.stages
        if stage.stage_id == "production_facing_candidate_portfolio"
    )
    assert task.status == "MATERIALIZATION_REQUIRED"
    assert candidate.state == "NEEDS_DETERMINISTIC_MATERIALIZATION"
    assert candidate.expected_llm_calls == 0
    assert candidate.missing_prerequisites == []


def test_caller_exclusion_is_recorded_without_inspection(tmp_path: Path) -> None:
    canonical = _canonical(tmp_path, "G04")
    _write(canonical / "malformed.json", {"anything": "here"})

    plan = build_scientific_verifier_materialization_plan(
        benchmark_root=tmp_path,
        case_keys=["G01", "G04"],
        exclude_case_keys=["G04"],
    )

    by_case = {task.case_key: task for task in plan.tasks}
    assert by_case["G04"].status == "EXCLUDED_BY_CALLER"
    assert by_case["G04"].stages[0].stage_id == "caller_exclusion"
    assert plan.selected_case_keys == ["G01"]
    assert plan.excluded_case_keys == ["G04"]
    assert plan.llm_calls_performed == 0
    assert plan.retrieval_performed is False


def test_multiple_exact_candidate_portfolios_fail_closed(tmp_path: Path) -> None:
    canonical = _canonical(tmp_path, "K02")
    _context(canonical)
    for index in (1, 2):
        _write(
            canonical / f"candidate_{index}.json",
            {
                "schema_version": "production-facing-scientific-candidate-portfolio-v1",
                "portfolio_id": f"candidate:{index}",
                "source_context_id": "ctx:1",
            },
        )

    plan = build_scientific_verifier_materialization_plan(
        benchmark_root=tmp_path,
        case_keys=["K02"],
    )

    task = plan.tasks[0]
    assert task.status == "AMBIGUOUS_LINEAGE"
    stage = task.stages[0]
    assert stage.stage_id == "production_facing_candidate_portfolio"
    assert stage.state == "AMBIGUOUS_LINEAGE"


def test_existing_candidate_plans_atomic_llm_as_next_authoritative_stage(tmp_path: Path) -> None:
    canonical = _canonical(tmp_path, "K03")
    _context(canonical)
    _candidate(canonical)

    plan = build_scientific_verifier_materialization_plan(
        benchmark_root=tmp_path,
        case_keys=["K03"],
    )

    task = plan.tasks[0]
    atomic = next(stage for stage in task.stages if stage.stage_id == "atomic_cross_lane_synthesis")
    assert task.status == "MATERIALIZATION_REQUIRED"
    assert atomic.state == "NEEDS_LLM_MATERIALIZATION"
    assert atomic.expected_llm_calls == 1
    assert task.planned_minimum_llm_calls == 1


def test_candidate_is_narrowed_by_exact_cross_lane_lineage(tmp_path: Path) -> None:
    canonical = _canonical(tmp_path, "G02")
    _context(canonical)
    _write(
        canonical / "scientific_cross_lane_reasoning_portfolio.json",
        {
            "schema_version": "cross-lane-scientific-reasoning-shadow-portfolio-v1",
            "portfolio_id": "cross:1",
            "source_context_id": "ctx:1",
        },
    )
    _write(
        canonical / "candidate_current.json",
        {
            "schema_version": "production-facing-scientific-candidate-portfolio-v1",
            "portfolio_id": "candidate:current",
            "source_context_id": "ctx:1",
            "source_cross_lane_portfolio_id": "cross:1",
        },
    )
    _write(
        canonical / "candidate_stale.json",
        {
            "schema_version": "production-facing-scientific-candidate-portfolio-v1",
            "portfolio_id": "candidate:stale",
            "source_context_id": "ctx:1",
            "source_cross_lane_portfolio_id": "cross:stale",
        },
    )

    plan = build_scientific_verifier_materialization_plan(
        benchmark_root=tmp_path,
        case_keys=["G02"],
    )

    task = plan.tasks[0]
    stage = next(
        row
        for row in task.stages
        if row.stage_id == "production_facing_candidate_portfolio"
    )
    assert stage.state == "REUSE_MATCHED"
    assert stage.selected_artifacts[0].object_id == "candidate:current"
    assert any("candidate_stale.json" in path for path in stage.ignored_stale_or_mismatched_artifacts)


def test_reframing_shadow_is_resolved_by_cross_lane_candidate_ids(tmp_path: Path) -> None:
    canonical = _canonical(tmp_path, "K03")
    _context(canonical)
    _write(
        canonical / "novelty_refinement_a6.portfolio.json",
        {
            "schema_version": "hypothesis-portfolio-v1",
            "portfolio_id": "alpha6:1",
            "source_context_id": "ctx:1",
            "hypotheses": [{"hypothesis_id": "source:1"}],
        },
    )
    _write(
        canonical / "scientific_cross_lane_reasoning_portfolio.json",
        {
            "schema_version": "cross-lane-scientific-reasoning-shadow-portfolio-v1",
            "portfolio_id": "cross:1",
            "source_context_id": "ctx:1",
            "relational_lane": {"source_portfolio_id": "alpha6:1"},
            "entries": [
                {
                    "lane_id": "RELATIONAL_DISCOVERY",
                    "source_object_id": "source:1",
                },
                {
                    "lane_id": "SCIENTIFIC_REFRAMING",
                    "source_object_id": "reframe:good:candidate",
                },
            ],
        },
    )
    _write(
        canonical / "scientific_reframing_shadow_good.json",
        {
            "schema_version": "scientific-reframing-shadow-report-v1",
            "report_id": "reframe:good",
            "source_context_id": "ctx:1",
            "candidates": [{"reframe_id": "reframe:good:candidate"}],
        },
    )
    _write(
        canonical / "scientific_reframing_shadow_stale.json",
        {
            "schema_version": "scientific-reframing-shadow-report-v1",
            "report_id": "reframe:stale",
            "source_context_id": "ctx:1",
            "candidates": [{"reframe_id": "reframe:stale:candidate"}],
        },
    )

    plan = build_scientific_verifier_materialization_plan(
        benchmark_root=tmp_path,
        case_keys=["K03"],
    )

    task = plan.tasks[0]
    stage = next(
        row
        for row in task.stages
        if row.stage_id == "production_facing_candidate_portfolio"
    )
    assert stage.state == "NEEDS_DETERMINISTIC_MATERIALIZATION"
    selected = {row.object_id for row in stage.selected_artifacts}
    assert "cross:1" in selected
    assert "alpha6:1" in selected
    assert "reframe:good" in selected
    assert "reframe:stale" not in selected
