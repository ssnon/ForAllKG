from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.reframing.multitask_replication_preflight import (
    build_multitask_replication_preflight,
)


def _write(path: Path, payload: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if payload is None:
        path.write_text("{}\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _base(root: Path, case: str = "G03") -> Path:
    canonical = root / case / "replicate_01" / "canonical"
    _write(canonical / "explorer.packet.json", {"schema_version": "packet-v1"})
    _write(canonical / "hypothesis.context.json", {"schema_version": "hypothesis-context-v1"})
    _write(
        canonical / "hypothesis_axis_a4.portfolio.json",
        {"schema_version": "hypothesis-portfolio-v1", "hypotheses": [{}, {}, {}]},
    )
    return canonical


def test_blocked_when_base_artifacts_are_missing(tmp_path: Path):
    report = build_multitask_replication_preflight(
        benchmark_root=tmp_path,
        case_keys=["G02"],
    )
    task = report.tasks[0]
    assert task.status == "blocked_missing_base"
    assert set(task.missing_base_artifacts) == {
        "explorer_packet",
        "hypothesis_context",
        "relational_portfolio",
    }


def test_base_ready_without_shadow_requests_generation(tmp_path: Path):
    _base(tmp_path)
    report = build_multitask_replication_preflight(
        benchmark_root=tmp_path,
        case_keys=["G03"],
    )
    task = report.tasks[0]
    assert task.status == "ready_for_reframe_generation"
    assert "run_scientific_reframing_shadow_if_trigger_eligible" in task.next_actions
    assert task.candidate_counts["relational_portfolio"] == 3


def test_s38_shadow_variant_is_discovered(tmp_path: Path):
    canonical = _base(tmp_path)
    _write(
        canonical / "scientific_reframing_shadow_s38.json",
        {"schema_version": "scientific-reframing-shadow-report-v1", "candidates": [{}, {}]},
    )
    report = build_multitask_replication_preflight(
        benchmark_root=tmp_path,
        case_keys=["G03"],
    )
    task = report.tasks[0]
    shadow = next(row for row in task.artifacts if row.artifact_key == "reframing_shadow")
    assert shadow.state == "present"
    assert shadow.path and shadow.path.endswith("scientific_reframing_shadow_s38.json")
    assert task.candidate_counts["reframing_shadow"] == 2


def test_existing_reframing_without_unified_portfolio_requests_extension(tmp_path: Path):
    canonical = _base(tmp_path)
    _write(canonical / "scientific_reframing_shadow.json", {"candidates": [{}]})
    report = build_multitask_replication_preflight(
        benchmark_root=tmp_path,
        case_keys=["G03"],
    )
    task = report.tasks[0]
    assert task.status == "ready_to_extend_reframing"
    assert "build_reasoning_mode_contrast_from_available_operator_outputs" in task.next_actions
    assert "build_unified_reframing_portfolio" in task.next_actions
    assert not any("proxy_challenge" in action for action in task.next_actions)
    assert any("PROXY_CHALLENGE is optional" in note for note in task.diagnostic_notes)
    assert any("CONTRADICTION_RESOLUTION is optional" in note for note in task.diagnostic_notes)


def test_reframing_portfolio_without_cross_lane_requests_cross_lane(tmp_path: Path):
    canonical = _base(tmp_path)
    _write(canonical / "scientific_reframing_shadow.json", {"candidates": [{}]})
    _write(
        canonical / "scientific_reframing_reasoning_portfolio.json",
        {"candidate_ids": ["r1"]},
    )
    report = build_multitask_replication_preflight(
        benchmark_root=tmp_path,
        case_keys=["G03"],
    )
    assert report.tasks[0].status == "ready_for_cross_lane_assembly"


def test_cross_lane_without_normalized_packet_requests_ablation_build(tmp_path: Path):
    canonical = _base(tmp_path)
    _write(canonical / "scientific_reframing_shadow.json", {"candidates": [{}]})
    _write(canonical / "scientific_reframing_reasoning_portfolio.json", {"candidate_ids": ["r1"]})
    _write(canonical / "scientific_cross_lane_reasoning_portfolio.json", {"entries": [{}, {}, {}, {}]})
    report = build_multitask_replication_preflight(
        benchmark_root=tmp_path,
        case_keys=["G03"],
    )
    task = report.tasks[0]
    assert task.status == "ready_for_ablation_build"
    assert "build_schema_normalized_ablation_packet" in task.next_actions


def test_normalized_packet_without_matched_packet_marks_cardinality_followup(tmp_path: Path):
    canonical = _base(tmp_path)
    _write(canonical / "scientific_reframing_shadow.json", {"candidates": [{}]})
    _write(canonical / "scientific_reframing_reasoning_portfolio.json", {"candidate_ids": ["r1"]})
    _write(canonical / "scientific_cross_lane_reasoning_portfolio.json", {"entries": [{}]})
    _write(canonical / "scientific_reasoning_ablation_schema_normalized_packet.json", {"comparison_count": 3})
    report = build_multitask_replication_preflight(
        benchmark_root=tmp_path,
        case_keys=["G03"],
    )
    task = report.tasks[0]
    assert task.status == "ready_for_blind_evaluation"
    assert "build_generalized_matched_count_ablation" in task.next_actions
    assert any("min(relational_count, reframing_count)" in note for note in task.diagnostic_notes)


def test_completed_matched_evaluation_is_detected(tmp_path: Path):
    canonical = _base(tmp_path)
    _write(canonical / "scientific_reframing_shadow.json", {"candidates": [{}]})
    _write(canonical / "scientific_reframing_reasoning_portfolio.json", {"candidate_ids": ["r1"]})
    _write(canonical / "scientific_cross_lane_reasoning_portfolio.json", {"entries": [{}]})
    _write(canonical / "scientific_reasoning_ablation_schema_normalized_packet.json", {"comparison_count": 3})
    _write(canonical / "scientific_reasoning_ablation_generalized_matched_packet.json", {"comparison_count": 4})
    _write(canonical / "scientific_reasoning_ablation_generalized_matched_unblinded_evaluation.json", {"comparisons": []})
    report = build_multitask_replication_preflight(
        benchmark_root=tmp_path,
        case_keys=["G03"],
    )
    assert report.tasks[0].status == "replication_evaluation_materialized"


def test_multiple_cases_are_deterministic_and_not_ranked(tmp_path: Path):
    _base(tmp_path, "G02")
    _base(tmp_path, "K06")
    first = build_multitask_replication_preflight(
        benchmark_root=tmp_path,
        case_keys=["K06", "G02"],
    )
    second = build_multitask_replication_preflight(
        benchmark_root=tmp_path,
        case_keys=["K06", "G02"],
    )
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert [row.task_key for row in first.tasks] == [
        "G02/replicate_01",
        "K06/replicate_01",
    ]
    assert first.scientific_task_ranking_performed is False
    assert first.replication_success_established is False
