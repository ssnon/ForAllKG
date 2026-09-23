from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.discovery.scientific_verifier_unseen_validation import (
    build_unseen_validation_preflight,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ready_case(root: Path, case: str = "K99_holdout") -> Path:
    canonical = root / case / "replicate_01" / "canonical"
    candidate_id = "production_facing_scientific_candidate_portfolio:p1"
    atomic_report_id = "atomic_cross_lane_scientific_synthesis:r1"
    atomic_portfolio_id = "hypothesis_portfolio:h1"
    hypothesis_ids = ["hypothesis:a", "hypothesis:b"]

    _write(
        canonical / "scientific_pre_n10_candidate_portfolio.json",
        {
            "schema_version": "production-facing-scientific-candidate-portfolio-v1",
            "portfolio_id": candidate_id,
        },
    )
    _write(
        canonical / "scientific_atomic_cross_lane_v2.report.json",
        {
            "schema_version": "atomic-cross-lane-scientific-synthesis-report-v1",
            "report_id": atomic_report_id,
            "source_candidate_portfolio_id": candidate_id,
            "hypotheses": [
                {"hypothesis_id": value}
                for value in hypothesis_ids
            ],
        },
    )
    _write(
        canonical / "scientific_atomic_cross_lane_v2.portfolio.json",
        {
            "schema_version": "hypothesis-portfolio-v1",
            "portfolio_id": atomic_portfolio_id,
            "domain_profile_id": "sers_au_ag",
            "hypotheses": [
                {"hypothesis_id": value}
                for value in hypothesis_ids
            ],
        },
    )
    _write(
        canonical / "scientific_atomic_grounded_identity_v2.annotation.json",
        {
            "schema_version": "grounded-identity-constituent-annotation-report-v1",
            "source_candidate_portfolio_id": candidate_id,
            "source_atomic_synthesis_report_id": atomic_report_id,
            "annotations": [],
        },
    )
    _write(
        canonical / "scientific_atomic_n10_external.report.json",
        {
            "schema_version": "external-novelty-report-v1",
            "source_portfolio_id": atomic_portfolio_id,
            "cards": [
                {"hypothesis_id": value}
                for value in hypothesis_ids
            ],
        },
    )
    return canonical


def test_ready_case_requires_exact_lineage_and_external_hypothesis_set(tmp_path: Path):
    _ready_case(tmp_path)

    report = build_unseen_validation_preflight(
        benchmark_root=tmp_path,
        excluded_case_keys=set(),
    )

    assert report.ready_task_count == 1
    row = report.tasks[0]
    assert row.status == "READY"
    assert row.ready_for_full_shadow_validation is True
    assert row.atomic_hypothesis_count == 2
    assert row.source_candidate_lineage_match is True
    assert row.atomic_hypothesis_lineage_match is True
    assert row.grounded_annotation_lineage_match is True
    assert row.matching_external_novelty_report_count == 1
    assert row.selected_external_novelty_report_path.endswith(
        "scientific_atomic_n10_external.report.json"
    )


def test_caller_declared_calibration_case_is_excluded_without_inspection(tmp_path: Path):
    _ready_case(tmp_path, case="G04_ordered_stochastic_transferability")

    report = build_unseen_validation_preflight(
        benchmark_root=tmp_path,
        excluded_case_keys={"G04_ordered_stochastic_transferability"},
    )

    assert report.excluded_task_count == 1
    row = report.tasks[0]
    assert row.status == "EXCLUDED_BY_CALLER"
    assert row.caller_declared_excluded is True
    assert row.ready_for_full_shadow_validation is False


def test_wrong_external_hypothesis_set_does_not_count_as_match(tmp_path: Path):
    canonical = _ready_case(tmp_path)
    external = canonical / "scientific_atomic_n10_external.report.json"
    payload = json.loads(external.read_text(encoding="utf-8"))
    payload["cards"] = [{"hypothesis_id": "hypothesis:a"}]
    _write(external, payload)

    report = build_unseen_validation_preflight(
        benchmark_root=tmp_path,
        excluded_case_keys=set(),
    )

    row = report.tasks[0]
    assert row.status == "EXTERNAL_NOVELTY_NOT_RESOLVED"
    assert row.matching_external_novelty_report_count == 0
    assert row.selected_external_novelty_report_path is None


def test_external_source_portfolio_mismatch_fails_closed(tmp_path: Path):
    canonical = _ready_case(tmp_path)
    external = canonical / "scientific_atomic_n10_external.report.json"
    payload = json.loads(external.read_text(encoding="utf-8"))
    payload["source_portfolio_id"] = "hypothesis_portfolio:wrong"
    _write(external, payload)

    report = build_unseen_validation_preflight(
        benchmark_root=tmp_path,
        excluded_case_keys=set(),
    )

    assert report.tasks[0].status == "EXTERNAL_NOVELTY_NOT_RESOLVED"


def test_preferred_canonical_external_report_resolves_multiple_full_matches(tmp_path: Path):
    canonical = _ready_case(tmp_path)
    source = json.loads(
        (canonical / "scientific_atomic_n10_external.report.json").read_text(
            encoding="utf-8"
        )
    )
    _write(canonical / "nested" / "another.report.json", source)

    report = build_unseen_validation_preflight(
        benchmark_root=tmp_path,
        excluded_case_keys=set(),
    )

    row = report.tasks[0]
    assert row.status == "READY"
    assert row.matching_external_novelty_report_count == 2
    assert row.selected_external_novelty_report_path.endswith(
        "scientific_atomic_n10_external.report.json"
    )
    assert (
        "preferred_canonical_external_report_selected_among_multiple_full_matches"
        in row.diagnostic_notes
    )


def test_lineage_mismatch_blocks_ready_state(tmp_path: Path):
    canonical = _ready_case(tmp_path)
    annotation = canonical / "scientific_atomic_grounded_identity_v2.annotation.json"
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["source_atomic_synthesis_report_id"] = "wrong-report"
    _write(annotation, payload)

    report = build_unseen_validation_preflight(
        benchmark_root=tmp_path,
        excluded_case_keys=set(),
    )

    row = report.tasks[0]
    assert row.status == "LINEAGE_MISMATCH"
    assert "grounded_identity_annotation_lineage_mismatch" in row.lineage_problem_codes


def test_missing_shadow_inputs_are_reported_without_materialization(tmp_path: Path):
    canonical = tmp_path / "K01_gap_size_hotspot" / "replicate_01" / "canonical"
    canonical.mkdir(parents=True)

    report = build_unseen_validation_preflight(
        benchmark_root=tmp_path,
        excluded_case_keys=set(),
    )

    row = report.tasks[0]
    assert row.status == "INCOMPLETE_INPUTS"
    assert set(row.missing_required_files) == {
        "source_candidate_portfolio",
        "atomic_report",
        "atomic_hypothesis_portfolio",
        "grounded_identity_annotation",
    }
    assert report.llm_calls_performed == 0
    assert report.retrieval_performed is False
