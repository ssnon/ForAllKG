from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.reframing.benchmark_coverage import (
    BenchmarkTaskMaterialization,
    inspect_benchmark_coverage,
)


def _materialize(
    root: Path,
    task_key: str,
    *,
    packet: bool = True,
    report: bool = True,
    context: bool = True,
) -> Path:
    canonical = root / task_key / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    if packet:
        (canonical / "explorer.packet.json").write_text("{}", encoding="utf-8")
    if report:
        (canonical / "explorer.report.json").write_text("{}", encoding="utf-8")
    if context:
        (canonical / "hypothesis.context.json").write_text("{}", encoding="utf-8")
    return canonical


def test_name_number_is_diagnostic_not_expected_count(tmp_path: Path) -> None:
    root = tmp_path / "s28_sers18_novelty_benchmark_v1"
    _materialize(root, "K01_case/replicate_01")

    result = inspect_benchmark_coverage(benchmark_root=root)

    assert 18 in result.benchmark_name_numeric_tokens
    assert result.benchmark_name_numbers_are_expectations is False
    assert result.expected_task_count is None
    assert result.expected_task_count_source == "none"
    assert result.coverage_status == "no_explicit_expectation"
    assert result.task_count_shortfall is None


def test_discovers_partial_canonical_not_visible_to_0014(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    _materialize(root, "K01_case/replicate_01")
    _materialize(
        root,
        "K02_partial/replicate_01",
        packet=False,
        report=True,
        context=True,
    )

    result = inspect_benchmark_coverage(benchmark_root=root)

    assert result.audit_discoverable_task_count == 1
    assert result.fully_auditable_task_count == 1
    assert result.incomplete_materialization_count == 1
    assert result.partially_materialized_canonical_count == 1
    assert result.missing_canonical_count == 0
    partial = next(row for row in result.task_materializations if row.case_key == "K02_partial")
    assert partial.audit_discoverable is False
    assert partial.missing_required_artifacts == ["explorer.packet.json"]


def test_replicate_without_canonical_is_reported_as_missing_materialization(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bench"
    (root / "K03_missing" / "replicate_01").mkdir(parents=True)

    result = inspect_benchmark_coverage(benchmark_root=root)

    assert len(result.task_materializations) == 1
    assert result.incomplete_materialization_count == 1
    assert result.missing_canonical_count == 1
    row = result.task_materializations[0]
    assert row.canonical_dir_exists is False
    assert row.audit_discoverable is False
    assert row.fully_auditable is False
    assert row.missing_required_artifacts == [
        "explorer.packet.json",
        "explorer.report.json",
        "hypothesis.context.json",
    ]


def test_prior_audit_is_reconciled_with_filesystem(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    _materialize(root, "K01_case/replicate_01")
    _materialize(root, "K02_case/replicate_01")
    prior = root / "scientific_reframing_trigger_audit.json"
    prior.write_text(
        json.dumps(
            {
                "task_audits": [
                    {"task_key": "K01_case/replicate_01"},
                    {"task_key": "OLD_case/replicate_01"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = inspect_benchmark_coverage(benchmark_root=root)

    assert result.prior_audit_present is True
    assert result.prior_audited_task_count == 2
    assert result.filesystem_discoverable_missing_from_prior_audit == [
        "K02_case/replicate_01"
    ]
    assert result.prior_audit_tasks_missing_from_filesystem == [
        "OLD_case/replicate_01"
    ]


def test_explicit_expected_count_can_report_shortfall(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    _materialize(root, "K01_case/replicate_01")
    _materialize(root, "K02_case/replicate_01")

    result = inspect_benchmark_coverage(
        benchmark_root=root,
        expected_task_count=3,
    )

    assert result.expected_task_count_source == "cli"
    assert result.task_count_shortfall == 1
    assert result.coverage_status == "shortfall_against_explicit_expectation"


def test_explicit_expected_count_can_be_complete(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    _materialize(root, "K01_case/replicate_01")
    _materialize(root, "K02_case/replicate_01")

    result = inspect_benchmark_coverage(
        benchmark_root=root,
        expected_task_count=2,
    )

    assert result.task_count_shortfall == 0
    assert result.coverage_status == "complete_against_explicit_expectation"


def test_metadata_candidates_are_listed_but_not_interpreted(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    _materialize(root, "K01_case/replicate_01")
    (root / "benchmark_manifest.json").write_text(
        json.dumps({"expected": 99}), encoding="utf-8"
    )

    result = inspect_benchmark_coverage(benchmark_root=root)

    assert result.metadata_json_candidates == ["benchmark_manifest.json"]
    assert result.expected_task_count is None


def test_materialization_rejects_inconsistent_missing_list() -> None:
    with pytest.raises(ValidationError):
        BenchmarkTaskMaterialization(
            task_key="K01/replicate_01",
            case_key="K01",
            replicate_key="replicate_01",
            canonical_dir="/bench/K01/replicate_01/canonical",
            canonical_dir_exists=True,
            explorer_packet_present=False,
            explorer_report_present=True,
            hypothesis_context_present=True,
            audit_discoverable=False,
            fully_auditable=False,
            missing_required_artifacts=[],
        )
