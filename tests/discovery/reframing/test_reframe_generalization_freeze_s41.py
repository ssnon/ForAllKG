from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.reframing.benchmark_audit import (
    BenchmarkTaskReframeAudit,
    ScientificReframeBenchmarkAuditReport,
    summarize_benchmark_task_audits,
)
from pipeline_core.discovery.reframing.benchmark_synthesis import (
    ScientificReframeBenchmarkSynthesis,
)
from pipeline_core.discovery.reframing.generalization_validation import (
    SEMANTICS_FINGERPRINT_PATHS,
    ScientificReframeCalibrationFreeze,
    audit_validation_cohort,
    build_calibration_freeze,
    semantics_fingerprint,
)


def _repo(root: Path) -> Path:
    for relative in SEMANTICS_FINGERPRINT_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\nVALUE = 1\n", encoding="utf-8")
    return root


def _audit(root: Path, *, prefix: str, task_count: int = 2) -> ScientificReframeBenchmarkAuditReport:
    rows = [
        BenchmarkTaskReframeAudit(
            task_key=f"{prefix}{i:02d}/replicate_01",
            case_key=f"{prefix}{i:02d}",
            replicate_key="replicate_01",
            canonical_dir=str(root / f"{prefix}{i:02d}" / "replicate_01" / "canonical"),
            status="complete",
            task_id=f"task:{prefix}{i:02d}",
            trigger_pattern="none",
        )
        for i in range(1, task_count + 1)
    ]
    return summarize_benchmark_task_audits(
        benchmark_root=str(root),
        extraction_roots=["/corpus"],
        task_audits=rows,
    )


def _synthesis(audit: ScientificReframeBenchmarkAuditReport) -> ScientificReframeBenchmarkSynthesis:
    return ScientificReframeBenchmarkSynthesis(
        synthesis_id="synthesis:cal",
        source_execution_id="execution:cal",
        source_benchmark_audit_id=audit.audit_id,
        benchmark_root=audit.benchmark_root,
        triggered_task_count=0,
        completed_task_count=0,
        candidate_count=0,
        operator_summaries=[],
        slot_counts={},
        candidates=[],
        lexical_overlap_diagnostics=[],
        lexical_overlap_threshold=0.35,
    )


def _freeze(tmp_path: Path) -> tuple[ScientificReframeCalibrationFreeze, Path]:
    repo = _repo(tmp_path / "repo")
    audit = _audit(tmp_path / "calibration", prefix="C")
    freeze = build_calibration_freeze(
        audit=audit,
        synthesis=_synthesis(audit),
        calibration_domain_label="sers",
        repository_root=repo,
    )
    return freeze, repo


def test_freeze_records_complete_calibration_identity_and_semantics(tmp_path: Path) -> None:
    freeze, _ = _freeze(tmp_path)
    assert freeze.calibration_domain_label == "sers"
    assert freeze.calibration_task_count == 2
    assert freeze.case_keys == ["C01", "C02"]
    assert freeze.task_ids == ["task:C01", "task:C02"]
    assert len(freeze.semantics_fingerprint) == 64
    assert set(freeze.semantics_file_sha256) == set(SEMANTICS_FINGERPRINT_PATHS)
    assert freeze.validation_sets_must_not_inform_tuning_before_evaluation is True
    assert freeze.model_configuration_fingerprinted is False
    assert freeze.direct_generation_quality_comparison_requires_same_model_configuration is True


def test_freeze_rejects_synthesis_from_different_audit(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    audit = _audit(tmp_path / "calibration", prefix="C")
    synthesis = _synthesis(audit).model_copy(update={"source_benchmark_audit_id": "audit:other"})
    with pytest.raises(ValueError, match="descend"):
        build_calibration_freeze(
            audit=audit,
            synthesis=synthesis,
            calibration_domain_label="sers",
            repository_root=repo,
        )


def test_semantics_fingerprint_is_deterministic(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    first, first_files = semantics_fingerprint(repo)
    second, second_files = semantics_fingerprint(repo)
    assert first == second
    assert first_files == second_files


def test_same_domain_holdout_without_overlap_is_comparable(tmp_path: Path) -> None:
    freeze, repo = _freeze(tmp_path)
    validation = _audit(tmp_path / "holdout", prefix="H")
    result = audit_validation_cohort(
        freeze=freeze,
        validation_audit=validation,
        cohort_role="holdout_same_domain",
        validation_domain_label="sers",
        repository_root=repo,
    )
    assert result.comparability_status == "comparable_clean"
    assert result.validation_evaluable_without_contamination is True
    assert result.overlap_detected is False


def test_overlap_is_reported_as_contamination(tmp_path: Path) -> None:
    freeze, repo = _freeze(tmp_path)
    validation = _audit(tmp_path / "holdout", prefix="H", task_count=1)
    row = validation.task_audits[0].model_copy(
        update={"case_key": "C01", "task_id": "task:C01"}
    )
    validation = summarize_benchmark_task_audits(
        benchmark_root=validation.benchmark_root,
        extraction_roots=validation.extraction_roots,
        task_audits=[row],
    )
    result = audit_validation_cohort(
        freeze=freeze,
        validation_audit=validation,
        cohort_role="holdout_same_domain",
        validation_domain_label="sers",
        repository_root=repo,
    )
    assert result.comparability_status == "contaminated_overlap"
    assert result.validation_evaluable_without_contamination is False
    assert result.overlapping_case_keys == ["C01"]
    assert result.calibration_data_reused_for_validation is True


def test_cross_domain_role_requires_different_domain_label(tmp_path: Path) -> None:
    freeze, repo = _freeze(tmp_path)
    validation = _audit(tmp_path / "other", prefix="X")
    result = audit_validation_cohort(
        freeze=freeze,
        validation_audit=validation,
        cohort_role="validation_cross_domain",
        validation_domain_label="sers",
        repository_root=repo,
    )
    assert result.comparability_status == "role_domain_mismatch"
    assert result.validation_evaluable_without_contamination is False


def test_semantic_drift_invalidates_direct_comparison(tmp_path: Path) -> None:
    freeze, repo = _freeze(tmp_path)
    validation = _audit(tmp_path / "holdout", prefix="H")
    changed = repo / SEMANTICS_FINGERPRINT_PATHS[0]
    changed.write_text(changed.read_text(encoding="utf-8") + "VALUE = 2\n", encoding="utf-8")
    result = audit_validation_cohort(
        freeze=freeze,
        validation_audit=validation,
        cohort_role="holdout_same_domain",
        validation_domain_label="sers",
        repository_root=repo,
    )
    assert result.comparability_status == "semantic_drift"
    assert result.semantics_unchanged is False
    assert result.validation_evaluable_without_contamination is False


def test_freeze_model_rejects_duplicate_calibration_tasks() -> None:
    with pytest.raises(ValidationError):
        ScientificReframeCalibrationFreeze(
            freeze_id="freeze:x",
            calibration_domain_label="sers",
            calibration_benchmark_root="/bench",
            source_audit_id="audit:x",
            source_synthesis_id="synthesis:x",
            calibration_task_count=2,
            task_keys=["K01", "K01"],
            case_keys=["K01"],
            task_ids=["task:1"],
            semantics_fingerprint="a" * 64,
            semantics_file_sha256={"x.py": "b" * 64},
        )
