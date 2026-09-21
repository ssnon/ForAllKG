from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.reframing.benchmark_audit import (
    BenchmarkTaskReframeAudit,
    summarize_benchmark_task_audits,
)
from pipeline_core.discovery.reframing.cross_domain_validation import (
    CrossDomainTaskSelection,
    ScientificReframeCrossDomainValidationReport,
    build_cross_domain_validation_report,
    select_cross_domain_tasks,
)
from pipeline_core.discovery.reframing.generalization_probe import (
    CrossDomainCandidateCohort,
    ScientificReframeGeneralizationPreflight,
    ValidationTaskInventoryRow,
)


def _preflight(*, unchanged: bool = True, overlap: bool = False, domain: str = "dac_her"):
    task = ValidationTaskInventoryRow(
        canonical_dir="/eval/D01/canonical",
        inferred_benchmark_root="/eval",
        task_key="D01",
        case_key="D01",
        replicate_key="default",
        artifact_presence={
            "explorer.packet.json": True,
            "explorer.report.json": True,
            "hypothesis.context.json": True,
        },
        complete_triplet=True,
        task_id="task:d01",
        inferred_domain_labels=[domain],
        canonical_domain_labels=[domain],
        frozen_task_id_overlap=overlap,
        frozen_case_key_overlap=False,
        any_frozen_overlap=overlap,
    )
    cohorts = [] if overlap else [
        CrossDomainCandidateCohort(
            canonical_domain_label=domain,
            task_count=1,
            case_count=1,
            task_keys=["D01"],
            canonical_dirs=["/eval/D01/canonical"],
            inferred_benchmark_roots=["/eval"],
            raw_domain_labels=[domain],
        )
    ]
    return ScientificReframeGeneralizationPreflight(
        preflight_id="preflight:test",
        source_freeze_id="freeze:test",
        calibration_domain_label="sers",
        search_root="/eval",
        frozen_semantics_fingerprint="a" * 64,
        current_semantics_fingerprint=("a" * 64 if unchanged else "b" * 64),
        semantics_unchanged=unchanged,
        source_domain_markers=["sers"],
        source_domain_marker_hits=[],
        source_domain_marker_hit_count=0,
        marker_hit_counts={},
        discovered_task_materialization_count=1,
        complete_triplet_count=1,
        incomplete_materialization_count=0,
        task_rows=[task],
        benchmark_inventories=[],
        cross_domain_candidate_roots=[],
        cross_domain_candidate_cohorts=cohorts,
        unclassified_candidate_roots=[],
        calibration_overlap_detected=overlap,
    )


def _audit(task_key: str = "D01", *, status: str = "complete"):
    if status == "complete":
        row = BenchmarkTaskReframeAudit(
            task_key=task_key,
            case_key="D01",
            replicate_key="default",
            canonical_dir="/eval/D01/canonical",
            status="complete",
            task_id="task:d01",
            trigger_pattern="latent_only",
            trigger_signal_by_operator={"LATENT_VARIABLE": True, "REGIME_BOUNDARY": False},
            readiness_status_by_operator={"LATENT_VARIABLE": "ready_now"},
            readiness_trigger_alignment_by_operator={"LATENT_VARIABLE": "ready_and_triggered"},
            tension_type_counts={"MECHANISTIC": 1},
            direct_trigger_signal_kind_counts={"MEDIATION_GAP": 1},
        )
    else:
        row = BenchmarkTaskReframeAudit(
            task_key=task_key,
            case_key="D01",
            replicate_key="default",
            canonical_dir="/eval/D01/canonical",
            status="error",
            task_id="task:d01",
            error_type="ValueError",
            error_message="boom",
            trigger_pattern="unavailable",
        )
    return summarize_benchmark_task_audits(
        benchmark_root="cross-domain:dac_her",
        extraction_roots=["/data_dac"],
        task_audits=[row],
    )


def test_selects_uncontaminated_cross_domain_task():
    selected = select_cross_domain_tasks(_preflight(), validation_domain_label="dac_her")
    assert len(selected) == 1
    assert selected[0].task_key == "D01"
    assert selected[0].canonical_domain_label == "dac_her"


def test_rejects_source_domain_as_cross_domain():
    with pytest.raises(ValueError, match="differ"):
        select_cross_domain_tasks(_preflight(), validation_domain_label="sers")


def test_rejects_semantic_drift():
    with pytest.raises(ValueError, match="semantics changed"):
        select_cross_domain_tasks(_preflight(unchanged=False), validation_domain_label="dac_her")


def test_rejects_missing_cross_domain_cohort():
    with pytest.raises(ValueError, match="no cross-domain candidate cohort"):
        select_cross_domain_tasks(_preflight(), validation_domain_label="catalysis_mechanism")


def test_builds_single_task_smoke_report_without_accuracy_claim():
    preflight = _preflight()
    selected = select_cross_domain_tasks(preflight, validation_domain_label="dac_her")
    report = build_cross_domain_validation_report(
        preflight=preflight,
        validation_domain_label="dac_her",
        extraction_roots=["/data_dac"],
        selected_tasks=selected,
        audit=_audit(),
    )
    assert report.validation_scale == "single_task_smoke_test"
    assert report.triggered_task_count == 1
    assert report.cross_domain_accuracy_evaluated is False
    assert report.generalization_claim_established is False
    assert report.llm_calls_performed == 0


def test_report_carries_trigger_and_direct_signal_summaries():
    preflight = _preflight()
    selected = select_cross_domain_tasks(preflight, validation_domain_label="dac_her")
    report = build_cross_domain_validation_report(
        preflight=preflight,
        validation_domain_label="dac_her",
        extraction_roots=["/data_dac"],
        selected_tasks=selected,
        audit=_audit(),
    )
    assert report.trigger_pattern_counts == {"latent_only": 1}
    assert report.tension_type_task_counts == {"MECHANISTIC": 1}
    assert report.direct_trigger_signal_task_counts == {"MEDIATION_GAP": 1}


def test_report_rejects_audit_identity_mismatch():
    preflight = _preflight()
    selected = select_cross_domain_tasks(preflight, validation_domain_label="dac_her")
    with pytest.raises(ValueError, match="identities"):
        build_cross_domain_validation_report(
            preflight=preflight,
            validation_domain_label="dac_her",
            extraction_roots=["/data_dac"],
            selected_tasks=selected,
            audit=_audit(task_key="OTHER"),
        )


def test_report_schema_rejects_false_semantics_unchanged():
    preflight = _preflight()
    selected = select_cross_domain_tasks(preflight, validation_domain_label="dac_her")
    report = build_cross_domain_validation_report(
        preflight=preflight,
        validation_domain_label="dac_her",
        extraction_roots=["/data_dac"],
        selected_tasks=selected,
        audit=_audit(),
    )
    payload = report.model_dump(mode="json")
    payload["semantics_unchanged"] = False
    with pytest.raises(ValidationError):
        ScientificReframeCrossDomainValidationReport.model_validate(payload)
