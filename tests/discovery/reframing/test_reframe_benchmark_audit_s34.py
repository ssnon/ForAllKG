from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.reframing.benchmark_audit import (
    BenchmarkTaskReframeAudit,
    ScientificReframeBenchmarkAuditReport,
    operator_ready_without_mandatory_enrichment,
    readiness_trigger_alignment,
    summarize_benchmark_task_audits,
    trigger_pattern_from_signals,
)


def _complete(
    task_key: str,
    *,
    pattern: str = "latent_only",
    tension_counts: dict[str, int] | None = None,
    measurement: bool = False,
    proxy: bool = False,
) -> BenchmarkTaskReframeAudit:
    latent_signal = pattern in {"latent_only", "latent_and_regime"}
    regime_signal = pattern in {"regime_only", "latent_and_regime"}
    return BenchmarkTaskReframeAudit(
        task_key=task_key,
        case_key=task_key.split("/")[0],
        replicate_key=task_key.split("/")[-1],
        canonical_dir=f"/bench/{task_key}/canonical",
        status="complete",
        task_id=f"task:{task_key}",
        selected_premise_count=3,
        selected_gap_count=1,
        grounded_source_chunk_count=5,
        grounded_semantic_record_count=20,
        tension_witness_count=sum((tension_counts or {}).values()),
        tension_type_counts=tension_counts or {},
        condition_example_count=4,
        condition_signature_count=3,
        condition_paper_count=2,
        readiness_status_by_operator={
            "LATENT_VARIABLE": "ready_now",
            "REGIME_BOUNDARY": "ready_with_optional_enrichment",
            "PROXY_CHALLENGE": "targeted_enrichment_required",
        },
        mandatory_backfill_count_by_operator={
            "LATENT_VARIABLE": 0,
            "REGIME_BOUNDARY": 0,
            "PROXY_CHALLENGE": 3,
        },
        optional_backfill_count_by_operator={
            "LATENT_VARIABLE": 0,
            "REGIME_BOUNDARY": 2,
            "PROXY_CHALLENGE": 0,
        },
        trigger_decision_by_operator={
            "LATENT_VARIABLE": "triggered" if latent_signal else "not_triggered",
            "REGIME_BOUNDARY": "triggered" if regime_signal else "not_triggered",
        },
        trigger_signal_by_operator={
            "LATENT_VARIABLE": latent_signal,
            "REGIME_BOUNDARY": regime_signal,
        },
        readiness_trigger_alignment_by_operator={
            "LATENT_VARIABLE": readiness_trigger_alignment(
                readiness_status="ready_now", trigger_signal=latent_signal
            ),
            "REGIME_BOUNDARY": readiness_trigger_alignment(
                readiness_status="ready_with_optional_enrichment",
                trigger_signal=regime_signal,
            ),
            "PROXY_CHALLENGE": "no_trigger_contract",
        },
        trigger_pattern=pattern,
        measurement_tension_present=measurement,
        proxy_enrichment_candidate=proxy,
        proxy_candidate_reason="diagnostic" if proxy else None,
    )


def _error(task_key: str) -> BenchmarkTaskReframeAudit:
    return BenchmarkTaskReframeAudit(
        task_key=task_key,
        case_key=task_key.split("/")[0],
        replicate_key=task_key.split("/")[-1],
        canonical_dir=f"/bench/{task_key}/canonical",
        status="error",
        error_type="FileNotFoundError",
        error_message="missing context",
        trigger_pattern="unavailable",
    )


def test_trigger_pattern_covers_all_boolean_combinations() -> None:
    assert trigger_pattern_from_signals(latent=False, regime=False) == "none"
    assert trigger_pattern_from_signals(latent=True, regime=False) == "latent_only"
    assert trigger_pattern_from_signals(latent=False, regime=True) == "regime_only"
    assert trigger_pattern_from_signals(latent=True, regime=True) == "latent_and_regime"
    assert trigger_pattern_from_signals(latent=None, regime=True) == "unavailable"


def test_readiness_trigger_alignment_preserves_the_two_axes() -> None:
    assert readiness_trigger_alignment(
        readiness_status="ready_now", trigger_signal=True
    ) == "ready_and_triggered"
    assert readiness_trigger_alignment(
        readiness_status="ready_with_optional_enrichment", trigger_signal=False
    ) == "ready_not_triggered"
    assert readiness_trigger_alignment(
        readiness_status="targeted_enrichment_required", trigger_signal=True
    ) == "triggered_not_ready"
    assert readiness_trigger_alignment(
        readiness_status="blocked_insufficient_substrate", trigger_signal=False
    ) == "not_ready_not_triggered"
    assert readiness_trigger_alignment(
        readiness_status="targeted_enrichment_required",
        trigger_signal=None,
        has_trigger_contract=False,
    ) == "no_trigger_contract"


def test_ready_without_mandatory_enrichment_is_fail_closed() -> None:
    assert operator_ready_without_mandatory_enrichment("ready_now")
    assert operator_ready_without_mandatory_enrichment(
        "ready_with_optional_enrichment"
    )
    assert not operator_ready_without_mandatory_enrichment(
        "targeted_enrichment_required"
    )
    assert not operator_ready_without_mandatory_enrichment(
        "blocked_insufficient_substrate"
    )


def test_summary_counts_patterns_taxonomy_and_errors() -> None:
    rows = [
        _complete(
            "K01/replicate_01",
            pattern="latent_only",
            tension_counts={"CONTEXT": 2, "MECHANISTIC": 1},
        ),
        _complete(
            "K02/replicate_01",
            pattern="none",
            tension_counts={"MEASUREMENT": 1},
            measurement=True,
            proxy=True,
        ),
        _error("K03/replicate_01"),
    ]
    report = summarize_benchmark_task_audits(
        benchmark_root="/bench",
        extraction_roots=["/corpus/a", "/corpus/b"],
        task_audits=rows,
    )
    assert report.discovered_task_count == 3
    assert report.complete_task_count == 2
    assert report.error_task_count == 1
    assert report.unique_case_count == 3
    assert report.trigger_pattern_counts == {"latent_only": 1, "none": 1}
    assert report.tension_type_task_counts == {
        "CONTEXT": 1,
        "MEASUREMENT": 1,
        "MECHANISTIC": 1,
    }
    assert report.tension_type_witness_counts["CONTEXT"] == 2
    assert report.measurement_tension_task_count == 1
    assert report.proxy_enrichment_candidate_task_count == 1


def test_summary_excludes_error_rows_from_scientific_aggregates() -> None:
    report = summarize_benchmark_task_audits(
        benchmark_root="/bench",
        extraction_roots=["/corpus"],
        task_audits=[_error("K01/replicate_01")],
    )
    assert report.trigger_pattern_counts == {}
    assert report.tension_type_task_counts == {}
    assert report.total_grounded_source_chunk_count == 0


def test_audit_id_is_deterministic_under_input_order() -> None:
    a = _complete("K01/replicate_01")
    b = _complete("K02/replicate_01", pattern="none")
    first = summarize_benchmark_task_audits(
        benchmark_root="/bench",
        extraction_roots=["/corpus"],
        task_audits=[a, b],
    )
    second = summarize_benchmark_task_audits(
        benchmark_root="/bench",
        extraction_roots=["/corpus"],
        task_audits=[b, a],
    )
    assert first.audit_id == second.audit_id
    assert [row.task_key for row in first.task_audits] == [
        "K01/replicate_01",
        "K02/replicate_01",
    ]


def test_error_task_requires_unavailable_trigger_pattern() -> None:
    with pytest.raises(ValidationError):
        BenchmarkTaskReframeAudit(
            task_key="K01/replicate_01",
            case_key="K01",
            replicate_key="replicate_01",
            canonical_dir="/bench/K01/replicate_01/canonical",
            status="error",
            error_type="ValueError",
            trigger_pattern="none",
        )


def test_report_rejects_duplicate_task_keys_and_has_no_ranking_authority() -> None:
    row = _complete("K01/replicate_01")
    base = summarize_benchmark_task_audits(
        benchmark_root="/bench",
        extraction_roots=["/corpus"],
        task_audits=[row],
    )
    assert base.llm_calls_performed == 0
    assert base.candidate_ranking_performed is False
    assert base.overall_score_computed is False
    assert base.production_selection_changed is False
    with pytest.raises(ValidationError):
        ScientificReframeBenchmarkAuditReport(
            **base.model_dump(exclude={"task_audits", "discovered_task_count"}),
            task_audits=[row, row],
            discovered_task_count=2,
        )
