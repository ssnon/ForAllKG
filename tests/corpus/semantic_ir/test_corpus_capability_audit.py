from __future__ import annotations

import json

import pytest

from pipeline_core.corpus.semantic_ir.capability import (
    CapabilityManifest,
    CapabilityRecord,
)
from pipeline_core.corpus.semantic_ir.capability_audit import (
    CapabilityAuditMetric,
    SemanticIRCapabilityAuditResult,
)
from pipeline_core.corpus.semantic_ir.corpus_audit import (
    ExtractionAttemptDiscoveryResult,
    DiscoveredExtractionAttempt,
    aggregate_semantic_ir_capability_audits,
    discover_extraction_attempts,
)


def _metric(
    capability: str,
    *,
    state: str,
    mode: str,
    applicable: int,
    supported: int,
) -> CapabilityAuditMetric:
    return CapabilityAuditMetric(
        capability=capability,
        state=state,
        assessment_mode=mode,
        applicable_count=applicable,
        supported_count=supported,
        coverage_fraction=(
            supported / applicable if applicable else None
        ),
        assessed_from=[capability],
    )


def _audit(
    paper_id: str,
    measurement_metric: CapabilityAuditMetric,
) -> SemanticIRCapabilityAuditResult:
    gap = _metric(
        "proxy_semantics",
        state="absent",
        mode="explicit_gap",
        applicable=1,
        supported=0,
    )
    metrics = {
        "measurement_condition_coverage": measurement_metric,
        "proxy_semantics": gap,
    }
    manifest = CapabilityManifest(
        manifest_id=f"manifest:{paper_id}",
        scope_kind="paper",
        scope_id=paper_id,
        capabilities={
            name: CapabilityRecord(
                name=name,
                state=metric.state,
                assessed_from=metric.assessed_from,
            )
            for name, metric in metrics.items()
        },
    )
    return SemanticIRCapabilityAuditResult(
        bundle_id=f"bundle:{paper_id}",
        paper_id=paper_id,
        manifest=manifest,
        metrics=metrics,
        object_counts={"measurement": measurement_metric.applicable_count},
    )


def _discovery(*paper_ids: str) -> ExtractionAttemptDiscoveryResult:
    attempts = [
        DiscoveredExtractionAttempt(
            paper_id=paper_id,
            active_chunks_path=f"/tmp/{paper_id}/active_chunks.json",
            attempt_directory=f"/tmp/{paper_id}",
        )
        for paper_id in paper_ids
    ]
    return ExtractionAttemptDiscoveryResult(
        root="/tmp",
        duplicate_paper_policy="error",
        discovered_active_chunks_count=len(attempts),
        selected_attempt_count=len(attempts),
        paper_count=len(attempts),
        duplicate_paper_count=0,
        attempts=attempts,
    )


def test_corpus_aggregate_weights_semantic_objects_not_paper_means():
    audits = [
        _audit(
            "P1",
            _metric(
                "measurement_condition_coverage",
                state="complete",
                mode="observed_coverage",
                applicable=1,
                supported=1,
            ),
        ),
        _audit(
            "P2",
            _metric(
                "measurement_condition_coverage",
                state="partial",
                mode="observed_coverage",
                applicable=2,
                supported=1,
            ),
        ),
    ]

    result = aggregate_semantic_ir_capability_audits(
        corpus_id="sers-test",
        root="/tmp",
        discovery=_discovery("P1", "P2"),
        audits=audits,
    )

    metric = result.metrics["measurement_condition_coverage"]
    assert metric.state == "partial"
    assert metric.supported_count == 2
    assert metric.applicable_count == 3
    assert metric.coverage_fraction == pytest.approx(2 / 3)
    assert metric.paper_state_counts == {"complete": 1, "partial": 1}


def test_explicit_gap_remains_schema_gap_at_corpus_level():
    audits = [
        _audit(
            "P1",
            _metric(
                "measurement_condition_coverage",
                state="unknown",
                mode="observed_coverage",
                applicable=0,
                supported=0,
            ),
        ),
        _audit(
            "P2",
            _metric(
                "measurement_condition_coverage",
                state="unknown",
                mode="observed_coverage",
                applicable=0,
                supported=0,
            ),
        ),
    ]

    result = aggregate_semantic_ir_capability_audits(
        corpus_id="sers-test",
        root="/tmp",
        discovery=_discovery("P1", "P2"),
        audits=audits,
    )

    gap = result.metrics["proxy_semantics"]
    assert gap.state == "absent"
    assert gap.assessment_mode == "explicit_gap"
    assert gap.paper_assessed_count == 2
    assert result.negative_evidence_inferred is False
    assert result.llm_calls_performed == 0


def _write_attempt(path, *, paper_id: str, attempt_id: str):
    path.mkdir(parents=True)
    (path / "active_chunks.json").write_text(
        json.dumps({
            "paper_id": paper_id,
            "run_id": "run",
            "attempt_id": attempt_id,
            "active_chunk_count": 0,
            "chunks": [],
        }),
        encoding="utf-8",
    )


def test_attempt_discovery_fails_closed_on_duplicate_papers(tmp_path):
    _write_attempt(tmp_path / "a", paper_id="P", attempt_id="20260101")
    _write_attempt(tmp_path / "b", paper_id="P", attempt_id="20260102")

    with pytest.raises(ValueError, match="multiple extraction attempts"):
        discover_extraction_attempts(tmp_path)


def test_attempt_discovery_can_select_latest_attempt(tmp_path):
    _write_attempt(tmp_path / "a", paper_id="P", attempt_id="20260101")
    _write_attempt(tmp_path / "b", paper_id="P", attempt_id="20260102")
    _write_attempt(tmp_path / "c", paper_id="Q", attempt_id="20260101")

    result = discover_extraction_attempts(
        tmp_path,
        duplicate_paper_policy="latest_attempt",
    )

    assert result.discovered_active_chunks_count == 3
    assert result.selected_attempt_count == 2
    assert result.paper_count == 2
    assert result.duplicate_paper_count == 1
    selected = {row.paper_id: row for row in result.attempts}
    assert selected["P"].attempt_id == "20260102"
