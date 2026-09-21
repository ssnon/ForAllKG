from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.semantic_ir.capability import (
    CapabilityManifest,
    CapabilityRecord,
    CapabilityState,
)
from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectRef,
)
from pipeline_core.corpus.semantic_ir.task_gap import (
    TaskLocalCapabilityGapReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


TaskCapabilityAssessmentMode = Literal[
    "structural_support",
    "task_local_observed_coverage",
    "explicit_gap",
]


class TaskCapabilityMetric(StrictModel):
    capability: str = Field(min_length=1)
    state: CapabilityState
    assessment_mode: TaskCapabilityAssessmentMode
    applicable_count: int = Field(ge=0)
    supported_count: int = Field(ge=0)
    coverage_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    detail: str = ""
    assessed_from: list[str] = Field(default_factory=list)

    negative_evidence_inferred: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "TaskCapabilityMetric":
        if self.supported_count > self.applicable_count:
            raise ValueError("supported_count cannot exceed applicable_count")
        if self.applicable_count == 0:
            if self.coverage_fraction is not None:
                raise ValueError(
                    "coverage_fraction must be null when applicable_count is zero"
                )
        else:
            expected = self.supported_count / self.applicable_count
            if self.coverage_fraction is None:
                raise ValueError(
                    "coverage_fraction is required when applicable_count > 0"
                )
            if abs(self.coverage_fraction - expected) > 1e-12:
                raise ValueError(
                    "coverage_fraction must equal supported/applicable"
                )
        return self


class TaskCapabilitySnapshot(StrictModel):
    schema_version: Literal[
        "task-semantic-capability-snapshot-v1"
    ] = "task-semantic-capability-snapshot-v1"

    scope_id: str = Field(min_length=1)
    manifest: CapabilityManifest
    metrics: dict[str, TaskCapabilityMetric] = Field(default_factory=dict)
    source_chunk_count: int = Field(ge=0)
    semantic_record_count: int = Field(ge=0)

    task_local_only: Literal[True] = True
    global_coverage_extrapolated: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False
    negative_evidence_inferred: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_alignment(self) -> "TaskCapabilitySnapshot":
        if self.manifest.scope_kind != "task":
            raise ValueError("task capability manifest must use task scope kind")
        if self.manifest.scope_id != self.scope_id:
            raise ValueError("manifest scope_id must equal task scope_id")
        if set(self.metrics) != set(self.manifest.capabilities):
            raise ValueError("metric and manifest capability keys must match")
        for name, metric in self.metrics.items():
            if metric.capability != name:
                raise ValueError("metric key must equal metric.capability")
            if self.manifest.capabilities[name].state != metric.state:
                raise ValueError("manifest state must equal metric state")
        return self


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _coverage_state(applicable: int, supported: int) -> CapabilityState:
    if applicable == 0:
        return "unknown"
    if supported == 0:
        return "absent"
    if supported == applicable:
        return "complete"
    return "partial"


def _coverage_metric(
    *,
    capability: str,
    applicable: int,
    supported: int,
    detail: str,
    assessed_from: list[str],
) -> TaskCapabilityMetric:
    return TaskCapabilityMetric(
        capability=capability,
        state=_coverage_state(applicable, supported),
        assessment_mode="task_local_observed_coverage",
        applicable_count=applicable,
        supported_count=supported,
        coverage_fraction=(supported / applicable if applicable else None),
        detail=detail,
        assessed_from=assessed_from,
    )


def _structural_metric(
    capability: str,
    detail: str,
) -> TaskCapabilityMetric:
    return TaskCapabilityMetric(
        capability=capability,
        state="complete",
        assessment_mode="structural_support",
        applicable_count=1,
        supported_count=1,
        coverage_fraction=1.0,
        detail=detail,
        assessed_from=["semantic-ir-v1 canonical payload schema"],
    )


def _explicit_gap_metric(
    capability: str,
    detail: str,
) -> TaskCapabilityMetric:
    return TaskCapabilityMetric(
        capability=capability,
        state="absent",
        assessment_mode="explicit_gap",
        applicable_count=1,
        supported_count=0,
        coverage_fraction=0.0,
        detail=detail,
        assessed_from=["semantic-ir-v1 canonical payload schema"],
    )


def _dedupe_chunks(refs: list[SemanticObjectRef]) -> list[SemanticObjectRef]:
    by_key = {}
    for ref in refs:
        if ref.object_kind != "chunk":
            raise ValueError("task source scope may contain only chunk refs")
        by_key.setdefault(ref.identity_key(), ref)
    return [by_key[key] for key in sorted(by_key)]


def _selected_records(
    *,
    bundles: dict[str, SemanticIRBundle],
    chunks: list[SemanticObjectRef],
) -> list[SemanticIRRecord]:
    chunk_keys = {ref.identity_key() for ref in chunks}
    rows: list[SemanticIRRecord] = []
    for bundle in bundles.values():
        for record in bundle.records:
            source = record.source_chunk_ref
            if source is not None and source.identity_key() in chunk_keys:
                rows.append(record)
    return rows


def _nonblank(payload: dict, key: str) -> bool:
    value = payload.get(key)
    return isinstance(value, str) and bool(value.strip())


def build_task_capability_snapshot(
    *,
    scope_id: str,
    bundles: dict[str, SemanticIRBundle],
    task_source_chunks: list[SemanticObjectRef],
    task_gap_report: TaskLocalCapabilityGapReport,
) -> TaskCapabilitySnapshot:
    """Audit only the already-grounded task source scope.

    The snapshot is data-availability metadata. It never turns missing extracted
    structure into a scientific absence claim and does not execute enrichment.
    """

    chunks = _dedupe_chunks(task_source_chunks)
    records = _selected_records(bundles=bundles, chunks=chunks)
    by_kind: dict[str, list[SemanticIRRecord]] = defaultdict(list)
    for record in records:
        by_kind[record.ref.object_kind].append(record)

    metrics: dict[str, TaskCapabilityMetric] = {}

    for capability, detail in (
        (
            "observation_claim_representation",
            "ObservationClaim remains a distinct semantic object kind.",
        ),
        (
            "mechanism_claim_representation",
            "MechanismClaim remains distinct from direct observations.",
        ),
        (
            "measurement_representation",
            "Measurement remains a distinct scalar-result object kind.",
        ),
        (
            "experiment_representation",
            "Experiment remains a distinct setup/method object kind.",
        ),
        (
            "calculation_representation",
            "Calculation remains a distinct computational object kind.",
        ),
        (
            "measurement_group_representation",
            "MeasurementGroup remains a distinct comparison container.",
        ),
    ):
        metrics[capability] = _structural_metric(capability, detail)

    recoverable = sum(record.source_chunk_ref is not None for record in records)
    metrics["source_chunk_recovery"] = _coverage_metric(
        capability="source_chunk_recovery",
        applicable=len(records),
        supported=recoverable,
        detail="Grounded semantic records retaining a persisted source chunk.",
        assessed_from=["SemanticIRRecord.source_chunk_ref"],
    )

    edges = by_kind["edge"]
    relation_supported = sum(
        _nonblank(record.payload, "source")
        and _nonblank(record.payload, "relation")
        and _nonblank(record.payload, "target")
        for record in edges
    )
    metrics["relation_structure"] = _coverage_metric(
        capability="relation_structure",
        applicable=len(edges),
        supported=relation_supported,
        detail="Grounded directed source-relation-target edge structure.",
        assessed_from=["edge.source", "edge.relation", "edge.target"],
    )

    provenance_supported = 0
    for record in edges:
        pointers = record.payload.get("evidence_pointers")
        if not isinstance(pointers, list) or not pointers:
            continue
        if all(
            isinstance(pointer, dict)
            and _nonblank(pointer, "document_id")
            and _nonblank(pointer, "document_role")
            for pointer in pointers
        ):
            provenance_supported += 1
    metrics["edge_provenance"] = _coverage_metric(
        capability="edge_provenance",
        applicable=len(edges),
        supported=provenance_supported,
        detail="Grounded edges retaining document-level evidence pointers.",
        assessed_from=["edge.evidence_pointers"],
    )

    measurements = by_kind["measurement"]
    identity_supported = sum(
        all(
            _nonblank(record.payload, field)
            for field in ("id", "metric_id", "metric", "subject_id", "source_expression")
        )
        for record in measurements
    )
    metrics["measurement_identity"] = _coverage_metric(
        capability="measurement_identity",
        applicable=len(measurements),
        supported=identity_supported,
        detail="Grounded measurements with metric, subject, and source identity.",
        assessed_from=[
            "measurement.id",
            "measurement.metric_id",
            "measurement.metric",
            "measurement.subject_id",
            "measurement.source_expression",
        ],
    )

    records_by_chunk_id: dict[tuple[str, str], list[SemanticIRRecord]] = defaultdict(list)
    for record in records:
        if record.ref.chunk_id is not None:
            records_by_chunk_id[(record.ref.paper_id, record.ref.chunk_id)].append(record)

    provider_supported = 0
    for measurement in measurements:
        chunk_id = measurement.ref.chunk_id
        if chunk_id is None:
            continue
        local_records = records_by_chunk_id[(measurement.ref.paper_id, chunk_id)]
        node_kind = {
            row.ref.object_id: row.ref.object_kind
            for row in local_records
            if row.ref.object_kind != "edge"
        }
        if any(
            row.ref.object_kind == "edge"
            and str(row.payload.get("relation", "")) == "HAS_MEASUREMENT"
            and str(row.payload.get("target", "")) == measurement.ref.object_id
            and node_kind.get(str(row.payload.get("source", "")))
            in {"experiment", "calculation"}
            for row in local_records
        ):
            provider_supported += 1
    metrics["measurement_provider_linkage"] = _coverage_metric(
        capability="measurement_provider_linkage",
        applicable=len(measurements),
        supported=provider_supported,
        detail="Grounded measurements linked to an Experiment or Calculation.",
        assessed_from=["HAS_MEASUREMENT"],
    )

    groups = by_kind["measurement_group"]
    group_supported = sum(
        isinstance(record.payload.get("member_measurement_ids"), list)
        and len(record.payload.get("member_measurement_ids") or []) >= 2
        for record in groups
    )
    metrics["measurement_group_structure"] = _coverage_metric(
        capability="measurement_group_structure",
        applicable=len(groups),
        supported=group_supported,
        detail="Grounded measurement groups preserving at least two members.",
        assessed_from=["measurement_group.member_measurement_ids"],
    )

    claims = by_kind["observation_claim"] + by_kind["mechanism_claim"]
    support_count = 0
    target_count = 0
    for claim in claims:
        chunk_id = claim.ref.chunk_id
        if chunk_id is None:
            continue
        local_records = records_by_chunk_id[(claim.ref.paper_id, chunk_id)]
        if any(
            row.ref.object_kind == "edge"
            and str(row.payload.get("relation", "")) == "SUPPORTS_CLAIM"
            and str(row.payload.get("target", "")) == claim.ref.object_id
            for row in local_records
        ):
            support_count += 1
        if any(
            row.ref.object_kind == "edge"
            and str(row.payload.get("relation", "")) == "APPLIES_TO"
            and str(row.payload.get("source", "")) == claim.ref.object_id
            for row in local_records
        ):
            target_count += 1
    metrics["claim_support_linkage"] = _coverage_metric(
        capability="claim_support_linkage",
        applicable=len(claims),
        supported=support_count,
        detail="Grounded claims with SUPPORTS_CLAIM evidence linkage.",
        assessed_from=["SUPPORTS_CLAIM"],
    )
    metrics["claim_application_target"] = _coverage_metric(
        capability="claim_application_target",
        applicable=len(claims),
        supported=target_count,
        detail="Grounded claims with an APPLIES_TO scientific target.",
        assessed_from=["APPLIES_TO"],
    )

    for name, gap_metric in task_gap_report.metrics.items():
        metrics[name] = TaskCapabilityMetric(
            capability=name,
            state=_coverage_state(
                gap_metric.applicable_record_count,
                gap_metric.supported_record_count,
            ),
            assessment_mode="task_local_observed_coverage",
            applicable_count=gap_metric.applicable_record_count,
            supported_count=gap_metric.supported_record_count,
            coverage_fraction=gap_metric.coverage_fraction,
            detail=(
                "Grounded task-local structured-condition coverage; missing "
                "values are extraction gaps, not scientific negative evidence."
            ),
            assessed_from=["task-local source chunks", "conditions"],
        )

    metrics["proxy_semantics"] = _explicit_gap_metric(
        "proxy_semantics",
        "Canonical Semantic IR does not explicitly encode whether one observable is a proxy for another construct.",
    )
    metrics["regime_semantics"] = _explicit_gap_metric(
        "regime_semantics",
        "Canonical Semantic IR does not explicitly encode response-regime boundaries as evidence objects.",
    )
    metrics["author_rationale"] = _explicit_gap_metric(
        "author_rationale",
        "Canonical Semantic IR does not explicitly encode why authors selected a method or experimental design.",
    )

    capabilities = {
        name: CapabilityRecord(
            name=name,
            state=metric.state,
            detail=metric.detail,
            assessed_from=metric.assessed_from,
        )
        for name, metric in sorted(metrics.items())
    }
    manifest_payload = {
        name: {
            "state": metric.state,
            "mode": metric.assessment_mode,
            "applicable": metric.applicable_count,
            "supported": metric.supported_count,
        }
        for name, metric in sorted(metrics.items())
    }
    manifest = CapabilityManifest(
        manifest_id=_stable_id(
            "task_semantic_capability_manifest",
            scope_id,
            _canonical_json(manifest_payload),
        ),
        scope_kind="task",
        scope_id=scope_id,
        capabilities=capabilities,
    )
    return TaskCapabilitySnapshot(
        scope_id=scope_id,
        manifest=manifest,
        metrics=metrics,
        source_chunk_count=len(chunks),
        semantic_record_count=len(records),
    )
