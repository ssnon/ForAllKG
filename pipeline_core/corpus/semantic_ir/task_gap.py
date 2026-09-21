from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticObjectRef,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


TaskLocalInspectableCapability = Literal[
    "measurement_condition_coverage",
    "experiment_condition_coverage",
    "calculation_condition_coverage",
]

_CAPABILITY_KIND = {
    "measurement_condition_coverage": "measurement",
    "experiment_condition_coverage": "experiment",
    "calculation_condition_coverage": "calculation",
}


class TaskLocalCapabilityGapWitness(StrictModel):
    capability: TaskLocalInspectableCapability
    paper_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source_chunk_ref: SemanticObjectRef
    applicable_record_count: int = Field(ge=1)
    missing_record_count: int = Field(ge=1)
    missing_semantic_refs: list[SemanticObjectRef] = Field(min_length=1)
    reason: str = Field(min_length=1)

    diagnostic_only: Literal[True] = True
    negative_evidence_inferred: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False


class TaskLocalCapabilityMetric(StrictModel):
    capability: TaskLocalInspectableCapability
    applicable_record_count: int = Field(ge=0)
    supported_record_count: int = Field(ge=0)
    missing_record_count: int = Field(ge=0)
    source_chunk_count: int = Field(ge=0)
    witness_chunk_count: int = Field(ge=0)
    coverage_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_counts(self) -> "TaskLocalCapabilityMetric":
        if self.supported_record_count + self.missing_record_count != self.applicable_record_count:
            raise ValueError("supported + missing must equal applicable")
        if self.witness_chunk_count > self.source_chunk_count:
            raise ValueError("witness_chunk_count cannot exceed source_chunk_count")
        if self.applicable_record_count == 0:
            if self.coverage_fraction is not None:
                raise ValueError("coverage_fraction must be null when no records apply")
        else:
            expected = self.supported_record_count / self.applicable_record_count
            if self.coverage_fraction is None or abs(self.coverage_fraction - expected) > 1e-12:
                raise ValueError("coverage_fraction must equal supported/applicable")
        return self


class TaskLocalCapabilityGapReport(StrictModel):
    schema_version: Literal[
        "task-local-capability-gap-report-v1"
    ] = "task-local-capability-gap-report-v1"

    source_chunk_count: int = Field(ge=0)
    metrics: dict[str, TaskLocalCapabilityMetric] = Field(default_factory=dict)
    witnesses: list[TaskLocalCapabilityGapWitness] = Field(default_factory=list)

    task_local_only: Literal[True] = True
    global_coverage_extrapolated: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False
    negative_evidence_inferred: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False


def _dedupe_chunks(refs: list[SemanticObjectRef]) -> list[SemanticObjectRef]:
    by_key = {ref.identity_key(): ref for ref in refs}
    return [by_key[key] for key in sorted(by_key)]


def detect_task_local_capability_gaps(
    *,
    bundles: dict[str, SemanticIRBundle],
    task_source_chunks: list[SemanticObjectRef],
    capabilities: list[TaskLocalInspectableCapability],
) -> TaskLocalCapabilityGapReport:
    """
    Inspect only already-selected task source chunks for structured-condition gaps.

    This is a local data-availability diagnostic, not a claim that the source
    paper omitted the condition scientifically. An empty structured field may
    reflect extraction coverage rather than source absence.
    """

    chunks = _dedupe_chunks(task_source_chunks)
    chunk_keys = {ref.identity_key(): ref for ref in chunks}
    records_by_chunk: dict[tuple[str, str | None, str, str], list] = defaultdict(list)

    for bundle in bundles.values():
        for record in bundle.records:
            source = record.source_chunk_ref
            if source is None or source.identity_key() not in chunk_keys:
                continue
            records_by_chunk[source.identity_key()].append(record)

    metrics: dict[str, TaskLocalCapabilityMetric] = {}
    witnesses: list[TaskLocalCapabilityGapWitness] = []

    for capability in capabilities:
        kind = _CAPABILITY_KIND[capability]
        applicable_total = 0
        supported_total = 0
        missing_total = 0
        witness_chunks = 0

        for chunk in chunks:
            applicable = [
                record
                for record in records_by_chunk.get(chunk.identity_key(), [])
                if record.ref.object_kind == kind
            ]
            if not applicable:
                continue
            missing = [
                record
                for record in applicable
                if not isinstance(record.payload.get("conditions"), list)
                or len(record.payload.get("conditions") or []) == 0
            ]
            supported = len(applicable) - len(missing)
            applicable_total += len(applicable)
            supported_total += supported
            missing_total += len(missing)

            if missing:
                witness_chunks += 1
                witnesses.append(
                    TaskLocalCapabilityGapWitness(
                        capability=capability,
                        paper_id=chunk.paper_id,
                        chunk_id=str(chunk.chunk_id),
                        source_chunk_ref=chunk,
                        applicable_record_count=len(applicable),
                        missing_record_count=len(missing),
                        missing_semantic_refs=[record.ref for record in missing],
                        reason=(
                            f"Task-local source chunk contains {len(missing)} of "
                            f"{len(applicable)} {kind} records without structured "
                            "conditions; targeted enrichment may be needed if the "
                            "operator requires those conditions."
                        ),
                    )
                )

        metrics[capability] = TaskLocalCapabilityMetric(
            capability=capability,
            applicable_record_count=applicable_total,
            supported_record_count=supported_total,
            missing_record_count=missing_total,
            source_chunk_count=len(chunks),
            witness_chunk_count=witness_chunks,
            coverage_fraction=(
                supported_total / applicable_total
                if applicable_total else None
            ),
        )

    witnesses.sort(
        key=lambda row: (
            row.capability,
            row.paper_id,
            row.chunk_id,
        )
    )
    return TaskLocalCapabilityGapReport(
        source_chunk_count=len(chunks),
        metrics=metrics,
        witnesses=witnesses,
    )
