from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.corpus.semantic_ir.backfill import (
    BackfillPlan,
    BackfillTarget,
    build_backfill_plan,
)
from pipeline_core.corpus.semantic_ir.capability import (
    CapabilityManifest,
    CapabilityRequirement,
    CapabilityRequirementCheck,
    evaluate_capability_requirements,
)
from pipeline_core.corpus.semantic_ir.capability_audit import (
    CapabilityAuditMetric,
)
from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticObjectRef,
)
from pipeline_core.corpus.semantic_ir.task_gap import (
    TaskLocalCapabilityGapWitness,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeferredBackfillCapability(StrictModel):
    capability: str = Field(min_length=1)
    observed_state: str = Field(min_length=1)
    assessment_mode: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    task_local_gap_witness_required: Literal[True] = True


class TaskScopedBackfillPlanResult(StrictModel):
    schema_version: Literal[
        "task-scoped-semantic-backfill-plan-v1"
    ] = "task-scoped-semantic-backfill-plan-v1"

    scope_id: str = Field(min_length=1)
    operator_id: str = Field(min_length=1)
    capability_check: CapabilityRequirementCheck
    plan: BackfillPlan
    task_source_chunk_count: int = Field(ge=0)
    deferred_capabilities: list[DeferredBackfillCapability] = Field(
        default_factory=list
    )

    execution_performed: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False
    global_partial_coverage_auto_backfilled: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    selection_authority_created: Literal[False] = False


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _dedupe_chunk_refs(
    refs: list[SemanticObjectRef],
) -> list[SemanticObjectRef]:
    by_key = {}
    for ref in refs:
        if ref.object_kind != "chunk":
            raise ValueError("task source scope may contain only chunk refs")
        by_key.setdefault(ref.identity_key(), ref)
    return [by_key[key] for key in sorted(by_key)]


def source_chunks_for_semantic_refs(
    *,
    bundle: SemanticIRBundle,
    refs: list[SemanticObjectRef],
) -> list[SemanticObjectRef]:
    """Map task-relevant semantic objects back to persisted source chunks."""

    records_by_key = {
        record.ref.identity_key(): record
        for record in bundle.records
    }
    source_by_key = {
        ref.identity_key(): ref
        for ref in bundle.source_chunks
    }
    selected: list[SemanticObjectRef] = []

    for ref in refs:
        if ref.paper_id != bundle.paper_id:
            raise ValueError("task semantic ref is outside the paper bundle")
        if ref.object_kind == "chunk":
            source = source_by_key.get(ref.identity_key())
            if source is None:
                raise ValueError(
                    f"task chunk ref is not present in bundle: {ref.object_id}"
                )
            selected.append(source)
            continue

        record = records_by_key.get(ref.identity_key())
        if record is None:
            raise ValueError(
                "task semantic ref is not present in bundle: "
                + str(ref.identity_key())
            )
        if record.source_chunk_ref is None:
            raise ValueError(
                "task semantic ref has no recoverable source chunk: "
                + str(ref.identity_key())
            )
        selected.append(record.source_chunk_ref)

    return _dedupe_chunk_refs(selected)


def plan_task_scoped_backfill(
    *,
    scope_id: str,
    operator_id: str,
    manifest: CapabilityManifest,
    metrics: dict[str, CapabilityAuditMetric],
    requirements: list[CapabilityRequirement],
    task_source_chunks: list[SemanticObjectRef],
    task_local_gap_witnesses: list[TaskLocalCapabilityGapWitness] | None = None,
) -> TaskScopedBackfillPlanResult:
    """
    Plan only explicit-schema-gap enrichment for a task-local source scope.

    Observed corpus/paper coverage gaps are intentionally not expanded into
    backfill targets without a later task-local gap witness. This prevents a
    45% condition-coverage statistic, for example, from causing every task
    source chunk to be re-extracted.
    """

    source_chunks = _dedupe_chunk_refs(task_source_chunks)
    check = evaluate_capability_requirements(
        manifest=manifest,
        requirements=requirements,
    )
    requirement_by_capability = {
        requirement.capability: requirement
        for requirement in requirements
    }

    witness_rows = list(task_local_gap_witnesses or [])
    witness_chunks_by_capability: dict[str, set[tuple[str, str | None, str, str]]] = {}
    for witness in witness_rows:
        witness_chunks_by_capability.setdefault(witness.capability, set()).add(
            witness.source_chunk_ref.identity_key()
        )

    per_chunk_capabilities: dict[
        tuple[str, str | None, str, str], set[str]
    ] = {ref.identity_key(): set() for ref in source_chunks}
    deferred: list[DeferredBackfillCapability] = []

    for assessment in check.assessments:
        if assessment.satisfied:
            continue
        metric = metrics.get(assessment.capability)
        if metric is not None and metric.assessment_mode == "explicit_gap":
            if source_chunks:
                for key in per_chunk_capabilities:
                    per_chunk_capabilities[key].add(assessment.capability)
            else:
                deferred.append(
                    DeferredBackfillCapability(
                        capability=assessment.capability,
                        observed_state=assessment.observed_state,
                        assessment_mode=metric.assessment_mode,
                        reason=(
                            "The capability is an explicit schema gap, but no "
                            "task-local source chunks were supplied."
                        ),
                    )
                )
            continue

        witnessed_keys = witness_chunks_by_capability.get(
            assessment.capability, set()
        )
        if witnessed_keys:
            for key in witnessed_keys:
                if key in per_chunk_capabilities:
                    per_chunk_capabilities[key].add(assessment.capability)
            continue

        deferred.append(
            DeferredBackfillCapability(
                capability=assessment.capability,
                observed_state=assessment.observed_state,
                assessment_mode=(
                    metric.assessment_mode if metric is not None else "undeclared"
                ),
                reason=(
                    "Global or paper-level capability state is insufficient to "
                    "identify which task-local source chunks actually need "
                    "enrichment. A task-local gap witness is required before "
                    "creating extraction targets."
                ),
            )
        )

    source_by_key = {ref.identity_key(): ref for ref in source_chunks}
    targets = []
    for key in sorted(per_chunk_capabilities):
        capabilities = sorted(per_chunk_capabilities[key])
        if not capabilities:
            continue
        reasons = [
            requirement_by_capability[capability].reason
            for capability in capabilities
        ]
        targets.append(
            BackfillTarget(
                source_chunk_ref=source_by_key[key],
                missing_capabilities=capabilities,
                reasons=reasons,
            )
        )

    plan = build_backfill_plan(
        plan_id=_stable_id(
            "task_semantic_backfill_plan",
            scope_id,
            operator_id,
            manifest.manifest_id,
            ",".join(sorted({
                capability
                for capabilities in per_chunk_capabilities.values()
                for capability in capabilities
            })),
            ",".join(
                ref.object_id
                for ref in source_chunks
            ),
        ),
        requirements=requirements,
        targets=targets,
    )

    return TaskScopedBackfillPlanResult(
        scope_id=scope_id,
        operator_id=operator_id,
        capability_check=check,
        plan=plan,
        task_source_chunk_count=len(source_chunks),
        deferred_capabilities=sorted(
            deferred,
            key=lambda row: row.capability,
        ),
    )
