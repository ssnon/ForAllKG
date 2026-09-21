from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.semantic_ir.backfill import (
    BackfillPlan,
    BackfillTarget,
    build_backfill_plan,
)
from pipeline_core.corpus.semantic_ir.capability import (
    CapabilityRequirement,
    CapabilityRequirementCheck,
    evaluate_capability_requirements,
)
from pipeline_core.corpus.semantic_ir.schema import SemanticIRBundle, SemanticObjectRef
from pipeline_core.corpus.semantic_ir.task_capability import TaskCapabilitySnapshot
from pipeline_core.corpus.semantic_ir.task_gap import TaskLocalCapabilityGapReport
from pipeline_core.discovery.reframing.operator_contracts import (
    ReframingOperatorContract,
    ReframingOperatorId,
    get_reframing_operator_contracts,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


OperatorReadinessStatus = Literal[
    "ready_now",
    "ready_with_optional_enrichment",
    "targeted_enrichment_required",
    "blocked_insufficient_substrate",
]


class ConditionFamilyReadiness(StrictModel):
    capability: str
    state: str
    applicable_count: int = Field(ge=0)
    supported_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    witness_chunk_count: int = Field(ge=0)


class OperatorReadinessAssessment(StrictModel):
    schema_version: Literal[
        "reframing-operator-readiness-v1"
    ] = "reframing-operator-readiness-v1"

    operator_id: ReframingOperatorId
    status: OperatorReadinessStatus
    hard_requirement_check: CapabilityRequirementCheck
    condition_families: list[ConditionFamilyReadiness] = Field(default_factory=list)
    mandatory_backfill_plan: BackfillPlan
    optional_backfill_plan: BackfillPlan
    reasons: list[str] = Field(default_factory=list)
    task_source_chunk_count: int = Field(ge=0)
    measurement_bearing_chunk_count: int = Field(ge=0)

    shadow_only: Literal[True] = True
    operator_execution_performed: Literal[False] = False
    scientific_trigger_evaluated: Literal[False] = False
    reframe_quality_evaluated: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    production_selection_changed: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False


class ReframingOperatorReadinessReport(StrictModel):
    schema_version: Literal[
        "reframing-operator-readiness-report-v1"
    ] = "reframing-operator-readiness-report-v1"

    scope_id: str = Field(min_length=1)
    task_capabilities: TaskCapabilitySnapshot
    assessments: list[OperatorReadinessAssessment] = Field(min_length=1)
    unresolved_grounded_node_count: int = Field(default=0, ge=0)
    ambiguous_grounded_node_count: int = Field(default=0, ge=0)
    grounded_scope_fully_resolved: bool = True

    shadow_only: Literal[True] = True
    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_scope(self) -> "ReframingOperatorReadinessReport":
        if self.task_capabilities.scope_id != self.scope_id:
            raise ValueError("task capability scope must match report scope")
        expected_complete = (
            self.unresolved_grounded_node_count == 0
            and self.ambiguous_grounded_node_count == 0
        )
        if self.grounded_scope_fully_resolved != expected_complete:
            raise ValueError(
                "grounded_scope_fully_resolved must match unresolved/ambiguous counts"
            )
        ids = [row.operator_id for row in self.assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("operator assessments must be unique")
        return self


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _empty_plan(
    *,
    scope_id: str,
    operator_id: str,
    lane: str,
) -> BackfillPlan:
    return build_backfill_plan(
        plan_id=_stable_id("reframe_backfill_plan", scope_id, operator_id, lane),
        requirements=[],
        targets=[],
    )


def _dedupe_chunks(refs: list[SemanticObjectRef]) -> list[SemanticObjectRef]:
    by_key = {ref.identity_key(): ref for ref in refs}
    return [by_key[key] for key in sorted(by_key)]


def _measurement_bearing_chunks(
    *,
    bundles: dict[str, SemanticIRBundle],
    task_source_chunks: list[SemanticObjectRef],
) -> list[SemanticObjectRef]:
    chunks = _dedupe_chunks(task_source_chunks)
    chunk_by_key = {ref.identity_key(): ref for ref in chunks}
    selected: dict[tuple[str, str | None, str, str], SemanticObjectRef] = {}
    for bundle in bundles.values():
        for record in bundle.records:
            source = record.source_chunk_ref
            if source is None or source.identity_key() not in chunk_by_key:
                continue
            if record.ref.object_kind != "measurement":
                continue
            selected[source.identity_key()] = chunk_by_key[source.identity_key()]
    return [selected[key] for key in sorted(selected)]


def _witness_targets(
    *,
    report: TaskLocalCapabilityGapReport,
    capabilities: set[str],
    reason_prefix: str,
) -> list[BackfillTarget]:
    targets: list[BackfillTarget] = []
    for witness in report.witnesses:
        if witness.capability not in capabilities:
            continue
        targets.append(
            BackfillTarget(
                source_chunk_ref=witness.source_chunk_ref,
                missing_capabilities=[witness.capability],
                reasons=[f"{reason_prefix}: {witness.reason}"],
            )
        )
    return targets


def _explicit_gap_plan(
    *,
    scope_id: str,
    contract: ReframingOperatorContract,
    measurement_chunks: list[SemanticObjectRef],
) -> BackfillPlan:
    capabilities = sorted(set(contract.mandatory_explicit_gap_capabilities))
    if not capabilities:
        return _empty_plan(
            scope_id=scope_id,
            operator_id=contract.operator_id,
            lane="mandatory",
        )

    requirements = [
        CapabilityRequirement(
            capability=capability,
            minimum_state="complete",
            reason=(
                f"{contract.operator_id} requires task-local {capability} annotations before operator execution."
            ),
        )
        for capability in capabilities
    ]
    targets = [
        BackfillTarget(
            source_chunk_ref=chunk,
            missing_capabilities=capabilities,
            reasons=[
                f"{contract.operator_id} explicit semantic enrichment is restricted to measurement-bearing grounded chunks."
            ],
        )
        for chunk in measurement_chunks
    ]
    return build_backfill_plan(
        plan_id=_stable_id(
            "reframe_backfill_plan",
            scope_id,
            contract.operator_id,
            "mandatory",
            ",".join(capabilities),
        ),
        requirements=requirements,
        targets=targets,
    )


def _condition_rows(
    *,
    contract: ReframingOperatorContract,
    gap_report: TaskLocalCapabilityGapReport,
    task_capabilities: TaskCapabilitySnapshot,
) -> list[ConditionFamilyReadiness]:
    policy = contract.task_local_condition_policy
    if policy is None:
        return []
    rows = []
    for capability in policy.capabilities:
        metric = gap_report.metrics.get(capability)
        snapshot_metric = task_capabilities.metrics.get(capability)
        if metric is None or snapshot_metric is None:
            rows.append(
                ConditionFamilyReadiness(
                    capability=capability,
                    state="undeclared",
                    applicable_count=0,
                    supported_count=0,
                    missing_count=0,
                    witness_chunk_count=0,
                )
            )
            continue
        rows.append(
            ConditionFamilyReadiness(
                capability=capability,
                state=snapshot_metric.state,
                applicable_count=metric.applicable_record_count,
                supported_count=metric.supported_record_count,
                missing_count=metric.missing_record_count,
                witness_chunk_count=metric.witness_chunk_count,
            )
        )
    return rows


def _assess_one(
    *,
    scope_id: str,
    contract: ReframingOperatorContract,
    task_capabilities: TaskCapabilitySnapshot,
    gap_report: TaskLocalCapabilityGapReport,
    bundles: dict[str, SemanticIRBundle],
    task_source_chunks: list[SemanticObjectRef],
) -> OperatorReadinessAssessment:
    hard_check = evaluate_capability_requirements(
        manifest=task_capabilities.manifest,
        requirements=contract.hard_requirements,
    )
    source_chunks = _dedupe_chunks(task_source_chunks)
    measurement_chunks = _measurement_bearing_chunks(
        bundles=bundles,
        task_source_chunks=source_chunks,
    )
    condition_rows = _condition_rows(
        contract=contract,
        gap_report=gap_report,
        task_capabilities=task_capabilities,
    )

    mandatory_plan = _empty_plan(
        scope_id=scope_id,
        operator_id=contract.operator_id,
        lane="mandatory",
    )
    optional_plan = _empty_plan(
        scope_id=scope_id,
        operator_id=contract.operator_id,
        lane="optional",
    )
    reasons: list[str] = []

    if hard_check.backfill_required_count:
        missing = [
            row.capability
            for row in hard_check.assessments
            if not row.satisfied
        ]
        status: OperatorReadinessStatus = "blocked_insufficient_substrate"
        reasons.append(
            "Hard grounded-evidence capabilities are insufficient: "
            + ", ".join(sorted(missing))
        )
        return OperatorReadinessAssessment(
            operator_id=contract.operator_id,
            status=status,
            hard_requirement_check=hard_check,
            condition_families=condition_rows,
            mandatory_backfill_plan=mandatory_plan,
            optional_backfill_plan=optional_plan,
            reasons=reasons,
            task_source_chunk_count=len(source_chunks),
            measurement_bearing_chunk_count=len(measurement_chunks),
        )

    if contract.mandatory_explicit_gap_capabilities:
        unsatisfied = [
            capability
            for capability in contract.mandatory_explicit_gap_capabilities
            if task_capabilities.manifest.capabilities.get(capability) is None
            or task_capabilities.manifest.capabilities[capability].state != "complete"
        ]
        if unsatisfied:
            if (
                contract.explicit_gap_target_selector
                == "measurement_bearing_chunks"
            ):
                if not measurement_chunks:
                    return OperatorReadinessAssessment(
                        operator_id=contract.operator_id,
                        status="blocked_insufficient_substrate",
                        hard_requirement_check=hard_check,
                        condition_families=condition_rows,
                        mandatory_backfill_plan=mandatory_plan,
                        optional_backfill_plan=optional_plan,
                        reasons=[
                            "Explicit proxy-semantic enrichment is required, but the grounded task scope contains no measurement-bearing source chunks."
                        ],
                        task_source_chunk_count=len(source_chunks),
                        measurement_bearing_chunk_count=0,
                    )
                mandatory_plan = _explicit_gap_plan(
                    scope_id=scope_id,
                    contract=contract,
                    measurement_chunks=measurement_chunks,
                )
                return OperatorReadinessAssessment(
                    operator_id=contract.operator_id,
                    status="targeted_enrichment_required",
                    hard_requirement_check=hard_check,
                    condition_families=condition_rows,
                    mandatory_backfill_plan=mandatory_plan,
                    optional_backfill_plan=optional_plan,
                    reasons=[
                        "Canonical Semantic IR lacks explicit "
                        + ", ".join(sorted(unsatisfied))
                        + "; enrichment is restricted to grounded measurement-bearing chunks."
                    ],
                    task_source_chunk_count=len(source_chunks),
                    measurement_bearing_chunk_count=len(measurement_chunks),
                )

    policy = contract.task_local_condition_policy
    if policy is not None:
        supported_families = sum(row.supported_count > 0 for row in condition_rows)
        incomplete_caps = {
            row.capability
            for row in condition_rows
            if row.missing_count > 0
        }
        if supported_families < policy.minimum_supported_families:
            mandatory_targets = _witness_targets(
                report=gap_report,
                capabilities=set(policy.capabilities),
                reason_prefix=(
                    f"{contract.operator_id} requires at least "
                    f"{policy.minimum_supported_families} condition-bearing evidence family"
                ),
            )
            if mandatory_targets:
                requirements = [
                    CapabilityRequirement(
                        capability=capability,
                        minimum_state="partial",
                        reason=(
                            f"{contract.operator_id} requires task-local structured conditions in at least one evidence family."
                        ),
                    )
                    for capability in policy.capabilities
                ]
                mandatory_plan = build_backfill_plan(
                    plan_id=_stable_id(
                        "reframe_backfill_plan",
                        scope_id,
                        contract.operator_id,
                        "mandatory-condition",
                    ),
                    requirements=requirements,
                    targets=mandatory_targets,
                )
                return OperatorReadinessAssessment(
                    operator_id=contract.operator_id,
                    status="targeted_enrichment_required",
                    hard_requirement_check=hard_check,
                    condition_families=condition_rows,
                    mandatory_backfill_plan=mandatory_plan,
                    optional_backfill_plan=optional_plan,
                    reasons=[
                        "No condition evidence family currently meets the operator minimum; task-local gap witnesses identify targeted enrichment chunks."
                    ],
                    task_source_chunk_count=len(source_chunks),
                    measurement_bearing_chunk_count=len(measurement_chunks),
                )
            return OperatorReadinessAssessment(
                operator_id=contract.operator_id,
                status="blocked_insufficient_substrate",
                hard_requirement_check=hard_check,
                condition_families=condition_rows,
                mandatory_backfill_plan=mandatory_plan,
                optional_backfill_plan=optional_plan,
                reasons=[
                    "No condition-bearing evidence family is currently available and no task-local witness can localize safe enrichment targets."
                ],
                task_source_chunk_count=len(source_chunks),
                measurement_bearing_chunk_count=len(measurement_chunks),
            )

        if incomplete_caps:
            optional_targets = _witness_targets(
                report=gap_report,
                capabilities=incomplete_caps,
                reason_prefix=(
                    f"Optional {contract.operator_id} coverage completion"
                ),
            )
            requirements = [
                CapabilityRequirement(
                    capability=capability,
                    minimum_state="complete",
                    reason=(
                        f"Optional enrichment can improve {contract.operator_id} coverage for task-local missing structured conditions."
                    ),
                )
                for capability in sorted(incomplete_caps)
            ]
            optional_plan = build_backfill_plan(
                plan_id=_stable_id(
                    "reframe_backfill_plan",
                    scope_id,
                    contract.operator_id,
                    "optional-condition",
                ),
                requirements=requirements,
                targets=optional_targets,
            )
            reasons.append(
                "At least one condition evidence family is usable now; missing condition fields are optional targeted enrichment, not a production blocker."
            )
            return OperatorReadinessAssessment(
                operator_id=contract.operator_id,
                status="ready_with_optional_enrichment",
                hard_requirement_check=hard_check,
                condition_families=condition_rows,
                mandatory_backfill_plan=mandatory_plan,
                optional_backfill_plan=optional_plan,
                reasons=reasons,
                task_source_chunk_count=len(source_chunks),
                measurement_bearing_chunk_count=len(measurement_chunks),
            )

    return OperatorReadinessAssessment(
        operator_id=contract.operator_id,
        status="ready_now",
        hard_requirement_check=hard_check,
        condition_families=condition_rows,
        mandatory_backfill_plan=mandatory_plan,
        optional_backfill_plan=optional_plan,
        reasons=[
            "Grounded semantic substrate satisfies the operator capability contract without enrichment."
        ],
        task_source_chunk_count=len(source_chunks),
        measurement_bearing_chunk_count=len(measurement_chunks),
    )


def assess_reframing_operator_readiness(
    *,
    scope_id: str,
    task_capabilities: TaskCapabilitySnapshot,
    gap_report: TaskLocalCapabilityGapReport,
    bundles: dict[str, SemanticIRBundle],
    task_source_chunks: list[SemanticObjectRef],
    contracts: list[ReframingOperatorContract] | None = None,
    unresolved_grounded_node_count: int = 0,
    ambiguous_grounded_node_count: int = 0,
) -> ReframingOperatorReadinessReport:
    selected_contracts = contracts or get_reframing_operator_contracts()
    assessments = [
        _assess_one(
            scope_id=scope_id,
            contract=contract,
            task_capabilities=task_capabilities,
            gap_report=gap_report,
            bundles=bundles,
            task_source_chunks=task_source_chunks,
        )
        for contract in selected_contracts
    ]
    if ambiguous_grounded_node_count:
        blocked = []
        for row in assessments:
            blocked.append(
                row.model_copy(
                    update={
                        "status": "blocked_insufficient_substrate",
                        "reasons": [
                            *row.reasons,
                            f"{ambiguous_grounded_node_count} grounded paper-local node(s) remain truly ambiguous; operator execution is fail-closed.",
                        ],
                    }
                )
            )
        assessments = blocked
    elif unresolved_grounded_node_count:
        warned = []
        for row in assessments:
            warned.append(
                row.model_copy(
                    update={
                        "reasons": [
                            *row.reasons,
                            f"{unresolved_grounded_node_count} grounded paper-local node(s) are unresolved; readiness applies only to the recovered grounded substrate.",
                        ]
                    }
                )
            )
        assessments = warned

    return ReframingOperatorReadinessReport(
        scope_id=scope_id,
        task_capabilities=task_capabilities,
        assessments=assessments,
        unresolved_grounded_node_count=unresolved_grounded_node_count,
        ambiguous_grounded_node_count=ambiguous_grounded_node_count,
        grounded_scope_fully_resolved=(
            unresolved_grounded_node_count == 0
            and ambiguous_grounded_node_count == 0
        ),
    )
