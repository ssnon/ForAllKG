from __future__ import annotations

import hashlib
import json
from collections import Counter
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
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CapabilityAssessmentMode = Literal[
    "structural_support",
    "observed_coverage",
    "explicit_gap",
]


class CapabilityAuditMetric(StrictModel):
    """How one capability state was derived from an existing IR bundle."""

    capability: str = Field(min_length=1)
    state: CapabilityState
    assessment_mode: CapabilityAssessmentMode
    applicable_count: int = Field(ge=0)
    supported_count: int = Field(ge=0)
    coverage_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    detail: str = ""
    assessed_from: list[str] = Field(default_factory=list)

    negative_evidence_inferred: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "CapabilityAuditMetric":
        if self.supported_count > self.applicable_count:
            raise ValueError("supported_count cannot exceed applicable_count")
        if self.applicable_count == 0 and self.coverage_fraction is not None:
            raise ValueError(
                "coverage_fraction must be null when applicable_count is zero"
            )
        if self.applicable_count > 0:
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


class SemanticIRCapabilityAuditResult(StrictModel):
    schema_version: Literal[
        "semantic-ir-capability-audit-v1"
    ] = "semantic-ir-capability-audit-v1"

    bundle_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    manifest: CapabilityManifest
    metrics: dict[str, CapabilityAuditMetric]
    object_counts: dict[str, int] = Field(default_factory=dict)

    audit_only: Literal[True] = True
    llm_calls_performed: Literal[0] = 0
    source_text_loaded: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    negative_evidence_inferred: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    selection_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest_alignment(self) -> "SemanticIRCapabilityAuditResult":
        if self.manifest.scope_kind != "paper":
            raise ValueError("SemanticIR bundle audit must produce a paper manifest")
        if self.manifest.scope_id != self.paper_id:
            raise ValueError("manifest scope_id must equal audited paper_id")
        if set(self.metrics) != set(self.manifest.capabilities):
            raise ValueError(
                "metric and capability mappings must contain identical keys"
            )
        for key, metric in self.metrics.items():
            if key != metric.capability:
                raise ValueError("metric key must equal metric.capability")
            if self.manifest.capabilities[key].state != metric.state:
                raise ValueError(
                    "manifest capability state must equal audit metric state"
                )
        return self


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


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
) -> CapabilityAuditMetric:
    return CapabilityAuditMetric(
        capability=capability,
        state=_coverage_state(applicable, supported),
        assessment_mode="observed_coverage",
        applicable_count=applicable,
        supported_count=supported,
        coverage_fraction=(
            supported / applicable if applicable > 0 else None
        ),
        detail=detail,
        assessed_from=assessed_from,
    )


def _structural_metric(
    *,
    capability: str,
    detail: str,
    assessed_from: list[str],
) -> CapabilityAuditMetric:
    return CapabilityAuditMetric(
        capability=capability,
        state="complete",
        assessment_mode="structural_support",
        applicable_count=1,
        supported_count=1,
        coverage_fraction=1.0,
        detail=detail,
        assessed_from=assessed_from,
    )


def _explicit_gap_metric(
    *,
    capability: str,
    detail: str,
) -> CapabilityAuditMetric:
    return CapabilityAuditMetric(
        capability=capability,
        state="absent",
        assessment_mode="explicit_gap",
        applicable_count=1,
        supported_count=0,
        coverage_fraction=0.0,
        detail=detail,
        assessed_from=["semantic-ir-v1 canonical payload schema"],
    )


def _records_of_kind(
    bundle: SemanticIRBundle,
    kind: str,
) -> list[SemanticIRRecord]:
    return [record for record in bundle.records if record.ref.object_kind == kind]


def _nonblank(payload: dict, key: str) -> bool:
    value = payload.get(key)
    return isinstance(value, str) and bool(value.strip())


def _node_kind_index(bundle: SemanticIRBundle) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for record in bundle.records:
        if record.ref.object_kind in {"chunk", "edge", "paper", "other"}:
            continue
        if record.ref.chunk_id is None:
            continue
        index[(record.ref.chunk_id, record.ref.object_id)] = record.ref.object_kind
    return index


def _edges_by_chunk(bundle: SemanticIRBundle) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for record in _records_of_kind(bundle, "edge"):
        if record.ref.chunk_id is None:
            continue
        rows.setdefault(record.ref.chunk_id, []).append(record.payload)
    return rows


def audit_semantic_ir_capabilities(
    bundle: SemanticIRBundle,
) -> SemanticIRCapabilityAuditResult:
    """
    Audit reusable scientific capabilities already encoded in one paper IR.

    This is deliberately conservative. Empty applicable populations produce
    ``unknown`` rather than ``absent`` for observed-coverage capabilities, and
    an absent semantic capability never becomes scientific negative evidence.
    """

    object_counts = Counter(record.ref.object_kind for record in bundle.records)
    node_kinds = _node_kind_index(bundle)
    edges_by_chunk = _edges_by_chunk(bundle)

    metrics: dict[str, CapabilityAuditMetric] = {}

    # These are guarantees of the current existing-extraction adapter rather
    # than claims that a paper contains at least one object of every kind.
    for capability, detail in (
        (
            "observation_claim_representation",
            "ObservationClaim objects are preserved as a distinct IR kind.",
        ),
        (
            "mechanism_claim_representation",
            "MechanismClaim objects are preserved separately from observations.",
        ),
        (
            "measurement_representation",
            "Measurement objects are preserved as distinct scalar-result records.",
        ),
        (
            "experiment_representation",
            "Experiment objects are preserved as distinct method/setup records.",
        ),
        (
            "calculation_representation",
            "Calculation objects are preserved as distinct computational records.",
        ),
        (
            "measurement_group_representation",
            "MeasurementGroup objects remain separate non-destructive containers.",
        ),
    ):
        metrics[capability] = _structural_metric(
            capability=capability,
            detail=detail,
            assessed_from=["SemanticIRRecord.ref.object_kind", "payload"],
        )

    source_ref_keys = {
        source.identity_key()
        for source in bundle.source_chunks
        if source.source_path and source.object_kind == "chunk"
    }
    source_supported = sum(
        record.source_chunk_ref is not None
        and record.source_chunk_ref.identity_key() in source_ref_keys
        for record in bundle.records
    )
    metrics["source_chunk_recovery"] = _coverage_metric(
        capability="source_chunk_recovery",
        applicable=len(bundle.records),
        supported=source_supported,
        detail=(
            "Coverage of semantic records that can be traced to a persisted "
            "source chunk without loading or copying source text."
        ),
        assessed_from=["SemanticIRRecord.source_chunk_ref", "source_chunks"],
    )

    edges = _records_of_kind(bundle, "edge")
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
        detail="Coverage of directed source-relation-target structure on edges.",
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
        detail=(
            "Coverage of edges retaining at least one document-level evidence "
            "pointer. This does not assert that page/asset locators are complete."
        ),
        assessed_from=["edge.evidence_pointers"],
    )

    measurements = _records_of_kind(bundle, "measurement")
    measurement_identity_supported = sum(
        all(
            _nonblank(record.payload, field)
            for field in (
                "id",
                "metric_id",
                "metric",
                "subject_id",
                "source_expression",
            )
        )
        for record in measurements
    )
    metrics["measurement_identity"] = _coverage_metric(
        capability="measurement_identity",
        applicable=len(measurements),
        supported=measurement_identity_supported,
        detail=(
            "Coverage of measurements with registry/human metric identity, "
            "subject identity, and source expression."
        ),
        assessed_from=[
            "measurement.id",
            "measurement.metric_id",
            "measurement.metric",
            "measurement.subject_id",
            "measurement.source_expression",
        ],
    )

    provider_supported = 0
    for measurement in measurements:
        chunk_id = measurement.ref.chunk_id
        if chunk_id is None:
            continue
        providers = {
            str(edge.get("source"))
            for edge in edges_by_chunk.get(chunk_id, [])
            if edge.get("relation") == "HAS_MEASUREMENT"
            and edge.get("target") == measurement.ref.object_id
            and node_kinds.get((chunk_id, str(edge.get("source"))))
            in {"experiment", "calculation"}
        }
        if providers:
            provider_supported += 1
    metrics["measurement_provider_linkage"] = _coverage_metric(
        capability="measurement_provider_linkage",
        applicable=len(measurements),
        supported=provider_supported,
        detail=(
            "Coverage of measurements with an incoming HAS_MEASUREMENT edge "
            "from an Experiment or Calculation in the same source chunk."
        ),
        assessed_from=["HAS_MEASUREMENT", "Experiment", "Calculation"],
    )

    measurement_condition_supported = sum(
        isinstance(record.payload.get("conditions"), list)
        and bool(record.payload.get("conditions"))
        for record in measurements
    )
    metrics["measurement_condition_coverage"] = _coverage_metric(
        capability="measurement_condition_coverage",
        applicable=len(measurements),
        supported=measurement_condition_supported,
        detail=(
            "Observed coverage of measurements carrying at least one structured "
            "Condition. Empty condition lists are not interpreted as evidence "
            "that no experimental condition existed in the source."
        ),
        assessed_from=["measurement.conditions"],
    )

    experiments = _records_of_kind(bundle, "experiment")
    experiment_condition_supported = sum(
        isinstance(record.payload.get("conditions"), list)
        and bool(record.payload.get("conditions"))
        for record in experiments
    )
    metrics["experiment_condition_coverage"] = _coverage_metric(
        capability="experiment_condition_coverage",
        applicable=len(experiments),
        supported=experiment_condition_supported,
        detail=(
            "Observed coverage of experiments carrying at least one structured "
            "Condition."
        ),
        assessed_from=["experiment.conditions"],
    )

    calculations = _records_of_kind(bundle, "calculation")
    calculation_condition_supported = sum(
        isinstance(record.payload.get("conditions"), list)
        and bool(record.payload.get("conditions"))
        for record in calculations
    )
    metrics["calculation_condition_coverage"] = _coverage_metric(
        capability="calculation_condition_coverage",
        applicable=len(calculations),
        supported=calculation_condition_supported,
        detail=(
            "Observed coverage of calculations carrying at least one structured "
            "Condition."
        ),
        assessed_from=["calculation.conditions"],
    )

    groups = _records_of_kind(bundle, "measurement_group")
    group_supported = 0
    for group in groups:
        chunk_id = group.ref.chunk_id
        if chunk_id is None:
            continue
        members = group.payload.get("member_measurement_ids")
        if not isinstance(members, list) or len(members) < 2:
            continue
        membership_edges = {
            str(edge.get("source"))
            for edge in edges_by_chunk.get(chunk_id, [])
            if edge.get("relation") == "IN_MEASUREMENT_GROUP"
            and edge.get("target") == group.ref.object_id
        }
        if all(
            isinstance(member, str)
            and node_kinds.get((chunk_id, member)) == "measurement"
            and member in membership_edges
            for member in members
        ):
            group_supported += 1
    metrics["measurement_group_structure"] = _coverage_metric(
        capability="measurement_group_structure",
        applicable=len(groups),
        supported=group_supported,
        detail=(
            "Coverage of MeasurementGroup records whose members are retained as "
            "distinct measurements with explicit membership edges."
        ),
        assessed_from=[
            "measurement_group.member_measurement_ids",
            "IN_MEASUREMENT_GROUP",
        ],
    )

    claims = [
        record
        for record in bundle.records
        if record.ref.object_kind in {"observation_claim", "mechanism_claim"}
    ]
    support_linkage_supported = 0
    target_linkage_supported = 0
    for claim in claims:
        chunk_id = claim.ref.chunk_id
        if chunk_id is None:
            continue
        chunk_edges = edges_by_chunk.get(chunk_id, [])
        direct_support = any(
            edge.get("relation") == "SUPPORTS_CLAIM"
            and edge.get("target") == claim.ref.object_id
            for edge in chunk_edges
        )
        interpreted_support = (
            claim.ref.object_kind == "mechanism_claim"
            and any(
                edge.get("relation") == "INTERPRETED_AS"
                and edge.get("target") == claim.ref.object_id
                for edge in chunk_edges
            )
        )
        if direct_support or interpreted_support:
            support_linkage_supported += 1
        if any(
            edge.get("relation") == "APPLIES_TO"
            and edge.get("source") == claim.ref.object_id
            for edge in chunk_edges
        ):
            target_linkage_supported += 1

    metrics["claim_support_linkage"] = _coverage_metric(
        capability="claim_support_linkage",
        applicable=len(claims),
        supported=support_linkage_supported,
        detail=(
            "Coverage of claims with direct SUPPORTS_CLAIM evidence or, for "
            "mechanism claims, an observation INTERPRETED_AS that mechanism."
        ),
        assessed_from=["SUPPORTS_CLAIM", "INTERPRETED_AS"],
    )
    metrics["claim_application_target"] = _coverage_metric(
        capability="claim_application_target",
        applicable=len(claims),
        supported=target_linkage_supported,
        detail="Coverage of claims retaining an explicit APPLIES_TO target.",
        assessed_from=["APPLIES_TO"],
    )

    # Deliberate gaps in the current canonical extraction schema. These are
    # reasons to consider targeted enrichment, never evidence that the
    # scientific concepts themselves are absent from the paper.
    metrics["proxy_semantics"] = _explicit_gap_metric(
        capability="proxy_semantics",
        detail=(
            "No dedicated canonical field states that one observable is being "
            "used as a proxy for a distinct scientific construct."
        ),
    )
    metrics["regime_semantics"] = _explicit_gap_metric(
        capability="regime_semantics",
        detail=(
            "No dedicated canonical representation encodes regime boundaries, "
            "piecewise response laws, or regime-transition conditions."
        ),
    )
    metrics["author_rationale"] = _explicit_gap_metric(
        capability="author_rationale",
        detail=(
            "Mechanism claims preserve scientific interpretations, but the "
            "canonical schema has no dedicated representation for why authors "
            "selected a method, comparison, or experimental design."
        ),
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
            "state": record.state,
            "detail": record.detail,
            "assessed_from": record.assessed_from,
        }
        for name, record in sorted(capabilities.items())
    }
    manifest_id = _stable_id(
        "semantic_capability_manifest",
        bundle.bundle_id,
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    manifest = CapabilityManifest(
        manifest_id=manifest_id,
        scope_kind="paper",
        scope_id=bundle.paper_id,
        capabilities=capabilities,
    )

    return SemanticIRCapabilityAuditResult(
        bundle_id=bundle.bundle_id,
        paper_id=bundle.paper_id,
        manifest=manifest,
        metrics={name: metrics[name] for name in sorted(metrics)},
        object_counts=dict(sorted(object_counts.items())),
    )
