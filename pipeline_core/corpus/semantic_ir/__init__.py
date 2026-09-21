"""Reusable semantic intermediate-representation primitives."""

from pipeline_core.corpus.semantic_ir.annotation import SemanticAnnotation
from pipeline_core.corpus.semantic_ir.backfill import (
    BackfillPlan,
    BackfillTarget,
    build_backfill_plan,
)
from pipeline_core.corpus.semantic_ir.corpus_audit import (
    CorpusCapabilityAggregateMetric,
    CorpusPaperCapabilitySummary,
    CorpusSemanticCapabilityAuditResult,
    DiscoveredExtractionAttempt,
    ExtractionAttemptDiscoveryResult,
    aggregate_semantic_ir_capability_audits,
    audit_existing_extraction_corpus,
    discover_extraction_attempts,
)
from pipeline_core.corpus.semantic_ir.existing_extraction import (
    ExistingExtractionIRImportResult,
    load_existing_extraction_semantic_ir,
)
from pipeline_core.corpus.semantic_ir.capability_audit import (
    CapabilityAuditMetric,
    SemanticIRCapabilityAuditResult,
    audit_semantic_ir_capabilities,
)
from pipeline_core.corpus.semantic_ir.capability import (
    CapabilityManifest,
    CapabilityRecord,
    CapabilityRequirement,
    CapabilityRequirementAssessment,
    CapabilityRequirementCheck,
    evaluate_capability_requirements,
)
from pipeline_core.corpus.semantic_ir.task_gap import (
    TaskLocalCapabilityGapReport,
    TaskLocalCapabilityGapWitness,
    TaskLocalCapabilityMetric,
    detect_task_local_capability_gaps,
)
from pipeline_core.corpus.semantic_ir.task_scope import (
    ResolvedTaskSemanticObject,
    TaskSemanticScopeResolution,
    resolve_graph_explorer_task_scope,
    resolve_graph_explorer_task_scope_from_bundles,
)
from pipeline_core.corpus.semantic_ir.task_backfill import (
    DeferredBackfillCapability,
    TaskScopedBackfillPlanResult,
    plan_task_scoped_backfill,
    source_chunks_for_semantic_refs,
)
from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectRef,
)

__all__ = [
    "ResolvedTaskSemanticObject",
    "TaskLocalCapabilityGapReport",
    "TaskLocalCapabilityGapWitness",
    "TaskLocalCapabilityMetric",
    "TaskSemanticScopeResolution",
    "CorpusCapabilityAggregateMetric",
    "CorpusPaperCapabilitySummary",
    "CorpusSemanticCapabilityAuditResult",
    "DeferredBackfillCapability",
    "DiscoveredExtractionAttempt",
    "ExtractionAttemptDiscoveryResult",
    "BackfillPlan",
    "BackfillTarget",
    "CapabilityAuditMetric",
    "CapabilityManifest",
    "ExistingExtractionIRImportResult",
    "CapabilityRecord",
    "CapabilityRequirement",
    "CapabilityRequirementAssessment",
    "CapabilityRequirementCheck",
    "SemanticAnnotation",
    "SemanticIRCapabilityAuditResult",
    "SemanticIRBundle",
    "SemanticIRRecord",
    "SemanticObjectRef",
    "TaskScopedBackfillPlanResult",
    "aggregate_semantic_ir_capability_audits",
    "audit_existing_extraction_corpus",
    "audit_semantic_ir_capabilities",
    "build_backfill_plan",
    "discover_extraction_attempts",
    "evaluate_capability_requirements",
    "load_existing_extraction_semantic_ir",
    "plan_task_scoped_backfill",
    "source_chunks_for_semantic_refs",
    "detect_task_local_capability_gaps",
    "resolve_graph_explorer_task_scope",
    "resolve_graph_explorer_task_scope_from_bundles",
]
