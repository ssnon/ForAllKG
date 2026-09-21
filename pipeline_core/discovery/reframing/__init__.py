"""Shadow-only scientific reframing infrastructure."""

from pipeline_core.discovery.reframing.evidence_tension import (
    EvidenceLevelTensionWitness,
    ScientificEvidenceTensionReport,
    extract_evidence_level_tensions,
)
from pipeline_core.discovery.reframing.operator_contracts import (
    ReframingOperatorContract,
    ReframingOperatorId,
    TaskLocalConditionPolicy,
    get_reframing_operator_contracts,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    DirectScientificTriggerSignal,
    ReframeOperatorTriggerAssessment,
    ScientificReframeTriggerReport,
    ScientificTensionWitness,
)
from pipeline_core.discovery.reframing.trigger_detection import (
    detect_scientific_reframe_triggers,
)
from pipeline_core.discovery.reframing.operator_readiness import (
    OperatorReadinessAssessment,
    ReframingOperatorReadinessReport,
    assess_reframing_operator_readiness,
)

__all__ = [
    "triggered_operator_ids",
    "summarize_selective_execution",
    "resumable_shadow_report",
    "resumable_portfolio_report",
    "resumable_critic_report",
    "SelectiveExecutionStage",
    "ScientificReframeSelectiveExecutionReport",
    "BenchmarkTaskSelectiveExecution",
    "EvidenceLevelTensionWitness",
    "ScientificEvidenceTensionReport",
    "extract_evidence_level_tensions",
    "OperatorReadinessAssessment",
    "ReframingOperatorContract",
    "ReframingOperatorId",
    "ReframingOperatorReadinessReport",
    "TaskLocalConditionPolicy",
    "DirectScientificTriggerSignal",
    "ReframeOperatorTriggerAssessment",
    "ScientificReframeTriggerReport",
    "ScientificTensionWitness",
    "detect_scientific_reframe_triggers",
    "assess_reframing_operator_readiness",
    "get_reframing_operator_contracts",
    "ScientificReframeTriggerReviewEvidencePack",
    "TaskTriggerReviewEvidence",
    "build_task_trigger_review_evidence",
    "build_trigger_review_evidence_pack",
    "BenchmarkReframeCandidateSynthesis",
    "LexicalOverlapDiagnostic",
    "OperatorSynthesisSummary",
    "ScientificReframeBenchmarkSynthesis",
    "build_benchmark_reframe_synthesis",
    "lexical_construct_tokens",
    "ScientificReframeCalibrationFreeze",
    "ScientificReframeValidationCohortAudit",
    "build_calibration_freeze",
    "audit_validation_cohort",
    "semantics_fingerprint",
]

from pipeline_core.discovery.reframing.review_evidence import (
    ScientificReframeTriggerReviewEvidencePack,
    TaskTriggerReviewEvidence,
    build_task_trigger_review_evidence,
    build_trigger_review_evidence_pack,
)

from pipeline_core.discovery.reframing.selective_execution import (
    BenchmarkTaskSelectiveExecution,
    ScientificReframeSelectiveExecutionReport,
    SelectiveExecutionStage,
    resumable_critic_report,
    resumable_portfolio_report,
    resumable_shadow_report,
    summarize_selective_execution,
    triggered_operator_ids,
)

from pipeline_core.discovery.reframing.benchmark_synthesis import (
    BenchmarkReframeCandidateSynthesis,
    LexicalOverlapDiagnostic,
    OperatorSynthesisSummary,
    ScientificReframeBenchmarkSynthesis,
    build_benchmark_reframe_synthesis,
    lexical_construct_tokens,
)

from pipeline_core.discovery.reframing.generalization_validation import (
    ScientificReframeCalibrationFreeze,
    ScientificReframeValidationCohortAudit,
    audit_validation_cohort,
    build_calibration_freeze,
    semantics_fingerprint,
)
