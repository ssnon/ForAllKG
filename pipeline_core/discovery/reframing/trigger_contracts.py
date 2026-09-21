from __future__ import annotations

import hashlib
import json
from typing import Literal

from pipeline_core.discovery.reframing.evidence_tension import (
    EvidenceTensionType,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.reframe_contracts import (
    ImplementedReframeOperatorId,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


NormalizedTensionType = Literal[
    "CONTEXT_DEPENDENCY",
    "QUALITATIVE_DIFFERENCE",
    "MAGNITUDE_DIFFERENCE",
    "POTENTIAL_CONFLICT",
    "INSUFFICIENT_CONTEXT",
]

TriggerDecision = Literal[
    "triggered",
    "not_triggered",
    "insufficient_trigger_evidence",
]

DirectTriggerSignalKind = Literal[
    "MEDIATION_GAP",
    "NON_MONOTONIC_RESPONSE",
    "FINITE_OPTIMUM",
    "SATURATION_OR_PLATEAU",
    "EXPLICIT_THRESHOLD",
    "MECHANISM_SWITCH",
]

DirectTriggerStrength = Literal["supporting", "sufficient"]


class ScientificTensionWitness(StrictModel):
    witness_id: str = Field(min_length=1)
    source_tension_id: str = Field(min_length=1)
    source_tension_type: str = Field(min_length=1)
    normalized_tension_type: NormalizedTensionType
    focal_statement_id: str | None = None
    side_a_statement_ids: list[str] = Field(default_factory=list)
    side_b_statement_ids: list[str] = Field(default_factory=list)
    paper_ids: list[str] = Field(default_factory=list)
    grounded_statement_ids: list[str] = Field(default_factory=list)
    distinct_claim_kinds: list[str] = Field(default_factory=list)
    evidence_tension_types: list[EvidenceTensionType] = Field(default_factory=list)
    classification_bases: list[str] = Field(default_factory=list)
    relevant_condition_signature_count: int = Field(ge=0, default=0)
    side_a_response_families: list[str] = Field(default_factory=list)
    side_b_response_families: list[str] = Field(default_factory=list)
    paired_grounded_sides: bool = False
    paired_response_signal: bool = False
    independence_basis: Literal[
        "cross_paper",
        "mixed_claim_kind",
        "single_family",
        "unknown",
    ] = "unknown"
    independent_family_signal: bool = False

    diagnostic_only: Literal[True] = True
    contrast_is_not_conflict_authority: Literal[True] = True
    scientific_conflict_authority: Literal[False] = False


class DirectScientificTriggerSignal(StrictModel):
    signal_id: str = Field(min_length=1)
    kind: DirectTriggerSignalKind
    target_operator_id: ImplementedReframeOperatorId
    strength: DirectTriggerStrength
    statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    diagnostic_only: Literal[True] = True
    scientific_authority: Literal[False] = False
    regime_boundary_authority: Literal[False] = False
    latent_variable_authority: Literal[False] = False


class ConditionDiversitySignal(StrictModel):
    example_count: int = Field(ge=0)
    paper_count: int = Field(ge=0)
    distinct_condition_signature_count: int = Field(ge=0)
    object_kinds: list[str] = Field(default_factory=list)
    condition_names: list[str] = Field(default_factory=list)
    diversity_present: bool = False

    missing_conditions_are_negative_evidence: Literal[False] = False
    boundary_authority: Literal[False] = False


class ReframeOperatorTriggerAssessment(StrictModel):
    operator_id: ImplementedReframeOperatorId
    decision: TriggerDecision
    witness_ids: list[str] = Field(default_factory=list)
    direct_signal_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    condition_diversity_required: bool = False
    independent_family_required: bool = False
    scientific_trigger_signal: bool = False

    production_selection_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


class ScientificReframeTriggerReport(StrictModel):
    schema_version: Literal[
        "scientific-reframe-trigger-report-v1"
    ] = "scientific-reframe-trigger-report-v1"

    report_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    source_explorer_report_id: str | None = None
    source_explorer_report_sha256: str = Field(min_length=1)
    source_evidence_sha256: str = Field(min_length=1)
    source_evidence_tension_report_id: str | None = None
    trigger_resolution_mode: Literal[
        "legacy_explorer_tension",
        "evidence_level_refined",
    ] = "legacy_explorer_tension"

    tension_witnesses: list[ScientificTensionWitness] = Field(default_factory=list)
    direct_trigger_signals: list[DirectScientificTriggerSignal] = Field(default_factory=list)
    condition_diversity: ConditionDiversitySignal
    assessments: list[ReframeOperatorTriggerAssessment]

    llm_calls_performed: Literal[0] = 0
    deterministic_trigger_detection: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    overall_score_computed: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False
    scientific_conflict_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_unique_operator_assessments(self) -> "ScientificReframeTriggerReport":
        operators = [row.operator_id for row in self.assessments]
        if len(operators) != len(set(operators)):
            raise ValueError("trigger assessments must have unique operator_id values")
        witness_ids = [row.witness_id for row in self.tension_witnesses]
        if len(witness_ids) != len(set(witness_ids)):
            raise ValueError("tension witness IDs must be unique")
        known = set(witness_ids)
        signal_ids = [row.signal_id for row in self.direct_trigger_signals]
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError("direct trigger signal IDs must be unique")
        known_signals = set(signal_ids)
        for assessment in self.assessments:
            unknown = set(assessment.witness_ids) - known
            if unknown:
                raise ValueError(
                    "trigger assessment references unknown witness IDs: "
                    + ", ".join(sorted(unknown))
                )
            unknown_signals = set(assessment.direct_signal_ids) - known_signals
            if unknown_signals:
                raise ValueError(
                    "trigger assessment references unknown direct signal IDs: "
                    + ", ".join(sorted(unknown_signals))
                )
        return self


def stable_trigger_report_id(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"scientific_reframe_trigger:{digest}"
