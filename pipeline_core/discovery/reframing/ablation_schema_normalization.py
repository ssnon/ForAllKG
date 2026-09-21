from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.ablation_evaluation import (
    AblationDimensionSpec,
    BlindAblationArm,
    BlindAblationComparison,
    BlindPrediction,
    BlindScientificCandidate,
    ScientificReasoningAblationPacket,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateNormalizationStats(StrictModel):
    candidate_alias: str
    baseline_model_removed: bool
    alternative_or_resolution_model_removed: bool
    dedicated_discriminating_test_removed: bool
    unresolved_question_count_removed: int = Field(ge=0)
    source_prediction_count: int = Field(ge=0)
    normalized_prediction_count: int = Field(ge=0)
    contrast_prediction_count_downprojected: int = Field(ge=0)
    relational_prediction_count_downprojected: int = Field(ge=0)


class ArmNormalizationStats(StrictModel):
    arm_alias: Literal["ARM_A", "ARM_B"]
    candidate_count: int = Field(ge=0)
    candidates: list[CandidateNormalizationStats]

    @model_validator(mode="after")
    def validate_count(self) -> "ArmNormalizationStats":
        if self.candidate_count != len(self.candidates):
            raise ValueError("normalization arm candidate_count mismatch")
        return self


class ComparisonNormalizationStats(StrictModel):
    comparison_alias: str
    arm_a: ArmNormalizationStats
    arm_b: ArmNormalizationStats
    candidate_count_difference_preserved: int = Field(ge=0)


class ScientificReasoningSchemaNormalizationReport(StrictModel):
    schema_version: Literal[
        "scientific-reasoning-ablation-schema-normalization-v1"
    ] = "scientific-reasoning-ablation-schema-normalization-v1"

    report_id: str
    source_packet_id: str
    source_packet_sha256: str = Field(min_length=64, max_length=64)
    normalized_packet_id: str
    normalized_packet_sha256: str = Field(min_length=64, max_length=64)
    comparisons: list[ComparisonNormalizationStats]

    llm_calls_performed: Literal[0] = 0
    deterministic_projection_only: Literal[True] = True
    source_candidate_identity_preserved_by_alias: Literal[True] = True
    blind_key_compatibility_preserved: Literal[True] = True
    source_conditions_remain_hidden: Literal[True] = True
    baseline_alternative_fields_removed_from_all_candidates: Literal[True] = True
    dedicated_discriminating_test_fields_removed_from_all_candidates: Literal[True] = True
    unresolved_question_fields_removed_from_all_candidates: Literal[True] = True
    prediction_schema_normalized_to_common_shape: Literal[True] = True
    schema_field_parity_targeted: Literal[True] = True
    candidate_count_parity_targeted: Literal[False] = False
    payload_length_parity_targeted: Literal[False] = False
    scientific_quality_judgment_performed: Literal[False] = False
    evaluator_judgments_modified: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical_json(parts).encode('utf-8')).hexdigest()[:20]}"



# Keep the dimension ordering explicit rather than depending on source-schema internals.
_NORMALIZED_DIMENSION_ORDER = (
    "task_relevance_and_coverage",
    "evidence_discipline",
    "explanatory_gain",
    "differential_prediction_quality",
    "falsifiability",
    "discriminating_experiment_quality",
    "portfolio_complementarity",
)

_NORMALIZED_DIMENSION_QUESTIONS = {
    "task_relevance_and_coverage": (
        "How directly and comprehensively does the portfolio address the stated scientific task "
        "without drifting to adjacent questions?"
    ),
    "evidence_discipline": (
        "Are the proposal, rationale, predictions, and falsifiers proportionate to the supplied "
        "evidence, with unresolved points remaining hypothetical?"
    ),
    "explanatory_gain": (
        "Does the common-denominator proposal and rationale add a non-trivial explanatory "
        "representation rather than merely restating or renaming observations?"
    ),
    "differential_prediction_quality": (
        "Using only the common-denominator prediction and falsifier fields shown for both arms, do "
        "the proposed observable outcomes materially help distinguish plausible explanations?"
    ),
    "falsifiability": (
        "Are there credible observable outcomes that would count against the proposed scientific "
        "explanations?"
    ),
    "discriminating_experiment_quality": (
        "Using only the common-denominator proposal, predictions, and falsifiers shown for both arms, "
        "is there a credible testable contrast that could distinguish explanations? Do not penalize "
        "either arm for the absence of a dedicated test-design field; that field was intentionally "
        "removed from every candidate."
    ),
    "portfolio_complementarity": (
        "Do the candidates contribute complementary scientific lines of reasoning rather than surface "
        "paraphrases of the same proposal?"
    ),
}


def normalized_dimension_specs() -> list[AblationDimensionSpec]:
    return [
        AblationDimensionSpec(
            dimension_id=dimension_id,
            question=_NORMALIZED_DIMENSION_QUESTIONS[dimension_id],
        )
        for dimension_id in _NORMALIZED_DIMENSION_ORDER
    ]


def _normalize_prediction(row: BlindPrediction) -> tuple[BlindPrediction, str]:
    if row.expected_direction is not None and row.rationale is not None:
        summary = row.rationale.strip()
        direction = row.expected_direction.strip()
        if direction and direction != "unspecified":
            summary = f"Expected qualitative direction: {direction}. {summary}"
        return (
            BlindPrediction(
                observable=row.observable,
                expected_direction="unspecified",
                rationale=summary,
            ),
            "relational",
        )

    if row.alternative_expectation is None:
        raise ValueError("contrast prediction lacks alternative expectation")
    return (
        BlindPrediction(
            observable=row.observable,
            expected_direction="unspecified",
            rationale=row.alternative_expectation.strip(),
        ),
        "contrast",
    )


def _normalize_candidate(
    candidate: BlindScientificCandidate,
) -> tuple[BlindScientificCandidate, CandidateNormalizationStats]:
    predictions: list[BlindPrediction] = []
    relational_count = 0
    contrast_count = 0
    for row in candidate.predictions:
        normalized, kind = _normalize_prediction(row)
        predictions.append(normalized)
        if kind == "relational":
            relational_count += 1
        else:
            contrast_count += 1

    normalized = candidate.model_copy(
        update={
            "baseline_model_summary": None,
            "alternative_or_resolution_summary": None,
            "predictions": predictions,
            "discriminating_test": None,
            "unresolved_questions": [],
        }
    )
    # Revalidate so the output remains consumable by the unchanged 0036 evaluator.
    normalized = BlindScientificCandidate.model_validate(
        normalized.model_dump(mode="python")
    )
    return normalized, CandidateNormalizationStats(
        candidate_alias=candidate.candidate_alias,
        baseline_model_removed=candidate.baseline_model_summary is not None,
        alternative_or_resolution_model_removed=(
            candidate.alternative_or_resolution_summary is not None
        ),
        dedicated_discriminating_test_removed=candidate.discriminating_test is not None,
        unresolved_question_count_removed=len(candidate.unresolved_questions),
        source_prediction_count=len(candidate.predictions),
        normalized_prediction_count=len(predictions),
        contrast_prediction_count_downprojected=contrast_count,
        relational_prediction_count_downprojected=relational_count,
    )


def _normalize_arm(
    arm: BlindAblationArm,
) -> tuple[BlindAblationArm, ArmNormalizationStats]:
    candidates: list[BlindScientificCandidate] = []
    stats: list[CandidateNormalizationStats] = []
    for candidate in arm.candidates:
        normalized, row = _normalize_candidate(candidate)
        candidates.append(normalized)
        stats.append(row)
    normalized_arm = arm.model_copy(update={"candidates": candidates})
    normalized_arm = BlindAblationArm.model_validate(
        normalized_arm.model_dump(mode="python")
    )
    return normalized_arm, ArmNormalizationStats(
        arm_alias=arm.arm_alias,
        candidate_count=len(candidates),
        candidates=stats,
    )


def build_schema_normalized_ablation_packet(
    packet: ScientificReasoningAblationPacket,
) -> tuple[ScientificReasoningAblationPacket, ScientificReasoningSchemaNormalizationReport]:
    comparisons: list[BlindAblationComparison] = []
    comparison_stats: list[ComparisonNormalizationStats] = []
    for comparison in packet.comparisons:
        arm_a, stats_a = _normalize_arm(comparison.arm_a)
        arm_b, stats_b = _normalize_arm(comparison.arm_b)
        cautions = list(comparison.interpretation_cautions)
        extra_caution = (
            "Schema-normalized common-denominator view: explicit baseline/alternative model fields, "
            "dedicated discriminating-test fields, unresolved-question fields, and contrast-specific "
            "prediction subfields were removed from every candidate. Candidate-count and payload-size "
            "asymmetries may still remain."
        )
        if extra_caution not in cautions:
            cautions.append(extra_caution)
        normalized_comparison = comparison.model_copy(
            update={
                "arm_a": arm_a,
                "arm_b": arm_b,
                "interpretation_cautions": cautions,
            }
        )
        normalized_comparison = BlindAblationComparison.model_validate(
            normalized_comparison.model_dump(mode="python")
        )
        comparisons.append(normalized_comparison)
        comparison_stats.append(
            ComparisonNormalizationStats(
                comparison_alias=comparison.comparison_alias,
                arm_a=stats_a,
                arm_b=stats_b,
                candidate_count_difference_preserved=(
                    comparison.absolute_candidate_count_difference
                ),
            )
        )

    normalized_packet = packet.model_copy(
        update={
            "evaluation_dimensions": normalized_dimension_specs(),
            "comparisons": comparisons,
        }
    )
    normalized_packet = ScientificReasoningAblationPacket.model_validate(
        normalized_packet.model_dump(mode="python")
    )

    source_sha = _sha256(packet.model_dump(mode="json"))
    normalized_sha = _sha256(normalized_packet.model_dump(mode="json"))
    report = ScientificReasoningSchemaNormalizationReport(
        report_id=_stable_id(
            "scientific_reasoning_ablation_schema_normalization",
            packet.packet_id,
            source_sha,
            normalized_sha,
        ),
        source_packet_id=packet.packet_id,
        source_packet_sha256=source_sha,
        normalized_packet_id=normalized_packet.packet_id,
        normalized_packet_sha256=normalized_sha,
        comparisons=comparison_stats,
    )
    return normalized_packet, report
