from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.ablation_blind_evaluator import (
    BlindScientificReasoningEvaluationReport,
    UnblindedScientificReasoningEvaluationReport,
)
from pipeline_core.discovery.reframing.ablation_evaluation import (
    AblationCondition,
    AblationDimensionId,
    BlindAblationArm,
    BlindScientificCandidate,
    ScientificReasoningAblationPacket,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


PresentationAsymmetryLevel = Literal[
    "low_observed_presentation_asymmetry",
    "moderate_presentation_asymmetry",
    "high_presentation_asymmetry",
]


class ArmPresentationProfile(StrictModel):
    arm_alias: Literal["ARM_A", "ARM_B"]
    condition: AblationCondition
    candidate_count: int = Field(ge=0)
    total_payload_characters: int = Field(ge=0)
    mean_payload_characters_per_candidate: float = Field(ge=0)
    candidates_with_explicit_baseline_model: int = Field(ge=0)
    candidates_with_explicit_alternative_or_resolution_model: int = Field(ge=0)
    candidates_with_dedicated_discriminating_test: int = Field(ge=0)
    candidates_with_unresolved_questions: int = Field(ge=0)
    relational_shape_prediction_count: int = Field(ge=0)
    contrast_shape_prediction_count: int = Field(ge=0)
    falsifier_count: int = Field(ge=0)

    explicit_baseline_fraction: float = Field(ge=0, le=1)
    explicit_alternative_fraction: float = Field(ge=0, le=1)
    dedicated_discriminating_test_fraction: float = Field(ge=0, le=1)
    unresolved_question_fraction: float = Field(ge=0, le=1)
    contrast_prediction_fraction: float = Field(ge=0, le=1)
    mean_falsifier_count_per_candidate: float = Field(ge=0)


class DimensionPresentationAudit(StrictModel):
    dimension_id: AblationDimensionId
    observed_preference: str
    asymmetry_level: PresentationAsymmetryLevel
    asymmetry_signals: list[str] = Field(default_factory=list)
    interpretation: str = Field(min_length=1)
    judgment_reversal_inferred: Literal[False] = False
    scientific_quality_conclusion_authorized: Literal[False] = False


class ComparisonPresentationValidityAudit(StrictModel):
    comparison_alias: str
    comparison_kind: str
    arm_a: ArmPresentationProfile
    arm_b: ArmPresentationProfile
    candidate_count_difference: int = Field(ge=0)
    payload_character_ratio: float = Field(ge=1)
    dimensions: list[DimensionPresentationAudit]
    high_asymmetry_dimension_count: int = Field(ge=0)
    moderate_asymmetry_dimension_count: int = Field(ge=0)
    low_asymmetry_dimension_count: int = Field(ge=0)

    presentation_parity_established: bool
    performance_superiority_established: Literal[False] = False


class ScientificReasoningAblationValidityAudit(StrictModel):
    schema_version: Literal[
        "scientific-reasoning-ablation-validity-audit-v1"
    ] = "scientific-reasoning-ablation-validity-audit-v1"

    audit_id: str
    source_packet_id: str
    source_blind_report_id: str
    source_unblinded_report_id: str
    judge_model_name: str
    comparison_audits: list[ComparisonPresentationValidityAudit]

    comparison_count: int = Field(ge=0)
    high_asymmetry_dimension_count: int = Field(ge=0)
    moderate_asymmetry_dimension_count: int = Field(ge=0)
    low_asymmetry_dimension_count: int = Field(ge=0)

    llm_calls_performed: Literal[0] = 0
    post_hoc_diagnostic_only: Literal[True] = True
    blind_evaluation_reused_unchanged: Literal[True] = True
    evaluator_judgments_modified: Literal[False] = False
    presentation_parity_established: bool
    schema_parity_established: bool
    candidate_count_parity_established: bool
    judge_replication_performed: Literal[False] = False
    multi_task_validation_performed: Literal[False] = False
    multi_model_validation_performed: Literal[False] = False
    scientific_reasoning_superiority_established: Literal[False] = False
    results_suitable_for_directional_signal_only: Literal[True] = True
    overall_score_computed: Literal[False] = False
    overall_winner_selected: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReasoningAblationValidityAudit":
        if self.comparison_count != len(self.comparison_audits):
            raise ValueError("comparison_count must match comparison_audits")
        rows = [d for c in self.comparison_audits for d in c.dimensions]
        high = sum(d.asymmetry_level == "high_presentation_asymmetry" for d in rows)
        moderate = sum(
            d.asymmetry_level == "moderate_presentation_asymmetry" for d in rows
        )
        low = sum(
            d.asymmetry_level == "low_observed_presentation_asymmetry" for d in rows
        )
        if (high, moderate, low) != (
            self.high_asymmetry_dimension_count,
            self.moderate_asymmetry_dimension_count,
            self.low_asymmetry_dimension_count,
        ):
            raise ValueError("dimension asymmetry counts do not match comparison audits")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(_canonical_json(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _candidate_payload_characters(candidate: BlindScientificCandidate) -> int:
    payload = candidate.model_dump(
        mode="json",
        exclude={
            "candidate_alias",
            "source_candidate_identity_hidden",
            "source_lane_identity_hidden",
            "source_operator_identity_hidden",
            "source_schema_identity_hidden",
        },
    )
    return len(_canonical_json(payload))


def _arm_profile(
    *,
    arm: BlindAblationArm,
    condition: AblationCondition,
) -> ArmPresentationProfile:
    candidates = list(arm.candidates)
    count = len(candidates)
    payload_chars = sum(_candidate_payload_characters(c) for c in candidates)
    baseline = sum(c.baseline_model_summary is not None for c in candidates)
    alternative = sum(
        c.alternative_or_resolution_summary is not None for c in candidates
    )
    tests = sum(c.discriminating_test is not None for c in candidates)
    unresolved = sum(bool(c.unresolved_questions) for c in candidates)
    relational_predictions = 0
    contrast_predictions = 0
    falsifiers = 0
    for candidate in candidates:
        falsifiers += len(candidate.falsifiers)
        for prediction in candidate.predictions:
            if prediction.expected_direction is not None and prediction.rationale is not None:
                relational_predictions += 1
            if (
                prediction.baseline_expectation is not None
                and prediction.alternative_expectation is not None
                and prediction.discriminating_outcome is not None
            ):
                contrast_predictions += 1
    total_predictions = relational_predictions + contrast_predictions

    def fraction(value: int) -> float:
        return round(value / count, 6) if count else 0.0

    return ArmPresentationProfile(
        arm_alias=arm.arm_alias,
        condition=condition,
        candidate_count=count,
        total_payload_characters=payload_chars,
        mean_payload_characters_per_candidate=round(payload_chars / count, 3)
        if count
        else 0.0,
        candidates_with_explicit_baseline_model=baseline,
        candidates_with_explicit_alternative_or_resolution_model=alternative,
        candidates_with_dedicated_discriminating_test=tests,
        candidates_with_unresolved_questions=unresolved,
        relational_shape_prediction_count=relational_predictions,
        contrast_shape_prediction_count=contrast_predictions,
        falsifier_count=falsifiers,
        explicit_baseline_fraction=fraction(baseline),
        explicit_alternative_fraction=fraction(alternative),
        dedicated_discriminating_test_fraction=fraction(tests),
        unresolved_question_fraction=fraction(unresolved),
        contrast_prediction_fraction=(
            round(contrast_predictions / total_predictions, 6)
            if total_predictions
            else 0.0
        ),
        mean_falsifier_count_per_candidate=round(falsifiers / count, 3)
        if count
        else 0.0,
    )


def _ratio(a: int, b: int) -> float:
    if a == b == 0:
        return 1.0
    small = max(1, min(a, b))
    return round(max(a, b) / small, 6)


def _dimension_signals(
    *,
    dimension_id: AblationDimensionId,
    a: ArmPresentationProfile,
    b: ArmPresentationProfile,
) -> tuple[PresentationAsymmetryLevel, list[str], str]:
    signals: list[str] = []
    high = False
    moderate = False

    count_diff = abs(a.candidate_count - b.candidate_count)
    payload_ratio = _ratio(a.total_payload_characters, b.total_payload_characters)
    baseline_gap = abs(a.explicit_baseline_fraction - b.explicit_baseline_fraction)
    alt_gap = abs(a.explicit_alternative_fraction - b.explicit_alternative_fraction)
    test_gap = abs(
        a.dedicated_discriminating_test_fraction
        - b.dedicated_discriminating_test_fraction
    )
    prediction_gap = abs(a.contrast_prediction_fraction - b.contrast_prediction_fraction)
    unresolved_gap = abs(a.unresolved_question_fraction - b.unresolved_question_fraction)
    falsifier_gap = abs(
        a.mean_falsifier_count_per_candidate - b.mean_falsifier_count_per_candidate
    )

    if dimension_id in {"task_relevance_and_coverage", "portfolio_complementarity"}:
        if count_diff:
            signals.append("candidate_count_asymmetry")
            moderate = True
        if payload_ratio >= 1.5:
            signals.append("portfolio_payload_volume_asymmetry")
            moderate = True

    if dimension_id == "evidence_discipline":
        if unresolved_gap >= 0.5:
            signals.append("explicit_uncertainty_field_asymmetry")
            moderate = True
        if payload_ratio >= 1.75:
            signals.append("qualification_payload_asymmetry")
            moderate = True

    if dimension_id == "explanatory_gain":
        if baseline_gap >= 0.5 or alt_gap >= 0.5:
            signals.append("explicit_model_contrast_schema_asymmetry")
            high = True
        if count_diff:
            signals.append("candidate_count_asymmetry")
            moderate = True
        if payload_ratio >= 1.5:
            signals.append("explanatory_payload_volume_asymmetry")
            moderate = True

    if dimension_id == "differential_prediction_quality":
        if prediction_gap >= 0.5:
            signals.append("prediction_schema_asymmetry")
            high = True
        if baseline_gap >= 0.5:
            signals.append("baseline_alternative_exposure_asymmetry")
            high = True

    if dimension_id == "falsifiability":
        if falsifier_gap >= 1.0:
            signals.append("falsifier_volume_asymmetry")
            moderate = True
        if payload_ratio >= 2.0:
            signals.append("falsifier_context_payload_asymmetry")
            moderate = True

    if dimension_id == "discriminating_experiment_quality":
        if test_gap >= 0.5:
            signals.append("dedicated_test_schema_asymmetry")
            high = True
        if prediction_gap >= 0.5:
            signals.append("prediction_discrimination_schema_asymmetry")
            high = True

    if high:
        level: PresentationAsymmetryLevel = "high_presentation_asymmetry"
        interpretation = (
            "This dimension is materially exposed to source-schema/presentation differences. "
            "The observed blind preference is retained, but it should not be interpreted as "
            "a clean estimate of reasoning quality for this dimension."
        )
    elif moderate:
        level = "moderate_presentation_asymmetry"
        interpretation = (
            "This dimension has a meaningful presentation or portfolio-size asymmetry. "
            "The blind preference is a directional signal, not a schema-controlled performance result."
        )
    else:
        level = "low_observed_presentation_asymmetry"
        interpretation = (
            "No large deterministic presentation asymmetry was detected for the audited fields. "
            "This does not establish full evaluation parity or scientific superiority."
        )
    return level, signals, interpretation


def audit_scientific_reasoning_ablation_validity(
    *,
    packet: ScientificReasoningAblationPacket,
    blind_report: BlindScientificReasoningEvaluationReport,
    unblinded_report: UnblindedScientificReasoningEvaluationReport,
) -> ScientificReasoningAblationValidityAudit:
    if blind_report.source_packet_id != packet.packet_id:
        raise ValueError("blind report source_packet_id mismatch")
    if unblinded_report.source_packet_id != packet.packet_id:
        raise ValueError("unblinded report source_packet_id mismatch")
    if unblinded_report.source_blind_report_id != blind_report.report_id:
        raise ValueError("unblinded report source_blind_report_id mismatch")

    packet_by_alias = {c.comparison_alias: c for c in packet.comparisons}
    blind_by_alias = {
        c.comparison_alias: c for c in blind_report.comparison_evaluations
    }
    unblinded_by_alias = {c.comparison_alias: c for c in unblinded_report.comparisons}
    if set(packet_by_alias) != set(blind_by_alias) or set(packet_by_alias) != set(
        unblinded_by_alias
    ):
        raise ValueError("packet/blind/unblinded comparison alias mismatch")

    audits: list[ComparisonPresentationValidityAudit] = []
    all_count_equal = True
    any_schema_asymmetry = False
    for alias in [c.comparison_alias for c in packet.comparisons]:
        comparison = packet_by_alias[alias]
        blind = blind_by_alias[alias]
        unblinded = unblinded_by_alias[alias]
        if unblinded.arm_a_condition == unblinded.arm_b_condition:
            raise ValueError("unblinded comparison maps both arms to the same condition")
        blind_judgments = {d.dimension_id: d for d in blind.dimensions}
        for row in unblinded.dimensions:
            source = blind_judgments.get(row.dimension_id)
            if source is None:
                raise ValueError(
                    f"blind comparison {alias} missing dimension {row.dimension_id}"
                )
            if row.blind_preference != source.preference:
                raise ValueError(
                    f"unblinded comparison {alias} changed blind preference for "
                    f"{row.dimension_id}"
                )
            expected = (
                unblinded.arm_a_condition
                if source.preference == "ARM_A"
                else unblinded.arm_b_condition
                if source.preference == "ARM_B"
                else source.preference
            )
            if row.preferred_condition != expected:
                raise ValueError(
                    f"unblinded comparison {alias} condition mapping mismatch for "
                    f"{row.dimension_id}"
                )
        arm_a = _arm_profile(arm=comparison.arm_a, condition=unblinded.arm_a_condition)
        arm_b = _arm_profile(arm=comparison.arm_b, condition=unblinded.arm_b_condition)
        all_count_equal = all_count_equal and arm_a.candidate_count == arm_b.candidate_count

        judgments = {d.dimension_id: d for d in unblinded.dimensions}
        dimension_rows: list[DimensionPresentationAudit] = []
        for spec in packet.evaluation_dimensions:
            judgment = judgments.get(spec.dimension_id)
            if judgment is None:
                raise ValueError(
                    f"unblinded comparison {alias} missing dimension {spec.dimension_id}"
                )
            level, signals, interpretation = _dimension_signals(
                dimension_id=spec.dimension_id,
                a=arm_a,
                b=arm_b,
            )
            if any("schema_asymmetry" in signal for signal in signals):
                any_schema_asymmetry = True
            dimension_rows.append(
                DimensionPresentationAudit(
                    dimension_id=spec.dimension_id,
                    observed_preference=judgment.preferred_condition,
                    asymmetry_level=level,
                    asymmetry_signals=signals,
                    interpretation=interpretation,
                )
            )

        levels = [row.asymmetry_level for row in dimension_rows]
        audits.append(
            ComparisonPresentationValidityAudit(
                comparison_alias=alias,
                comparison_kind=unblinded.comparison_kind,
                arm_a=arm_a,
                arm_b=arm_b,
                candidate_count_difference=abs(
                    arm_a.candidate_count - arm_b.candidate_count
                ),
                payload_character_ratio=_ratio(
                    arm_a.total_payload_characters, arm_b.total_payload_characters
                ),
                dimensions=dimension_rows,
                high_asymmetry_dimension_count=sum(
                    level == "high_presentation_asymmetry" for level in levels
                ),
                moderate_asymmetry_dimension_count=sum(
                    level == "moderate_presentation_asymmetry" for level in levels
                ),
                low_asymmetry_dimension_count=sum(
                    level == "low_observed_presentation_asymmetry" for level in levels
                ),
                presentation_parity_established=all(
                    level == "low_observed_presentation_asymmetry" for level in levels
                ),
            )
        )

    rows = [d for c in audits for d in c.dimensions]
    high = sum(d.asymmetry_level == "high_presentation_asymmetry" for d in rows)
    moderate = sum(
        d.asymmetry_level == "moderate_presentation_asymmetry" for d in rows
    )
    low = sum(
        d.asymmetry_level == "low_observed_presentation_asymmetry" for d in rows
    )
    parity = high == 0 and moderate == 0
    return ScientificReasoningAblationValidityAudit(
        audit_id=_stable_id(
            "scientific_reasoning_ablation_validity_audit",
            packet.packet_id,
            blind_report.report_id,
            unblinded_report.report_id,
        ),
        source_packet_id=packet.packet_id,
        source_blind_report_id=blind_report.report_id,
        source_unblinded_report_id=unblinded_report.report_id,
        judge_model_name=blind_report.model_name,
        comparison_audits=audits,
        comparison_count=len(audits),
        high_asymmetry_dimension_count=high,
        moderate_asymmetry_dimension_count=moderate,
        low_asymmetry_dimension_count=low,
        presentation_parity_established=parity,
        schema_parity_established=not any_schema_asymmetry,
        candidate_count_parity_established=all_count_equal,
    )
