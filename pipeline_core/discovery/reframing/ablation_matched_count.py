from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.ablation_evaluation import (
    AblationArmKey,
    AblationComparisonKey,
    AblationCondition,
    BlindAblationArm,
    BlindAblationComparison,
    BlindScientificCandidate,
    ScientificReasoningAblationBlindKey,
    ScientificReasoningAblationPacket,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MatchedCountComparisonBuild(StrictModel):
    comparison_alias: str
    source_comparison_alias: str
    omitted_reframing_candidate_alias: str
    omitted_reframing_source_candidate_id: str
    relational_candidate_count: int = Field(ge=1)
    reframing_candidate_count: int = Field(ge=1)
    candidate_count_equal: Literal[True] = True
    arm_a_condition: AblationCondition
    arm_b_condition: AblationCondition
    arm_a_payload_characters: int = Field(ge=1)
    arm_b_payload_characters: int = Field(ge=1)
    payload_character_ratio: float = Field(ge=1.0)

    @model_validator(mode="after")
    def validate_counts(self) -> "MatchedCountComparisonBuild":
        if self.relational_candidate_count != self.reframing_candidate_count:
            raise ValueError("matched-count comparison must use equal candidate counts")
        return self


class ScientificReasoningMatchedCountBuildReport(StrictModel):
    schema_version: Literal[
        "scientific-reasoning-ablation-matched-count-build-v1"
    ] = "scientific-reasoning-ablation-matched-count-build-v1"

    report_id: str
    source_packet_id: str
    source_packet_sha256: str = Field(min_length=64, max_length=64)
    source_key_packet_id: str
    matched_packet_id: str
    matched_packet_sha256: str = Field(min_length=64, max_length=64)
    comparisons: list[MatchedCountComparisonBuild]

    source_relational_candidate_count: int = Field(ge=1)
    source_reframing_candidate_count: int = Field(ge=1)
    matched_candidate_count_per_arm: int = Field(ge=1)
    leave_one_out_comparison_count: int = Field(ge=1)

    llm_calls_performed: Literal[0] = 0
    deterministic_subset_construction_only: Literal[True] = True
    source_packet_reused_without_mutation: Literal[True] = True
    schema_normalized_input_required: Literal[True] = True
    candidate_count_parity_established: Literal[True] = True
    reframing_leave_one_out_coverage_complete: Literal[True] = True
    relational_candidates_reused_in_every_comparison: Literal[True] = True
    payload_length_parity_targeted: Literal[False] = False
    judge_replication_performed: Literal[False] = False
    multi_task_validation_performed: Literal[False] = False
    source_condition_labels_hidden_from_evaluator_packet: Literal[True] = True
    blind_key_must_not_be_supplied_to_evaluator: Literal[True] = True
    scientific_quality_judgment_performed: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "ScientificReasoningMatchedCountBuildReport":
        if self.leave_one_out_comparison_count != len(self.comparisons):
            raise ValueError("leave_one_out_comparison_count mismatch")
        if self.source_reframing_candidate_count != self.source_relational_candidate_count + 1:
            raise ValueError(
                "matched-count v1 expects reframing count to equal relational count plus one"
            )
        if self.matched_candidate_count_per_arm != self.source_relational_candidate_count:
            raise ValueError("matched count must equal relational source candidate count")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    payload = _canonical_json(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _payload_characters(candidates: list[BlindScientificCandidate]) -> int:
    return sum(len(_canonical_json(row.model_dump(mode="json"))) for row in candidates)


def _arm_key_for_condition(
    comparison: AblationComparisonKey,
    condition: AblationCondition,
) -> AblationArmKey:
    matches = [row for row in comparison.arms if row.condition == condition]
    if len(matches) != 1:
        raise ValueError(
            f"source key must map condition {condition} exactly once in "
            f"{comparison.comparison_alias}"
        )
    return matches[0]


def _arm_for_alias(
    comparison: BlindAblationComparison,
    arm_alias: str,
) -> BlindAblationArm:
    if arm_alias == "ARM_A":
        return comparison.arm_a
    if arm_alias == "ARM_B":
        return comparison.arm_b
    raise ValueError(f"unsupported arm alias: {arm_alias}")


def _condition_arm(
    *,
    packet_comparison: BlindAblationComparison,
    key_comparison: AblationComparisonKey,
    condition: AblationCondition,
) -> tuple[BlindAblationArm, AblationArmKey]:
    key_arm = _arm_key_for_condition(key_comparison, condition)
    return _arm_for_alias(packet_comparison, key_arm.arm_alias), key_arm


def _assert_schema_normalized(packet: ScientificReasoningAblationPacket) -> None:
    for comparison in packet.comparisons:
        for arm in (comparison.arm_a, comparison.arm_b):
            for candidate in arm.candidates:
                if candidate.baseline_model_summary is not None:
                    raise ValueError(
                        "matched-count experiment requires schema-normalized packet: "
                        "baseline_model_summary remains present"
                    )
                if candidate.alternative_or_resolution_summary is not None:
                    raise ValueError(
                        "matched-count experiment requires schema-normalized packet: "
                        "alternative_or_resolution_summary remains present"
                    )
                if candidate.discriminating_test is not None:
                    raise ValueError(
                        "matched-count experiment requires schema-normalized packet: "
                        "dedicated discriminating_test remains present"
                    )
                if candidate.unresolved_questions:
                    raise ValueError(
                        "matched-count experiment requires schema-normalized packet: "
                        "unresolved_questions remain present"
                    )
                for prediction in candidate.predictions:
                    if prediction.expected_direction is None or prediction.rationale is None:
                        raise ValueError(
                            "matched-count experiment requires common-shape normalized predictions"
                        )
                    if any(
                        value is not None
                        for value in (
                            prediction.baseline_expectation,
                            prediction.alternative_expectation,
                            prediction.discriminating_outcome,
                        )
                    ):
                        raise ValueError(
                            "matched-count experiment requires contrast-specific prediction fields "
                            "to be removed"
                        )


def _condition_goes_first(packet_id: str, omitted_alias: str) -> bool:
    digest = hashlib.sha256(f"{packet_id}|{omitted_alias}|arm-order-v1".encode("utf-8")).digest()
    return bool(digest[0] & 1)


def _comparison_cautions(omitted_alias: str) -> list[str]:
    return [
        "Candidate counts are exactly matched across arms for this comparison.",
        "The reframing arm is a deterministic leave-one-out subset; one blinded reframing candidate "
        f"is omitted for this replicate ({omitted_alias} in the source blind packet).",
        "Schema-specific baseline/alternative/test fields were already removed by the source "
        "schema-normalization step.",
        "Payload length is recorded but is not forcibly equalized; do not infer quality from response "
        "length or verbosity.",
        "Treat each comparison as an independent robustness probe rather than a vote toward an overall "
        "scientific winner.",
    ]


def build_matched_count_ablation_packet(
    *,
    packet: ScientificReasoningAblationPacket,
    key: ScientificReasoningAblationBlindKey,
) -> tuple[
    ScientificReasoningAblationPacket,
    ScientificReasoningAblationBlindKey,
    ScientificReasoningMatchedCountBuildReport,
]:
    if key.packet_id != packet.packet_id:
        raise ValueError("blind key packet_id mismatch")
    _assert_schema_normalized(packet)

    source_key_rows = [
        row for row in key.comparisons if row.comparison_kind == "RELATIONAL_VS_REFRAMING"
    ]
    if len(source_key_rows) != 1:
        raise ValueError(
            "matched-count v1 requires exactly one RELATIONAL_VS_REFRAMING source comparison"
        )
    source_key = source_key_rows[0]
    packet_by_alias = {row.comparison_alias: row for row in packet.comparisons}
    if source_key.comparison_alias not in packet_by_alias:
        raise ValueError("source RELATIONAL_VS_REFRAMING comparison absent from packet")
    source_comparison = packet_by_alias[source_key.comparison_alias]

    relational_arm, relational_key_arm = _condition_arm(
        packet_comparison=source_comparison,
        key_comparison=source_key,
        condition="RELATIONAL_ONLY",
    )
    reframing_arm, reframing_key_arm = _condition_arm(
        packet_comparison=source_comparison,
        key_comparison=source_key,
        condition="REFRAMING_ONLY",
    )

    relational_count = len(relational_arm.candidates)
    reframing_count = len(reframing_arm.candidates)
    if relational_count < 1:
        raise ValueError("matched-count experiment requires relational candidates")
    if reframing_count != relational_count + 1:
        raise ValueError(
            "matched-count v1 requires reframing candidate count to equal relational count plus one; "
            f"got relational={relational_count}, reframing={reframing_count}"
        )

    missing_relational_keys = sorted(
        set(row.candidate_alias for row in relational_arm.candidates)
        - set(relational_key_arm.candidate_alias_to_source_id)
    )
    missing_reframing_keys = sorted(
        set(row.candidate_alias for row in reframing_arm.candidates)
        - set(reframing_key_arm.candidate_alias_to_source_id)
    )
    if missing_relational_keys or missing_reframing_keys:
        raise ValueError(
            "blind key candidate mapping incomplete; "
            f"relational_missing={missing_relational_keys}, "
            f"reframing_missing={missing_reframing_keys}"
        )

    omitted_aliases = [row.candidate_alias for row in reframing_arm.candidates]
    matched_packet_id = _stable_id(
        "scientific_reasoning_ablation_matched_count_packet",
        packet.packet_id,
        *omitted_aliases,
    )

    comparisons: list[BlindAblationComparison] = []
    comparison_keys: list[AblationComparisonKey] = []
    build_rows: list[MatchedCountComparisonBuild] = []

    for index, omitted in enumerate(reframing_arm.candidates, start=1):
        comparison_alias = f"COMPARISON_{index:02d}"
        reframing_subset = [
            row for row in reframing_arm.candidates if row.candidate_alias != omitted.candidate_alias
        ]
        if len(reframing_subset) != relational_count:
            raise ValueError("leave-one-out subset failed to achieve candidate-count parity")

        relational_candidates = list(relational_arm.candidates)
        reframing_candidates = list(reframing_subset)
        relational_first = _condition_goes_first(packet.packet_id, omitted.candidate_alias)
        if relational_first:
            arm_a_condition: AblationCondition = "RELATIONAL_ONLY"
            arm_b_condition: AblationCondition = "REFRAMING_ONLY"
            arm_a_candidates = relational_candidates
            arm_b_candidates = reframing_candidates
            arm_a_source_map = relational_key_arm.candidate_alias_to_source_id
            arm_b_source_map = reframing_key_arm.candidate_alias_to_source_id
        else:
            arm_a_condition = "REFRAMING_ONLY"
            arm_b_condition = "RELATIONAL_ONLY"
            arm_a_candidates = reframing_candidates
            arm_b_candidates = relational_candidates
            arm_a_source_map = reframing_key_arm.candidate_alias_to_source_id
            arm_b_source_map = relational_key_arm.candidate_alias_to_source_id

        arm_a = BlindAblationArm(
            arm_alias="ARM_A",
            candidates=arm_a_candidates,
            candidate_count=len(arm_a_candidates),
        )
        arm_b = BlindAblationArm(
            arm_alias="ARM_B",
            candidates=arm_b_candidates,
            candidate_count=len(arm_b_candidates),
        )
        comparison_id = _stable_id(
            "scientific_reasoning_ablation_matched_count_comparison",
            packet.packet_id,
            omitted.candidate_alias,
        )
        comparisons.append(
            BlindAblationComparison(
                comparison_id=comparison_id,
                comparison_alias=comparison_alias,
                arm_a=arm_a,
                arm_b=arm_b,
                candidate_count_equal=True,
                absolute_candidate_count_difference=0,
                interpretation_cautions=_comparison_cautions(omitted.candidate_alias),
            )
        )

        arm_a_mapping = {
            row.candidate_alias: arm_a_source_map[row.candidate_alias]
            for row in arm_a_candidates
        }
        arm_b_mapping = {
            row.candidate_alias: arm_b_source_map[row.candidate_alias]
            for row in arm_b_candidates
        }
        comparison_keys.append(
            AblationComparisonKey(
                comparison_id=comparison_id,
                comparison_alias=comparison_alias,
                comparison_kind="RELATIONAL_VS_REFRAMING",
                arms=[
                    AblationArmKey(
                        arm_alias="ARM_A",
                        condition=arm_a_condition,
                        candidate_alias_to_source_id=arm_a_mapping,
                    ),
                    AblationArmKey(
                        arm_alias="ARM_B",
                        condition=arm_b_condition,
                        candidate_alias_to_source_id=arm_b_mapping,
                    ),
                ],
            )
        )

        arm_a_payload = _payload_characters(arm_a_candidates)
        arm_b_payload = _payload_characters(arm_b_candidates)
        payload_ratio = max(arm_a_payload, arm_b_payload) / min(arm_a_payload, arm_b_payload)
        build_rows.append(
            MatchedCountComparisonBuild(
                comparison_alias=comparison_alias,
                source_comparison_alias=source_comparison.comparison_alias,
                omitted_reframing_candidate_alias=omitted.candidate_alias,
                omitted_reframing_source_candidate_id=(
                    reframing_key_arm.candidate_alias_to_source_id[omitted.candidate_alias]
                ),
                relational_candidate_count=relational_count,
                reframing_candidate_count=len(reframing_subset),
                arm_a_condition=arm_a_condition,
                arm_b_condition=arm_b_condition,
                arm_a_payload_characters=arm_a_payload,
                arm_b_payload_characters=arm_b_payload,
                payload_character_ratio=round(payload_ratio, 6),
            )
        )

    matched_packet = ScientificReasoningAblationPacket(
        packet_id=matched_packet_id,
        task_alias=packet.task_alias,
        question=packet.question,
        evidence_statements=packet.evidence_statements,
        evaluation_dimensions=packet.evaluation_dimensions,
        comparisons=comparisons,
        comparison_count=len(comparisons),
        evidence_statement_count=packet.evidence_statement_count,
        source_candidate_count=packet.source_candidate_count,
    )

    matched_key = key.model_copy(
        update={
            "packet_id": matched_packet.packet_id,
            "comparisons": comparison_keys,
        }
    )
    matched_key = ScientificReasoningAblationBlindKey.model_validate(
        matched_key.model_dump(mode="python")
    )

    source_packet_sha = _sha256(packet.model_dump(mode="json"))
    matched_packet_sha = _sha256(matched_packet.model_dump(mode="json"))
    report = ScientificReasoningMatchedCountBuildReport(
        report_id=_stable_id(
            "scientific_reasoning_ablation_matched_count_build",
            packet.packet_id,
            matched_packet.packet_id,
            *[
                f"{row.comparison_alias}:{row.omitted_reframing_source_candidate_id}"
                for row in build_rows
            ],
        ),
        source_packet_id=packet.packet_id,
        source_packet_sha256=source_packet_sha,
        source_key_packet_id=key.packet_id,
        matched_packet_id=matched_packet.packet_id,
        matched_packet_sha256=matched_packet_sha,
        comparisons=build_rows,
        source_relational_candidate_count=relational_count,
        source_reframing_candidate_count=reframing_count,
        matched_candidate_count_per_arm=relational_count,
        leave_one_out_comparison_count=len(build_rows),
    )
    return matched_packet, matched_key, report
