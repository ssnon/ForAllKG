from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
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


class GeneralizedMatchedComparisonBuild(StrictModel):
    comparison_alias: str
    source_comparison_alias: str
    matched_candidate_count: int = Field(ge=1)
    arm_a_condition: AblationCondition
    arm_b_condition: AblationCondition
    arm_a_source_candidate_ids: list[str] = Field(min_length=1)
    arm_b_source_candidate_ids: list[str] = Field(min_length=1)
    omitted_source_candidate_ids: dict[str, list[str]] = Field(default_factory=dict)
    arm_a_payload_characters: int = Field(ge=1)
    arm_b_payload_characters: int = Field(ge=1)
    payload_character_ratio: float = Field(ge=1.0)
    candidate_count_equal: Literal[True] = True

    @model_validator(mode="after")
    def validate_counts(self) -> "GeneralizedMatchedComparisonBuild":
        if len(self.arm_a_source_candidate_ids) != self.matched_candidate_count:
            raise ValueError("ARM_A source candidate count does not match matched count")
        if len(self.arm_b_source_candidate_ids) != self.matched_candidate_count:
            raise ValueError("ARM_B source candidate count does not match matched count")
        return self


class ScientificReasoningGeneralizedMatchedBuildReport(StrictModel):
    schema_version: Literal[
        "scientific-reasoning-ablation-generalized-matched-build-v1"
    ] = "scientific-reasoning-ablation-generalized-matched-build-v1"

    report_id: str
    source_packet_id: str
    source_packet_sha256: str = Field(min_length=64, max_length=64)
    source_key_packet_id: str
    matched_packet_id: str
    matched_packet_sha256: str = Field(min_length=64, max_length=64)
    source_candidate_counts: dict[str, int]
    matched_candidate_count_per_arm: int = Field(ge=1)
    total_possible_subset_comparisons: int = Field(ge=1)
    materialized_comparison_count: int = Field(ge=1)
    max_comparisons: int = Field(ge=1)
    subset_schedule_complete: bool
    candidate_exposure_counts: dict[str, dict[str, int]]
    comparisons: list[GeneralizedMatchedComparisonBuild]

    llm_calls_performed: Literal[0] = 0
    deterministic_subset_construction_only: Literal[True] = True
    schema_normalized_input_required: Literal[True] = True
    candidate_count_parity_established: Literal[True] = True
    blind_packet_condition_labels_hidden: Literal[True] = True
    blind_packet_source_ids_hidden: Literal[True] = True
    neutral_blind_cautions_required: Literal[True] = True
    source_packet_reused_without_mutation: Literal[True] = True
    payload_length_parity_targeted: Literal[False] = False
    judge_replication_performed: Literal[False] = False
    scientific_quality_judgment_performed: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "ScientificReasoningGeneralizedMatchedBuildReport":
        if self.materialized_comparison_count != len(self.comparisons):
            raise ValueError("materialized_comparison_count mismatch")
        if self.total_possible_subset_comparisons < self.materialized_comparison_count:
            raise ValueError("materialized comparisons exceed possible subset comparisons")
        if self.subset_schedule_complete != (
            self.total_possible_subset_comparisons == self.materialized_comparison_count
        ):
            raise ValueError("subset_schedule_complete inconsistent with comparison counts")
        return self


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256(_canonical_json(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _payload_characters(candidates: list[BlindScientificCandidate]) -> int:
    return sum(len(_canonical_json(row.model_dump(mode="json"))) for row in candidates)


def _source_comparison(
    packet: ScientificReasoningAblationPacket,
    key: ScientificReasoningAblationBlindKey,
) -> tuple[BlindAblationComparison, AblationComparisonKey]:
    key_rows = [
        row for row in key.comparisons if row.comparison_kind == "RELATIONAL_VS_REFRAMING"
    ]
    if len(key_rows) != 1:
        raise ValueError(
            "generalized matched-count v1 requires exactly one "
            "RELATIONAL_VS_REFRAMING source comparison"
        )
    key_row = key_rows[0]
    packet_rows = {
        row.comparison_alias: row for row in packet.comparisons
    }
    comparison = packet_rows.get(key_row.comparison_alias)
    if comparison is None:
        raise ValueError("source RELATIONAL_VS_REFRAMING comparison absent from packet")
    return comparison, key_row


def _key_arm(
    comparison: AblationComparisonKey,
    condition: AblationCondition,
) -> AblationArmKey:
    matches = [row for row in comparison.arms if row.condition == condition]
    if len(matches) != 1:
        raise ValueError(
            f"blind key must map {condition} exactly once in {comparison.comparison_alias}"
        )
    return matches[0]


def _packet_arm(
    comparison: BlindAblationComparison,
    arm_alias: str,
) -> BlindAblationArm:
    if arm_alias == "ARM_A":
        return comparison.arm_a
    if arm_alias == "ARM_B":
        return comparison.arm_b
    raise ValueError(f"unsupported arm alias {arm_alias!r}")


def _condition_arm(
    *,
    packet_comparison: BlindAblationComparison,
    key_comparison: AblationComparisonKey,
    condition: AblationCondition,
) -> tuple[BlindAblationArm, AblationArmKey]:
    key_arm = _key_arm(key_comparison, condition)
    return _packet_arm(packet_comparison, key_arm.arm_alias), key_arm


def _assert_schema_normalized(packet: ScientificReasoningAblationPacket) -> None:
    for comparison in packet.comparisons:
        for arm in (comparison.arm_a, comparison.arm_b):
            for candidate in arm.candidates:
                if candidate.baseline_model_summary is not None:
                    raise ValueError(
                        "generalized matched-count experiment requires schema-normalized packet: "
                        "baseline_model_summary remains present"
                    )
                if candidate.alternative_or_resolution_summary is not None:
                    raise ValueError(
                        "generalized matched-count experiment requires schema-normalized packet: "
                        "alternative_or_resolution_summary remains present"
                    )
                if candidate.discriminating_test is not None:
                    raise ValueError(
                        "generalized matched-count experiment requires schema-normalized packet: "
                        "dedicated discriminating_test remains present"
                    )
                if candidate.unresolved_questions:
                    raise ValueError(
                        "generalized matched-count experiment requires schema-normalized packet: "
                        "unresolved_questions remain present"
                    )
                for prediction in candidate.predictions:
                    if prediction.expected_direction is None or prediction.rationale is None:
                        raise ValueError(
                            "generalized matched-count experiment requires common-shape predictions"
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
                            "generalized matched-count experiment requires contrast-specific "
                            "prediction fields to be removed"
                        )


def _assert_candidate_mapping_complete(
    arm: BlindAblationArm,
    key_arm: AblationArmKey,
) -> None:
    aliases = {row.candidate_alias for row in arm.candidates}
    mapping = set(key_arm.candidate_alias_to_source_id)
    missing = sorted(aliases - mapping)
    if missing:
        raise ValueError(
            "blind key candidate mapping incomplete for source arm: " + ", ".join(missing)
        )


def _stable_subset_key(
    *,
    packet_id: str,
    condition: AblationCondition,
    aliases: tuple[str, ...],
) -> str:
    return hashlib.sha256(
        _canonical_json((packet_id, condition, aliases, "subset-order-v1")).encode("utf-8")
    ).hexdigest()


def _subsets(
    *,
    packet_id: str,
    condition: AblationCondition,
    candidates: list[BlindScientificCandidate],
    matched_count: int,
    max_comparisons: int,
) -> tuple[list[tuple[str, ...]], int, bool]:
    aliases = tuple(row.candidate_alias for row in candidates)
    if len(aliases) == matched_count:
        return [aliases], 1, True
    all_subsets = list(itertools.combinations(aliases, matched_count))
    total = len(all_subsets)
    ordered = sorted(
        all_subsets,
        key=lambda row: _stable_subset_key(
            packet_id=packet_id,
            condition=condition,
            aliases=row,
        ),
    )
    materialized = ordered[:max_comparisons]
    return materialized, total, len(materialized) == total


def _neutral_cautions(*, subset_schedule_complete: bool) -> list[str]:
    cautions = [
        "Candidate counts are exactly matched across the two anonymized arms.",
        "One anonymized source portfolio may be represented by a deterministic subset for this "
        "comparison; judge only the scientific content shown in the current arms.",
        "Source-schema-specific model and test fields were removed before this matched-count step.",
        "Payload length is recorded diagnostically but is not forcibly equalized; do not infer "
        "quality from response length or verbosity.",
        "Treat each comparison as an independent robustness probe; no overall winner is requested.",
    ]
    if not subset_schedule_complete:
        cautions.append(
            "The deterministic subset schedule is capped; this comparison is part of a sampled, "
            "not exhaustive, subset audit."
        )
    return cautions


def _assert_neutral_blind_packet(packet: ScientificReasoningAblationPacket) -> None:
    forbidden_serialized = (
        "RELATIONAL_ONLY",
        "REFRAMING_ONLY",
        "COMBINED",
        "RELATIONAL_DISCOVERY",
        "SCIENTIFIC_REFRAMING",
        "hypothesis:",
        "scientific_reframe:",
        "scientific_proxy_challenge:",
        "scientific_contradiction_resolution:",
    )
    payload = _canonical_json(packet.model_dump(mode="json"))
    leaked = [token for token in forbidden_serialized if token in payload]
    if leaked:
        raise ValueError(
            "blind matched-count packet leaks source condition or candidate identity: "
            + ", ".join(leaked)
        )
    forbidden_caution_words = ("relational", "reframing", "combined")
    for comparison in packet.comparisons:
        caution_text = " ".join(comparison.interpretation_cautions).lower()
        leaked_words = [word for word in forbidden_caution_words if word in caution_text]
        if leaked_words:
            raise ValueError(
                f"blind caution leaks source-condition semantics in {comparison.comparison_alias}: "
                + ", ".join(leaked_words)
            )


def _arm_order(packet_id: str, subset_signature: str) -> bool:
    digest = hashlib.sha256(
        f"{packet_id}|{subset_signature}|matched-arm-order-v2".encode("utf-8")
    ).digest()
    return bool(digest[0] & 1)


def build_generalized_matched_count_ablation_packet(
    *,
    packet: ScientificReasoningAblationPacket,
    key: ScientificReasoningAblationBlindKey,
    max_comparisons: int = 12,
) -> tuple[
    ScientificReasoningAblationPacket,
    ScientificReasoningAblationBlindKey,
    ScientificReasoningGeneralizedMatchedBuildReport,
]:
    if max_comparisons < 1:
        raise ValueError("max_comparisons must be >= 1")
    if key.packet_id != packet.packet_id:
        raise ValueError("blind key packet_id mismatch")
    _assert_schema_normalized(packet)

    source_comparison, source_key = _source_comparison(packet, key)
    relational_arm, relational_key = _condition_arm(
        packet_comparison=source_comparison,
        key_comparison=source_key,
        condition="RELATIONAL_ONLY",
    )
    reframing_arm, reframing_key = _condition_arm(
        packet_comparison=source_comparison,
        key_comparison=source_key,
        condition="REFRAMING_ONLY",
    )
    _assert_candidate_mapping_complete(relational_arm, relational_key)
    _assert_candidate_mapping_complete(reframing_arm, reframing_key)

    condition_arms = {
        "RELATIONAL_ONLY": (relational_arm, relational_key),
        "REFRAMING_ONLY": (reframing_arm, reframing_key),
    }
    source_counts = {
        condition: len(arm.candidates)
        for condition, (arm, _) in condition_arms.items()
    }
    if min(source_counts.values()) < 1:
        raise ValueError("generalized matched-count experiment requires candidates in both arms")
    matched_count = min(source_counts.values())

    subset_rows: dict[AblationCondition, list[tuple[str, ...]]] = {}
    possible_rows: dict[AblationCondition, int] = {}
    complete_rows: dict[AblationCondition, bool] = {}
    for condition, (arm, _) in condition_arms.items():
        subsets, possible, complete = _subsets(
            packet_id=packet.packet_id,
            condition=condition,
            candidates=list(arm.candidates),
            matched_count=matched_count,
            max_comparisons=max_comparisons,
        )
        subset_rows[condition] = subsets
        possible_rows[condition] = possible
        complete_rows[condition] = complete

    # Because matched_count is the minimum source cardinality, at most one condition has
    # multiple subsets. Equal-cardinality inputs therefore produce exactly one comparison.
    if len(subset_rows["RELATIONAL_ONLY"]) > 1 and len(subset_rows["REFRAMING_ONLY"]) > 1:
        raise ValueError("unexpected dual subset expansion for minimum-cardinality matching")
    if len(subset_rows["RELATIONAL_ONLY"]) > 1:
        schedule = [
            (row, subset_rows["REFRAMING_ONLY"][0])
            for row in subset_rows["RELATIONAL_ONLY"]
        ]
    elif len(subset_rows["REFRAMING_ONLY"]) > 1:
        schedule = [
            (subset_rows["RELATIONAL_ONLY"][0], row)
            for row in subset_rows["REFRAMING_ONLY"]
        ]
    else:
        schedule = [
            (subset_rows["RELATIONAL_ONLY"][0], subset_rows["REFRAMING_ONLY"][0])
        ]

    total_possible = max(
        possible_rows["RELATIONAL_ONLY"], possible_rows["REFRAMING_ONLY"]
    )
    schedule_complete = all(complete_rows.values())
    matched_packet_id = _stable_id(
        "scientific_reasoning_ablation_generalized_matched_packet",
        packet.packet_id,
        matched_count,
        total_possible,
        max_comparisons,
    )

    comparisons: list[BlindAblationComparison] = []
    comparison_keys: list[AblationComparisonKey] = []
    report_rows: list[GeneralizedMatchedComparisonBuild] = []
    exposure: dict[str, Counter[str]] = {
        "RELATIONAL_ONLY": Counter(),
        "REFRAMING_ONLY": Counter(),
    }

    relational_candidates = {row.candidate_alias: row for row in relational_arm.candidates}
    reframing_candidates = {row.candidate_alias: row for row in reframing_arm.candidates}
    all_source_ids = {
        "RELATIONAL_ONLY": set(relational_key.candidate_alias_to_source_id.values()),
        "REFRAMING_ONLY": set(reframing_key.candidate_alias_to_source_id.values()),
    }

    for index, (rel_aliases, ref_aliases) in enumerate(schedule, start=1):
        rel_candidates = [relational_candidates[alias] for alias in rel_aliases]
        ref_candidates = [reframing_candidates[alias] for alias in ref_aliases]
        rel_source_map = {
            alias: relational_key.candidate_alias_to_source_id[alias] for alias in rel_aliases
        }
        ref_source_map = {
            alias: reframing_key.candidate_alias_to_source_id[alias] for alias in ref_aliases
        }
        for source_id in rel_source_map.values():
            exposure["RELATIONAL_ONLY"][source_id] += 1
        for source_id in ref_source_map.values():
            exposure["REFRAMING_ONLY"][source_id] += 1

        subset_signature = _canonical_json((rel_aliases, ref_aliases))
        relational_first = _arm_order(packet.packet_id, subset_signature)
        if relational_first:
            arm_a_condition: AblationCondition = "RELATIONAL_ONLY"
            arm_b_condition: AblationCondition = "REFRAMING_ONLY"
            arm_a_candidates, arm_b_candidates = rel_candidates, ref_candidates
            arm_a_map, arm_b_map = rel_source_map, ref_source_map
        else:
            arm_a_condition = "REFRAMING_ONLY"
            arm_b_condition = "RELATIONAL_ONLY"
            arm_a_candidates, arm_b_candidates = ref_candidates, rel_candidates
            arm_a_map, arm_b_map = ref_source_map, rel_source_map

        comparison_alias = f"COMPARISON_{index:02d}"
        comparison_id = _stable_id(
            "scientific_reasoning_ablation_generalized_matched_comparison",
            packet.packet_id,
            rel_aliases,
            ref_aliases,
        )
        arm_a = BlindAblationArm(
            arm_alias="ARM_A",
            candidates=arm_a_candidates,
            candidate_count=matched_count,
        )
        arm_b = BlindAblationArm(
            arm_alias="ARM_B",
            candidates=arm_b_candidates,
            candidate_count=matched_count,
        )
        comparisons.append(
            BlindAblationComparison(
                comparison_id=comparison_id,
                comparison_alias=comparison_alias,
                arm_a=arm_a,
                arm_b=arm_b,
                candidate_count_equal=True,
                absolute_candidate_count_difference=0,
                interpretation_cautions=_neutral_cautions(
                    subset_schedule_complete=schedule_complete
                ),
            )
        )
        comparison_keys.append(
            AblationComparisonKey(
                comparison_id=comparison_id,
                comparison_alias=comparison_alias,
                comparison_kind="RELATIONAL_VS_REFRAMING",
                arms=[
                    AblationArmKey(
                        arm_alias="ARM_A",
                        condition=arm_a_condition,
                        candidate_alias_to_source_id=arm_a_map,
                    ),
                    AblationArmKey(
                        arm_alias="ARM_B",
                        condition=arm_b_condition,
                        candidate_alias_to_source_id=arm_b_map,
                    ),
                ],
            )
        )

        arm_a_payload = _payload_characters(arm_a_candidates)
        arm_b_payload = _payload_characters(arm_b_candidates)
        ratio = max(arm_a_payload, arm_b_payload) / min(arm_a_payload, arm_b_payload)
        included_by_condition = {
            "RELATIONAL_ONLY": set(rel_source_map.values()),
            "REFRAMING_ONLY": set(ref_source_map.values()),
        }
        omitted = {
            condition: sorted(all_source_ids[condition] - included)
            for condition, included in included_by_condition.items()
            if all_source_ids[condition] - included
        }
        report_rows.append(
            GeneralizedMatchedComparisonBuild(
                comparison_alias=comparison_alias,
                source_comparison_alias=source_comparison.comparison_alias,
                matched_candidate_count=matched_count,
                arm_a_condition=arm_a_condition,
                arm_b_condition=arm_b_condition,
                arm_a_source_candidate_ids=list(arm_a_map.values()),
                arm_b_source_candidate_ids=list(arm_b_map.values()),
                omitted_source_candidate_ids=omitted,
                arm_a_payload_characters=arm_a_payload,
                arm_b_payload_characters=arm_b_payload,
                payload_character_ratio=round(ratio, 6),
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
    _assert_neutral_blind_packet(matched_packet)

    matched_key = key.model_copy(
        update={
            "packet_id": matched_packet.packet_id,
            "comparisons": comparison_keys,
        }
    )
    matched_key = ScientificReasoningAblationBlindKey.model_validate(
        matched_key.model_dump(mode="python")
    )

    source_sha = _sha256(packet.model_dump(mode="json"))
    matched_sha = _sha256(matched_packet.model_dump(mode="json"))
    report = ScientificReasoningGeneralizedMatchedBuildReport(
        report_id=_stable_id(
            "scientific_reasoning_ablation_generalized_matched_build",
            packet.packet_id,
            matched_packet.packet_id,
            *[
                (row.comparison_alias, row.omitted_source_candidate_ids)
                for row in report_rows
            ],
        ),
        source_packet_id=packet.packet_id,
        source_packet_sha256=source_sha,
        source_key_packet_id=key.packet_id,
        matched_packet_id=matched_packet.packet_id,
        matched_packet_sha256=matched_sha,
        source_candidate_counts=source_counts,
        matched_candidate_count_per_arm=matched_count,
        total_possible_subset_comparisons=total_possible,
        materialized_comparison_count=len(comparisons),
        max_comparisons=max_comparisons,
        subset_schedule_complete=schedule_complete,
        candidate_exposure_counts={
            condition: dict(sorted(counts.items()))
            for condition, counts in exposure.items()
        },
        comparisons=report_rows,
    )
    return matched_packet, matched_key, report


__all__ = [
    "GeneralizedMatchedComparisonBuild",
    "ScientificReasoningGeneralizedMatchedBuildReport",
    "build_generalized_matched_count_ablation_packet",
]
