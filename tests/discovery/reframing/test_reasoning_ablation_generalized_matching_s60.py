from __future__ import annotations

import json

import pytest

from pipeline_core.discovery.reframing.ablation_evaluation import (
    AblationArmKey,
    AblationComparisonKey,
    BlindAblationArm,
    BlindAblationComparison,
    BlindEvidenceStatement,
    BlindFalsifier,
    BlindPrediction,
    BlindScientificCandidate,
    ScientificReasoningAblationBlindKey,
    ScientificReasoningAblationPacket,
)
from pipeline_core.discovery.reframing.ablation_generalized_matching import (
    build_generalized_matched_count_ablation_packet,
)
from pipeline_core.discovery.reframing.ablation_schema_normalization import (
    normalized_dimension_specs,
)


def _candidate(alias: str, label: str) -> BlindScientificCandidate:
    return BlindScientificCandidate(
        candidate_alias=alias,
        title=f"Candidate {alias}",
        scientific_proposal=f"Scientific proposal {label}",
        reasoning_rationale=f"Reasoning {label}",
        premise_evidence_aliases=["E01"],
        gap_evidence_aliases=["E02"],
        predictions=[
            BlindPrediction(
                observable=f"observable {alias}",
                expected_direction="unspecified",
                rationale=f"prediction {label}",
            )
        ],
        falsifiers=[
            BlindFalsifier(
                observable=f"observable {alias}",
                falsifying_outcome=f"falsifier {label}",
            )
        ],
    )


def _fixture(rel_count: int, ref_count: int):
    rel = [_candidate(f"C{i:02d}", f"lane-a-{i}") for i in range(1, rel_count + 1)]
    ref = [_candidate(f"C{i:02d}", f"lane-b-{i}") for i in range(1, ref_count + 1)]
    comparison = BlindAblationComparison(
        comparison_id="comparison:source",
        comparison_alias="COMPARISON_02",
        arm_a=BlindAblationArm(arm_alias="ARM_A", candidates=rel, candidate_count=rel_count),
        arm_b=BlindAblationArm(arm_alias="ARM_B", candidates=ref, candidate_count=ref_count),
        candidate_count_equal=rel_count == ref_count,
        absolute_candidate_count_difference=abs(rel_count - ref_count),
        interpretation_cautions=["Source normalized comparison."],
    )
    packet = ScientificReasoningAblationPacket(
        packet_id="packet:normalized",
        task_alias="TASK_01",
        question="How should the scientific response be explained?",
        evidence_statements=[
            BlindEvidenceStatement(
                evidence_alias="E01",
                text="reported evidence",
                epistemic_role="reported",
                claim_kind="observation",
                paper_count=1,
                requires_verification=False,
                eligible_as_positive_premise=True,
                eligible_as_gap=False,
            ),
            BlindEvidenceStatement(
                evidence_alias="E02",
                text="unresolved gap",
                epistemic_role="unresolved",
                claim_kind="gap",
                paper_count=1,
                requires_verification=True,
                eligible_as_positive_premise=False,
                eligible_as_gap=True,
            ),
        ],
        evaluation_dimensions=normalized_dimension_specs(),
        comparisons=[comparison],
        comparison_count=1,
        evidence_statement_count=2,
        source_candidate_count=rel_count + ref_count,
    )
    rel_map = {f"C{i:02d}": f"hypothesis:h{i}" for i in range(1, rel_count + 1)}
    ref_map = {f"C{i:02d}": f"scientific_reframe:r{i}" for i in range(1, ref_count + 1)}
    key = ScientificReasoningAblationBlindKey(
        packet_id=packet.packet_id,
        source_task_id="task:source",
        source_context_id="context:source",
        source_context_sha256="a" * 64,
        source_cross_lane_portfolio_id="cross:source",
        source_cross_lane_portfolio_file_sha256="b" * 64,
        source_context_file_sha256="c" * 64,
        source_relational_portfolio_file_sha256="d" * 64,
        source_reframe_shadow_file_sha256="e" * 64,
        source_proxy_shadow_file_sha256="f" * 64,
        source_contradiction_shadow_file_sha256="1" * 64,
        evidence_alias_to_statement_id={"E01": "stmt:1", "E02": "stmt:2"},
        condition_candidate_source_ids={
            "RELATIONAL_ONLY": list(rel_map.values()),
            "REFRAMING_ONLY": list(ref_map.values()),
            "COMBINED": [*rel_map.values(), *ref_map.values()],
        },
        comparisons=[
            AblationComparisonKey(
                comparison_id=comparison.comparison_id,
                comparison_alias=comparison.comparison_alias,
                comparison_kind="RELATIONAL_VS_REFRAMING",
                arms=[
                    AblationArmKey(
                        arm_alias="ARM_A",
                        condition="RELATIONAL_ONLY",
                        candidate_alias_to_source_id=rel_map,
                    ),
                    AblationArmKey(
                        arm_alias="ARM_B",
                        condition="REFRAMING_ONLY",
                        candidate_alias_to_source_id=ref_map,
                    ),
                ],
            )
        ],
    )
    return packet, key


def test_two_vs_one_builds_two_one_by_one_comparisons():
    packet, key = _fixture(2, 1)
    matched, _, report = build_generalized_matched_count_ablation_packet(
        packet=packet, key=key
    )
    assert matched.comparison_count == 2
    assert report.matched_candidate_count_per_arm == 1
    assert report.total_possible_subset_comparisons == 2
    assert report.subset_schedule_complete is True
    assert all(row.arm_a.candidate_count == row.arm_b.candidate_count == 1 for row in matched.comparisons)


def test_three_vs_two_builds_three_two_by_two_comparisons():
    packet, key = _fixture(3, 2)
    matched, _, report = build_generalized_matched_count_ablation_packet(
        packet=packet, key=key
    )
    assert matched.comparison_count == 3
    assert report.matched_candidate_count_per_arm == 2
    assert report.total_possible_subset_comparisons == 3
    assert report.subset_schedule_complete is True
    assert all(row.arm_a.candidate_count == row.arm_b.candidate_count == 2 for row in matched.comparisons)


def test_equal_cardinality_builds_one_full_comparison():
    packet, key = _fixture(2, 2)
    matched, _, report = build_generalized_matched_count_ablation_packet(
        packet=packet, key=key
    )
    assert matched.comparison_count == 1
    assert report.total_possible_subset_comparisons == 1
    assert report.subset_schedule_complete is True


def test_subset_cap_is_explicit_and_deterministic():
    packet, key = _fixture(5, 2)
    first = build_generalized_matched_count_ablation_packet(
        packet=packet, key=key, max_comparisons=4
    )
    second = build_generalized_matched_count_ablation_packet(
        packet=packet, key=key, max_comparisons=4
    )
    assert first[0].comparison_count == 4
    assert first[2].total_possible_subset_comparisons == 10
    assert first[2].subset_schedule_complete is False
    assert first[0].model_dump(mode="json") == second[0].model_dump(mode="json")
    assert first[2].model_dump(mode="json") == second[2].model_dump(mode="json")


def test_candidate_exposure_counts_are_reported():
    packet, key = _fixture(3, 2)
    _, _, report = build_generalized_matched_count_ablation_packet(packet=packet, key=key)
    rel = report.candidate_exposure_counts["RELATIONAL_ONLY"]
    ref = report.candidate_exposure_counts["REFRAMING_ONLY"]
    assert set(rel) == {"hypothesis:h1", "hypothesis:h2", "hypothesis:h3"}
    assert set(ref) == {"scientific_reframe:r1", "scientific_reframe:r2"}
    assert sum(rel.values()) == 6
    assert sum(ref.values()) == 6


def test_blind_packet_contains_no_condition_labels_source_ids_or_lane_words_in_cautions():
    packet, key = _fixture(3, 2)
    matched, _, _ = build_generalized_matched_count_ablation_packet(packet=packet, key=key)
    payload = json.dumps(matched.model_dump(mode="json"), ensure_ascii=False)
    for forbidden in (
        "RELATIONAL_ONLY",
        "REFRAMING_ONLY",
        "COMBINED",
        "hypothesis:h",
        "scientific_reframe:r",
    ):
        assert forbidden not in payload
    caution_text = " ".join(
        caution
        for comparison in matched.comparisons
        for caution in comparison.interpretation_cautions
    ).lower()
    assert "relational" not in caution_text
    assert "reframing" not in caution_text
    assert "combined" not in caution_text


def test_new_key_preserves_subset_source_mapping_and_updates_packet_id():
    packet, key = _fixture(2, 1)
    matched, matched_key, _ = build_generalized_matched_count_ablation_packet(
        packet=packet, key=key
    )
    assert matched_key.packet_id == matched.packet_id
    assert len(matched_key.comparisons) == 2
    for comparison in matched_key.comparisons:
        counts = {row.condition: len(row.candidate_alias_to_source_id) for row in comparison.arms}
        assert counts == {"RELATIONAL_ONLY": 1, "REFRAMING_ONLY": 1}


def test_non_normalized_input_fails_closed():
    packet, key = _fixture(2, 1)
    candidate = packet.comparisons[0].arm_a.candidates[0].model_copy(
        update={"baseline_model_summary": "baseline"}
    )
    arm = packet.comparisons[0].arm_a.model_copy(
        update={"candidates": [candidate, *packet.comparisons[0].arm_a.candidates[1:]]}
    )
    comparison = packet.comparisons[0].model_copy(update={"arm_a": arm})
    packet = packet.model_copy(update={"comparisons": [comparison]})
    with pytest.raises(ValueError, match="schema-normalized"):
        build_generalized_matched_count_ablation_packet(packet=packet, key=key)


def test_source_packet_is_not_mutated():
    packet, key = _fixture(3, 2)
    before = packet.model_dump(mode="json")
    build_generalized_matched_count_ablation_packet(packet=packet, key=key)
    assert packet.model_dump(mode="json") == before


def test_invalid_max_comparisons_fails_closed():
    packet, key = _fixture(3, 2)
    with pytest.raises(ValueError, match="max_comparisons"):
        build_generalized_matched_count_ablation_packet(
            packet=packet, key=key, max_comparisons=0
        )
