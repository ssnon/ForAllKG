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
from pipeline_core.discovery.reframing.ablation_matched_count import (
    build_matched_count_ablation_packet,
)
from pipeline_core.discovery.reframing.ablation_schema_normalization import (
    normalized_dimension_specs,
)


def _candidate(alias: str, text: str) -> BlindScientificCandidate:
    return BlindScientificCandidate(
        candidate_alias=alias,
        title=f"Candidate {alias}",
        scientific_proposal=text,
        reasoning_rationale=f"Reasoning for {text}",
        premise_evidence_aliases=["E01"],
        gap_evidence_aliases=["E02"],
        predictions=[
            BlindPrediction(
                observable=f"observable {alias}",
                expected_direction="unspecified",
                rationale=f"prediction {text}",
            )
        ],
        falsifiers=[
            BlindFalsifier(
                observable=f"observable {alias}",
                falsifying_outcome=f"falsifier {text}",
            )
        ],
    )


def _packet() -> ScientificReasoningAblationPacket:
    relational = [_candidate(f"C0{i}", f"relational {i}") for i in range(1, 4)]
    reframing = [_candidate(f"C0{i}", f"reframing {i}") for i in range(1, 5)]
    comparison = BlindAblationComparison(
        comparison_id="comparison:source",
        comparison_alias="COMPARISON_02",
        arm_a=BlindAblationArm(
            arm_alias="ARM_A", candidates=relational, candidate_count=3
        ),
        arm_b=BlindAblationArm(
            arm_alias="ARM_B", candidates=reframing, candidate_count=4
        ),
        candidate_count_equal=False,
        absolute_candidate_count_difference=1,
        interpretation_cautions=["source normalized comparison"],
    )
    return ScientificReasoningAblationPacket(
        packet_id="packet:normalized",
        task_alias="TASK_01",
        question="How should the mechanism be explained?",
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
        source_candidate_count=7,
    )


def _key() -> ScientificReasoningAblationBlindKey:
    return ScientificReasoningAblationBlindKey(
        packet_id="packet:normalized",
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
            "RELATIONAL_ONLY": ["rel:1", "rel:2", "rel:3"],
            "REFRAMING_ONLY": ["ref:1", "ref:2", "ref:3", "ref:4"],
            "COMBINED": [
                "rel:1",
                "rel:2",
                "rel:3",
                "ref:1",
                "ref:2",
                "ref:3",
                "ref:4",
            ],
        },
        comparisons=[
            AblationComparisonKey(
                comparison_id="comparison:source",
                comparison_alias="COMPARISON_02",
                comparison_kind="RELATIONAL_VS_REFRAMING",
                arms=[
                    AblationArmKey(
                        arm_alias="ARM_A",
                        condition="RELATIONAL_ONLY",
                        candidate_alias_to_source_id={
                            "C01": "rel:1",
                            "C02": "rel:2",
                            "C03": "rel:3",
                        },
                    ),
                    AblationArmKey(
                        arm_alias="ARM_B",
                        condition="REFRAMING_ONLY",
                        candidate_alias_to_source_id={
                            "C01": "ref:1",
                            "C02": "ref:2",
                            "C03": "ref:3",
                            "C04": "ref:4",
                        },
                    ),
                ],
            )
        ],
    )


def test_builds_four_three_by_three_leave_one_out_comparisons():
    matched, _, report = build_matched_count_ablation_packet(packet=_packet(), key=_key())
    assert matched.comparison_count == 4
    assert report.leave_one_out_comparison_count == 4
    for row in matched.comparisons:
        assert row.arm_a.candidate_count == 3
        assert row.arm_b.candidate_count == 3
        assert row.candidate_count_equal is True
        assert row.absolute_candidate_count_difference == 0


def test_each_reframing_candidate_is_omitted_exactly_once():
    _, _, report = build_matched_count_ablation_packet(packet=_packet(), key=_key())
    assert sorted(row.omitted_reframing_source_candidate_id for row in report.comparisons) == [
        "ref:1",
        "ref:2",
        "ref:3",
        "ref:4",
    ]


def test_relational_candidates_are_reused_in_every_comparison():
    _, matched_key, _ = build_matched_count_ablation_packet(packet=_packet(), key=_key())
    for comparison in matched_key.comparisons:
        relational = next(row for row in comparison.arms if row.condition == "RELATIONAL_ONLY")
        assert set(relational.candidate_alias_to_source_id.values()) == {
            "rel:1",
            "rel:2",
            "rel:3",
        }


def test_output_key_tracks_new_packet_and_subset_mappings():
    matched, matched_key, _ = build_matched_count_ablation_packet(packet=_packet(), key=_key())
    assert matched_key.packet_id == matched.packet_id
    assert {row.comparison_alias for row in matched_key.comparisons} == {
        "COMPARISON_01",
        "COMPARISON_02",
        "COMPARISON_03",
        "COMPARISON_04",
    }
    for comparison in matched_key.comparisons:
        reframing = next(row for row in comparison.arms if row.condition == "REFRAMING_ONLY")
        assert len(reframing.candidate_alias_to_source_id) == 3


def test_packet_does_not_leak_condition_or_source_candidate_ids():
    matched, _, _ = build_matched_count_ablation_packet(packet=_packet(), key=_key())
    payload = json.dumps(matched.model_dump(mode="json"), sort_keys=True)
    assert "RELATIONAL_ONLY" not in payload
    assert "REFRAMING_ONLY" not in payload
    assert "rel:1" not in payload
    assert "ref:1" not in payload


def test_builder_is_deterministic():
    first = build_matched_count_ablation_packet(packet=_packet(), key=_key())
    second = build_matched_count_ablation_packet(packet=_packet(), key=_key())
    assert first[0].model_dump(mode="json") == second[0].model_dump(mode="json")
    assert first[1].model_dump(mode="json") == second[1].model_dump(mode="json")
    assert first[2].model_dump(mode="json") == second[2].model_dump(mode="json")


def test_rejects_non_normalized_candidate_shape():
    packet = _packet()
    candidate = packet.comparisons[0].arm_b.candidates[0].model_copy(
        update={"baseline_model_summary": "baseline"}
    )
    arm_b = packet.comparisons[0].arm_b.model_copy(
        update={
            "candidates": [candidate, *packet.comparisons[0].arm_b.candidates[1:]],
        }
    )
    comparison = packet.comparisons[0].model_copy(update={"arm_b": arm_b})
    packet = packet.model_copy(update={"comparisons": [comparison]})
    with pytest.raises(ValueError, match="schema-normalized"):
        build_matched_count_ablation_packet(packet=packet, key=_key())


def test_rejects_wrong_candidate_count_relation():
    packet = _packet()
    arm_b = packet.comparisons[0].arm_b.model_copy(
        update={
            "candidates": packet.comparisons[0].arm_b.candidates[:3],
            "candidate_count": 3,
        }
    )
    comparison = packet.comparisons[0].model_copy(
        update={
            "arm_b": arm_b,
            "candidate_count_equal": True,
            "absolute_candidate_count_difference": 0,
        }
    )
    packet = packet.model_copy(update={"comparisons": [comparison]})
    with pytest.raises(ValueError, match="reframing candidate count"):
        build_matched_count_ablation_packet(packet=packet, key=_key())


def test_source_packet_is_not_mutated():
    packet = _packet()
    before = packet.model_dump(mode="json")
    build_matched_count_ablation_packet(packet=packet, key=_key())
    assert packet.model_dump(mode="json") == before
