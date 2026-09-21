from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline_core.discovery.reframing.ablation_blind_evaluator import (
    AblationBlindEvaluationGeneration,
    BlindCandidateReference,
    BlindComparisonJudgmentDraft,
    BlindDimensionJudgmentDraft,
    build_ablation_blind_evaluation_prompt,
    compile_blind_comparison_evaluation,
    run_blind_scientific_reasoning_evaluation,
    unblind_scientific_reasoning_evaluation,
)
from pipeline_core.discovery.reframing.ablation_evaluation import (
    AblationArmKey,
    AblationComparisonKey,
    BlindAblationArm,
    BlindAblationComparison,
    BlindEvidenceStatement,
    BlindPrediction,
    BlindScientificCandidate,
    ScientificReasoningAblationBlindKey,
    ScientificReasoningAblationPacket,
    _evaluation_dimensions,
)


def _candidate(alias: str, *, evidence: str = "E01") -> BlindScientificCandidate:
    return BlindScientificCandidate(
        candidate_alias=alias,
        title="A scientific proposal",
        scientific_proposal="A conditional explanation of the observed response.",
        reasoning_rationale="The supplied observations motivate a discriminating model.",
        premise_evidence_aliases=[evidence],
        predictions=[
            BlindPrediction(
                observable="response maximum",
                expected_direction="shift",
                rationale="the response should move under the perturbation",
            )
        ],
        falsifiers=[{"falsifying_outcome": "the response never changes"}],
    )


def _packet() -> ScientificReasoningAblationPacket:
    arm_a = BlindAblationArm(
        arm_alias="ARM_A",
        candidates=[_candidate("C01")],
        candidate_count=1,
    )
    arm_b = BlindAblationArm(
        arm_alias="ARM_B",
        candidates=[_candidate("C02")],
        candidate_count=1,
    )
    comparison = BlindAblationComparison(
        comparison_id="comparison:1",
        comparison_alias="COMPARISON_01",
        arm_a=arm_a,
        arm_b=arm_b,
        candidate_count_equal=True,
        absolute_candidate_count_difference=0,
        interpretation_cautions=["candidate count is not a quality signal"],
    )
    return ScientificReasoningAblationPacket(
        packet_id="packet:blind",
        task_alias="TASK_01",
        question="How does the local environment alter the response?",
        evidence_statements=[
            BlindEvidenceStatement(
                evidence_alias="E01",
                text="The response shifts under one condition.",
                epistemic_role="reported",
                claim_kind="observation",
                paper_count=1,
                requires_verification=False,
                eligible_as_positive_premise=True,
                eligible_as_gap=False,
            )
        ],
        evaluation_dimensions=_evaluation_dimensions(),
        comparisons=[comparison],
        comparison_count=1,
        evidence_statement_count=1,
        source_candidate_count=2,
    )


def _draft(
    *,
    preference: str = "ARM_A",
    candidate_alias: str = "C01",
    comparison_alias: str = "COMPARISON_01",
) -> BlindComparisonJudgmentDraft:
    dimensions = []
    for spec in _evaluation_dimensions():
        dimensions.append(
            BlindDimensionJudgmentDraft(
                dimension_id=spec.dimension_id,
                preference=preference,
                rationale="This arm provides the clearer evidence-linked discrimination.",
                supporting_evidence_aliases=["E01"],
                supporting_candidate_refs=[
                    BlindCandidateReference(
                        arm_alias="ARM_A" if preference == "ARM_A" else "ARM_B",
                        candidate_alias=candidate_alias,
                    )
                ] if preference in {"ARM_A", "ARM_B"} else [],
            )
        )
    return BlindComparisonJudgmentDraft(
        comparison_alias=comparison_alias,
        dimensions=dimensions,
    )


class _Backend:
    backend_name = "fake"
    model_name = "fake-model"

    def __init__(self, draft):
        self.draft = draft
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return AblationBlindEvaluationGeneration(
            draft=self.draft,
            input_tokens=100,
            output_tokens=50,
            response_id="r1",
            elapsed_seconds=0.1,
        )


def _key() -> ScientificReasoningAblationBlindKey:
    return ScientificReasoningAblationBlindKey(
        packet_id="packet:blind",
        source_task_id="task:1",
        source_context_id="ctx:1",
        source_context_sha256="c" * 64,
        source_cross_lane_portfolio_id="cross:1",
        source_cross_lane_portfolio_file_sha256="a" * 64,
        source_context_file_sha256="b" * 64,
        source_relational_portfolio_file_sha256="d" * 64,
        source_reframe_shadow_file_sha256="e" * 64,
        source_proxy_shadow_file_sha256="f" * 64,
        source_contradiction_shadow_file_sha256="1" * 64,
        evidence_alias_to_statement_id={"E01": "stmt:1"},
        condition_candidate_source_ids={
            "RELATIONAL_ONLY": ["hypothesis:h1"],
            "REFRAMING_ONLY": ["scientific_reframe:r1"],
            "COMBINED": ["hypothesis:h1", "scientific_reframe:r1"],
        },
        comparisons=[
            AblationComparisonKey(
                comparison_id="comparison:1",
                comparison_alias="COMPARISON_01",
                comparison_kind="RELATIONAL_VS_REFRAMING",
                arms=[
                    AblationArmKey(
                        arm_alias="ARM_A",
                        condition="RELATIONAL_ONLY",
                        candidate_alias_to_source_id={"C01": "hypothesis:h1"},
                    ),
                    AblationArmKey(
                        arm_alias="ARM_B",
                        condition="REFRAMING_ONLY",
                        candidate_alias_to_source_id={"C02": "scientific_reframe:r1"},
                    ),
                ],
            )
        ],
    )


def test_prompt_contains_no_condition_labels_or_source_ids():
    prompt = build_ablation_blind_evaluation_prompt(
        packet=_packet(), comparison=_packet().comparisons[0]
    )
    combined = prompt.system_prompt + prompt.user_prompt
    assert "RELATIONAL_ONLY" not in combined
    assert "REFRAMING_ONLY" not in combined
    assert "COMBINED" not in combined
    assert "hypothesis:h1" not in combined
    assert "scientific_reframe:r1" not in combined
    assert "candidate count" in combined.lower()


def test_compile_requires_all_dimensions_exactly_once():
    packet = _packet()
    draft = _draft()
    draft.dimensions = draft.dimensions[:-1]
    with pytest.raises(ValueError, match="every evaluation dimension"):
        compile_blind_comparison_evaluation(
            packet=packet,
            comparison=packet.comparisons[0],
            draft=draft,
        )


def test_compile_rejects_unknown_evidence_alias():
    packet = _packet()
    draft = _draft()
    draft.dimensions[0].supporting_evidence_aliases = ["E99"]
    with pytest.raises(ValueError, match="unknown evidence aliases"):
        compile_blind_comparison_evaluation(
            packet=packet,
            comparison=packet.comparisons[0],
            draft=draft,
        )


def test_compile_rejects_candidate_alias_outside_arm():
    packet = _packet()
    draft = _draft()
    draft.dimensions[0].supporting_candidate_refs = [
        BlindCandidateReference(arm_alias="ARM_A", candidate_alias="C99")
    ]
    with pytest.raises(ValueError, match="outside declared arm"):
        compile_blind_comparison_evaluation(
            packet=packet,
            comparison=packet.comparisons[0],
            draft=draft,
        )


def test_arm_preference_requires_support_from_that_arm():
    packet = _packet()
    draft = _draft()
    draft.dimensions[0].supporting_candidate_refs = [
        BlindCandidateReference(arm_alias="ARM_B", candidate_alias="C02")
    ]
    with pytest.raises(ValueError, match="ARM_A preference requires"):
        compile_blind_comparison_evaluation(
            packet=packet,
            comparison=packet.comparisons[0],
            draft=draft,
        )


def test_blind_runner_makes_one_call_per_comparison_and_no_overall_winner():
    packet = _packet()
    backend = _Backend(_draft())
    report = run_blind_scientific_reasoning_evaluation(packet=packet, backend=backend)
    assert backend.calls == 1
    assert report.llm_calls_performed == 1
    assert report.blind_key_loaded_by_evaluator is False
    assert report.overall_score_computed is False
    assert report.overall_winner_selected is False
    assert report.scientific_quality_ranking_performed is False


def test_unblind_maps_arm_preference_to_condition_and_source_ids():
    packet = _packet()
    blind = run_blind_scientific_reasoning_evaluation(
        packet=packet, backend=_Backend(_draft())
    )
    unblinded = unblind_scientific_reasoning_evaluation(
        packet=packet,
        key=_key(),
        blind_report=blind,
        source_blind_report_sha256="9" * 64,
    )
    row = unblinded.comparisons[0].dimensions[0]
    assert row.blind_preference == "ARM_A"
    assert row.preferred_condition == "RELATIONAL_ONLY"
    assert row.supporting_statement_ids == ["stmt:1"]
    assert row.supporting_candidate_refs[0].source_candidate_id == "hypothesis:h1"
    assert unblinded.overall_winner_selected is False


def test_unblind_maps_tie_without_inventing_condition_winner():
    packet = _packet()
    blind = run_blind_scientific_reasoning_evaluation(
        packet=packet, backend=_Backend(_draft(preference="TIE", candidate_alias="C01"))
    )
    unblinded = unblind_scientific_reasoning_evaluation(
        packet=packet,
        key=_key(),
        blind_report=blind,
        source_blind_report_sha256="9" * 64,
    )
    assert all(
        row.preferred_condition == "TIE"
        for row in unblinded.comparisons[0].dimensions
    )


def test_unblind_rejects_wrong_packet_id():
    packet = _packet()
    blind = run_blind_scientific_reasoning_evaluation(
        packet=packet, backend=_Backend(_draft())
    )
    key = _key().model_copy(update={"packet_id": "packet:wrong"})
    with pytest.raises(ValueError, match="blind key packet_id mismatch"):
        unblind_scientific_reasoning_evaluation(
            packet=packet,
            key=key,
            blind_report=blind,
            source_blind_report_sha256="9" * 64,
        )


def test_unblind_rejects_tampered_blind_packet_hash():
    packet = _packet()
    blind = run_blind_scientific_reasoning_evaluation(
        packet=packet, backend=_Backend(_draft())
    )
    blind = blind.model_copy(update={"source_packet_sha256": "0" * 64})
    with pytest.raises(ValueError, match="source_packet_sha256 mismatch"):
        unblind_scientific_reasoning_evaluation(
            packet=packet,
            key=_key(),
            blind_report=blind,
            source_blind_report_sha256="9" * 64,
        )


def test_preference_counts_are_descriptive_not_composite_score():
    packet = _packet()
    blind = run_blind_scientific_reasoning_evaluation(
        packet=packet, backend=_Backend(_draft())
    )
    unblinded = unblind_scientific_reasoning_evaluation(
        packet=packet,
        key=_key(),
        blind_report=blind,
        source_blind_report_sha256="9" * 64,
    )
    comparison = unblinded.comparisons[0]
    assert comparison.preference_counts == {"RELATIONAL_ONLY": 7}
    assert comparison.composite_score_computed is False
    assert comparison.overall_winner_selected is False
