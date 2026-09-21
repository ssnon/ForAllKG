from __future__ import annotations

import json

import pytest

from pipeline_core.discovery.reframing.final_output_blind_evaluation import (
    FinalOutputDimensionJudgment,
    FinalOutputEvaluationDraft,
    FinalOutputEvaluationGeneration,
    build_blind_final_output_evaluation_packet,
    build_final_output_evaluation_prompt,
    run_blind_final_output_evaluation,
    unblind_final_output_evaluation,
)
from pipeline_core.discovery.reframing.integrated_hypothesis_shadow import (
    IntegratedHypothesisDiscriminatingTest,
    IntegratedHypothesisFalsifier,
    IntegratedHypothesisPrediction,
    IntegratedScientificHypothesisShadowPortfolio,
    IntegratedShadowHypothesis,
    IntegratedShadowLineage,
)


DIMENSIONS = [
    "task_relevance_and_coverage",
    "mechanistic_explanatory_gain",
    "prediction_specificity_and_differentiation",
    "falsifiability_and_discriminating_tests",
    "internal_coherence",
    "focus_and_redundancy_control",
    "practical_research_value",
]


def _legacy(entry_id: str = "entry:legacy") -> IntegratedShadowHypothesis:
    return IntegratedShadowHypothesis(
        entry_id=entry_id,
        origin="LEGACY_RELATIONAL",
        title="Matrix changes ranking",
        hypothesis_statement="Matrix composition changes SERS design ranking.",
        reasoning_rationale="Observed matrix effects motivate a conditional relation.",
        premise_statement_ids=["p1"],
        gap_statement_ids=["g1"],
        assumptions=["matched analyte level"],
        predictions=[
            IntegratedHypothesisPrediction(
                observable="design ranking",
                expected_result="qualitative_change",
                rationale="ranking should differ by matrix",
                source_candidate_ids=["pc:rel"],
            )
        ],
        falsifiers=[
            IntegratedHypothesisFalsifier(
                observable="design ranking",
                falsifying_outcome="ranking remains invariant",
                source_candidate_ids=["pc:rel"],
            )
        ],
        source_candidate_ids=["pc:rel"],
        source_object_ids=["hypothesis:1"],
        source_lanes=["RELATIONAL_DISCOVERY"],
    )


def _synthesis(entry_id: str = "entry:synth") -> IntegratedShadowHypothesis:
    return IntegratedShadowHypothesis(
        entry_id=entry_id,
        origin="CROSS_LANE_SYNTHESIS",
        title="Interfacial state mediates ranking",
        hypothesis_statement="Matrix-conditioned interfacial state mediates design-rank changes.",
        reasoning_rationale="This connects the conditional relation to a measurable local-state mechanism.",
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        assumptions=["interfacial state is measurable"],
        predictions=[
            IntegratedHypothesisPrediction(
                observable="rank and local accumulation",
                expected_result="covariation",
                rationale="mediation predicts coupled changes",
                source_candidate_ids=["pc:rel", "pc:ref"],
            )
        ],
        falsifiers=[
            IntegratedHypothesisFalsifier(
                observable="rank and local accumulation",
                falsifying_outcome="no covariation is observed",
                source_candidate_ids=["pc:rel", "pc:ref"],
            )
        ],
        discriminating_test=IntegratedHypothesisDiscriminatingTest(
            test_design="Compare matched matrices and measure local accumulation independently.",
            primary_observables=["design ranking", "local accumulation"],
            source_candidate_ids=["pc:rel", "pc:ref"],
            favoring_outcomes=["covariation favors mediation"],
        ),
        unresolved_questions=["which matrix constituent dominates"],
        source_candidate_ids=["pc:rel", "pc:ref"],
        source_object_ids=["hypothesis:1", "scientific_reframe:1"],
        source_lanes=["RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"],
        synthesis_kind="complementary_mechanism_integration",
    )


def _portfolio(*, additions: bool = True) -> IntegratedScientificHypothesisShadowPortfolio:
    legacy = [_legacy()]
    synthesized = [_synthesis()] if additions else []
    return IntegratedScientificHypothesisShadowPortfolio(
        portfolio_id="integrated-shadow:1",
        source_candidate_portfolio_id="pcp:1",
        source_candidate_portfolio_sha256="a" * 64,
        source_synthesis_report_id="synth-report:1",
        source_synthesis_report_sha256="b" * 64,
        source_task_id="task:1",
        source_context_id="ctx:1",
        question="Does matrix alter SERS design ranking?",
        legacy_hypotheses=legacy,
        integrated_hypotheses=[*legacy, *synthesized],
        synthesized_additions=synthesized,
        lineage_validation=IntegratedShadowLineage(),
        legacy_hypothesis_count=1,
        integrated_hypothesis_count=1 + len(synthesized),
        synthesized_addition_count=len(synthesized),
        raw_reframing_source_candidate_count=1,
        source_candidate_count=2,
        source_kind_counts={"relational_hypothesis": 1, "scientific_reframe": 1},
        synthesis_covered_relational_candidate_ids=["pc:rel"] if additions else [],
        synthesis_covered_reframing_candidate_ids=["pc:ref"] if additions else [],
        synthesis_uncovered_relational_candidate_ids=[] if additions else ["pc:rel"],
        synthesis_uncovered_reframing_candidate_ids=[] if additions else ["pc:ref"],
        shadow_output_differs_from_legacy=additions,
    )


class FakeBackend:
    backend_name = "fake"
    model_name = "fake-model"

    def __init__(self, preference: str = "ARM_A") -> None:
        self.preference = preference

    def evaluate(self, prompt):
        return FinalOutputEvaluationGeneration(
            draft=FinalOutputEvaluationDraft(
                judgments=[
                    FinalOutputDimensionJudgment(
                        dimension_id=dimension,
                        preference=self.preference,
                        rationale=f"reason for {dimension}",
                    )
                    for dimension in DIMENSIONS
                ]
            ),
            input_tokens=100,
            output_tokens=50,
        )


def test_build_packet_hides_condition_labels_and_source_ids():
    packet, _ = build_blind_final_output_evaluation_packet(_portfolio())
    payload = json.dumps(packet.model_dump(mode="json"), ensure_ascii=False).lower()
    assert "legacy_final_view" not in payload
    assert "integrated_final_view" not in payload
    assert "hypothesis:1" not in payload
    assert "scientific_reframe:1" not in payload
    assert "pc:rel" not in payload
    assert "pc:ref" not in payload


def test_packet_preserves_real_portfolio_count_difference():
    packet, _ = build_blind_final_output_evaluation_packet(_portfolio())
    assert sorted([packet.arm_a.hypothesis_count, packet.arm_b.hypothesis_count]) == [1, 2]
    assert packet.candidate_count_treated_as_quality_signal is False


def test_key_maps_both_conditions_exactly_once():
    _, key = build_blind_final_output_evaluation_packet(_portfolio())
    assert {row.condition for row in key.arms} == {
        "LEGACY_FINAL_VIEW",
        "INTEGRATED_FINAL_VIEW",
    }


def test_blind_hypothesis_drops_origin_and_lineage_fields():
    packet, _ = build_blind_final_output_evaluation_packet(_portfolio())
    payload = packet.arm_a.model_dump(mode="json")
    text = json.dumps(payload)
    assert "origin" not in text
    assert "source_candidate_ids" not in text
    assert "source_object_ids" not in text
    assert "source_lanes" not in text


def test_prompt_explicitly_neutralizes_candidate_count_signal():
    packet, _ = build_blind_final_output_evaluation_packet(_portfolio())
    prompt = build_final_output_evaluation_prompt(packet)
    joined = (prompt.system_prompt + prompt.user_prompt).lower()
    assert "portfolio size is not a quality signal" in joined
    assert "overall winner" in joined


def test_evaluator_returns_only_per_dimension_results():
    packet, _ = build_blind_final_output_evaluation_packet(_portfolio())
    report, _ = run_blind_final_output_evaluation(packet=packet, backend=FakeBackend())
    assert len(report.judgments) == 7
    assert report.preference_counts == {"ARM_A": 7}
    assert report.overall_score_computed is False
    assert report.overall_winner_selected is False


def test_unblind_maps_arm_preference_to_condition():
    packet, key = build_blind_final_output_evaluation_packet(_portfolio())
    preferred_arm = next(row.arm_alias for row in key.arms if row.condition == "INTEGRATED_FINAL_VIEW")
    blind, _ = run_blind_final_output_evaluation(
        packet=packet,
        backend=FakeBackend(preferred_arm),
    )
    report = unblind_final_output_evaluation(packet=packet, blind_report=blind, key=key)
    assert report.condition_preference_counts == {"INTEGRATED_FINAL_VIEW": 7}


def test_unblind_preserves_tie_without_forcing_condition():
    packet, key = build_blind_final_output_evaluation_packet(_portfolio())
    blind, _ = run_blind_final_output_evaluation(packet=packet, backend=FakeBackend("TIE"))
    report = unblind_final_output_evaluation(packet=packet, blind_report=blind, key=key)
    assert report.condition_preference_counts == {"TIE": 7}


def test_builder_rejects_no_difference_from_legacy():
    with pytest.raises(ValueError, match="differ"):
        build_blind_final_output_evaluation_packet(_portfolio(additions=False))


def test_evaluator_rejects_missing_dimension():
    packet, _ = build_blind_final_output_evaluation_packet(_portfolio())

    class BadBackend(FakeBackend):
        def evaluate(self, prompt):
            return FinalOutputEvaluationGeneration(
                draft=FinalOutputEvaluationDraft(
                    judgments=[
                        FinalOutputDimensionJudgment(
                            dimension_id=dimension,
                            preference="ARM_A",
                            rationale="reason",
                        )
                        for dimension in DIMENSIONS[:-1]
                    ]
                )
            )

    with pytest.raises(Exception):
        run_blind_final_output_evaluation(packet=packet, backend=BadBackend())


def test_builder_is_deterministic():
    first = build_blind_final_output_evaluation_packet(_portfolio())
    second = build_blind_final_output_evaluation_packet(_portfolio())
    assert first[0].model_dump(mode="json") == second[0].model_dump(mode="json")
    assert first[1].model_dump(mode="json") == second[1].model_dump(mode="json")
