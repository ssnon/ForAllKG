from __future__ import annotations

from pipeline_core.discovery.reframing.ablation_blind_evaluator import (
    BlindComparisonEvaluation,
    BlindDimensionJudgment,
    BlindJudgeCallRecord,
    BlindScientificReasoningEvaluationReport,
    UnblindedComparisonEvaluation,
    UnblindedDimensionJudgment,
    UnblindedScientificReasoningEvaluationReport,
)
from pipeline_core.discovery.reframing.ablation_evaluation import (
    BlindAblationArm,
    BlindAblationComparison,
    BlindDiscriminatingTest,
    BlindEvidenceStatement,
    BlindPrediction,
    BlindScientificCandidate,
    ScientificReasoningAblationPacket,
    _evaluation_dimensions,
)
from pipeline_core.discovery.reframing.ablation_validity_audit import (
    audit_scientific_reasoning_ablation_validity,
)


def _relational(alias: str) -> BlindScientificCandidate:
    return BlindScientificCandidate(
        candidate_alias=alias,
        title="Relational proposal",
        scientific_proposal="A context-dependent relation is proposed.",
        reasoning_rationale="The evidence motivates a bounded relational extension.",
        premise_evidence_aliases=["E01"],
        predictions=[
            BlindPrediction(
                observable="response",
                expected_direction="shift",
                rationale="the response should shift under the context change",
            )
        ],
        falsifiers=[{"observable": "response", "falsifying_outcome": "no change"}],
    )


def _reframe(alias: str) -> BlindScientificCandidate:
    return BlindScientificCandidate(
        candidate_alias=alias,
        title="Reframing proposal",
        scientific_proposal="An alternative explanatory model is proposed.",
        baseline_model_summary="One rule governs the response.",
        alternative_or_resolution_summary="A different representation explains the response.",
        reasoning_rationale="Challenge the baseline representation.",
        premise_evidence_aliases=["E01"],
        predictions=[
            BlindPrediction(
                observable="response",
                baseline_expectation="tracks the baseline",
                alternative_expectation="can diverge",
                discriminating_outcome="divergence favors the alternative",
            )
        ],
        falsifiers=[{"falsifying_outcome": "the baseline always predicts the response"}],
        discriminating_test=BlindDiscriminatingTest(
            test_design="Measure both explanatory observables under matched conditions.",
            primary_observables=["response"],
            baseline_favoring_outcome="tracking",
            alternative_favoring_outcome="divergence",
        ),
        unresolved_questions=["Which mechanism causes the divergence?"],
    )


def _packet() -> ScientificReasoningAblationPacket:
    comparison = BlindAblationComparison(
        comparison_id="comparison:1",
        comparison_alias="COMPARISON_01",
        arm_a=BlindAblationArm(
            arm_alias="ARM_A",
            candidates=[_relational("C01")],
            candidate_count=1,
        ),
        arm_b=BlindAblationArm(
            arm_alias="ARM_B",
            candidates=[_reframe("C01")],
            candidate_count=1,
        ),
        candidate_count_equal=True,
        absolute_candidate_count_difference=0,
        interpretation_cautions=["candidate count is not a quality signal"],
    )
    return ScientificReasoningAblationPacket(
        packet_id="packet:1",
        task_alias="TASK_01",
        question="How does context alter response?",
        evidence_statements=[
            BlindEvidenceStatement(
                evidence_alias="E01",
                text="The response differs across contexts.",
                epistemic_role="reported",
                claim_kind="observation",
                paper_count=2,
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


def _blind() -> BlindScientificReasoningEvaluationReport:
    dimensions = [
        BlindDimensionJudgment(
            judgment_id=f"j:{spec.dimension_id}",
            dimension_id=spec.dimension_id,
            preference="ARM_B",
            rationale="ARM_B is preferred for this dimension.",
            supporting_evidence_aliases=["E01"],
            supporting_candidate_refs=[],
        )
        for spec in _evaluation_dimensions()
    ]
    return BlindScientificReasoningEvaluationReport(
        report_id="blind:1",
        source_packet_id="packet:1",
        source_packet_sha256="a" * 64,
        backend_name="fake",
        model_name="judge-model",
        comparison_evaluations=[
            BlindComparisonEvaluation(
                comparison_alias="COMPARISON_01",
                arm_a_candidate_count=1,
                arm_b_candidate_count=1,
                absolute_candidate_count_difference=0,
                dimensions=dimensions,
            )
        ],
        call_records=[
            BlindJudgeCallRecord(comparison_alias="COMPARISON_01")
        ],
        llm_calls_performed=1,
    )


def _unblinded() -> UnblindedScientificReasoningEvaluationReport:
    dimensions = [
        UnblindedDimensionJudgment(
            dimension_id=spec.dimension_id,
            blind_preference="ARM_B",
            preferred_condition="REFRAMING_ONLY",
            rationale="ARM_B is preferred for this dimension.",
            supporting_statement_ids=["stmt:1"],
        )
        for spec in _evaluation_dimensions()
    ]
    return UnblindedScientificReasoningEvaluationReport(
        report_id="unblind:1",
        source_blind_report_id="blind:1",
        source_blind_report_sha256="b" * 64,
        source_packet_id="packet:1",
        source_packet_sha256="a" * 64,
        source_task_id="task:1",
        source_context_id="ctx:1",
        source_context_sha256="c" * 64,
        comparisons=[
            UnblindedComparisonEvaluation(
                comparison_alias="COMPARISON_01",
                comparison_kind="RELATIONAL_VS_REFRAMING",
                arm_a_condition="RELATIONAL_ONLY",
                arm_b_condition="REFRAMING_ONLY",
                dimensions=dimensions,
                preference_counts={"REFRAMING_ONLY": 7},
            )
        ],
    )


def test_audit_flags_schema_asymmetry_without_changing_judgment():
    audit = audit_scientific_reasoning_ablation_validity(
        packet=_packet(), blind_report=_blind(), unblinded_report=_unblinded()
    )
    comparison = audit.comparison_audits[0]
    by_dimension = {row.dimension_id: row for row in comparison.dimensions}
    assert (
        by_dimension["differential_prediction_quality"].asymmetry_level
        == "high_presentation_asymmetry"
    )
    assert "prediction_schema_asymmetry" in by_dimension[
        "differential_prediction_quality"
    ].asymmetry_signals
    assert (
        by_dimension["discriminating_experiment_quality"].asymmetry_level
        == "high_presentation_asymmetry"
    )
    assert "dedicated_test_schema_asymmetry" in by_dimension[
        "discriminating_experiment_quality"
    ].asymmetry_signals
    assert by_dimension["differential_prediction_quality"].observed_preference == (
        "REFRAMING_ONLY"
    )
    assert by_dimension["differential_prediction_quality"].judgment_reversal_inferred is False


def test_audit_does_not_claim_superiority_or_parity():
    audit = audit_scientific_reasoning_ablation_validity(
        packet=_packet(), blind_report=_blind(), unblinded_report=_unblinded()
    )
    assert audit.llm_calls_performed == 0
    assert audit.evaluator_judgments_modified is False
    assert audit.scientific_reasoning_superiority_established is False
    assert audit.schema_parity_established is False
    assert audit.presentation_parity_established is False
    assert audit.results_suitable_for_directional_signal_only is True
    assert audit.overall_score_computed is False
    assert audit.overall_winner_selected is False


def test_candidate_count_parity_is_reported_separately_from_schema_parity():
    audit = audit_scientific_reasoning_ablation_validity(
        packet=_packet(), blind_report=_blind(), unblinded_report=_unblinded()
    )
    assert audit.candidate_count_parity_established is True
    assert audit.schema_parity_established is False


def test_lineage_mismatch_fails_closed():
    blind = _blind().model_copy(update={"source_packet_id": "packet:other"})
    try:
        audit_scientific_reasoning_ablation_validity(
            packet=_packet(), blind_report=blind, unblinded_report=_unblinded()
        )
    except ValueError as exc:
        assert "source_packet_id mismatch" in str(exc)
    else:
        raise AssertionError("expected lineage mismatch to fail closed")


def test_unblinding_preference_tamper_fails_closed():
    report = _unblinded()
    changed = report.comparisons[0].dimensions[0].model_copy(
        update={"preferred_condition": "RELATIONAL_ONLY"}
    )
    dimensions = [changed, *report.comparisons[0].dimensions[1:]]
    comparison = report.comparisons[0].model_copy(update={"dimensions": dimensions})
    report = report.model_copy(update={"comparisons": [comparison]})
    try:
        audit_scientific_reasoning_ablation_validity(
            packet=_packet(), blind_report=_blind(), unblinded_report=report
        )
    except ValueError as exc:
        assert "condition mapping mismatch" in str(exc)
    else:
        raise AssertionError("expected unblinding preference tamper to fail closed")


def test_blind_preference_tamper_fails_closed():
    report = _unblinded()
    changed = report.comparisons[0].dimensions[0].model_copy(
        update={"blind_preference": "ARM_A"}
    )
    comparison = report.comparisons[0].model_copy(
        update={"dimensions": [changed, *report.comparisons[0].dimensions[1:]]}
    )
    report = report.model_copy(update={"comparisons": [comparison]})
    try:
        audit_scientific_reasoning_ablation_validity(
            packet=_packet(), blind_report=_blind(), unblinded_report=report
        )
    except ValueError as exc:
        assert "changed blind preference" in str(exc)
    else:
        raise AssertionError("expected blind preference tamper to fail closed")


def test_low_asymmetry_is_possible_for_falsifiability():
    audit = audit_scientific_reasoning_ablation_validity(
        packet=_packet(), blind_report=_blind(), unblinded_report=_unblinded()
    )
    row = {
        item.dimension_id: item for item in audit.comparison_audits[0].dimensions
    }["falsifiability"]
    assert row.asymmetry_level == "low_observed_presentation_asymmetry"
    assert row.scientific_quality_conclusion_authorized is False


def test_single_judge_and_single_task_limits_are_explicit():
    audit = audit_scientific_reasoning_ablation_validity(
        packet=_packet(), blind_report=_blind(), unblinded_report=_unblinded()
    )
    assert audit.judge_model_name == "judge-model"
    assert audit.judge_replication_performed is False
    assert audit.multi_task_validation_performed is False
    assert audit.multi_model_validation_performed is False
