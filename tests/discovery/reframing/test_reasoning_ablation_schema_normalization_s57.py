from __future__ import annotations

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
from pipeline_core.discovery.reframing.ablation_schema_normalization import (
    build_schema_normalized_ablation_packet,
)


def _relational(alias: str) -> BlindScientificCandidate:
    return BlindScientificCandidate(
        candidate_alias=alias,
        title="Relational proposal",
        scientific_proposal="A context-dependent relation is proposed.",
        reasoning_rationale="The evidence motivates a bounded extension.",
        assumptions=["matched conditions"],
        premise_evidence_aliases=["E01"],
        predictions=[
            BlindPrediction(
                observable="response",
                expected_direction="shift",
                rationale="the response should shift with context",
            )
        ],
        falsifiers=[{"observable": "response", "falsifying_outcome": "no shift"}],
    )


def _reframe(alias: str) -> BlindScientificCandidate:
    return BlindScientificCandidate(
        candidate_alias=alias,
        title="Reframing proposal",
        scientific_proposal="A competing explanation is proposed.",
        baseline_model_summary="The response follows one rule.",
        alternative_or_resolution_summary="The response can follow another rule.",
        reasoning_rationale="Challenge the one-rule representation.",
        assumptions=["matched conditions"],
        premise_evidence_aliases=["E01"],
        predictions=[
            BlindPrediction(
                observable="response",
                baseline_expectation="tracks the baseline",
                alternative_expectation="diverges under context change",
                discriminating_outcome="divergence favors the alternative",
            )
        ],
        falsifiers=[{"falsifying_outcome": "the baseline always holds"}],
        discriminating_test=BlindDiscriminatingTest(
            test_design="Measure both explanatory observables.",
            primary_observables=["response"],
            baseline_favoring_outcome="tracking",
            alternative_favoring_outcome="divergence",
        ),
        unresolved_questions=["Which mechanism causes divergence?"],
    )


def _packet() -> ScientificReasoningAblationPacket:
    comparison = BlindAblationComparison(
        comparison_id="comparison:1",
        comparison_alias="COMPARISON_01",
        arm_a=BlindAblationArm(
            arm_alias="ARM_A", candidates=[_relational("C01")], candidate_count=1
        ),
        arm_b=BlindAblationArm(
            arm_alias="ARM_B", candidates=[_reframe("C01")], candidate_count=1
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


def test_normalization_removes_schema_rich_fields_symmetrically():
    normalized, report = build_schema_normalized_ablation_packet(_packet())
    candidates = [
        *normalized.comparisons[0].arm_a.candidates,
        *normalized.comparisons[0].arm_b.candidates,
    ]
    assert all(row.baseline_model_summary is None for row in candidates)
    assert all(row.alternative_or_resolution_summary is None for row in candidates)
    assert all(row.discriminating_test is None for row in candidates)
    assert all(row.unresolved_questions == [] for row in candidates)
    assert report.schema_field_parity_targeted is True


def test_all_predictions_use_common_relational_shape_after_normalization():
    normalized, _ = build_schema_normalized_ablation_packet(_packet())
    predictions = [
        prediction
        for comparison in normalized.comparisons
        for arm in (comparison.arm_a, comparison.arm_b)
        for candidate in arm.candidates
        for prediction in candidate.predictions
    ]
    assert predictions
    assert all(row.expected_direction == "unspecified" for row in predictions)
    assert all(row.rationale for row in predictions)
    assert all(row.baseline_expectation is None for row in predictions)
    assert all(row.alternative_expectation is None for row in predictions)
    assert all(row.discriminating_outcome is None for row in predictions)


def test_contrast_prediction_keeps_only_proposed_pattern_not_baseline_or_discriminant():
    normalized, report = build_schema_normalized_ablation_packet(_packet())
    reframe = normalized.comparisons[0].arm_b.candidates[0]
    assert reframe.predictions[0].rationale == "diverges under context change"
    stats = report.comparisons[0].arm_b.candidates[0]
    assert stats.contrast_prediction_count_downprojected == 1
    assert stats.relational_prediction_count_downprojected == 0


def test_relational_prediction_preserves_direction_inside_common_rationale():
    normalized, _ = build_schema_normalized_ablation_packet(_packet())
    relational = normalized.comparisons[0].arm_a.candidates[0]
    assert "Expected qualitative direction: shift." in relational.predictions[0].rationale


def test_blind_identity_and_key_compatibility_coordinates_are_preserved():
    source = _packet()
    normalized, report = build_schema_normalized_ablation_packet(source)
    assert normalized.packet_id == source.packet_id
    assert normalized.comparisons[0].comparison_id == source.comparisons[0].comparison_id
    assert normalized.comparisons[0].comparison_alias == "COMPARISON_01"
    assert normalized.comparisons[0].arm_a.candidates[0].candidate_alias == "C01"
    assert normalized.comparisons[0].arm_b.candidates[0].candidate_alias == "C01"
    assert report.blind_key_compatibility_preserved is True


def test_evidence_view_and_candidate_counts_are_unchanged():
    source = _packet()
    normalized, report = build_schema_normalized_ablation_packet(source)
    assert normalized.evidence_statements == source.evidence_statements
    assert normalized.source_candidate_count == source.source_candidate_count
    assert normalized.comparisons[0].absolute_candidate_count_difference == 0
    assert report.candidate_count_parity_targeted is False


def test_normalized_dimensions_explicitly_remove_dedicated_test_field_penalty():
    normalized, _ = build_schema_normalized_ablation_packet(_packet())
    by_id = {row.dimension_id: row for row in normalized.evaluation_dimensions}
    question = by_id["discriminating_experiment_quality"].question
    assert "dedicated test-design field" in question
    assert "intentionally removed" in question


def test_normalization_report_is_diagnostic_only():
    _, report = build_schema_normalized_ablation_packet(_packet())
    assert report.llm_calls_performed == 0
    assert report.deterministic_projection_only is True
    assert report.scientific_quality_judgment_performed is False
    assert report.evaluator_judgments_modified is False
    assert report.production_selection_changed is False
    assert report.canonical_graph_mutated is False


def test_source_packet_is_not_mutated():
    source = _packet()
    before = source.model_dump(mode="json")
    build_schema_normalized_ablation_packet(source)
    assert source.model_dump(mode="json") == before
