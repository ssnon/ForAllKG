import pytest

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
)
from pipeline_core.discovery.nonobviousness_canonical_candidate import (
    N11CanonicalCandidateCompiler,
)
from pipeline_core.discovery.nonobviousness_operator_generation_contracts import (
    N11OperatorCandidateDraft,
    N11OperatorFalsificationDraft,
    N11OperatorGenerationDraft,
    N11OperatorPredictionDraft,
)
from pipeline_core.discovery.nonobviousness_operator_generation_validation import (
    N11OperatorGenerationValidation,
)


def context():
    return HypothesisContext(
        context_id="context:test",
        context_sha256="sha-context",
        source_packet_id="packet:test",
        source_packet_sha256="sha-packet",
        source_report_id="report:test",
        source_report_sha256="sha-report",
        task_id="task:test",
        question="How does spacing relate to SERS?",
        corpus_id="corpus:test",
        domain_profile_id="sers_test",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="stmt:baseline",
                text=(
                    "Spacing affects plasmon coupling."
                ),
                epistemic_role="reported",
                claim_kind="mechanism",
                paper_ids=["paper:A"],
                eligible_as_premise=True,
            ),
            HypothesisEvidenceStatement(
                statement_id="stmt:gap",
                text=(
                    "The spacing-to-SERS mechanism "
                    "remains unresolved."
                ),
                epistemic_role="unresolved",
                claim_kind="scope_limit",
                paper_ids=["paper:A"],
                eligible_as_gap=True,
            ),
        ],
    )


def draft():
    candidate = N11OperatorCandidateDraft(
        local_id="candidate:1",
        title="Relative mechanism contribution",
        hypothesis_statement=(
            "Spacing may alter the relative contribution "
            "of electromagnetic and chemical enhancement."
        ),
        operator="RELATIVE_CONTRIBUTION_SHIFT",
        hypothesis_type="cross_evidence_synthesis",
        baseline_premise_statement_ids=[
            "stmt:baseline"
        ],
        supplemental_mechanism_node_ids=[
            "node:supplemental"
        ],
        gap_statement_ids=[
            "stmt:gap"
        ],
        shared_component_ids=[
            "component:shared"
        ],
        supplemental_only_component_ids=[
            "component:chemical"
        ],
        relative_contribution_claim=(
            "Spacing may change the relative contribution "
            "of electromagnetic and chemical enhancement."
        ),
        inferential_bridge=(
            "The components are separately grounded; "
            "their spacing-dependent relative weighting "
            "is inferred."
        ),
        predicted_observations=[
            N11OperatorPredictionDraft(
                local_id="prediction:1",
                observable=(
                    "relative mechanistic signature balance"
                ),
                expected_direction="shift",
                rationale=(
                    "Relative weighting should change."
                ),
            )
        ],
        discriminating_observation_local_id=(
            "prediction:1"
        ),
        falsification_criteria=[
            N11OperatorFalsificationDraft(
                local_id="falsifier:1",
                prediction_local_id="prediction:1",
                falsifying_outcome=(
                    "The relative mechanistic balance "
                    "does not change."
                ),
            )
        ],
        assumptions=[],
        generated_relation_status=(
            "INFERENCE_NOT_REPORTED"
        ),
        task_to_supplemental_relation_grounded=False,
    )

    return N11OperatorGenerationDraft(
        candidate=candidate,
        abstention_reason=None,
    )


def validation():
    return N11OperatorGenerationValidation(
        passes=True,
        issues=[],
    )


def supplemental():
    return [
        {
            "node_id":
                "node:supplemental",
            "label":
                "EM plus chemical enhancement",
            "node_text":
                (
                    "SERS was interpreted through "
                    "electromagnetic and chemical enhancement."
                ),
            "source_paper_id":
                "paper:B",
        }
    ]


def compile_candidate():
    return (
        N11CanonicalCandidateCompiler()
        .compile(
            context=context(),
            source_hypothesis_id=(
                "hypothesis:source"
            ),
            draft=draft(),
            validation=validation(),
            supplemental_nodes=
                supplemental(),
        )
    )


def test_canonical_candidate_preserves_three_epistemic_lanes():
    row = compile_candidate()

    assert (
        row.baseline_premise_statement_ids
        == ["stmt:baseline"]
    )

    assert (
        row.gap_statement_ids
        == ["stmt:gap"]
    )

    assert [
        item.node_id
        for item
        in row.supplemental_evidence
    ] == [
        "node:supplemental"
    ]

    assert all(
        item.eligible_as_positive_premise
        is False
        for item
        in row.supplemental_evidence
    )

    assert (
        row.supplemental_promoted_to_positive_premise
        is False
    )

    assert (
        row.gap_promoted_to_positive_premise
        is False
    )


def test_canonical_candidate_tracks_cross_source_grounding_without_premise_promotion():
    row = compile_candidate()

    assert (
        row.baseline_source_paper_ids
        == ["paper:A"]
    )

    assert (
        row.supplemental_source_paper_ids
        == ["paper:B"]
    )

    assert row.cross_source_grounding is True


def test_canonical_candidate_has_no_production_authority_before_novelty_and_n10():
    row = compile_candidate()

    assert row.production_authority is False
    assert row.novelty_status == "NOT_ASSESSED"
    assert row.n10_status == "NOT_ASSESSED"


def test_falsifier_has_exact_canonical_prediction_reference():
    row = compile_candidate()

    assert (
        row.falsifiers[0].prediction_id
        == row.predictions[0].prediction_id
    )

    assert (
        row.discriminating_prediction_id
        == row.predictions[0].prediction_id
    )


def test_compilation_rejects_unvalidated_generation():
    bad = N11OperatorGenerationValidation(
        passes=False,
        issues=[],
    )

    with pytest.raises(
        ValueError,
        match="validation must pass",
    ):
        (
            N11CanonicalCandidateCompiler()
            .compile(
                context=context(),
                source_hypothesis_id=(
                    "hypothesis:source"
                ),
                draft=draft(),
                validation=bad,
                supplemental_nodes=
                    supplemental(),
            )
        )


def test_compilation_rejects_extra_supplemental_node():
    rows = supplemental() + [
        {
            "node_id":
                "node:extra",
            "node_text":
                "extra mechanism",
        }
    ]

    with pytest.raises(
        ValueError,
        match="exactly match",
    ):
        (
            N11CanonicalCandidateCompiler()
            .compile(
                context=context(),
                source_hypothesis_id=(
                    "hypothesis:source"
                ),
                draft=draft(),
                validation=validation(),
                supplemental_nodes=rows,
            )
        )


def test_candidate_identity_is_deterministic():
    first = compile_candidate()
    second = compile_candidate()

    assert (
        first.candidate_id
        == second.candidate_id
    )

    assert (
        first.model_dump(
            mode="json"
        )
        == second.model_dump(
            mode="json"
        )
    )
