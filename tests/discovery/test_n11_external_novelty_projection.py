from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
)
from pipeline_core.discovery.hypothesis_validation import (
    HypothesisValidator,
)
from pipeline_core.discovery.nonobviousness_canonical_candidate import (
    N11CanonicalCandidate,
    N11CanonicalFalsifier,
    N11CanonicalPrediction,
    N11CanonicalSupplementalEvidence,
)
from pipeline_core.discovery.nonobviousness_external_novelty_projection import (
    N11ExternalNoveltyProjectionCompiler,
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
                paper_ids=[
                    "paper:A"
                ],
                eligible_as_premise=True,
            ),
            HypothesisEvidenceStatement(
                statement_id="stmt:gap",
                text=(
                    "The spacing-to-SERS relation "
                    "remains unresolved."
                ),
                epistemic_role="unresolved",
                claim_kind="scope_limit",
                paper_ids=[
                    "paper:A"
                ],
                eligible_as_gap=True,
            ),
        ],
    )


def candidate():
    prediction = (
        N11CanonicalPrediction(
            prediction_id=
                "n11_prediction:test",

            source_local_id=
                "prediction:1",

            observable=(
                "relative electromagnetic-to-chemical "
                "mechanistic signature balance"
            ),

            expected_direction=
                "shift",

            rationale=(
                "Relative mechanistic weighting "
                "should change."
            ),

            discriminating=
                True,
        )
    )

    return N11CanonicalCandidate(
        candidate_id=
            "n11_candidate:test",

        source_hypothesis_id=
            "hypothesis:source",

        source_context_id=
            "context:test",

        source_context_sha256=
            "sha-context",

        source_report_id=
            "report:test",

        source_report_sha256=
            "sha-report",

        domain_profile_id=
            "sers_test",

        task_id=
            "task:test",

        question=
            "How does spacing relate to SERS?",

        operator=
            "RELATIVE_CONTRIBUTION_SHIFT",

        title=(
            "Spacing-dependent relative "
            "mechanistic contribution"
        ),

        hypothesis_statement=(
            "Spacing may change the relative contribution "
            "of electromagnetic and chemical enhancement."
        ),

        hypothesis_type=
            "cross_evidence_synthesis",

        baseline_premise_statement_ids=[
            "stmt:baseline"
        ],

        supplemental_evidence=[
            N11CanonicalSupplementalEvidence(
                node_id=
                    "node:supplemental",

                label=
                    "EM plus chemical mechanism",

                node_text=(
                    "The source describes electromagnetic "
                    "and chemical enhancement."
                ),

                source_paper_id=
                    "paper:B",
            )
        ],

        gap_statement_ids=[
            "stmt:gap"
        ],

        shared_component_ids=[
            "component:em"
        ],

        supplemental_only_component_ids=[
            "component:chemical"
        ],

        relative_contribution_claim=(
            "Spacing may change the relative contribution "
            "of electromagnetic and chemical enhancement."
        ),

        inferential_bridge=(
            "The mechanisms are separately grounded; "
            "their spacing-dependent relative weighting "
            "is inferred."
        ),

        predictions=[
            prediction
        ],

        discriminating_prediction_id=
            prediction.prediction_id,

        falsifiers=[
            N11CanonicalFalsifier(
                falsifier_id=
                    "n11_falsifier:test",

                source_local_id=
                    "falsifier:1",

                prediction_id=
                    prediction.prediction_id,

                falsifying_outcome=(
                    "The relative mechanistic balance "
                    "does not change."
                ),
            )
        ],

        assumptions=[],

        baseline_source_paper_ids=[
            "paper:A"
        ],

        supplemental_source_paper_ids=[
            "paper:B"
        ],

        gap_source_paper_ids=[
            "paper:A"
        ],

        cross_source_grounding=
            True,
    )


def projection():
    return (
        N11ExternalNoveltyProjectionCompiler()
        .compile(
            context=context(),
            candidate=candidate(),
        )
    )


def test_projection_keeps_supplemental_out_of_standard_premise_scope():
    row = projection()

    card = row.portfolio.hypotheses[0]

    assert (
        card.premise_statement_ids
        == ["stmt:baseline"]
    )

    assert (
        card.source_paper_ids
        == ["paper:A"]
    )

    assert (
        "paper:B"
        not in card.source_paper_ids
    )

    assert (
        row.supplemental_promoted_to_positive_premise
        is False
    )


def test_projection_preserves_standard_hypothesis_card_semantics():
    row = projection()

    result = (
        HypothesisValidator()
        .validate(
            context(),
            row.portfolio,
        )
    )

    assert result.passes
    assert result.errors == 0


def test_projection_reuses_exact_prediction_observable_for_falsifier():
    row = projection()

    card = row.portfolio.hypotheses[0]

    assert (
        card.falsification_criteria[0]
        .observable
        == card.predicted_observations[0]
        .observable
    )


def test_projection_is_not_production_authority():
    row = projection()

    assert (
        row.projection_role
        == "EXTERNAL_NOVELTY_INPUT_ONLY"
    )

    assert row.production_authority is False

    assert (
        row.supplemental_provenance_retained_upstream
        is True
    )


def test_projection_identity_is_deterministic():
    first = projection()
    second = projection()

    assert (
        first.model_dump(
            mode="json"
        )
        == second.model_dump(
            mode="json"
        )
    )
