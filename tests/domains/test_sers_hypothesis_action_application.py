import pytest

from domains.sers.hypothesis_action_application import (
    SERSG1ApplicationPlanBuilder,
    SERSG1ApplicationPlanError,
)
from pipeline_core.discovery.hypothesis_action_contracts import (
    G1ActionDecision,
    G1ActionDirective,
    G1FindingRef,
    G1FindingScope,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    PredictedObservation,
)


PORTFOLIO = "hypothesis_portfolio:test"
HYPOTHESIS = "hypothesis:test"


def _card():
    return HypothesisCard(
        hypothesis_id=
            HYPOTHESIS,

        domain_profile_id=
            "sers_au_ag",

        source_context_id=
            "context:test",

        source_context_sha256=
            "a" * 64,

        source_report_id=
            "report:test",

        source_report_sha256=
            "b" * 64,

        title=
            "Synthetic SERS hypothesis",

        hypothesis_statement=
            "A contextual moderator changes the response.",

        hypothesis_type=
            "context_dependency",

        premise_statement_ids=[
            "statement:1"
        ],

        gap_statement_ids=[],

        inferential_bridge=
            "The relation is proposed as a bounded extension.",

        predicted_observations=[
            PredictedObservation(
                observation_id=
                    "prediction:test",

                observable=
                    "The response differs qualitatively.",

                expected_direction=
                    "qualitative_change",

                rationale=
                    "This tests the moderator relation.",
            )
        ],

        falsification_criteria=[
            FalsificationCriterion(
                criterion_id=
                    "falsifier:test",

                observable=
                    "The response differs qualitatively.",

                falsifying_outcome=
                    "No context-dependent difference is observed.",
            )
        ],

        assumptions=[
            "The compared structures remain otherwise comparable."
        ],

        source_paper_ids=[
            "paper:1"
        ],

        gap_paper_ids=[],

        cross_paper_synthesis=
            False,

        candidate_dependency=
            "none",

        evidence_profile=
            HypothesisEvidenceProfile(
                premise_count=1,
                gap_count=0,
                source_paper_count=1,
                candidate_premise_count=0,
                reported_premise_count=1,
                synthesis_premise_count=0,
            ),
    )


def _finding(
    *,
    ref_id,
    scope,
    source_kind="context_review",
):
    return G1FindingRef(
        finding_ref_id=
            ref_id,

        source_kind=
            source_kind,

        source_artifact_id=
            "artifact:test",

        source_finding_id=
            "source:" + ref_id,

        source_status=
            "role_mismatch",

        authority=
            "actionable",

        source_portfolio_id=
            PORTFOLIO,

        source_hypothesis_ids=[
            HYPOTHESIS
        ],

        source_scope=
            scope,

        target_portfolio_id=
            PORTFOLIO,

        target_hypothesis_id=
            HYPOTHESIS,

        target_scope=
            scope,

        rationale=
            "synthetic actionable finding",
    )


def _decision(
    *,
    directives,
    findings,
):
    return G1ActionDecision(
        decision_id=
            "g1_action_decision:test",

        target_portfolio_id=
            PORTFOLIO,

        target_hypothesis_id=
            HYPOTHESIS,

        findings=
            findings,

        directives=
            directives,

        disposition=
            "repair_required",

        reason_codes=[
            "TEST_REPAIR"
        ],

        interpretation=
            "Synthetic repair-required decision.",

        mutation_applied=
            False,
    )


def test_reframe_binds_exact_source_text():
    scope = G1FindingScope(
        kind="central",
        hypothesis_ids=[
            HYPOTHESIS
        ],
        assertion_ids=[
            f"central:{HYPOTHESIS}"
        ],
    )

    finding = _finding(
        ref_id="finding:central",
        scope=scope,
    )

    directive = G1ActionDirective(
        directive_id=
            "directive:central",

        action=
            "reframe",

        target_scope=
            scope,

        finding_ref_ids=[
            finding.finding_ref_id
        ],

        rationale=
            "central reframe",
    )

    plan = (
        SERSG1ApplicationPlanBuilder()
        .build(
            source_card=
                _card(),

            decision=
                _decision(
                    directives=[
                        directive
                    ],
                    findings=[
                        finding
                    ],
                ),
        )
    )

    assert (
        plan.mutation_owner
        == "hypothesis_draft_backend_repair"
    )

    assert (
        plan.max_scientific_repair_calls
        == 1
    )

    assert (
        plan.pre_repair_axis_inference_review_required
        is True
    )

    assert len(
        plan.scientific_repair_constraints
    ) == 1

    source = (
        plan
        .scientific_repair_constraints[0]
        .source_assertions[0]
    )

    assert (
        source.assertion_id
        == f"central:{HYPOTHESIS}"
    )

    assert (
        source.assertion_text
        == _card().hypothesis_statement
    )


def test_prediction_reframe_is_bound_to_prediction():
    scope = G1FindingScope(
        kind="prediction",
        hypothesis_ids=[
            HYPOTHESIS
        ],
        assertion_ids=[
            "prediction:test"
        ],
    )

    finding = _finding(
        ref_id="finding:prediction",
        scope=scope,
    )

    directive = G1ActionDirective(
        directive_id=
            "directive:prediction",

        action=
            "reframe",

        target_scope=
            scope,

        finding_ref_ids=[
            finding.finding_ref_id
        ],

        rationale=
            "prediction reframe",
    )

    plan = (
        SERSG1ApplicationPlanBuilder()
        .build(
            source_card=_card(),
            decision=_decision(
                directives=[
                    directive
                ],
                findings=[
                    finding
                ],
            ),
        )
    )

    row = (
        plan
        .scientific_repair_constraints[0]
        .source_assertions[0]
    )

    assert (
        row.assertion_kind
        == "prediction"
    )

    assert (
        row.assertion_id
        == "prediction:test"
    )


def test_downgrade_is_metadata_not_text_mutation():
    scope = G1FindingScope(
        kind="hypothesis",
        hypothesis_ids=[
            HYPOTHESIS
        ],
    )

    finding = _finding(
        ref_id="finding:novelty",
        scope=scope,
        source_kind=
            "external_novelty",
    )

    directive = G1ActionDirective(
        directive_id=
            "directive:downgrade",

        action=
            "downgrade",

        target_scope=
            scope,

        finding_ref_ids=[
            finding.finding_ref_id
        ],

        rationale=
            "novelty downgrade",
    )

    plan = (
        SERSG1ApplicationPlanBuilder()
        .build(
            source_card=_card(),
            decision=_decision(
                directives=[
                    directive
                ],
                findings=[
                    finding
                ],
            ),
        )
    )

    assert (
        plan.scientific_repair_constraints
        == []
    )

    assert len(
        plan.novelty_disposition_constraints
    ) == 1

    row = (
        plan
        .novelty_disposition_constraints[0]
    )

    assert (
        row.storage_target
        == "application_artifact"
    )

    assert (
        row.scientific_text_mutation
        is False
    )


def test_missing_source_assertion_fails_closed():
    scope = G1FindingScope(
        kind="prediction",
        hypothesis_ids=[
            HYPOTHESIS
        ],
        assertion_ids=[
            "prediction:missing"
        ],
    )

    finding = _finding(
        ref_id="finding:missing",
        scope=scope,
    )

    directive = G1ActionDirective(
        directive_id=
            "directive:missing",

        action=
            "reframe",

        target_scope=
            scope,

        finding_ref_ids=[
            finding.finding_ref_id
        ],

        rationale=
            "missing target",
    )

    with pytest.raises(
        SERSG1ApplicationPlanError,
        match="absent from source card",
    ):
        (
            SERSG1ApplicationPlanBuilder()
            .build(
                source_card=_card(),
                decision=_decision(
                    directives=[
                        directive
                    ],
                    findings=[
                        finding
                    ],
                ),
            )
        )


def test_uncovered_actionable_finding_fails_closed():
    central_scope = G1FindingScope(
        kind="central",
        hypothesis_ids=[
            HYPOTHESIS
        ],
        assertion_ids=[
            f"central:{HYPOTHESIS}"
        ],
    )

    bridge_scope = G1FindingScope(
        kind="bridge",
        hypothesis_ids=[
            HYPOTHESIS
        ],
        assertion_ids=[
            f"bridge:{HYPOTHESIS}"
        ],
    )

    central = _finding(
        ref_id="finding:central",
        scope=central_scope,
    )

    bridge = _finding(
        ref_id="finding:bridge",
        scope=bridge_scope,
    )

    directive = G1ActionDirective(
        directive_id=
            "directive:central",

        action=
            "reframe",

        target_scope=
            central_scope,

        finding_ref_ids=[
            central.finding_ref_id
        ],

        rationale=
            "central only",
    )

    with pytest.raises(
        SERSG1ApplicationPlanError,
        match="exactly cover actionable",
    ):
        (
            SERSG1ApplicationPlanBuilder()
            .build(
                source_card=_card(),
                decision=_decision(
                    directives=[
                        directive
                    ],
                    findings=[
                        central,
                        bridge,
                    ],
                ),
            )
        )


def test_remove_assertion_not_yet_implemented():
    scope = G1FindingScope(
        kind="prediction",
        hypothesis_ids=[
            HYPOTHESIS
        ],
        assertion_ids=[
            "prediction:test"
        ],
    )

    finding = _finding(
        ref_id="finding:remove",
        scope=scope,
    )

    directive = G1ActionDirective(
        directive_id=
            "directive:remove",

        action=
            "remove_assertion",

        target_scope=
            scope,

        finding_ref_ids=[
            finding.finding_ref_id
        ],

        rationale=
            "remove",
    )

    with pytest.raises(
        SERSG1ApplicationPlanError,
        match="not implemented",
    ):
        (
            SERSG1ApplicationPlanBuilder()
            .build(
                source_card=_card(),
                decision=_decision(
                    directives=[
                        directive
                    ],
                    findings=[
                        finding
                    ],
                ),
            )
        )


def test_complete_revalidation_set_is_required():
    scope = G1FindingScope(
        kind="central",
        hypothesis_ids=[
            HYPOTHESIS
        ],
        assertion_ids=[
            f"central:{HYPOTHESIS}"
        ],
    )

    finding = _finding(
        ref_id="finding:central",
        scope=scope,
    )

    directive = G1ActionDirective(
        directive_id=
            "directive:central",

        action=
            "reframe",

        target_scope=
            scope,

        finding_ref_ids=[
            finding.finding_ref_id
        ],

        rationale=
            "central reframe",
    )

    plan = (
        SERSG1ApplicationPlanBuilder()
        .build(
            source_card=_card(),
            decision=_decision(
                directives=[
                    directive
                ],
                findings=[
                    finding
                ],
            ),
        )
    )

    assert set(
        plan.required_post_repair_checks
    ) == {
        "compile_validate",
        "axis_fidelity",
        "axis_inference",
        "context_review",
        "internal_novelty",
        "external_novelty",
        "semantic_review",
    }
