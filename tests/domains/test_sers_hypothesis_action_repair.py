import hashlib
import json

import pytest

from domains.sers.hypothesis_action_repair import (
    SERSG1RepairFeedbackError,
    SERSG1RepairInputBinder,
    SERSG1UnifiedRepairFeedbackBuilder,
)
from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.discovery_axis_inference_contracts import (
    AxisInferenceAssertionDraft,
    AxisInferenceReview,
)
from pipeline_core.discovery.hypothesis_action_application_contracts import (
    G1ApplicationAssertionSource,
    G1ApplicationPlan,
    G1NoveltyDispositionConstraint,
    G1ScientificRepairConstraint,
)
from pipeline_core.discovery.hypothesis_action_contracts import (
    G1FindingScope,
)
from pipeline_core.discovery.hypothesis_compiler import (
    HypothesisCompiler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservation,
    PredictedObservationDraft,
    FalsificationCriterionDraft,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyRefinementReport,
    RefinementAttempt,
)


FINAL_PORTFOLIO = "hypothesis_portfolio:final"
FINAL_HYPOTHESIS = "hypothesis:final"
FINAL_PREDICTION = "prediction:final"


def _canonical_json(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(
            mode="json"
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha_json(value):
    return hashlib.sha256(
        _canonical_json(
            value
        ).encode("utf-8")
    ).hexdigest()


def _sha_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _context():
    return HypothesisContext(
        context_id=
            "hypothesis_context:test",

        context_sha256=
            "a" * 64,

        source_packet_id=
            "packet:test",

        source_packet_sha256=
            "b" * 64,

        source_report_id=
            "report:test",

        source_report_sha256=
            "c" * 64,

        task_id=
            "task:test",

        question=
            "Synthetic question",

        corpus_id=
            "corpus:test",

        domain_profile_id=
            "sers_au_ag",

        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id=
                    "stmt:1",

                text=
                    "A grounded premise.",

                epistemic_role=
                    "reported",

                claim_kind=
                    "mechanistic",

                paper_ids=[
                    "paper:1"
                ],

                eligible_as_premise=
                    True,
            )
        ],
    )


def _proposal():
    return HypothesisProposalDraft(
        local_id=
            "AX1_hypothesis_1",

        title=
            "Synthetic hypothesis",

        hypothesis_statement=
            "A context-dependent relation changes the response.",

        hypothesis_type=
            "context_dependency",

        premise_statement_ids=[
            "stmt:1"
        ],

        inferential_bridge=
            "The relation is proposed as a bounded extension.",

        predicted_observations=[
            PredictedObservationDraft(
                local_id=
                    "pred_1",

                observable=
                    "The response differs qualitatively.",

                expected_direction=
                    "qualitative_change",

                rationale=
                    "This directly tests the relation.",
            )
        ],

        falsification_criteria=[
            FalsificationCriterionDraft(
                local_id=
                    "fals_1",

                observable=
                    "The response differs qualitatively.",

                falsifying_outcome=
                    "No difference is observed.",
            )
        ],

        assumptions=[
            "Compared structures remain otherwise comparable."
        ],
    )


def _original_card():
    portfolio = (
        HypothesisCompiler()
        .compile(
            _context(),
            HypothesisPortfolioDraft(
                hypotheses=[
                    _proposal()
                ]
            ),
        )
    )

    return portfolio.hypotheses[0]


def _final_card():
    original = _original_card()

    return original.model_copy(
        update={
            "hypothesis_id":
                FINAL_HYPOTHESIS,

            "predicted_observations": [
                PredictedObservation(
                    observation_id=
                        FINAL_PREDICTION,

                    observable=
                        original
                        .predicted_observations[0]
                        .observable,

                    expected_direction=
                        original
                        .predicted_observations[0]
                        .expected_direction,

                    rationale=
                        original
                        .predicted_observations[0]
                        .rationale,
                )
            ],

            "falsification_criteria": [
                FalsificationCriterion(
                    criterion_id=
                        "falsifier:final",

                    observable=
                        original
                        .falsification_criteria[0]
                        .observable,

                    falsifying_outcome=
                        original
                        .falsification_criteria[0]
                        .falsifying_outcome,
                )
            ],
        }
    )


def _axis():
    return DiscoveryAxis(
        axis_id=
            "discovery_axis:test",

        axis_rank=
            1,

        inspiration_id=
            "inspiration:test",

        source_path_id=
            "path:test",

        candidate_unit_id=
            "candidate:test",

        label=
            "Synthetic context axis",

        proposed_subject=
            "context",

        proposed_relation=
            "MODERATES",

        proposed_object=
            "response",

        rendered_path=
            "context -> response",

        source_mode=
            "candidate_unit",

        exploration_score=
            1.0,

        planner_score=
            1.0,

        mechanistic_continuity_band=
            "high",
    )


def _r6():
    original = _original_card()

    return NoveltyRefinementReport(
        report_id=
            "novelty_refinement_report:test",

        report_sha256=
            "d" * 64,

        source_portfolio_id=
            "hypothesis_portfolio:original",

        source_external_report_id=
            "external:test",

        source_gap_plan_id=
            "gap_plan:test",

        final_portfolio_id=
            FINAL_PORTFOLIO,

        attempts=[
            RefinementAttempt(
                original_hypothesis_id=
                    original.hypothesis_id,

                candidate_hypothesis_id=
                    original.hypothesis_id,

                final_hypothesis_id=
                    FINAL_HYPOTHESIS,

                gap_id=
                    "gap:test",

                action=
                    "keep",

                decision=
                    "kept_original",

                original_external_status=
                    "NEW_COMBINATION_OF_KNOWN_EFFECTS",

                final_external_status=
                    "NEW_COMBINATION_OF_KNOWN_EFFECTS",

                grounding_preserved=
                    True,

                refinement_generated=
                    False,

                interpretation=
                    "Kept original.",
            )
        ],

        kept_original_count=
            1,
    )


def _plan(
    *,
    target="prediction",
    novelty=False,
):
    card = _final_card()

    if target == "central":
        assertion_id = (
            f"central:{FINAL_HYPOTHESIS}"
        )

        text = card.hypothesis_statement

    elif target == "prediction":
        assertion_id = FINAL_PREDICTION

        prediction = (
            card.predicted_observations[0]
        )

        text = (
            "Observable: "
            + prediction.observable
            + "\nExpected direction: "
            + prediction.expected_direction
            + "\nRationale: "
            + prediction.rationale
        )

    else:
        raise AssertionError(target)

    scope = G1FindingScope(
        kind=target,

        hypothesis_ids=[
            FINAL_HYPOTHESIS
        ],

        assertion_ids=[
            assertion_id
        ],
    )

    scientific = [
        G1ScientificRepairConstraint(
            directive_id=
                "directive:g1",

            source_scope=
                scope,

            source_assertions=[
                G1ApplicationAssertionSource(
                    assertion_id=
                        assertion_id,

                    assertion_kind=
                        target,

                    assertion_text=
                        text,

                    assertion_text_sha256=
                        _sha_text(text),
                )
            ],

            finding_ref_ids=[
                "finding:g1"
            ],

            rationale=
                "Repair context-role semantics.",
        )
    ]

    novelty_rows = []

    if novelty:
        novelty_rows.append(
            G1NoveltyDispositionConstraint(
                directive_id=
                    "directive:novelty",

                finding_ref_ids=[
                    "finding:novelty"
                ],

                rationale=
                    "Historical novelty downgrade.",
            )
        )

    return G1ApplicationPlan(
        plan_id=
            "g1_application_plan:test",

        source_portfolio_id=
            FINAL_PORTFOLIO,

        source_hypothesis_id=
            FINAL_HYPOTHESIS,

        source_decision_id=
            "decision:test",

        source_card_sha256=
            _sha_json(card),

        scientific_repair_constraints=
            scientific,

        novelty_disposition_constraints=
            novelty_rows,
    )


def _d1(
    *,
    prediction_action="KEEP",
    prediction_source_class=
        "S_BOUNDED_SYNTHESIS",
    axis_id="discovery_axis:test",
    hypothesis_id=FINAL_HYPOTHESIS,
):
    card = _final_card()

    central = (
        AxisInferenceAssertionDraft(
            assertion_id=
                f"central:{hypothesis_id}",

            assertion_kind=
                "central_hypothesis",

            assertion_text=
                card.hypothesis_statement,

            source_class=
                "S_BOUNDED_SYNTHESIS",

            action=
                "KEEP",

            grounded_statement_ids=[
                "stmt:1"
            ],

            axis_basis=[
                "Synthetic context axis"
            ],

            rationale=
                "Central relation is bounded.",
        )
    )

    prediction = (
        AxisInferenceAssertionDraft(
            assertion_id=
                FINAL_PREDICTION,

            assertion_kind=
                "prediction",

            assertion_text=
                card
                .predicted_observations[0]
                .observable,

            source_class=
                prediction_source_class,

            action=
                prediction_action,

            grounded_statement_ids=[
                "stmt:1"
            ],

            axis_basis=[
                "Synthetic context axis"
            ],

            rationale=
                "Prediction-level D1 judgment.",
        )
    )

    status = (
        "reframe_required"
        if prediction_action
        in {
            "OPEN_DIRECTION",
            "REFRAME",
            "REMOVE",
        }
        else "pass"
    )

    return AxisInferenceReview(
        review_id=
            "axis_inference_review:test",

        axis_id=
            axis_id,

        hypothesis_id=
            hypothesis_id,

        source_context_id=
            _context().context_id,

        source_context_sha256=
            _context().context_sha256,

        critic_prompt_version=
            "axis-inference-critic-prompt-v1",

        critic_prompt_sha256=
            "e" * 64,

        status=
            status,

        assertions=[
            central,
            prediction,
        ],

        overall_risk=
            (
                "moderate"
                if status
                == "reframe_required"
                else "low"
            ),

        reason_codes=(
            ["open_direction_required"]
            if prediction_action
            == "OPEN_DIRECTION"
            else []
        ),

        interpretation=
            "Synthetic final-bound D1 review.",
    )


def _binding(plan=None):
    plan = plan or _plan()

    return (
        SERSG1RepairInputBinder()
        .bind(
            context=
                _context(),

            source_proposal=
                _proposal(),

            source_card=
                _final_card(),

            axis=
                _axis(),

            refinement_report=
                _r6(),

            application_plan=
                plan,
        )
    )


def test_authoritative_binding_closes_draft_r6_final_chain():
    binding = _binding()

    assert (
        binding.original_hypothesis_id
        == _original_card().hypothesis_id
    )

    assert (
        binding.final_hypothesis_id
        == FINAL_HYPOTHESIS
    )

    assert (
        binding.authoritative_draft_local_id
        == "AX1_hypothesis_1"
    )

    assert (
        binding.axis_id
        == "discovery_axis:test"
    )


def test_d1_keep_plus_g1_reframe_becomes_guarded_g1_reframe():
    plan = _plan(
        target="prediction"
    )

    feedback = (
        SERSG1UnifiedRepairFeedbackBuilder()
        .build(
            binding=_binding(plan),
            application_plan=plan,
            source_card=_final_card(),
            d1_review=_d1(),
        )
    )

    assert len(
        feedback.requirements
    ) == 1

    row = feedback.requirements[0]

    assert (
        row.source_assertion_id
        == FINAL_PREDICTION
    )

    assert (
        row.effective_requirement
        == "g1_reframe_with_d1_guard"
    )

    assert (
        FINAL_PREDICTION
        not in feedback
        .d1_preserve_assertion_ids
    )


def test_d1_open_direction_plus_g1_reframe_merges_once():
    plan = _plan(
        target="prediction"
    )

    feedback = (
        SERSG1UnifiedRepairFeedbackBuilder()
        .build(
            binding=_binding(plan),
            application_plan=plan,
            source_card=_final_card(),
            d1_review=_d1(
                prediction_action=
                    "OPEN_DIRECTION",
            ),
        )
    )

    row = feedback.requirements[0]

    assert (
        row.effective_requirement
        == "d1_and_g1_repair"
    )

    assert (
        row.d1_action
        == "OPEN_DIRECTION"
    )


def test_d1_remove_and_g1_other_scope_both_survive_merge():
    plan = _plan(
        target="central"
    )

    feedback = (
        SERSG1UnifiedRepairFeedbackBuilder()
        .build(
            binding=_binding(plan),
            application_plan=plan,
            source_card=_final_card(),
            d1_review=_d1(
                prediction_action=
                    "REMOVE",

                prediction_source_class=
                    "X_UNSUPPORTED_SPECIFICITY",
            ),
        )
    )

    kinds = {
        row.source_assertion_id:
            row.effective_requirement
        for row in feedback.requirements
    }

    assert kinds[
        f"central:{FINAL_HYPOTHESIS}"
    ] == "g1_reframe_with_d1_guard"

    assert kinds[
        FINAL_PREDICTION
    ] == "d1_repair"


def test_wrong_axis_d1_review_fails_closed():
    plan = _plan()

    with pytest.raises(
        SERSG1RepairFeedbackError,
        match="axis mismatch",
    ):
        (
            SERSG1UnifiedRepairFeedbackBuilder()
            .build(
                binding=_binding(plan),
                application_plan=plan,
                source_card=_final_card(),
                d1_review=_d1(
                    axis_id=
                        "discovery_axis:wrong"
                ),
            )
        )


def test_wrong_final_hypothesis_d1_review_fails_closed():
    plan = _plan()

    review = _d1()

    review = review.model_copy(
        update={
            "hypothesis_id":
                "hypothesis:wrong"
        }
    )

    with pytest.raises(
        SERSG1RepairFeedbackError,
        match="not bound to current final",
    ):
        (
            SERSG1UnifiedRepairFeedbackBuilder()
            .build(
                binding=_binding(plan),
                application_plan=plan,
                source_card=_final_card(),
                d1_review=review,
            )
        )


def test_novelty_downgrade_remains_metadata_only():
    plan = _plan(
        novelty=True
    )

    feedback = (
        SERSG1UnifiedRepairFeedbackBuilder()
        .build(
            binding=_binding(plan),
            application_plan=plan,
            source_card=_final_card(),
            d1_review=_d1(),
        )
    )

    assert (
        feedback
        .novelty_metadata_directive_ids
        == ["directive:novelty"]
    )

    assert (
        feedback
        .novelty_metadata_is_scientific_instruction
        is False
    )

    rendered = (
        SERSG1UnifiedRepairFeedbackBuilder()
        .render(feedback)
    )

    assert (
        "NOT A SCIENTIFIC REWRITE INSTRUCTION"
        in rendered
    )

    assert (
        "External novelty will be reassessed"
        in rendered
    )


def test_render_enforces_single_mutation_owner_and_call():
    plan = _plan()

    builder = (
        SERSG1UnifiedRepairFeedbackBuilder()
    )

    feedback = builder.build(
        binding=_binding(plan),
        application_plan=plan,
        source_card=_final_card(),
        d1_review=_d1(),
    )

    rendered = builder.render(
        feedback
    )

    assert (
        "ONE permitted scientific repair call"
        in rendered
    )

    assert (
        "Preserve premise_statement_ids"
        in rendered
    )

    assert (
        "satisfy BOTH constraints"
        in rendered
    )

def test_render_forbids_cross_assertion_specificity_migration():
    plan = _plan()

    builder = (
        SERSG1UnifiedRepairFeedbackBuilder()
    )

    feedback = builder.build(
        binding=_binding(plan),
        application_plan=plan,
        source_card=_final_card(),
        d1_review=_d1(
            prediction_action=
                "REFRAME",

            prediction_source_class=
                "X_UNSUPPORTED_SPECIFICITY",
        ),
    )

    rendered = builder.render(
        feedback
    )

    assert (
        "sers-g1-unified-repair-render-v1.1"
        in rendered
    )

    assert (
        "MUST NOT be moved or reintroduced"
        in rendered
    )

    assert (
        "inferential bridge"
        in rendered
    )

    assert (
        "falsification criterion"
        in rendered
    )

    assert (
        "non-migration rule"
        in rendered
    )

    assert (
        "not a new D1 scientific judgment"
        in rendered
    )
