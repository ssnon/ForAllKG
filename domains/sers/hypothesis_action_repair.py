from __future__ import annotations

import hashlib
import json

from domains.sers.hypothesis_context_contracts import (
    expected_hypothesis_context_assertions,
)
from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.discovery_axis_inference_contracts import (
    AxisInferenceReview,
)
from pipeline_core.discovery.discovery_axis_inference_prompt import (
    expected_assertions as expected_d1_assertions,
)
from pipeline_core.discovery.hypothesis_action_application_contracts import (
    G1ApplicationPlan,
)
from pipeline_core.discovery.hypothesis_action_repair_contracts import (
    G1RepairInputBinding,
    G1UnifiedRepairFeedback,
    G1UnifiedRepairRequirement,
)
from pipeline_core.discovery.hypothesis_compiler import (
    HypothesisCompiler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisContext,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyRefinementReport,
)


_POLICY_VERSION = (
    "sers-g1-unified-repair-feedback-v1"
)

_D1_REPAIR_ACTIONS = {
    "OPEN_DIRECTION",
    "REFRAME",
    "REMOVE",
}

_D1_KEEP_ACTIONS = {
    "KEEP",
    "KEEP_HYPOTHETICAL",
}


class SERSG1RepairFeedbackError(
    ValueError
):
    pass


def _canonical_json(
    value: object,
) -> str:
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


def _sha_json(
    value: object,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            value
        ).encode("utf-8")
    ).hexdigest()


def _sha_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    value: object,
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            value
        ).encode("utf-8")
    ).hexdigest()[:20]

    return f"{prefix}:{digest}"


def _proposal_scientific_payload(
    proposal: HypothesisProposalDraft,
) -> dict[str, object]:
    return {
        "title":
            proposal.title,

        "hypothesis_statement":
            proposal.hypothesis_statement,

        "hypothesis_type":
            proposal.hypothesis_type,

        "premise_statement_ids":
            sorted(
                proposal.premise_statement_ids
            ),

        "gap_statement_ids":
            sorted(
                proposal.gap_statement_ids
            ),

        "inferential_bridge":
            proposal.inferential_bridge,

        "predicted_observations": [
            {
                "observable":
                    row.observable,

                "expected_direction":
                    row.expected_direction,

                "rationale":
                    row.rationale,
            }
            for row
            in proposal.predicted_observations
        ],

        "falsification_criteria": [
            {
                "observable":
                    row.observable,

                "falsifying_outcome":
                    row.falsifying_outcome,
            }
            for row
            in proposal.falsification_criteria
        ],

        "assumptions":
            list(
                proposal.assumptions
            ),
    }


def _card_scientific_payload(
    card: HypothesisCard,
) -> dict[str, object]:
    return {
        "title":
            card.title,

        "hypothesis_statement":
            card.hypothesis_statement,

        "hypothesis_type":
            card.hypothesis_type,

        "premise_statement_ids":
            sorted(
                card.premise_statement_ids
            ),

        "gap_statement_ids":
            sorted(
                card.gap_statement_ids
            ),

        "inferential_bridge":
            card.inferential_bridge,

        "predicted_observations": [
            {
                "observable":
                    row.observable,

                "expected_direction":
                    row.expected_direction,

                "rationale":
                    row.rationale,
            }
            for row
            in card.predicted_observations
        ],

        "falsification_criteria": [
            {
                "observable":
                    row.observable,

                "falsifying_outcome":
                    row.falsifying_outcome,
            }
            for row
            in card.falsification_criteria
        ],

        "assumptions":
            list(
                card.assumptions
            ),
    }


class SERSG1RepairInputBinder:
    policy_version = _POLICY_VERSION

    def bind(
        self,
        *,
        context: HypothesisContext,
        source_proposal: HypothesisProposalDraft,
        source_card: HypothesisCard,
        axis: DiscoveryAxis,
        refinement_report: NoveltyRefinementReport,
        application_plan: G1ApplicationPlan,
    ) -> G1RepairInputBinding:

        if (
            source_card.source_context_id
            != context.context_id
            or source_card.source_context_sha256
            != context.context_sha256
        ):
            raise SERSG1RepairFeedbackError(
                "source card/context binding mismatch"
            )

        if (
            application_plan.source_hypothesis_id
            != source_card.hypothesis_id
        ):
            raise SERSG1RepairFeedbackError(
                "application plan/card "
                "hypothesis mismatch"
            )

        if (
            application_plan.source_portfolio_id
            != refinement_report
            .final_portfolio_id
        ):
            raise SERSG1RepairFeedbackError(
                "application plan/R6 final "
                "portfolio mismatch"
            )

        actual_card_sha = _sha_json(
            source_card
        )

        if (
            actual_card_sha
            != application_plan
            .source_card_sha256
        ):
            raise SERSG1RepairFeedbackError(
                "application plan source-card "
                "SHA mismatch"
            )

        single_draft = (
            HypothesisPortfolioDraft(
                hypotheses=[
                    source_proposal
                ]
            )
        )

        compiled = (
            HypothesisCompiler()
            .compile(
                context,
                single_draft,
            )
        )

        if (
            len(compiled.hypotheses)
            != 1
        ):
            raise SERSG1RepairFeedbackError(
                "authoritative source proposal "
                "did not compile to exactly "
                "one hypothesis"
            )

        original_card = (
            compiled.hypotheses[0]
        )

        attempts = [
            row
            for row
            in refinement_report.attempts
            if (
                row.final_hypothesis_id
                == source_card.hypothesis_id
            )
        ]

        if len(attempts) != 1:
            raise SERSG1RepairFeedbackError(
                "expected exactly one R6 attempt "
                "for final hypothesis"
            )

        attempt = attempts[0]

        if (
            attempt.decision
            != "kept_original"
        ):
            raise SERSG1RepairFeedbackError(
                "G1 repair-input v1 requires "
                "R6 kept_original survivor"
            )

        if (
            attempt.original_hypothesis_id
            != original_card.hypothesis_id
        ):
            raise SERSG1RepairFeedbackError(
                "authoritative draft does not "
                "compile to R6 original identity"
            )

        if (
            attempt.candidate_hypothesis_id
            != attempt.original_hypothesis_id
        ):
            raise SERSG1RepairFeedbackError(
                "kept_original R6 candidate "
                "identity drift"
            )

        proposal_payload = (
            _proposal_scientific_payload(
                source_proposal
            )
        )

        final_payload = (
            _card_scientific_payload(
                source_card
            )
        )

        if (
            proposal_payload
            != final_payload
        ):
            raise SERSG1RepairFeedbackError(
                "authoritative draft/final "
                "scientific payload mismatch"
            )

        payload_sha = _sha_json(
            proposal_payload
        )

        binding_payload = {
            "policy_version":
                self.policy_version,

            "application_plan_id":
                application_plan.plan_id,

            "source_portfolio_id":
                application_plan
                .source_portfolio_id,

            "original_hypothesis_id":
                original_card.hypothesis_id,

            "final_hypothesis_id":
                source_card.hypothesis_id,

            "authoritative_draft_local_id":
                source_proposal.local_id,

            "refinement_report_id":
                refinement_report.report_id,

            "axis_id":
                axis.axis_id,

            "context_id":
                context.context_id,

            "context_sha256":
                context.context_sha256,

            "source_card_sha256":
                actual_card_sha,

            "scientific_payload_sha256":
                payload_sha,
        }

        return G1RepairInputBinding(
            binding_id=
                _stable_id(
                    "g1_repair_input_binding",
                    binding_payload,
                ),

            application_plan_id=
                application_plan.plan_id,

            source_portfolio_id=
                application_plan
                .source_portfolio_id,

            original_hypothesis_id=
                original_card.hypothesis_id,

            final_hypothesis_id=
                source_card.hypothesis_id,

            authoritative_draft_local_id=
                source_proposal.local_id,

            refinement_report_id=
                refinement_report.report_id,

            axis_id=
                axis.axis_id,

            context_id=
                context.context_id,

            context_sha256=
                context.context_sha256,

            source_card_sha256=
                actual_card_sha,

            scientific_payload_sha256=
                payload_sha,
        )


class SERSG1UnifiedRepairFeedbackBuilder:
    policy_version = _POLICY_VERSION

    # Rendering has execution semantics distinct from the
    # structured merge identity. Version it explicitly so a
    # frozen repair prompt can record the exact instruction
    # surface without changing scientific feedback IDs.
    render_version = (
        "sers-g1-unified-repair-render-v1.1"
    )

    def build(
        self,
        *,
        binding: G1RepairInputBinding,
        application_plan: G1ApplicationPlan,
        source_card: HypothesisCard,
        d1_review: AxisInferenceReview,
    ) -> G1UnifiedRepairFeedback:

        if (
            binding.application_plan_id
            != application_plan.plan_id
        ):
            raise SERSG1RepairFeedbackError(
                "binding/application-plan mismatch"
            )

        if (
            binding.final_hypothesis_id
            != source_card.hypothesis_id
            or application_plan
            .source_hypothesis_id
            != source_card.hypothesis_id
        ):
            raise SERSG1RepairFeedbackError(
                "final hypothesis binding mismatch"
            )

        if (
            _sha_json(source_card)
            != binding.source_card_sha256
        ):
            raise SERSG1RepairFeedbackError(
                "source card changed after "
                "repair-input binding"
            )

        if (
            d1_review.hypothesis_id
            != source_card.hypothesis_id
        ):
            raise SERSG1RepairFeedbackError(
                "D1 review is not bound to "
                "current final hypothesis"
            )

        if (
            d1_review.axis_id
            != binding.axis_id
        ):
            raise SERSG1RepairFeedbackError(
                "D1 review axis mismatch"
            )

        if (
            d1_review.source_context_id
            != binding.context_id
            or d1_review.source_context_sha256
            != binding.context_sha256
        ):
            raise SERSG1RepairFeedbackError(
                "D1 review context mismatch"
            )

        expected_d1 = {
            row["assertion_id"]:
                row
            for row in (
                expected_d1_assertions(
                    source_card
                )
            )
        }

        actual_d1 = {
            row.assertion_id:
                row
            for row in d1_review.assertions
        }

        if (
            set(expected_d1)
            != set(actual_d1)
        ):
            raise SERSG1RepairFeedbackError(
                "D1 review assertion set "
                "does not exactly match "
                "current final card"
            )

        for assertion_id, expected in (
            expected_d1.items()
        ):
            actual = actual_d1[
                assertion_id
            ]

            if (
                actual.assertion_kind
                != expected[
                    "assertion_kind"
                ]
                or actual.assertion_text
                != expected[
                    "assertion_text"
                ]
            ):
                raise SERSG1RepairFeedbackError(
                    "D1 review assertion "
                    "identity/text mismatch: "
                    + assertion_id
                )

        full_catalog = {
            row["assertion_id"]:
                row
            for row in (
                expected_hypothesis_context_assertions(
                    source_card
                )
            )
        }

        g1_by_assertion = {}

        for constraint in (
            application_plan
            .scientific_repair_constraints
        ):
            for assertion in (
                constraint.source_assertions
            ):
                assertion_id = (
                    assertion.assertion_id
                )

                if (
                    assertion_id
                    in g1_by_assertion
                ):
                    raise SERSG1RepairFeedbackError(
                        "multiple G1 constraints "
                        "target one assertion"
                    )

                expected = full_catalog.get(
                    assertion_id
                )

                if expected is None:
                    raise SERSG1RepairFeedbackError(
                        "G1 source assertion absent "
                        "from final source card"
                    )

                if (
                    expected["assertion_text"]
                    != assertion.assertion_text
                ):
                    raise SERSG1RepairFeedbackError(
                        "G1 source assertion text drift"
                    )

                g1_by_assertion[
                    assertion_id
                ] = constraint

        d1_repair_ids = {
            row.assertion_id
            for row in d1_review.assertions
            if row.action
            in _D1_REPAIR_ACTIONS
        }

        target_ids = (
            set(g1_by_assertion)
            | d1_repair_ids
        )

        requirements = []

        for assertion_id in sorted(
            target_ids
        ):
            catalog_row = (
                full_catalog.get(
                    assertion_id
                )
            )

            if catalog_row is None:
                raise SERSG1RepairFeedbackError(
                    "repair target absent from "
                    "full assertion catalog: "
                    + assertion_id
                )

            d1_row = actual_d1.get(
                assertion_id
            )

            g1_constraint = (
                g1_by_assertion.get(
                    assertion_id
                )
            )

            has_g1 = (
                g1_constraint
                is not None
            )

            d1_action = (
                d1_row.action
                if d1_row is not None
                else None
            )

            if (
                d1_action
                in _D1_REPAIR_ACTIONS
                and has_g1
            ):
                effective = (
                    "d1_and_g1_repair"
                )

            elif (
                d1_action
                in _D1_REPAIR_ACTIONS
            ):
                effective = "d1_repair"

            elif (
                d1_action
                in _D1_KEEP_ACTIONS
                and has_g1
            ):
                effective = (
                    "g1_reframe_with_d1_guard"
                )

            elif has_g1:
                effective = "g1_reframe"

            else:
                raise SERSG1RepairFeedbackError(
                    "unreachable unified "
                    "repair target state"
                )

            directive_ids = (
                [
                    g1_constraint.directive_id
                ]
                if has_g1
                else []
            )

            rationales = (
                [
                    g1_constraint.rationale
                ]
                if has_g1
                else []
            )

            text = catalog_row[
                "assertion_text"
            ]

            requirements.append(
                G1UnifiedRepairRequirement(
                    source_assertion_id=
                        assertion_id,

                    assertion_kind=
                        catalog_row[
                            "assertion_kind"
                        ],

                    source_assertion_text=
                        text,

                    source_assertion_text_sha256=
                        _sha_text(text),

                    d1_source_class=(
                        d1_row.source_class
                        if d1_row is not None
                        else None
                    ),

                    d1_action=(
                        d1_action
                    ),

                    d1_rationale=(
                        d1_row.rationale
                        if d1_row is not None
                        else None
                    ),

                    g1_directive_ids=
                        directive_ids,

                    g1_rationales=
                        rationales,

                    effective_requirement=
                        effective,
                )
            )

        d1_preserve = sorted(
            row.assertion_id
            for row in d1_review.assertions
            if (
                row.action
                in _D1_KEEP_ACTIONS
                and row.assertion_id
                not in g1_by_assertion
            )
        )

        novelty_ids = sorted(
            row.directive_id
            for row in (
                application_plan
                .novelty_disposition_constraints
            )
        )

        feedback_payload = {
            "policy_version":
                self.policy_version,

            "binding_id":
                binding.binding_id,

            "application_plan_id":
                application_plan.plan_id,

            "d1_review_id":
                d1_review.review_id,

            "d1_status":
                d1_review.status,

            "requirements": [
                row.model_dump(
                    mode="json"
                )
                for row in requirements
            ],

            "d1_preserve":
                d1_preserve,

            "novelty_metadata":
                novelty_ids,
        }

        return G1UnifiedRepairFeedback(
            feedback_id=
                _stable_id(
                    "g1_unified_repair_feedback",
                    feedback_payload,
                ),

            binding=
                binding,

            pre_repair_d1_review_id=
                d1_review.review_id,

            pre_repair_d1_status=
                d1_review.status,

            requirements=
                requirements,

            d1_preserve_assertion_ids=
                d1_preserve,

            novelty_metadata_directive_ids=
                novelty_ids,
        )

    def render(
        self,
        feedback: G1UnifiedRepairFeedback,
    ) -> str:

        requirement_json = json.dumps(
            [
                row.model_dump(
                    mode="json"
                )
                for row
                in feedback.requirements
            ],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        preserve_json = json.dumps(
            feedback.d1_preserve_assertion_ids,
            ensure_ascii=False,
            indent=2,
        )

        novelty_json = json.dumps(
            feedback.novelty_metadata_directive_ids,
            ensure_ascii=False,
            indent=2,
        )

        b = feedback.binding

        return "\n".join([
            "UNIFIED D1 + G1 BOUNDED HYPOTHESIS REPAIR",
            "==========================================",
            (
                "Render policy: "
                f"{self.render_version}"
            ),
            "",
            "AUTHORITATIVE SOURCE BINDING",
            "----------------------------",
            f"Final hypothesis ID: {b.final_hypothesis_id}",
            f"R6 original hypothesis ID: {b.original_hypothesis_id}",
            (
                "Authoritative draft local_id: "
                f"{b.authoritative_draft_local_id}"
            ),
            f"Discovery axis ID: {b.axis_id}",
            f"Context ID: {b.context_id}",
            (
                "Application plan ID: "
                f"{b.application_plan_id}"
            ),
            (
                "Pre-repair D1 review ID: "
                f"{feedback.pre_repair_d1_review_id}"
            ),
            (
                "Pre-repair D1 status: "
                f"{feedback.pre_repair_d1_status}"
            ),
            "",
            "MUTATION BOUNDARY",
            "-----------------",
            (
                "- This is the ONE permitted scientific repair "
                "call for this hypothesis generation."
            ),
            (
                "- Return a complete replacement "
                "HypothesisPortfolioDraft containing exactly "
                "one repaired hypothesis, or abstain if the "
                "constraints cannot be satisfied."
            ),
            (
                "- Preserve the hypothesis local_id from the "
                "authoritative previous draft."
            ),
            (
                "- Preserve premise_statement_ids and "
                "gap_statement_ids exactly. Do not add, "
                "remove, or substitute evidence premises."
            ),
            (
                "- Preserve title and hypothesis_type unless "
                "a minimal wording adjustment is strictly "
                "necessary for consistency; do not change "
                "the scientific scope."
            ),
            (
                "- Preserve local_ids for retained predictions "
                "and falsification criteria."
            ),
            (
                "- Change only assertions required below, plus "
                "the minimum matching falsification wording "
                "needed for internal consistency."
            ),
            (
                "- Do not promote the discovery axis into "
                "positive evidence."
            ),
            (
                "- Do not add new external-novelty claims."
            ),
            "",
            "MERGE SEMANTICS",
            "---------------",
            (
                "- D1 KEEP / KEEP_HYPOTHETICAL means no "
                "inference-strength weakening is required. "
                "If G1 also targets that assertion, the text "
                "MAY and MUST be locally reframed for G1 while "
                "preserving the D1 epistemic-strength guard."
            ),
            (
                "- D1 OPEN_DIRECTION: remove unsupported sign, "
                "ordering, response shape, optimum, threshold, "
                "or directional specificity."
            ),
            (
                "- D1 REFRAME: retain only the minimum "
                "open-direction relation needed for the "
                "grounded-plus-axis synthesis."
            ),
            (
                "- D1 REMOVE: remove the unsupported "
                "prediction/consequence unless a weaker "
                "directly testable replacement is required "
                "for falsifiability."
            ),
            (
                "- G1 REFRAME: repair only the indicated "
                "context-role/context-compatibility scope. "
                "Do not invent new support."
            ),
            (
                "- When D1 and G1 target the same assertion, "
                "satisfy BOTH constraints in one local rewrite."
            ),
            (
                "- Unsupported specificity identified by D1 "
                "MUST NOT be moved or reintroduced into the "
                "central hypothesis, inferential bridge, "
                "assumptions, another prediction, rationale, "
                "or falsification criterion."
            ),
            (
                "- If a D1 repair weakens or removes "
                "prediction-level specificity, update other "
                "G1-targeted text and matching falsification "
                "wording only as needed to prevent that same "
                "unsupported specificity from surviving "
                "elsewhere. This is a non-migration rule, not "
                "a new D1 scientific judgment on assertions "
                "that D1 did not review."
            ),
            "",
            "UNIFIED SCIENTIFIC REQUIREMENTS",
            "-------------------------------",
            requirement_json,
            "",
            "D1 ASSERTIONS TO PRESERVE UNLESS NEEDED FOR",
            "FALSIFICATION-CONSISTENCY ONLY",
            "------------------------------------------",
            preserve_json,
            "",
            "NOVELTY METADATA — NOT A SCIENTIFIC REWRITE INSTRUCTION",
            "-------------------------------------------------------",
            novelty_json,
            (
                "The novelty directives above are historical "
                "application metadata only. Do NOT weaken, "
                "strengthen, or rewrite the scientific claim "
                "merely to satisfy them. External novelty will "
                "be reassessed on the new generation."
            ),
        ])
