from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pipeline_core.discovery.discovery_axis_contracts import (
    AxisAttemptRecord,
    AxisFidelityReview,
    DiscoveryAxis,
    DiscoveryAxisPlan,
    DiscoveryAxisSynthesisReport,
    DiscoveryHypothesisLineage,
)
from pipeline_core.discovery.discovery_axis_fidelity import DiscoveryAxisFidelityCritic
from pipeline_core.discovery.discovery_axis_inference_contracts import (
    AxisInferenceReview,
)
from pipeline_core.discovery.discovery_axis_inference_critic import (
    DiscoveryAxisInferenceCritic,
)
from pipeline_core.discovery.evidence_family_selection import EvidenceFamilyHierarchy
from pipeline_core.discovery.discovery_axis_prompt import DiscoveryAxisHypothesisPromptAssembler
from pipeline_core.discovery.dual_hypothesis_context import DualHypothesisContext
from pipeline_core.discovery.hypothesis_compiler import HypothesisCompileError, HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftBackend
from pipeline_core.discovery.hypothesis_prompt import HypothesisPrompt
from pipeline_core.discovery.hypothesis_runtime import HypothesisMakerAgentRuntime
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator
from pipeline_core.discovery.internal_novelty import (
    InternalNoveltyAssessor,
    InternalNoveltyCard,
    InternalNoveltyReport,
)


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _route_summary(card: InternalNoveltyCard) -> str:
    route = card.strongest_route_match
    if route is None:
        return ""
    return (
        f"route_id={route.route_id}; premise_coverage={route.premise_coverage:.2f}; "
        f"single_paper={route.single_paper}; statements={','.join(route.matched_statement_ids)}"
    )


def _namespace_proposal(
    proposal: HypothesisProposalDraft,
    *,
    prefix: str,
) -> HypothesisProposalDraft:
    predictions = [
        PredictedObservationDraft(
            local_id=f"{prefix}_{index}_{row.local_id}",
            observable=row.observable,
            expected_direction=row.expected_direction,
            rationale=row.rationale,
        )
        for index, row in enumerate(proposal.predicted_observations, start=1)
    ]
    falsifiers = [
        FalsificationCriterionDraft(
            local_id=f"{prefix}_{index}_{row.local_id}",
            observable=row.observable,
            falsifying_outcome=row.falsifying_outcome,
        )
        for index, row in enumerate(proposal.falsification_criteria, start=1)
    ]
    return HypothesisProposalDraft(
        local_id=f"{prefix}_{proposal.local_id}",
        title=proposal.title,
        hypothesis_statement=proposal.hypothesis_statement,
        hypothesis_type=proposal.hypothesis_type,
        premise_statement_ids=list(proposal.premise_statement_ids),
        gap_statement_ids=list(proposal.gap_statement_ids),
        inferential_bridge=proposal.inferential_bridge,
        predicted_observations=predictions,
        falsification_criteria=falsifiers,
        assumptions=list(proposal.assumptions),
    )


@dataclass(frozen=True)
class AcceptedAxisDraft:
    axis: DiscoveryAxis
    proposal: HypothesisProposalDraft
    fidelity: AxisFidelityReview
    inference: AxisInferenceReview | None
    internal_novelty_status: str
    fidelity_repaired: bool
    inference_repaired: bool
    novelty_repaired: bool


@dataclass(frozen=True)
class AxisPromptRecord:
    axis_id: str
    axis_rank: int
    prompt: HypothesisPrompt


@dataclass(frozen=True)
class DiscoveryAxisSynthesisOutcome:
    portfolio: HypothesisPortfolio
    report: DiscoveryAxisSynthesisReport
    internal_novelty_report: InternalNoveltyReport
    final_draft: HypothesisPortfolioDraft
    axis_prompts: tuple[AxisPromptRecord, ...]
    inference_reviews: tuple[AxisInferenceReview, ...]


class DiscoveryAxisSynthesisRuntime:
    """Per-axis discovery synthesis with bounded fidelity/novelty repair.

    Grounded compiler/validator semantics remain unchanged. Discovery lineage
    is assigned deterministically by the orchestrator rather than trusted to
    the LLM, so inspiration IDs can never become positive premise IDs.
    """

    def __init__(
        self,
        draft_backend: HypothesisDraftBackend,
        mapper: Any,
        *,
        compiler: HypothesisCompiler | None = None,
        validator: HypothesisValidator | None = None,
        fidelity_critic: DiscoveryAxisFidelityCritic | None = None,
        inference_critic: DiscoveryAxisInferenceCritic | None = None,
        novelty_assessor: InternalNoveltyAssessor | None = None,
        max_compile_repairs: int = 1,
        max_fidelity_repairs: int = 1,
        max_inference_repairs: int = 1,
        max_novelty_repairs: int = 1,
        reject_novelty_statuses: tuple[str, ...] = (
            "reconstructs_existing_corpus_claim",
            "reconstructs_existing_corpus_chain",
        ),
        family_hierarchy: EvidenceFamilyHierarchy | None = None,
    ) -> None:
        for name, value in {
            "max_compile_repairs": max_compile_repairs,
            "max_fidelity_repairs": max_fidelity_repairs,
            "max_inference_repairs": max_inference_repairs,
            "max_novelty_repairs": max_novelty_repairs,
        }.items():
            if value not in {0, 1}:
                raise ValueError(f"{name} must be 0 or 1 in alpha4")
        self.backend = draft_backend
        self.mapper = mapper
        self.compiler = compiler or HypothesisCompiler()
        self.validator = validator or HypothesisValidator()
        self.fidelity_critic = fidelity_critic or DiscoveryAxisFidelityCritic()
        self.inference_critic = inference_critic
        self.novelty_assessor = novelty_assessor or InternalNoveltyAssessor()
        self.max_compile_repairs = int(max_compile_repairs)
        self.max_fidelity_repairs = int(max_fidelity_repairs)
        self.max_inference_repairs = int(max_inference_repairs)
        self.max_novelty_repairs = int(max_novelty_repairs)
        self.reject_novelty_statuses = tuple(reject_novelty_statuses)
        self.family_hierarchy = family_hierarchy

    def _compile_validate(
        self,
        context: Any,
        draft: HypothesisPortfolioDraft,
    ) -> tuple[HypothesisPortfolio | None, list[str], list[str]]:
        try:
            portfolio = self.compiler.compile(context, draft)
        except HypothesisCompileError as exc:
            return None, [row.code for row in exc.issues], []
        validation = self.validator.validate(context, portfolio)
        if not validation.passes:
            return portfolio, [], [row.code for row in validation.issues if row.severity == "error"]
        return portfolio, [], []

    def _single_card(
        self,
        portfolio: HypothesisPortfolio | None,
    ) -> Any | None:
        if portfolio is None or len(portfolio.hypotheses) != 1:
            return None
        return portfolio.hypotheses[0]

    def _novelty_card(
        self,
        dual: DualHypothesisContext,
        portfolio: HypothesisPortfolio,
    ) -> InternalNoveltyCard:
        report = self.novelty_assessor.assess(dual, portfolio, self.mapper)
        if len(report.cards) != 1:
            raise RuntimeError("per-axis novelty assessment expected exactly one card")
        return report.cards[0]

    def run(
        self,
        dual: DualHypothesisContext,
        plan: DiscoveryAxisPlan,
    ) -> DiscoveryAxisSynthesisOutcome:
        if plan.source_dual_context_id != dual.dual_context_id:
            raise ValueError("axis plan dual_context_id mismatch")
        if plan.source_dual_context_sha256 != dual.dual_context_sha256:
            raise ValueError("axis plan dual_context_sha256 mismatch")

        context = dual.grounded_context
        accepted: list[AcceptedAxisDraft] = []
        attempts: list[AxisAttemptRecord] = []
        prompt_records: list[AxisPromptRecord] = []

        for axis in plan.axes:
            assembler = DiscoveryAxisHypothesisPromptAssembler(
                axis,
                family_hierarchy=self.family_hierarchy,
            )
            initial_runtime = HypothesisMakerAgentRuntime(
                self.backend,
                prompt_assembler=assembler,
                compiler=self.compiler,
                validator=self.validator,
                max_repairs=self.max_compile_repairs,
            )
            outcome = initial_runtime.run(context)
            prompt_records.append(
                AxisPromptRecord(axis_id=axis.axis_id, axis_rank=axis.axis_rank, prompt=outcome.prompt)
            )
            current_draft = outcome.final_draft
            generation_index = len(outcome.draft_history)

            if current_draft is None:
                attempts.append(
                    AxisAttemptRecord(
                        axis_id=axis.axis_id,
                        stage="initial",
                        generation_index=generation_index,
                        decision="compile_rejected",
                        compile_issue_codes=[row.code for row in outcome.compile_issues],
                    )
                )
                continue

            # Per-axis runs must contain zero (abstention) or exactly one proposal.
            if not current_draft.hypotheses:
                attempts.append(
                    AxisAttemptRecord(
                        axis_id=axis.axis_id,
                        stage="initial",
                        generation_index=generation_index,
                        decision="abstained",
                    )
                )
                continue

            if len(current_draft.hypotheses) != 1:
                feedback = "\n".join(
                    [
                        "PER-AXIS CARDINALITY REPAIR",
                        "===========================",
                        "This run is assigned exactly one discovery axis.",
                        "Return exactly ONE hypothesis for that axis, or abstain.",
                        "Do not return a portfolio of alternatives.",
                    ]
                )
                repaired = self.backend.repair(outcome.prompt, current_draft, feedback)
                current_draft = repaired.draft
                generation_index += 1

            portfolio, compile_codes, validation_codes = self._compile_validate(
                context, current_draft
            )
            card = self._single_card(portfolio)
            if card is None:
                decision = "compile_rejected" if compile_codes else "validation_rejected"
                if not current_draft.hypotheses:
                    decision = "abstained"
                attempts.append(
                    AxisAttemptRecord(
                        axis_id=axis.axis_id,
                        stage="initial",
                        generation_index=generation_index,
                        decision=decision,
                        compile_issue_codes=compile_codes,
                        validation_issue_codes=validation_codes,
                    )
                )
                continue

            fidelity_repaired = False
            inference_repaired = False
            novelty_repaired = False
            inference: AxisInferenceReview | None = None
            fidelity = self.fidelity_critic.review(axis, card, self.mapper.encoder)

            if fidelity.status == "fail" and self.max_fidelity_repairs:
                attempts.append(
                    AxisAttemptRecord(
                        axis_id=axis.axis_id,
                        stage="initial",
                        generation_index=generation_index,
                        decision="fidelity_rejected",
                        hypothesis_id=card.hypothesis_id,
                        title=card.title,
                        fidelity_status=fidelity.status,
                        repair_reason=";".join(fidelity.reason_codes),
                    )
                )
                feedback = assembler.fidelity_repair_feedback(
                    previous_draft=current_draft,
                    reason=fidelity.interpretation + " " + ",".join(fidelity.reason_codes),
                )
                repaired = self.backend.repair(outcome.prompt, current_draft, feedback)
                current_draft = repaired.draft
                generation_index += 1
                fidelity_repaired = True
                if not current_draft.hypotheses:
                    attempts.append(
                        AxisAttemptRecord(
                            axis_id=axis.axis_id,
                            stage="fidelity_repair",
                            generation_index=generation_index,
                            decision="abstained",
                            repair_reason="axis fidelity repair chose abstention",
                        )
                    )
                    continue
                portfolio, compile_codes, validation_codes = self._compile_validate(
                    context, current_draft
                )
                card = self._single_card(portfolio)
                if card is None:
                    attempts.append(
                        AxisAttemptRecord(
                            axis_id=axis.axis_id,
                            stage="fidelity_repair",
                            generation_index=generation_index,
                            decision=(
                                "compile_rejected" if compile_codes else "validation_rejected"
                            ),
                            compile_issue_codes=compile_codes,
                            validation_issue_codes=validation_codes,
                        )
                    )
                    continue
                fidelity = self.fidelity_critic.review(axis, card, self.mapper.encoder)

            if fidelity.status == "fail":
                attempts.append(
                    AxisAttemptRecord(
                        axis_id=axis.axis_id,
                        stage="fidelity_repair" if fidelity_repaired else "initial",
                        generation_index=generation_index,
                        decision="fidelity_rejected",
                        hypothesis_id=card.hypothesis_id,
                        title=card.title,
                        fidelity_status=fidelity.status,
                    )
                )
                continue

            # --------------------------------------------------------------
            # Discovery-axis inference-strength gate.
            #
            # This gate is optional at the library-runtime level so legacy
            # callers/tests can remain compatible. The production Alpha4
            # runner supplies it explicitly.
            # --------------------------------------------------------------
            if self.inference_critic is not None:
                inference_outcome = self.inference_critic.review(
                    context,
                    axis,
                    card,
                )
                inference = inference_outcome.review

                if (
                    inference.status == "reframe_required"
                    and self.max_inference_repairs
                ):
                    attempts.append(
                        AxisAttemptRecord(
                            axis_id=axis.axis_id,
                            stage=(
                                "fidelity_repair"
                                if fidelity_repaired
                                else "initial"
                            ),
                            generation_index=generation_index,
                            decision="inference_rejected",
                            hypothesis_id=card.hypothesis_id,
                            title=card.title,
                            fidelity_status=fidelity.status,
                            inference_status=inference.status,
                            repair_reason=";".join(
                                inference.reason_codes
                            )
                            or inference.interpretation,
                        )
                    )

                    feedback = assembler.inference_repair_feedback(
                        previous_draft=current_draft,
                        review=inference,
                    )

                    repaired = self.backend.repair(
                        outcome.prompt,
                        current_draft,
                        feedback,
                    )

                    current_draft = repaired.draft
                    generation_index += 1
                    inference_repaired = True

                    if not current_draft.hypotheses:
                        attempts.append(
                            AxisAttemptRecord(
                                axis_id=axis.axis_id,
                                stage="inference_repair",
                                generation_index=generation_index,
                                decision="abstained",
                                inference_status="reframe_required",
                                repair_reason=(
                                    "inference-strength repair chose abstention"
                                ),
                            )
                        )
                        continue

                    portfolio, compile_codes, validation_codes = (
                        self._compile_validate(
                            context,
                            current_draft,
                        )
                    )

                    card = self._single_card(portfolio)

                    if card is None:
                        attempts.append(
                            AxisAttemptRecord(
                                axis_id=axis.axis_id,
                                stage="inference_repair",
                                generation_index=generation_index,
                                decision=(
                                    "compile_rejected"
                                    if compile_codes
                                    else "validation_rejected"
                                ),
                                compile_issue_codes=compile_codes,
                                validation_issue_codes=validation_codes,
                                inference_status="reframe_required",
                            )
                        )
                        continue

                    # A repair must preserve assigned-axis fidelity.
                    fidelity = self.fidelity_critic.review(
                        axis,
                        card,
                        self.mapper.encoder,
                    )

                    if fidelity.status == "fail":
                        attempts.append(
                            AxisAttemptRecord(
                                axis_id=axis.axis_id,
                                stage="inference_repair",
                                generation_index=generation_index,
                                decision="fidelity_rejected",
                                hypothesis_id=card.hypothesis_id,
                                title=card.title,
                                fidelity_status=fidelity.status,
                                inference_status="reframe_required",
                                repair_reason=(
                                    "inference repair lost assigned-axis fidelity"
                                ),
                            )
                        )
                        continue

                    inference_outcome = self.inference_critic.review(
                        context,
                        axis,
                        card,
                    )
                    inference = inference_outcome.review

                if inference.status != "pass":
                    attempts.append(
                        AxisAttemptRecord(
                            axis_id=axis.axis_id,
                            stage=(
                                "inference_repair"
                                if inference_repaired
                                else "initial"
                            ),
                            generation_index=generation_index,
                            decision="inference_rejected",
                            hypothesis_id=card.hypothesis_id,
                            title=card.title,
                            fidelity_status=fidelity.status,
                            inference_status=inference.status,
                            repair_reason=(
                                "inference-strength review still requires "
                                "reframing after bounded repair"
                            ),
                        )
                    )
                    continue

            novelty = self._novelty_card(dual, portfolio)
            if novelty.status in self.reject_novelty_statuses and self.max_novelty_repairs:
                attempts.append(
                    AxisAttemptRecord(
                        axis_id=axis.axis_id,
                        stage="fidelity_repair" if fidelity_repaired else "initial",
                        generation_index=generation_index,
                        decision="novelty_rejected",
                        hypothesis_id=card.hypothesis_id,
                        title=card.title,
                        fidelity_status=fidelity.status,
                        internal_novelty_status=novelty.status,
                        repair_reason=novelty.interpretation,
                    )
                )
                feedback = assembler.novelty_repair_feedback(
                    previous_draft=current_draft,
                    novelty_status=novelty.status,
                    interpretation=novelty.interpretation,
                    route_summary=_route_summary(novelty),
                )
                repaired = self.backend.repair(outcome.prompt, current_draft, feedback)
                current_draft = repaired.draft
                generation_index += 1
                novelty_repaired = True
                if not current_draft.hypotheses:
                    attempts.append(
                        AxisAttemptRecord(
                            axis_id=axis.axis_id,
                            stage="novelty_repair",
                            generation_index=generation_index,
                            decision="abstained",
                            repair_reason="internal-novelty repair chose abstention",
                        )
                    )
                    continue
                portfolio, compile_codes, validation_codes = self._compile_validate(
                    context, current_draft
                )
                card = self._single_card(portfolio)
                if card is None:
                    attempts.append(
                        AxisAttemptRecord(
                            axis_id=axis.axis_id,
                            stage="novelty_repair",
                            generation_index=generation_index,
                            decision=(
                                "compile_rejected" if compile_codes else "validation_rejected"
                            ),
                            compile_issue_codes=compile_codes,
                            validation_issue_codes=validation_codes,
                        )
                    )
                    continue
                fidelity = self.fidelity_critic.review(axis, card, self.mapper.encoder)
                if fidelity.status == "fail":
                    attempts.append(
                        AxisAttemptRecord(
                            axis_id=axis.axis_id,
                            stage="novelty_repair",
                            generation_index=generation_index,
                            decision="fidelity_rejected",
                            hypothesis_id=card.hypothesis_id,
                            title=card.title,
                            fidelity_status=fidelity.status,
                            inference_status=(
                                inference.status
                                if inference is not None
                                else "not_assessed"
                            ),
                            repair_reason="novelty repair lost assigned-axis fidelity",
                        )
                    )
                    continue

                if self.inference_critic is not None:
                    inference_outcome = self.inference_critic.review(
                        context,
                        axis,
                        card,
                    )
                    inference = inference_outcome.review

                    if inference.status != "pass":
                        attempts.append(
                            AxisAttemptRecord(
                                axis_id=axis.axis_id,
                                stage="novelty_repair",
                                generation_index=generation_index,
                                decision="inference_rejected",
                                hypothesis_id=card.hypothesis_id,
                                title=card.title,
                                fidelity_status=fidelity.status,
                                inference_status=inference.status,
                                repair_reason=(
                                    "novelty repair introduced or retained "
                                    "unsupported inference specificity"
                                ),
                            )
                        )
                        continue

                novelty = self._novelty_card(dual, portfolio)

            if novelty.status in self.reject_novelty_statuses:
                attempts.append(
                    AxisAttemptRecord(
                        axis_id=axis.axis_id,
                        stage="novelty_repair" if novelty_repaired else "initial",
                        generation_index=generation_index,
                        decision="novelty_rejected",
                        hypothesis_id=card.hypothesis_id,
                        title=card.title,
                        fidelity_status=fidelity.status,
                        internal_novelty_status=novelty.status,
                        repair_reason="still reconstructs corpus prior art after bounded repair",
                    )
                )
                continue

            proposal = current_draft.hypotheses[0]
            accepted.append(
                AcceptedAxisDraft(
                    axis=axis,
                    proposal=proposal,
                    fidelity=fidelity,
                    inference=inference,
                    internal_novelty_status=str(novelty.status),
                    fidelity_repaired=fidelity_repaired,
                    inference_repaired=inference_repaired,
                    novelty_repaired=novelty_repaired,
                )
            )
            attempts.append(
                AxisAttemptRecord(
                    axis_id=axis.axis_id,
                    stage=(
                        "novelty_repair"
                        if novelty_repaired
                        else "fidelity_repair"
                        if fidelity_repaired
                        else "initial"
                    ),
                    generation_index=generation_index,
                    decision="accepted",
                    hypothesis_id=card.hypothesis_id,
                    title=card.title,
                    fidelity_status=fidelity.status,
                    inference_status=(
                        inference.status
                        if inference is not None
                        else "not_assessed"
                    ),
                    internal_novelty_status=novelty.status,
                )
            )

        namespaced = [
            _namespace_proposal(row.proposal, prefix=f"AX{row.axis.axis_rank}")
            for row in accepted
        ]
        final_draft = HypothesisPortfolioDraft(
            hypotheses=namespaced,
            abstention_reason=(
                None
                if namespaced
                else "No discovery axis survived grounded validation, axis-fidelity control, and corpus-internal novelty control."
            ),
        )
        final_portfolio = self.compiler.compile(context, final_draft)
        final_validation = self.validator.validate(context, final_portfolio)
        if not final_validation.passes:
            codes = [row.code for row in final_validation.issues if row.severity == "error"]
            raise RuntimeError(f"aggregate alpha4 portfolio failed deterministic validation: {codes}")

        final_novelty = self.novelty_assessor.assess(dual, final_portfolio, self.mapper)
        novelty_by_id = {row.hypothesis_id: row for row in final_novelty.cards}

        lineages: list[DiscoveryHypothesisLineage] = []
        for accepted_row, final_card in zip(accepted, final_portfolio.hypotheses, strict=True):
            final_card_novelty = novelty_by_id[final_card.hypothesis_id]
            lineages.append(
                DiscoveryHypothesisLineage(
                    hypothesis_id=final_card.hypothesis_id,
                    axis_id=accepted_row.axis.axis_id,
                    inspiration_id=accepted_row.axis.inspiration_id,
                    candidate_unit_id=accepted_row.axis.candidate_unit_id,
                    axis_fidelity_status=accepted_row.fidelity.status,
                    inference_status=(
                        accepted_row.inference.status
                        if accepted_row.inference is not None
                        else "not_assessed"
                    ),
                    internal_novelty_status=final_card_novelty.status,
                    fidelity_repaired=accepted_row.fidelity_repaired,
                    inference_repaired=accepted_row.inference_repaired,
                    novelty_repaired=accepted_row.novelty_repaired,
                )
            )

        portfolio_sha = _sha256_json(final_portfolio)
        report_id = _stable_id(
            "discovery_axis_synthesis_report",
            dual.dual_context_sha256,
            plan.plan_sha256,
            final_portfolio.portfolio_id,
            portfolio_sha,
        )
        report_payload = {
            "schema_version": "discovery-axis-synthesis-report-v1",
            "report_id": report_id,
            "source_dual_context_id": dual.dual_context_id,
            "source_dual_context_sha256": dual.dual_context_sha256,
            "axis_plan_id": plan.plan_id,
            "axis_plan_sha256": plan.plan_sha256,
            "final_portfolio_id": final_portfolio.portfolio_id,
            "final_portfolio_sha256": portfolio_sha,
            "attempted_axis_count": len(plan.axes),
            "accepted_hypothesis_count": len(final_portfolio.hypotheses),
            "lineages": [row.model_dump(mode="json") for row in lineages],
            "attempts": [row.model_dump(mode="json") for row in attempts],
            "external_novelty_status": "not_assessed",
            "policy_version": (
                "discovery-axis-synthesis-policy-v2"
                if self.inference_critic is not None
                else "discovery-axis-synthesis-policy-v1"
            ),
        }
        report = DiscoveryAxisSynthesisReport(
            **report_payload,
            report_sha256=_sha256_json(report_payload),
        )
        return DiscoveryAxisSynthesisOutcome(
            portfolio=final_portfolio,
            report=report,
            internal_novelty_report=final_novelty,
            final_draft=final_draft,
            axis_prompts=tuple(prompt_records),
            inference_reviews=tuple(
                row.inference
                for row in accepted
                if row.inference is not None
            ),
        )
