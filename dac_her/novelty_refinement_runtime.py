from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxisPlan, DiscoveryAxisSynthesisReport
from dac_her.discovery_axis_fidelity import DiscoveryAxisFidelityCritic
from pipeline_core.discovery.dual_hypothesis_context import DualHypothesisContext
from dac_her.external_novelty import ExternalNoveltyAssessor
from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyCard,
    ExternalNoveltyReport,
    LiteratureQueryPlan,
    PriorArtPacket,
)
from pipeline_core.discovery.hypothesis_compiler import HypothesisCompileError, HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisCard,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from dac_her.hypothesis_llm import HypothesisDraftBackend
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator
from dac_her.internal_novelty import InternalNoveltyAssessor
from pipeline_core.discovery.novelty_claim_decomposition import LiteratureQueryPlanner
from dac_her.novelty_gap_analysis import NoveltyGapAnalyzer
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyGapPlan,
    NoveltyRefinementReport,
    RefinementAttempt,
    TargetedSearchRecord,
)
from dac_her.novelty_refinement_prompt import NoveltyRefinementPromptAssembler
from dac_her.targeted_novelty_retrieval import TargetedNoveltyRetriever
from dac_her.prior_art_review_audit import (
    prior_art_review_audit_scope,
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _proposal_from_card(card: HypothesisCard, *, prefix: str) -> HypothesisProposalDraft:
    return HypothesisProposalDraft(
        local_id=f"{prefix}_{card.hypothesis_id.split(':')[-1]}",
        title=card.title,
        hypothesis_statement=card.hypothesis_statement,
        hypothesis_type=card.hypothesis_type,
        premise_statement_ids=list(card.premise_statement_ids),
        gap_statement_ids=list(card.gap_statement_ids),
        inferential_bridge=card.inferential_bridge,
        predicted_observations=[
            PredictedObservationDraft(
                local_id=f"{prefix}_p{i}",
                observable=x.observable,
                expected_direction=x.expected_direction,
                rationale=x.rationale,
            )
            for i, x in enumerate(card.predicted_observations, start=1)
        ],
        falsification_criteria=[
            FalsificationCriterionDraft(
                local_id=f"{prefix}_f{i}",
                observable=x.observable,
                falsifying_outcome=x.falsifying_outcome,
            )
            for i, x in enumerate(card.falsification_criteria, start=1)
        ],
        assumptions=list(card.assumptions),
    )


@dataclass(frozen=True)
class PerHypothesisExternalArtifacts:
    hypothesis_id: str
    query_plan: LiteratureQueryPlan
    prior_art: PriorArtPacket
    report: ExternalNoveltyReport


@dataclass(frozen=True)
class NoveltyRefinementOutcome:
    portfolio: HypothesisPortfolio
    gap_plan: NoveltyGapPlan
    report: NoveltyRefinementReport
    targeted_external_artifacts: tuple[PerHypothesisExternalArtifacts, ...]
    final_external_artifacts: tuple[PerHypothesisExternalArtifacts, ...]


class TargetedNoveltyRefinementRuntime:
    """Bounded alpha6 novelty refinement.

    One hypothesis can be regenerated at most once. External literature is never
    inserted into premise_statement_ids.
    """

    REJECT_INTERNAL = {
        "reconstructs_existing_corpus_claim",
        "reconstructs_existing_corpus_chain",
    }
    REJECT_EXTERNAL = {
        "WELL_ESTABLISHED",
        "CONFLICTING_PRIOR_ART",
    }

    def __init__(
        self,
        *,
        hypothesis_backend: HypothesisDraftBackend,
        external_assessor: ExternalNoveltyAssessor,
        targeted_retriever: TargetedNoveltyRetriever,
        mapper: Any,
        compiler: HypothesisCompiler | None = None,
        validator: HypothesisValidator | None = None,
        gap_analyzer: NoveltyGapAnalyzer | None = None,
        fidelity_critic: DiscoveryAxisFidelityCritic | None = None,
        internal_assessor: InternalNoveltyAssessor | None = None,
    ) -> None:
        self.hypothesis_backend = hypothesis_backend
        self.external_assessor = external_assessor
        self.targeted_retriever = targeted_retriever
        self.mapper = mapper
        self.compiler = compiler or HypothesisCompiler()
        self.validator = validator or HypothesisValidator()
        self.gap_analyzer = gap_analyzer or NoveltyGapAnalyzer()
        self.fidelity_critic = fidelity_critic or DiscoveryAxisFidelityCritic()
        self.internal_assessor = internal_assessor or InternalNoveltyAssessor()

    def _compile_one(
        self,
        dual: DualHypothesisContext,
        draft: HypothesisPortfolioDraft,
    ) -> tuple[HypothesisPortfolio | None, list[str]]:
        try:
            portfolio = self.compiler.compile(dual.grounded_context, draft)
        except HypothesisCompileError as exc:
            return None, [x.code for x in exc.issues]
        validation = self.validator.validate(dual.grounded_context, portfolio)
        if not validation.passes:
            return None, [
                x.code for x in validation.issues if x.severity == "error"
            ]
        if len(portfolio.hypotheses) != 1:
            return None, ["PER_AXIS_CARDINALITY"]
        return portfolio, []

    @staticmethod
    def _grounding_preserved(original: HypothesisCard, refined: HypothesisCard) -> bool:
        return (
            sorted(original.premise_statement_ids)
            == sorted(refined.premise_statement_ids)
            and sorted(original.gap_statement_ids)
            == sorted(refined.gap_statement_ids)
            and original.hypothesis_type == refined.hypothesis_type
        )

    @staticmethod
    def _lock_refinement_provenance(
        original: HypothesisCard,
        draft: HypothesisPortfolioDraft,
    ) -> HypothesisPortfolioDraft:
        """Make provenance an orchestrator-owned invariant.

        The LLM owns scientific wording, predictions, falsifiers, and assumptions.
        It does not own evidence lineage. Any premise/gap/type fields emitted by
        the model are overwritten from the already-grounded original hypothesis
        before deterministic compilation.
        """
        if len(draft.hypotheses) != 1:
            return draft
        row = draft.hypotheses[0]
        locked = row.model_copy(
            update={
                "premise_statement_ids": list(original.premise_statement_ids),
                "gap_statement_ids": list(original.gap_statement_ids),
                "hypothesis_type": original.hypothesis_type,
            }
        )
        return draft.model_copy(
            update={
                "hypotheses": [locked],
                "abstention_reason": None,
            }
        )

    @classmethod
    def _original_fallback_allowed(cls, targeted_status: str) -> bool:
        """Whether a failed optional refinement may keep the original.

        INSFFICIENT_SEARCH_EVIDENCE and LITERATURE_SUPPORTED_EXTENSION are not
        scientific rejection states. A refinement failure must not destroy an
        otherwise grounded candidate. Only a targeted assessment that finds the
        original WELL_ESTABLISHED or CONFLICTING_PRIOR_ART makes refinement
        destructive.
        """
        return targeted_status not in cls.REJECT_EXTERNAL

    def _fresh_external(
        self,
        portfolio: HypothesisPortfolio,
    ) -> PerHypothesisExternalArtifacts:
        decompositions = self.external_assessor.decompose_portfolio(portfolio)
        plan = LiteratureQueryPlanner().build(portfolio, decompositions)
        packet = self.targeted_retriever.retriever.retrieve(plan).packet
        with prior_art_review_audit_scope(
            assessment_kind="alpha6_fresh_final",
            focal_hypothesis_id=portfolio.hypotheses[0].hypothesis_id,
            source_portfolio_id=portfolio.portfolio_id,
            query_plan_id=plan.plan_id,
            prior_art_packet_id=packet.packet_id,
        ):
            report = self.external_assessor.assess(
                portfolio,
                plan,
                packet,
            )
        return PerHypothesisExternalArtifacts(
            hypothesis_id=portfolio.hypotheses[0].hypothesis_id,
            query_plan=plan,
            prior_art=packet,
            report=report,
        )

    def run(
        self,
        *,
        dual: DualHypothesisContext,
        portfolio: HypothesisPortfolio,
        lineage: DiscoveryAxisSynthesisReport,
        axis_plan: DiscoveryAxisPlan,
        external_report: ExternalNoveltyReport,
        external_query_plan: LiteratureQueryPlan,
        external_prior_art: PriorArtPacket,
    ) -> NoveltyRefinementOutcome:
        gap_plan = self.gap_analyzer.build(
            portfolio, external_report, external_query_plan
        )
        cards = {x.hypothesis_id: x for x in portfolio.hypotheses}
        lineage_by_h = {x.hypothesis_id: x for x in lineage.lineages}
        axis_by_id = {x.axis_id: x for x in axis_plan.axes}
        external_by_h = {x.hypothesis_id: x for x in external_report.cards}

        accepted_proposals: list[HypothesisProposalDraft] = []
        attempts: list[RefinementAttempt] = []
        search_records: list[TargetedSearchRecord] = []
        targeted_artifacts: list[PerHypothesisExternalArtifacts] = []
        final_external_artifacts: list[PerHypothesisExternalArtifacts] = []

        for index, gap in enumerate(gap_plan.gaps, start=1):
            original = cards[gap.hypothesis_id]
            source_external = external_by_h[gap.hypothesis_id]

            if gap.action == "keep":
                accepted_proposals.append(
                    _proposal_from_card(original, prefix=f"keep{index}")
                )
                attempts.append(
                    RefinementAttempt(
                        original_hypothesis_id=original.hypothesis_id,
                        final_hypothesis_id=original.hypothesis_id,
                        gap_id=gap.gap_id,
                        action=gap.action,
                        decision="kept_original",
                        original_external_status=source_external.status,
                        targeted_external_status=source_external.status,
                        final_external_status=source_external.status,
                        grounding_preserved=True,
                        refinement_generated=False,
                        interpretation=(
                            "The external novelty status did not require bounded refinement."
                        ),
                    )
                )
                continue

            if not gap.targeted_queries:
                if self._original_fallback_allowed(source_external.status):
                    accepted_proposals.append(
                        _proposal_from_card(original, prefix=f"fallback{index}")
                    )
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            final_hypothesis_id=original.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="kept_original",
                            original_external_status=source_external.status,
                            targeted_external_status=source_external.status,
                            final_external_status=source_external.status,
                            grounding_preserved=True,
                            refinement_generated=False,
                            reason_codes=[
                                "no_targeted_queries",
                                "non_destructive_original_fallback",
                            ],
                            interpretation=(
                                "No non-duplicate targeted query could be generated. "
                                "The grounded original is retained without any novelty upgrade."
                            ),
                        )
                    )
                else:
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="search_insufficient",
                            original_external_status=source_external.status,
                            grounding_preserved=True,
                            reason_codes=["no_targeted_queries"],
                            interpretation=(
                                "No non-duplicate targeted query could be generated for an "
                                "original hypothesis already in a destructive prior-art state."
                            ),
                        )
                    )
                continue

            targeted = self.targeted_retriever.retrieve(
                external_query_plan, external_prior_art, gap
            )
            with prior_art_review_audit_scope(
                assessment_kind="alpha6_targeted_reassessment",
                focal_hypothesis_id=original.hypothesis_id,
                gap_id=gap.gap_id,
                source_portfolio_id=portfolio.portfolio_id,
                query_plan_id=targeted.augmented_plan.plan_id,
                prior_art_packet_id=targeted.merged_packet.packet_id,
            ):
                reassessed = self.external_assessor.assess(
                    portfolio,
                    targeted.augmented_plan,
                    targeted.merged_packet,
                    lineage=lineage,
                )
            targeted_card = next(
                x for x in reassessed.cards
                if x.hypothesis_id == original.hypothesis_id
            )

            def keep_original_after_failed_refinement(
                failure_decision: str,
                *,
                reason_codes: list[str] | None = None,
                failure_interpretation: str,
                axis_fidelity_status: str | None = None,
                internal_novelty_status: str | None = None,
            ) -> None:
                accepted_proposals.append(
                    _proposal_from_card(original, prefix=f"fallback{index}")
                )
                attempts.append(
                    RefinementAttempt(
                        original_hypothesis_id=original.hypothesis_id,
                        final_hypothesis_id=original.hypothesis_id,
                        gap_id=gap.gap_id,
                        action=gap.action,
                        decision="kept_original",
                        original_external_status=source_external.status,
                        targeted_external_status=targeted_card.status,
                        final_external_status=targeted_card.status,
                        axis_fidelity_status=axis_fidelity_status,
                        internal_novelty_status=internal_novelty_status,
                        grounding_preserved=True,
                        refinement_generated=True,
                        reason_codes=sorted(
                            set(
                                [
                                    "non_destructive_original_fallback",
                                    f"refinement_failure:{failure_decision}",
                                    *(reason_codes or []),
                                ]
                            )
                        ),
                        interpretation=(
                            failure_interpretation
                            + " The optional refinement is discarded and the grounded "
                              "original is retained at its targeted external-novelty "
                              f"status ({targeted_card.status}); no novelty upgrade is claimed."
                        ),
                    )
                )
            targeted_artifacts.append(
                PerHypothesisExternalArtifacts(
                    hypothesis_id=original.hypothesis_id,
                    query_plan=targeted.augmented_plan,
                    prior_art=targeted.merged_packet,
                    report=reassessed,
                )
            )
            search_records.append(
                TargetedSearchRecord(
                    hypothesis_id=original.hypothesis_id,
                    gap_id=gap.gap_id,
                    query_plan_id=targeted.augmented_plan.plan_id,
                    prior_art_packet_id=targeted.merged_packet.packet_id,
                    external_report_id=reassessed.report_id,
                    external_status_after_search=targeted_card.status,
                    unique_work_count=targeted_card.coverage.unique_work_count,
                    abstract_work_count=targeted_card.coverage.abstract_work_count,
                    successful_query_count=targeted_card.coverage.successful_query_count,
                )
            )

            # If targeted search resolves an insufficient result into a positive
            # novelty category, keep the original rather than optimizing it further.
            if (
                source_external.status == "INSUFFICIENT_SEARCH_EVIDENCE"
                and targeted_card.status
                in {"NEW_COMBINATION_OF_KNOWN_EFFECTS", "PLAUSIBLY_NOVEL"}
            ):
                accepted_proposals.append(
                    _proposal_from_card(original, prefix=f"keep{index}")
                )
                attempts.append(
                    RefinementAttempt(
                        original_hypothesis_id=original.hypothesis_id,
                        final_hypothesis_id=original.hypothesis_id,
                        gap_id=gap.gap_id,
                        action=gap.action,
                        decision="kept_original",
                        original_external_status=source_external.status,
                        targeted_external_status=targeted_card.status,
                        final_external_status=targeted_card.status,
                        grounding_preserved=True,
                        refinement_generated=False,
                        interpretation=(
                            "Targeted search resolved the prior uncertainty without "
                            "requiring hypothesis regeneration."
                        ),
                    )
                )
                continue

            assembler = NoveltyRefinementPromptAssembler(
                original=original,
                gap=gap,
                targeted_card=targeted_card,
            )
            prompt = assembler.build(dual.grounded_context)
            generation = self.hypothesis_backend.generate(prompt)
            draft = generation.draft
            if not draft.hypotheses:
                if self._original_fallback_allowed(targeted_card.status):
                    keep_original_after_failed_refinement(
                        "abstained",
                        failure_interpretation="The bounded refinement model abstained.",
                    )
                else:
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="abstained",
                            original_external_status=source_external.status,
                            targeted_external_status=targeted_card.status,
                            grounding_preserved=True,
                            refinement_generated=True,
                            interpretation="The bounded refinement model abstained.",
                        )
                    )
                continue
            if len(draft.hypotheses) != 1:
                if self._original_fallback_allowed(targeted_card.status):
                    keep_original_after_failed_refinement(
                        "validation_rejected",
                        reason_codes=["per_hypothesis_cardinality_violation"],
                        failure_interpretation="Refinement returned more than one hypothesis.",
                    )
                else:
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="validation_rejected",
                            original_external_status=source_external.status,
                            targeted_external_status=targeted_card.status,
                            grounding_preserved=False,
                            refinement_generated=True,
                            reason_codes=["per_hypothesis_cardinality_violation"],
                            interpretation="Refinement returned more than one hypothesis.",
                        )
                    )
                continue

            # Provenance is orchestrator-owned. Do not let model-generated IDs
            # determine whether an otherwise valid refinement compiles.
            draft = self._lock_refinement_provenance(original, draft)
            compiled, issue_codes = self._compile_one(dual, draft)
            if compiled is None:
                if self._original_fallback_allowed(targeted_card.status):
                    keep_original_after_failed_refinement(
                        "compile_rejected",
                        reason_codes=issue_codes,
                        failure_interpretation=(
                            "Refined hypothesis failed deterministic compile/validation."
                        ),
                    )
                else:
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="compile_rejected",
                            original_external_status=source_external.status,
                            targeted_external_status=targeted_card.status,
                            grounding_preserved=False,
                            refinement_generated=True,
                            reason_codes=issue_codes,
                            interpretation=(
                                "Refined hypothesis failed deterministic compile/validation."
                            ),
                        )
                    )
                continue
            refined = compiled.hypotheses[0]

            if not self._grounding_preserved(original, refined):
                if self._original_fallback_allowed(targeted_card.status):
                    keep_original_after_failed_refinement(
                        "grounding_drift_rejected",
                        reason_codes=["premise_gap_or_type_changed_after_provenance_lock"],
                        failure_interpretation=(
                            "Refinement changed grounded evidence lineage or hypothesis type "
                            "even after deterministic provenance locking."
                        ),
                    )
                else:
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            final_hypothesis_id=refined.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="grounding_drift_rejected",
                            original_external_status=source_external.status,
                            targeted_external_status=targeted_card.status,
                            grounding_preserved=False,
                            refinement_generated=True,
                            reason_codes=["premise_gap_or_type_changed_after_provenance_lock"],
                            interpretation=(
                                "Refinement changed grounded evidence lineage or hypothesis type "
                                "even after deterministic provenance locking."
                            ),
                        )
                    )
                continue

            lineage_row = lineage_by_h.get(original.hypothesis_id)
            axis = axis_by_id.get(lineage_row.axis_id) if lineage_row else None
            fidelity_status = None
            if axis is not None:
                fidelity = self.fidelity_critic.review(
                    axis, refined, self.mapper.encoder
                )
                fidelity_status = fidelity.status
                if fidelity.status == "fail":
                    if self._original_fallback_allowed(targeted_card.status):
                        keep_original_after_failed_refinement(
                            "axis_fidelity_rejected",
                            reason_codes=list(fidelity.reason_codes),
                            failure_interpretation=(
                                "Refinement escaped the assigned discovery axis."
                            ),
                            axis_fidelity_status=fidelity.status,
                        )
                    else:
                        attempts.append(
                            RefinementAttempt(
                                original_hypothesis_id=original.hypothesis_id,
                                final_hypothesis_id=refined.hypothesis_id,
                                gap_id=gap.gap_id,
                                action=gap.action,
                                decision="axis_fidelity_rejected",
                                original_external_status=source_external.status,
                                targeted_external_status=targeted_card.status,
                                axis_fidelity_status=fidelity.status,
                                grounding_preserved=True,
                                refinement_generated=True,
                                reason_codes=list(fidelity.reason_codes),
                                interpretation=(
                                    "Refinement escaped the assigned discovery axis."
                                ),
                            )
                        )
                    continue

            internal = self.internal_assessor.assess(
                dual, compiled, self.mapper
            ).cards[0]
            if internal.status in self.REJECT_INTERNAL:
                if self._original_fallback_allowed(targeted_card.status):
                    keep_original_after_failed_refinement(
                        "internal_novelty_rejected",
                        reason_codes=list(internal.reason_codes),
                        failure_interpretation=internal.interpretation,
                        axis_fidelity_status=fidelity_status,
                        internal_novelty_status=internal.status,
                    )
                else:
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            final_hypothesis_id=refined.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="internal_novelty_rejected",
                            original_external_status=source_external.status,
                            targeted_external_status=targeted_card.status,
                            axis_fidelity_status=fidelity_status,
                            internal_novelty_status=internal.status,
                            grounding_preserved=True,
                            refinement_generated=True,
                            reason_codes=list(internal.reason_codes),
                            interpretation=internal.interpretation,
                        )
                    )
                continue

            fresh = self._fresh_external(compiled)
            final_external_artifacts.append(fresh)
            final_card = fresh.report.cards[0]
            if final_card.status in self.REJECT_EXTERNAL:
                if self._original_fallback_allowed(targeted_card.status):
                    keep_original_after_failed_refinement(
                        "external_novelty_rejected",
                        reason_codes=list(final_card.reason_codes),
                        failure_interpretation=(
                            "The refined wording hit direct/conflicting prior art under "
                            "a fresh search and is therefore discarded. "
                            + final_card.interpretation
                        ),
                        axis_fidelity_status=fidelity_status,
                        internal_novelty_status=internal.status,
                    )
                else:
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            final_hypothesis_id=refined.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="external_novelty_rejected",
                            original_external_status=source_external.status,
                            targeted_external_status=targeted_card.status,
                            final_external_status=final_card.status,
                            axis_fidelity_status=fidelity_status,
                            internal_novelty_status=internal.status,
                            grounding_preserved=True,
                            refinement_generated=True,
                            reason_codes=list(final_card.reason_codes),
                            interpretation=final_card.interpretation,
                        )
                    )
                continue

            accepted_proposals.append(
                _proposal_from_card(refined, prefix=f"refine{index}")
            )
            attempts.append(
                RefinementAttempt(
                    original_hypothesis_id=original.hypothesis_id,
                    final_hypothesis_id=refined.hypothesis_id,
                    gap_id=gap.gap_id,
                    action=gap.action,
                    decision="accepted_refinement",
                    original_external_status=source_external.status,
                    targeted_external_status=targeted_card.status,
                    final_external_status=final_card.status,
                    axis_fidelity_status=fidelity_status,
                    internal_novelty_status=internal.status,
                    grounding_preserved=True,
                    refinement_generated=True,
                    interpretation=(
                        "One bounded refinement preserved grounding and axis scope, "
                        "passed corpus-internal novelty, and avoided direct/conflicting "
                        "external prior art under a fresh search."
                    ),
                )
            )

        if not accepted_proposals:
            final_draft = HypothesisPortfolioDraft(
                hypotheses=[],
                abstention_reason=(
                    "No original or refined hypothesis survived bounded novelty-refinement gates."
                ),
            )
        else:
            final_draft = HypothesisPortfolioDraft(
                hypotheses=accepted_proposals,
                abstention_reason=None,
            )
        final_portfolio = self.compiler.compile(
            dual.grounded_context, final_draft
        )
        final_validation = self.validator.validate(
            dual.grounded_context, final_portfolio
        )
        if not final_validation.passes:
            raise RuntimeError(
                "final novelty-refined portfolio failed deterministic validation: "
                + ",".join(
                    x.code for x in final_validation.issues
                    if x.severity == "error"
                )
            )

        accepted_count = sum(
            x.decision == "accepted_refinement" for x in attempts
        )
        kept_count = sum(x.decision == "kept_original" for x in attempts)
        rejected_count = len(attempts) - accepted_count - kept_count
        report_id = _stable_id(
            "novelty_refinement_report",
            portfolio.portfolio_id,
            external_report.report_id,
            gap_plan.plan_id,
            final_portfolio.portfolio_id,
            *(f"{x.original_hypothesis_id}:{x.decision}" for x in attempts),
        )
        body = {
            "schema_version": "novelty-refinement-report-v1",
            "report_id": report_id,
            "source_portfolio_id": portfolio.portfolio_id,
            "source_external_report_id": external_report.report_id,
            "source_gap_plan_id": gap_plan.plan_id,
            "final_portfolio_id": final_portfolio.portfolio_id,
            "attempts": [x.model_dump(mode="json") for x in attempts],
            "targeted_searches": [x.model_dump(mode="json") for x in search_records],
            "accepted_refinement_count": accepted_count,
            "kept_original_count": kept_count,
            "rejected_count": rejected_count,
            "max_refinements_per_hypothesis": 1,
            "external_prior_art_can_be_positive_premise": False,
            "policy_version": "novelty-refinement-policy-v1",
        }
        report = NoveltyRefinementReport(
            **body, report_sha256=_sha256_json(body)
        )
        return NoveltyRefinementOutcome(
            portfolio=final_portfolio,
            gap_plan=gap_plan,
            report=report,
            targeted_external_artifacts=tuple(targeted_artifacts),
            final_external_artifacts=tuple(final_external_artifacts),
        )
