from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxisPlan, DiscoveryAxisSynthesisReport
from pipeline_core.discovery.discovery_axis_fidelity import DiscoveryAxisFidelityCritic
from pipeline_core.discovery.dual_hypothesis_context import DualHypothesisContext
from pipeline_core.discovery.external_novelty import ExternalNoveltyAssessor
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
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftBackend
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator
from pipeline_core.discovery.internal_novelty import InternalNoveltyAssessor
from pipeline_core.discovery.novelty_claim_decomposition import LiteratureQueryPlanner
from pipeline_core.discovery.novelty_gap_analysis import NoveltyGapAnalyzer
from pipeline_core.discovery.question_hypothesis_responsiveness import (
    HypothesisResponsivenessBackendProtocol,
    evaluate_hypothesis_task_preservation,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyGapPlan,
    NoveltyRefinementReport,
    RefinementAttempt,
    TargetedSearchRecord,
)
from pipeline_core.discovery.novelty_refinement_prompt import NoveltyRefinementPromptAssembler
from pipeline_core.discovery.novelty_reaxis_prompt import (
    FreshNoveltyReaxisPromptAssembler,
)
from pipeline_core.discovery.targeted_novelty_retrieval import TargetedNoveltyRetriever
from pipeline_core.discovery.prior_art_review_audit import (
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

    # Strong known-axis states that may justify ONE fresh-context re-axis.
    # INSUFFICIENT_SEARCH_EVIDENCE is intentionally absent: UNKNOWN is not
    # a scientific rejection and must not trigger novelty chasing.
    REAXIS_EXTERNAL = {
        "WELL_ESTABLISHED",
        "LITERATURE_SUPPORTED_EXTENSION",
    }

    # A fresh re-axis has not improved the known-axis problem if its own
    # final fresh search still lands in one of these states.
    REAXIS_REJECT_EXTERNAL = {
        "WELL_ESTABLISHED",
        "LITERATURE_SUPPORTED_EXTENSION",
        "CONFLICTING_PRIOR_ART",
    }

    # Targeted search can itself resolve an initially weak novelty status.
    # In that case do not regenerate merely for the sake of regeneration.
    RESOLVED_CANDIDATE_EXTERNAL = {
        "PLAUSIBLY_NOVEL",
        "NEW_COMBINATION_OF_KNOWN_EFFECTS",
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
    }

    SURVIVOR_DECISIONS = frozenset({
        "kept_original",
        "accepted_refinement",
        "accepted_reaxis",
    })

    def __init__(
        self,
        *,
        hypothesis_backend: HypothesisDraftBackend,
        external_assessor: ExternalNoveltyAssessor,
        targeted_retriever: TargetedNoveltyRetriever,
        mapper: Any,
        compiler: HypothesisCompiler | None = None,
        validator: HypothesisValidator | None = None,
        gap_analyzer: NoveltyGapAnalyzer,
        fidelity_critic: DiscoveryAxisFidelityCritic | None = None,
        internal_assessor: InternalNoveltyAssessor | None = None,
        task_responsiveness_backend: (
            HypothesisResponsivenessBackendProtocol
            | None
        ) = None,
    ) -> None:
        self.hypothesis_backend = hypothesis_backend
        self.external_assessor = external_assessor
        self.targeted_retriever = targeted_retriever
        self.mapper = mapper
        self.compiler = compiler or HypothesisCompiler()
        self.validator = validator or HypothesisValidator()
        self.gap_analyzer = gap_analyzer
        self.fidelity_critic = fidelity_critic or DiscoveryAxisFidelityCritic()
        self.internal_assessor = internal_assessor or InternalNoveltyAssessor()
        self.task_responsiveness_backend = task_responsiveness_backend

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

    @classmethod
    def _bind_final_hypothesis_ids(
        cls,
        attempts: list[RefinementAttempt],
        final_portfolio: HypothesisPortfolio,
    ) -> list[RefinementAttempt]:
        """Bind surviving attempts to actual final portfolio IDs.

        Attempt-stage compilation and final portfolio compilation are
        distinct identity namespaces because HypothesisCompiler includes
        local_id in the hypothesis ID digest. Never infer final membership
        from original/candidate identity.
        """
        survivor_indices = [
            index
            for index, attempt
            in enumerate(attempts)
            if (
                attempt.decision
                in cls.SURVIVOR_DECISIONS
            )
        ]

        final_cards = list(
            final_portfolio.hypotheses
        )

        if (
            len(survivor_indices)
            != len(final_cards)
        ):
            raise RuntimeError(
                "novelty refinement survivor/final-portfolio "
                "cardinality mismatch: "
                f"survivors={len(survivor_indices)}, "
                f"final={len(final_cards)}"
            )

        final_ids = [
            card.hypothesis_id
            for card in final_cards
        ]

        if (
            len(final_ids)
            != len(set(final_ids))
        ):
            raise RuntimeError(
                "final novelty-refined portfolio contains "
                "duplicate hypothesis IDs"
            )

        rebound = list(attempts)

        for attempt_index, card in zip(
            survivor_indices,
            final_cards,
            strict=True,
        ):
            attempt = rebound[
                attempt_index
            ]

            if (
                attempt.candidate_hypothesis_id
                is None
            ):
                raise RuntimeError(
                    "surviving refinement attempt is missing "
                    "candidate_hypothesis_id before final binding"
                )

            if (
                attempt.final_hypothesis_id
                is not None
            ):
                raise RuntimeError(
                    "final_hypothesis_id was populated before "
                    "final portfolio binding"
                )

            rebound[
                attempt_index
            ] = attempt.model_copy(
                update={
                    "final_hypothesis_id":
                        card.hypothesis_id,
                }
            )

        for attempt in rebound:
            if (
                attempt.decision
                not in cls.SURVIVOR_DECISIONS
                and attempt.final_hypothesis_id
                is not None
            ):
                raise RuntimeError(
                    "non-surviving refinement attempt claims "
                    "final portfolio membership"
                )

        return rebound

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

    @staticmethod
    def _validate_scientific_novelty_gate(
        scientific_novelty_gate: dict[str, Any] | None,
        portfolio: HypothesisPortfolio,
    ) -> dict[str, dict[str, Any]] | None:
        if scientific_novelty_gate is None:
            return None

        if (
            scientific_novelty_gate.get("schema_version")
            != "scientific-novelty-fallback-gate-v1"
        ):
            raise RuntimeError(
                "Unexpected scientific novelty fallback gate schema."
            )

        if (
            scientific_novelty_gate.get("production_authority")
            is not True
        ):
            raise RuntimeError(
                "Scientific novelty fallback gate lacks production authority."
            )

        rows = scientific_novelty_gate.get("gates")

        if not isinstance(rows, list):
            raise RuntimeError(
                "Scientific novelty fallback gates must be a list."
            )

        by_id: dict[str, dict[str, Any]] = {}

        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(
                    "Scientific novelty fallback gate row must be an object."
                )

            hypothesis_id = str(
                row.get("hypothesis_id") or ""
            ).strip()

            if not hypothesis_id:
                raise RuntimeError(
                    "Scientific novelty fallback gate row lacks hypothesis_id."
                )

            if hypothesis_id in by_id:
                raise RuntimeError(
                    "Duplicate scientific novelty fallback gate: "
                    f"{hypothesis_id}"
                )

            selection_class = str(
                row.get("selection_class") or ""
            )

            expected_allowed = (
                selection_class
                in {
                    "ELIGIBLE",
                    "CONDITIONAL",
                }
            )

            if (
                row.get("fallback_allowed")
                is not expected_allowed
            ):
                raise RuntimeError(
                    "Scientific novelty fallback gate is internally "
                    f"inconsistent for {hypothesis_id}."
                )

            by_id[hypothesis_id] = row

        portfolio_ids = {
            card.hypothesis_id
            for card in portfolio.hypotheses
        }

        if set(by_id) != portfolio_ids:
            raise RuntimeError(
                "Scientific novelty fallback gate hypothesis set "
                "does not match Alpha6 source portfolio."
            )

        return by_id

    @classmethod
    def _original_fallback_allowed(
        cls,
        targeted_status: str,
        *,
        hypothesis_id: str | None = None,
        scientific_gate_by_id: (
            dict[str, dict[str, Any]]
            | None
        ) = None,
    ) -> bool:
        """Whether Alpha6 may retain the original after failed refinement."""

        if targeted_status in cls.REJECT_EXTERNAL:
            return False

        if scientific_gate_by_id is None:
            # Backward-compatible behavior when production gate is absent.
            return True

        if hypothesis_id is None:
            raise RuntimeError(
                "Scientific novelty fallback gate requires hypothesis_id."
            )

        gate = scientific_gate_by_id.get(
            hypothesis_id
        )

        if gate is None:
            raise RuntimeError(
                "Missing scientific novelty fallback gate for "
                f"{hypothesis_id}."
            )

        return bool(
            gate["fallback_allowed"]
        )


    @staticmethod
    def _fresh_reaxis_safe_unused_premise_ids(
        dual: DualHypothesisContext,
        original: HypothesisCard,
    ) -> list[str]:
        """Return conservative unused positive premises for fresh re-axis.

        A fresh novelty re-axis is intentionally stricter than ordinary
        hypothesis generation: provisional/verification-required or explicitly
        restricted premises are not introduced merely to escape prior art.
        """
        original_ids = set(
            map(str, original.premise_statement_ids)
        )

        return sorted(
            str(row.statement_id)
            for row in dual.grounded_context.evidence_statements
            if (
                row.eligible_as_premise
                and not row.requires_verification
                and not row.premise_restrictions
                and row.epistemic_role
                in {"reported", "evidence_synthesis"}
                and str(row.statement_id)
                not in original_ids
            )
        )

    @classmethod
    def _should_attempt_fresh_reaxis(
        cls,
        targeted_status: str,
        unused_premise_ids: list[str],
    ) -> bool:
        return (
            targeted_status in cls.REAXIS_EXTERNAL
            and bool(unused_premise_ids)
        )

    @classmethod
    def _fresh_reaxis_grounding_valid(
        cls,
        dual: DualHypothesisContext,
        original: HypothesisCard,
        candidate: HypothesisCard,
    ) -> bool:
        safe_unused = set(
            cls._fresh_reaxis_safe_unused_premise_ids(
                dual,
                original,
            )
        )

        original_ids = set(
            map(str, original.premise_statement_ids)
        )

        allowed_premises = (
            original_ids
            | safe_unused
        )

        candidate_premises = set(
            map(str, candidate.premise_statement_ids)
        )

        if not candidate_premises:
            return False

        if not candidate_premises.issubset(
            allowed_premises
        ):
            return False

        # A fresh re-axis must actually introduce at least one previously
        # unused safe premise. Otherwise it is merely same-premise refinement.
        if not (
            candidate_premises
            & safe_unused
        ):
            return False

        allowed_gaps = {
            str(row.statement_id)
            for row in dual.grounded_context.evidence_statements
            if row.eligible_as_gap
        }

        candidate_gaps = set(
            map(str, candidate.gap_statement_ids)
        )

        if not candidate_gaps.issubset(
            allowed_gaps
        ):
            return False

        return True

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
        scientific_novelty_gate: dict[str, Any] | None = None,
    ) -> NoveltyRefinementOutcome:
        scientific_gate_by_id = (
            self._validate_scientific_novelty_gate(
                scientific_novelty_gate,
                portfolio,
            )
        )

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
                if self._original_fallback_allowed(
                    source_external.status,
                    hypothesis_id=original.hypothesis_id,
                    scientific_gate_by_id=scientific_gate_by_id,
                ):
                    accepted_proposals.append(
                        _proposal_from_card(
                            original,
                            prefix=f"keep{index}",
                        )
                    )
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            candidate_hypothesis_id=original.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="kept_original",
                            original_external_status=source_external.status,
                            targeted_external_status=source_external.status,
                            final_external_status=source_external.status,
                            grounding_preserved=True,
                            refinement_generated=False,
                            interpretation=(
                                "The external novelty status did not require "
                                "bounded refinement."
                            ),
                        )
                    )
                else:
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="scientific_novelty_rejected",
                            original_external_status=source_external.status,
                            targeted_external_status=source_external.status,
                            grounding_preserved=True,
                            refinement_generated=False,
                            reason_codes=[
                                "scientific_novelty_gate_blocked_original_fallback",
                            ],
                            interpretation=(
                                "External novelty alone would otherwise retain "
                                "the original hypothesis, but the authoritative "
                                "scientific-novelty gate disallows original fallback."
                            ),
                        )
                    )
                continue

            if not gap.targeted_queries:
                if self._original_fallback_allowed(
                    source_external.status,
                    hypothesis_id=original.hypothesis_id,
                    scientific_gate_by_id=scientific_gate_by_id,
                ):
                    accepted_proposals.append(
                        _proposal_from_card(original, prefix=f"fallback{index}")
                    )
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            candidate_hypothesis_id=original.hypothesis_id,
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
                        candidate_hypothesis_id=original.hypothesis_id,
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

            # If targeted search already resolves the original into a
            # search-bounded candidate category, do not regenerate merely for
            # novelty optimization. This applies regardless of the initial
            # status that triggered targeted search.
            if (
                targeted_card.status
                in self.RESOLVED_CANDIDATE_EXTERNAL
            ):
                if self._original_fallback_allowed(
                    targeted_card.status,
                    hypothesis_id=original.hypothesis_id,
                    scientific_gate_by_id=scientific_gate_by_id,
                ):
                    accepted_proposals.append(
                        _proposal_from_card(
                            original,
                            prefix=f"keep{index}",
                        )
                    )
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            candidate_hypothesis_id=original.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="kept_original",
                            original_external_status=source_external.status,
                            targeted_external_status=targeted_card.status,
                            final_external_status=targeted_card.status,
                            grounding_preserved=True,
                            refinement_generated=False,
                            generation_mode="none",
                            context_grounding_valid=True,
                            reason_codes=[
                                "targeted_search_resolved_candidate_status",
                            ],
                            interpretation=(
                                "Targeted search resolved the candidate without "
                                "requiring hypothesis regeneration; no additional "
                                "novelty optimization is performed."
                            ),
                        )
                    )
                else:
                    attempts.append(
                        RefinementAttempt(
                            original_hypothesis_id=original.hypothesis_id,
                            gap_id=gap.gap_id,
                            action=gap.action,
                            decision="scientific_novelty_rejected",
                            original_external_status=source_external.status,
                            targeted_external_status=targeted_card.status,
                            grounding_preserved=True,
                            refinement_generated=False,
                            generation_mode="none",
                            context_grounding_valid=True,
                            reason_codes=[
                                "targeted_search_resolved_candidate_status",
                                "scientific_novelty_gate_blocked_original_fallback",
                            ],
                            interpretation=(
                                "Targeted search reached an externally acceptable "
                                "candidate status, but the authoritative scientific-"
                                "novelty gate disallows retention of the original "
                                "hypothesis."
                            ),
                        )
                    )
                continue

            # ----------------------------------------------------------
            # Fresh-context novelty re-axis
            #
            # This path is intentionally distinct from same-premise
            # refinement. It may select a different subset of positive
            # premises from the SAME grounded HypothesisContext, but it
            # must introduce at least one previously unused, conservative
            # eligible premise.
            #
            # At most one fresh re-axis generation is attempted here.
            # If it fails, existing same-premise refinement remains the
            # bounded fallback path.
            # ----------------------------------------------------------

            unused_reaxis_ids = (
                self._fresh_reaxis_safe_unused_premise_ids(
                    dual,
                    original,
                )
            )

            if self._should_attempt_fresh_reaxis(
                targeted_card.status,
                unused_reaxis_ids,
            ):
                allowed_reaxis_ids = sorted(
                    set(
                        map(
                            str,
                            original.premise_statement_ids,
                        )
                    )
                    | set(unused_reaxis_ids)
                )

                reaxis_assembler = (
                    FreshNoveltyReaxisPromptAssembler(
                        original=original,
                        gap=gap,
                        targeted_card=targeted_card,
                        allowed_premise_ids=allowed_reaxis_ids,
                        required_unused_premise_ids=unused_reaxis_ids,
                    )
                )

                reaxis_prompt = reaxis_assembler.build(
                    dual.grounded_context
                )

                reaxis_generation = (
                    self.hypothesis_backend.generate(
                        reaxis_prompt
                    )
                )

                reaxis_draft = (
                    reaxis_generation.draft
                )

                reaxis_candidate_id = None
                reaxis_context_grounding_valid = False
                reaxis_internal_status = None
                reaxis_final_status = None
                reaxis_failure_decision = None
                reaxis_reason_codes = [
                    "fresh_context_reaxis",
                ]
                reaxis_failure_interpretation = ""

                if not reaxis_draft.hypotheses:
                    reaxis_failure_decision = "abstained"
                    reaxis_reason_codes.append(
                        "fresh_reaxis_abstained"
                    )
                    reaxis_failure_interpretation = (
                        "Fresh-context novelty re-axis abstained."
                    )

                elif len(reaxis_draft.hypotheses) != 1:
                    reaxis_failure_decision = (
                        "validation_rejected"
                    )
                    reaxis_reason_codes.append(
                        "fresh_reaxis_cardinality_violation"
                    )
                    reaxis_failure_interpretation = (
                        "Fresh-context novelty re-axis returned "
                        "more than one hypothesis."
                    )

                else:
                    (
                        reaxis_compiled,
                        reaxis_issue_codes,
                    ) = self._compile_one(
                        dual,
                        reaxis_draft,
                    )

                    if reaxis_compiled is None:
                        reaxis_failure_decision = (
                            "compile_rejected"
                        )
                        reaxis_reason_codes.extend(
                            reaxis_issue_codes
                        )
                        reaxis_failure_interpretation = (
                            "Fresh-context novelty re-axis failed "
                            "deterministic compile/validation."
                        )

                    else:
                        reaxis_card = (
                            reaxis_compiled.hypotheses[0]
                        )

                        reaxis_candidate_id = (
                            reaxis_card.hypothesis_id
                        )

                        reaxis_context_grounding_valid = (
                            self._fresh_reaxis_grounding_valid(
                                dual,
                                original,
                                reaxis_card,
                            )
                        )

                        if not reaxis_context_grounding_valid:
                            reaxis_failure_decision = (
                                "grounding_drift_rejected"
                            )
                            reaxis_reason_codes.append(
                                "fresh_reaxis_grounding_contract_failed"
                            )
                            reaxis_failure_interpretation = (
                                "Fresh re-axis did not use only allowed "
                                "same-context premises with at least one "
                                "previously unused safe premise."
                            )

                        else:
                            reaxis_internal = (
                                self.internal_assessor.assess(
                                    dual,
                                    reaxis_compiled,
                                    self.mapper,
                                ).cards[0]
                            )

                            reaxis_internal_status = (
                                reaxis_internal.status
                            )

                            if (
                                reaxis_internal.status
                                in self.REJECT_INTERNAL
                            ):
                                reaxis_failure_decision = (
                                    "internal_novelty_rejected"
                                )
                                reaxis_reason_codes.extend(
                                    reaxis_internal.reason_codes
                                )
                                reaxis_failure_interpretation = (
                                    "Fresh-context re-axis reconstructed "
                                    "existing internal corpus prior art. "
                                    + reaxis_internal.interpretation
                                )

                            else:
                                reaxis_fresh_external = (
                                    self._fresh_external(
                                        reaxis_compiled
                                    )
                                )

                                final_external_artifacts.append(
                                    reaxis_fresh_external
                                )

                                reaxis_final_card = (
                                    reaxis_fresh_external
                                    .report
                                    .cards[0]
                                )

                                reaxis_final_status = (
                                    reaxis_final_card.status
                                )

                                reaxis_task_assessment = None

                                if (
                                    self.task_responsiveness_backend
                                    is not None
                                ):
                                    (
                                        reaxis_task_assessment,
                                        _reaxis_task_stability,
                                    ) = (
                                        evaluate_hypothesis_task_preservation(
                                            question=(
                                                dual.grounded_context.question
                                            ),
                                            hypothesis=reaxis_card,
                                            backend=(
                                                self.task_responsiveness_backend
                                            ),
                                        )
                                    )

                                if (
                                    reaxis_final_card.status
                                    in self.REAXIS_REJECT_EXTERNAL
                                ):
                                    reaxis_failure_decision = (
                                        "external_novelty_rejected"
                                    )
                                    reaxis_reason_codes.extend(
                                        reaxis_final_card.reason_codes
                                    )
                                    reaxis_failure_interpretation = (
                                        "Fresh-context re-axis remained "
                                        "directly known/extension-like or "
                                        "conflicted under a fresh external "
                                        "search. "
                                        + reaxis_final_card.interpretation
                                    )

                                elif (
                                    reaxis_task_assessment
                                    is not None
                                    and reaxis_task_assessment.task_class
                                    not in {
                                        "DIRECT",
                                        "SUBORDINATE",
                                    }
                                ):
                                    reaxis_failure_decision = (
                                        "question_task_rejected"
                                    )

                                    reaxis_reason_codes.extend(
                                        [
                                            (
                                                "fresh_reaxis_question_"
                                                "task_preservation_failed"
                                            ),
                                            (
                                                "question_task_class_"
                                                + reaxis_task_assessment
                                                .task_class
                                                .lower()
                                            ),
                                        ]
                                    )

                                    reaxis_failure_interpretation = (
                                        "Fresh-context novelty re-axis was "
                                        "rejected because its primary "
                                        "scientific task did not stably "
                                        "preserve the original question. "
                                        "Task class: "
                                        + reaxis_task_assessment.task_class
                                        + "."
                                    )

                                else:
                                    accepted_proposals.append(
                                        _proposal_from_card(
                                            reaxis_card,
                                            prefix=f"reaxis{index}",
                                        )
                                    )

                                    attempts.append(
                                        RefinementAttempt(
                                            original_hypothesis_id=(
                                                original.hypothesis_id
                                            ),
                                            candidate_hypothesis_id=(
                                                reaxis_card.hypothesis_id
                                            ),
                                            gap_id=gap.gap_id,
                                            action=gap.action,
                                            decision="accepted_reaxis",
                                            original_external_status=(
                                                source_external.status
                                            ),
                                            targeted_external_status=(
                                                targeted_card.status
                                            ),
                                            final_external_status=(
                                                reaxis_final_card.status
                                            ),
                                            axis_fidelity_status=(
                                                "fresh_reaxis_context_bound"
                                            ),
                                            internal_novelty_status=(
                                                reaxis_internal.status
                                            ),
                                            grounding_preserved=(
                                                self._grounding_preserved(
                                                    original,
                                                    reaxis_card,
                                                )
                                            ),
                                            refinement_generated=True,
                                            generation_mode=(
                                                "fresh_context_reaxis"
                                            ),
                                            context_grounding_valid=True,
                                            reason_codes=[
                                                "fresh_context_reaxis",
                                                "used_unused_eligible_premise",
                                            ],
                                            interpretation=(
                                                "One bounded fresh-context "
                                                "re-axis used at least one "
                                                "previously unused eligible "
                                                "grounded premise, passed "
                                                "deterministic validation and "
                                                "internal novelty, and did not "
                                                "remain WELL_ESTABLISHED, "
                                                "LITERATURE_SUPPORTED_EXTENSION, "
                                                "or CONFLICTING_PRIOR_ART under "
                                                "its fresh external search. "
                                                "INSUFFICIENT_SEARCH_EVIDENCE, "
                                                "when present, is retained as "
                                                "uncertainty rather than treated "
                                                "as scientific rejection."
                                            ),
                                        )
                                    )

                                    continue

                # Record the failed fresh re-axis attempt, then fall through
                # to the existing same-premise refinement path.
                attempts.append(
                    RefinementAttempt(
                        original_hypothesis_id=(
                            original.hypothesis_id
                        ),
                        candidate_hypothesis_id=(
                            reaxis_candidate_id
                        ),
                        gap_id=gap.gap_id,
                        action=gap.action,
                        decision=reaxis_failure_decision,
                        original_external_status=(
                            source_external.status
                        ),
                        targeted_external_status=(
                            targeted_card.status
                        ),
                        final_external_status=(
                            reaxis_final_status
                        ),
                        axis_fidelity_status=(
                            "fresh_reaxis_context_bound"
                            if reaxis_candidate_id is not None
                            else None
                        ),
                        internal_novelty_status=(
                            reaxis_internal_status
                        ),
                        grounding_preserved=False,
                        refinement_generated=True,
                        generation_mode=(
                            "fresh_context_reaxis"
                        ),
                        context_grounding_valid=(
                            reaxis_context_grounding_valid
                        ),
                        reason_codes=sorted(
                            set(reaxis_reason_codes)
                        ),
                        interpretation=(
                            reaxis_failure_interpretation
                            + " Existing same-premise refinement "
                              "remains available as the bounded "
                              "fallback path."
                        ),
                    )
                )

            assembler = NoveltyRefinementPromptAssembler(
                original=original,
                gap=gap,
                targeted_card=targeted_card,
            )
            prompt = assembler.build(dual.grounded_context)
            generation = self.hypothesis_backend.generate(prompt)
            draft = generation.draft
            if not draft.hypotheses:
                if self._original_fallback_allowed(
                    targeted_card.status,
                    hypothesis_id=original.hypothesis_id,
                    scientific_gate_by_id=scientific_gate_by_id,
                ):
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
                if self._original_fallback_allowed(
                    targeted_card.status,
                    hypothesis_id=original.hypothesis_id,
                    scientific_gate_by_id=scientific_gate_by_id,
                ):
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
                if self._original_fallback_allowed(
                    targeted_card.status,
                    hypothesis_id=original.hypothesis_id,
                    scientific_gate_by_id=scientific_gate_by_id,
                ):
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
                if self._original_fallback_allowed(
                    targeted_card.status,
                    hypothesis_id=original.hypothesis_id,
                    scientific_gate_by_id=scientific_gate_by_id,
                ):
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
                            candidate_hypothesis_id=refined.hypothesis_id,
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
                    if self._original_fallback_allowed(
                    targeted_card.status,
                    hypothesis_id=original.hypothesis_id,
                    scientific_gate_by_id=scientific_gate_by_id,
                ):
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
                                candidate_hypothesis_id=refined.hypothesis_id,
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
                if self._original_fallback_allowed(
                    targeted_card.status,
                    hypothesis_id=original.hypothesis_id,
                    scientific_gate_by_id=scientific_gate_by_id,
                ):
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
                            candidate_hypothesis_id=refined.hypothesis_id,
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
                if self._original_fallback_allowed(
                    targeted_card.status,
                    hypothesis_id=original.hypothesis_id,
                    scientific_gate_by_id=scientific_gate_by_id,
                ):
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
                            candidate_hypothesis_id=refined.hypothesis_id,
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
                    candidate_hypothesis_id=refined.hypothesis_id,
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

        attempts = self._bind_final_hypothesis_ids(
            attempts,
            final_portfolio,
        )

        accepted_count = sum(
            x.decision == "accepted_refinement" for x in attempts
        )
        accepted_reaxis_count = sum(
            x.decision == "accepted_reaxis" for x in attempts
        )
        kept_count = sum(x.decision == "kept_original" for x in attempts)
        rejected_count = (
            len(attempts)
            - accepted_count
            - accepted_reaxis_count
            - kept_count
        )
        report_id = _stable_id(
            "novelty_refinement_report",
            portfolio.portfolio_id,
            external_report.report_id,
            gap_plan.plan_id,
            final_portfolio.portfolio_id,
            *(f"{x.original_hypothesis_id}:{x.decision}" for x in attempts),
        )
        body = {
            "schema_version": "novelty-refinement-report-v2",
            "report_id": report_id,
            "source_portfolio_id": portfolio.portfolio_id,
            "source_external_report_id": external_report.report_id,
            "source_gap_plan_id": gap_plan.plan_id,
            "final_portfolio_id": final_portfolio.portfolio_id,
            "attempts": [x.model_dump(mode="json") for x in attempts],
            "targeted_searches": [x.model_dump(mode="json") for x in search_records],
            "accepted_refinement_count": accepted_count,
            "accepted_reaxis_count": accepted_reaxis_count,
            "kept_original_count": kept_count,
            "rejected_count": rejected_count,
            "max_refinements_per_hypothesis": 1,
            "max_reaxes_per_hypothesis": 1,
            "external_prior_art_can_be_positive_premise": False,
            "policy_version": "novelty-refinement-policy-v2",
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
