from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import ExternalNoveltyStatus
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


GapAction = Literal[
    "keep",
    "targeted_search_then_refine",
    "targeted_search_only",
    "refine_away_from_conflict",
    "reject",
]
TargetedGapQueryRole = Literal[
    "relation_primary",
    "relation_variant",
    "scope_check",
    "exact_higher_order_verification",
]


class TargetedGapQuery(StrictModel):
    claim_id: str = Field(min_length=1)
    query_role: TargetedGapQueryRole
    query_text: str = Field(min_length=1)


RefinementDecision = Literal[
    "kept_original",
    "accepted_refinement",
    "accepted_reaxis",
    "abstained",
    "compile_rejected",
    "validation_rejected",
    "grounding_drift_rejected",
    "axis_fidelity_rejected",
    "internal_novelty_rejected",
    "external_novelty_rejected",
    "search_insufficient",
]


class NoveltyGap(StrictModel):
    gap_id: str
    hypothesis_id: str
    source_external_status: ExternalNoveltyStatus
    action: GapAction
    target_claim_ids: list[str] = Field(default_factory=list)
    differentiator: str
    already_known_boundary: list[str] = Field(default_factory=list)
    unresolved_boundary: list[str] = Field(default_factory=list)
    contextual_conflict_work_ids: list[str] = Field(default_factory=list)
    targeted_queries: list[TargetedGapQuery] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _targeted_query_claims_must_be_bound(self) -> "NoveltyGap":
        allowed = set(self.target_claim_ids)
        unknown = sorted({row.claim_id for row in self.targeted_queries} - allowed)
        if unknown:
            raise ValueError(
                "targeted query claim_id must be present in target_claim_ids: "
                f"{unknown}"
            )
        return self


class NoveltyGapPlan(StrictModel):
    schema_version: Literal["novelty-gap-plan-v2"] = "novelty-gap-plan-v2"
    plan_id: str
    plan_sha256: str
    source_portfolio_id: str
    source_external_report_id: str
    gaps: list[NoveltyGap] = Field(default_factory=list)
    policy_version: Literal["novelty-gap-policy-v2"] = "novelty-gap-policy-v2"


class TargetedSearchRecord(StrictModel):
    hypothesis_id: str
    gap_id: str
    query_plan_id: str
    prior_art_packet_id: str
    external_report_id: str
    external_status_after_search: ExternalNoveltyStatus
    unique_work_count: int
    abstract_work_count: int
    successful_query_count: int


GenerationMode = Literal[
    "none",
    "same_premise_refinement",
    "fresh_context_reaxis",
]


class RefinementAttempt(StrictModel):
    original_hypothesis_id: str

    # Identity of the hypothesis actually assessed during this
    # refinement attempt. For kept-original paths this is the
    # original hypothesis ID; for compiled refinements this is
    # the single-card compiled candidate ID. It is NOT necessarily
    # the identity assigned by the final portfolio compilation.
    candidate_hypothesis_id: str | None = None

    # Membership identity in NoveltyRefinementReport.final_portfolio_id.
    # This is populated only after the final portfolio is compiled.
    # Rejected/non-surviving attempts must leave it unset.
    final_hypothesis_id: str | None = None
    gap_id: str
    action: GapAction
    decision: RefinementDecision
    original_external_status: ExternalNoveltyStatus
    targeted_external_status: ExternalNoveltyStatus | None = None
    final_external_status: ExternalNoveltyStatus | None = None
    axis_fidelity_status: str | None = None
    internal_novelty_status: str | None = None
    grounding_preserved: bool = False
    refinement_generated: bool = False
    generation_mode: GenerationMode = "none"
    context_grounding_valid: bool = False
    reason_codes: list[str] = Field(default_factory=list)
    interpretation: str


class NoveltyRefinementReport(StrictModel):
    # v1 remains parseable for historical artifacts. v2 separates
    # attempt-stage candidate identity from actual final-portfolio
    # membership identity.
    schema_version: Literal[
        "novelty-refinement-report-v1",
        "novelty-refinement-report-v2",
    ] = "novelty-refinement-report-v2"
    report_id: str
    report_sha256: str
    source_portfolio_id: str
    source_external_report_id: str
    source_gap_plan_id: str
    final_portfolio_id: str
    attempts: list[RefinementAttempt] = Field(default_factory=list)
    targeted_searches: list[TargetedSearchRecord] = Field(default_factory=list)
    accepted_refinement_count: int = 0
    accepted_reaxis_count: int = 0
    kept_original_count: int = 0
    rejected_count: int = 0
    max_refinements_per_hypothesis: Literal[1] = 1
    max_reaxes_per_hypothesis: Literal[1] = 1
    external_prior_art_can_be_positive_premise: Literal[False] = False
    policy_version: Literal[
        "novelty-refinement-policy-v1",
        "novelty-refinement-policy-v2",
    ] = "novelty-refinement-policy-v2"

    @model_validator(mode="after")
    def _v2_identity_binding_consistency(
        self,
    ) -> "NoveltyRefinementReport":
        # Historical v1 encoded an attempt-stage identity in
        # final_hypothesis_id. Preserve its parseability exactly
        # as historical evidence; do not reinterpret it.
        if (
            self.schema_version
            == "novelty-refinement-report-v1"
        ):
            return self

        survivor_decisions = {
            "kept_original",
            "accepted_refinement",
            "accepted_reaxis",
        }

        final_ids: list[str] = []

        for attempt in self.attempts:
            survives = (
                attempt.decision
                in survivor_decisions
            )

            if survives:
                if (
                    attempt.candidate_hypothesis_id
                    is None
                ):
                    raise ValueError(
                        "v2 surviving refinement attempt "
                        "requires candidate_hypothesis_id"
                    )

                if (
                    attempt.final_hypothesis_id
                    is None
                ):
                    raise ValueError(
                        "v2 surviving refinement attempt "
                        "requires final_hypothesis_id"
                    )

                final_ids.append(
                    attempt.final_hypothesis_id
                )

                continue

            if (
                attempt.final_hypothesis_id
                is not None
            ):
                raise ValueError(
                    "v2 rejected/non-surviving refinement "
                    "attempt must not claim final portfolio "
                    "membership"
                )

        if (
            len(final_ids)
            != len(set(final_ids))
        ):
            raise ValueError(
                "v2 final_hypothesis_id values must be "
                "unique among surviving attempts"
            )

        expected_survivors = (
            self.accepted_refinement_count
            + self.accepted_reaxis_count
            + self.kept_original_count
        )

        if len(final_ids) != expected_survivors:
            raise ValueError(
                "v2 final hypothesis binding count does "
                "not match surviving attempt counts"
            )

        return self


class NoveltyRefinementArtifact(StrictModel):
    schema_version: Literal[
        "novelty-refinement-artifact-v1",
        "novelty-refinement-artifact-v2",
    ] = "novelty-refinement-artifact-v2"

    portfolio: HypothesisPortfolio
    report: NoveltyRefinementReport

    @model_validator(mode="after")
    def _final_portfolio_membership_consistency(
        self,
    ) -> "NoveltyRefinementArtifact":
        if (
            self.report.final_portfolio_id
            != self.portfolio.portfolio_id
        ):
            raise ValueError(
                "novelty refinement report final_portfolio_id "
                "does not match embedded portfolio"
            )

        # Historical v1 artifacts retain their historical identity
        # semantics and must remain parseable without reinterpretation.
        if (
            self.schema_version
            == "novelty-refinement-artifact-v1"
        ):
            return self

        if (
            self.report.schema_version
            != "novelty-refinement-report-v2"
        ):
            raise ValueError(
                "novelty-refinement-artifact-v2 requires "
                "novelty-refinement-report-v2"
            )

        portfolio_ids = [
            row.hypothesis_id
            for row in self.portfolio.hypotheses
        ]

        if (
            len(portfolio_ids)
            != len(set(portfolio_ids))
        ):
            raise ValueError(
                "final novelty-refined portfolio contains "
                "duplicate hypothesis IDs"
            )

        report_final_ids = [
            attempt.final_hypothesis_id
            for attempt in self.report.attempts
            if (
                attempt.decision
                in {
                    "kept_original",
                    "accepted_refinement",
                    "accepted_reaxis",
                }
            )
        ]

        if any(
            value is None
            for value in report_final_ids
        ):
            raise ValueError(
                "v2 surviving attempt is missing "
                "final portfolio membership ID"
            )

        if (
            set(report_final_ids)
            != set(portfolio_ids)
        ):
            raise ValueError(
                "v2 refinement final_hypothesis_id set "
                "does not equal final portfolio hypothesis ID set"
            )

        if (
            len(report_final_ids)
            != len(portfolio_ids)
        ):
            raise ValueError(
                "v2 refinement final hypothesis binding "
                "cardinality does not match final portfolio"
            )

        return self
