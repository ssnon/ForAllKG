from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from dac_her.external_novelty_contracts import ClaimPriorArtReview
from dac_her.novelty_refinement_contracts import GapAction, StrictModel


TargetedEvidenceState = Literal[
    "directly_covered",
    "partially_covered",
    "relational_gap_remains",
    "conflicted",
    "unresolved",
]

R0RefinementBasisKind = Literal[
    "qualify_partial_relation",
    "separate_known_components",
    "resolve_scope_matched_conflict",
]

R0Route = Literal[
    "pass_through_frozen",
    "pass_original_to_r2",
    "authorize_one_r1_then_r2",
    "terminal_reject",
]

_TARGETED_ACTIONS = {
    "targeted_search_then_refine",
    "targeted_search_only",
    "refine_away_from_conflict",
}


class R0RefinementBasis(StrictModel):
    schema_version: Literal["r0-refinement-basis-v1"] = "r0-refinement-basis-v1"
    basis_id: str = Field(min_length=1)
    basis_sha256: str = Field(min_length=64, max_length=64)
    target_claim_id: str = Field(min_length=1)
    basis_kind: R0RefinementBasisKind
    source_work_ids: list[str] = Field(min_length=1)
    epistemic_usage: Literal[
        "boundary_constraint_only_not_positive_premise"
    ] = "boundary_constraint_only_not_positive_premise"

    @model_validator(mode="after")
    def _validate_contract(self) -> "R0RefinementBasis":
        if len(self.source_work_ids) != len(set(self.source_work_ids)):
            raise ValueError("duplicate source_work_ids in R0 refinement basis")
        if self.source_work_ids != sorted(self.source_work_ids):
            raise ValueError("source_work_ids must be canonically sorted")
        return self


class TargetedEvidenceAssessment(StrictModel):
    schema_version: Literal["targeted-evidence-assessment-v1"] = (
        "targeted-evidence-assessment-v1"
    )
    assessment_id: str = Field(min_length=1)
    assessment_sha256: str = Field(min_length=64, max_length=64)
    hypothesis_id: str = Field(min_length=1)
    gap_id: str = Field(min_length=1)
    source_gap_plan_id: str = Field(min_length=1)
    source_gap_plan_sha256: str = Field(min_length=64, max_length=64)
    source_gap_action: GapAction
    source_t1_run_id: str = Field(min_length=1)
    source_t1_freeze_id: str = Field(min_length=1)
    source_t1_manifest_sha256: str = Field(min_length=64, max_length=64)
    target_claim_ids: list[str] = Field(min_length=1)
    claim_reviews: list[ClaimPriorArtReview] = Field(min_length=1)
    evidence_state: TargetedEvidenceState
    refinement_bases: list[R0RefinementBasis] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    literature_absence_claimed: Literal[False] = False
    external_prior_art_can_be_positive_premise: Literal[False] = False
    policy_version: Literal["r0-targeted-evidence-policy-v1"] = (
        "r0-targeted-evidence-policy-v1"
    )

    @model_validator(mode="after")
    def _validate_contract(self) -> "TargetedEvidenceAssessment":
        if self.source_gap_action not in _TARGETED_ACTIONS:
            raise ValueError(
                "targeted evidence assessment requires a targeted-search gap action"
            )

        target_ids = self.target_claim_ids
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("duplicate target_claim_ids")
        if target_ids != sorted(target_ids):
            raise ValueError("target_claim_ids must be canonically sorted")

        review_ids = [row.claim_id for row in self.claim_reviews]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("duplicate claim reviews")
        if review_ids != sorted(review_ids):
            raise ValueError("claim_reviews must be canonically sorted by claim_id")
        if set(review_ids) != set(target_ids):
            missing = sorted(set(target_ids) - set(review_ids))
            unexpected = sorted(set(review_ids) - set(target_ids))
            raise ValueError(
                "claim review set must exactly match target_claim_ids: "
                f"missing={missing}, unexpected={unexpected}"
            )

        for review in self.claim_reviews:
            if review.hypothesis_id != self.hypothesis_id:
                raise ValueError(
                    f"claim review hypothesis_id drift for {review.claim_id}"
                )
            if review.coverage.claim_id != review.claim_id:
                raise ValueError(
                    f"claim review coverage claim_id mismatch for {review.claim_id}"
                )

        basis_ids = [row.basis_id for row in self.refinement_bases]
        if len(basis_ids) != len(set(basis_ids)):
            raise ValueError("duplicate R0 refinement basis IDs")
        if basis_ids != sorted(basis_ids):
            raise ValueError("refinement_bases must be canonically sorted by basis_id")
        unknown_basis_claims = sorted(
            {row.target_claim_id for row in self.refinement_bases} - set(target_ids)
        )
        if unknown_basis_claims:
            raise ValueError(
                "refinement bases must target targeted claims: "
                f"{unknown_basis_claims}"
            )
        if self.source_gap_action == "targeted_search_only" and self.refinement_bases:
            raise ValueError(
                "targeted_search_only may not carry a refinement authorization basis"
            )

        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("reason_codes must be sorted and unique")
        return self


class R0ActionDecision(StrictModel):
    schema_version: Literal["r0-action-decision-v1"] = "r0-action-decision-v1"
    decision_id: str = Field(min_length=1)
    decision_sha256: str = Field(min_length=64, max_length=64)
    hypothesis_id: str = Field(min_length=1)
    gap_id: str = Field(min_length=1)
    source_gap_plan_id: str = Field(min_length=1)
    source_gap_plan_sha256: str = Field(min_length=64, max_length=64)
    source_gap_action: GapAction
    source_assessment_id: str | None = None
    source_assessment_sha256: str | None = None
    route: R0Route
    r1_authorized: bool
    r2_required: bool
    max_refinements_authorized: Literal[0, 1]
    reason_codes: list[str] = Field(default_factory=list)
    network_calls: Literal[0] = 0
    llm_calls: Literal[0] = 0
    hypothesis_rewritten: Literal[False] = False
    automatic_next_stage_authorized: Literal[False] = False
    external_prior_art_can_be_positive_premise: Literal[False] = False
    policy_version: Literal["r0-action-policy-v1"] = "r0-action-policy-v1"

    @model_validator(mode="after")
    def _validate_route_contract(self) -> "R0ActionDecision":
        targeted = self.source_gap_action in _TARGETED_ACTIONS
        if targeted:
            if self.source_assessment_id is None or self.source_assessment_sha256 is None:
                raise ValueError("targeted R0 decisions require a source assessment")
        elif self.source_assessment_id is not None or self.source_assessment_sha256 is not None:
            raise ValueError("keep/reject decisions may not carry a source assessment")

        if self.route == "authorize_one_r1_then_r2":
            if not self.r1_authorized or not self.r2_required:
                raise ValueError("R1 authorization must continue to R2")
            if self.max_refinements_authorized != 1:
                raise ValueError("R1 authorization must permit exactly one refinement")
        elif self.route == "terminal_reject":
            if self.r1_authorized or self.r2_required:
                raise ValueError("terminal reject may not authorize R1 or R2")
            if self.max_refinements_authorized != 0:
                raise ValueError("terminal reject must authorize zero refinements")
        else:
            if self.r1_authorized:
                raise ValueError("non-R1 route may not authorize R1")
            if not self.r2_required:
                raise ValueError("non-terminal pass route must require R2")
            if self.max_refinements_authorized != 0:
                raise ValueError("non-R1 route must authorize zero refinements")

        if self.source_gap_action == "targeted_search_only":
            if self.route == "authorize_one_r1_then_r2":
                raise ValueError("targeted_search_only can never authorize R1")
            if self.r1_authorized or self.max_refinements_authorized != 0:
                raise ValueError("targeted_search_only can never authorize refinement")

        if self.route == "terminal_reject" and self.source_gap_action != "reject":
            raise ValueError("only a frozen source reject action may be terminal in R0")
        if self.source_gap_action == "reject" and self.route != "terminal_reject":
            raise ValueError("frozen source reject must remain terminal in R0")
        if self.source_gap_action == "keep" and self.route != "pass_through_frozen":
            raise ValueError("frozen keep must pass through without refinement")
        if targeted and self.route == "pass_through_frozen":
            raise ValueError("targeted actions require an explicit post-search R0 route")

        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("reason_codes must be sorted and unique")
        return self
