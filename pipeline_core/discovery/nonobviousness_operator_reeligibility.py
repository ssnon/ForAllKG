from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.discovery.nonobviousness_grounded_claim_attachment import (
    N11GroundedClaimAttachmentResult,
)
from pipeline_core.discovery.nonobviousness_missing_bridge_contracts import (
    N11MissingBridgeOpportunity,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


N11RelativeContributionDecisionStatus = Literal[
    "ELIGIBLE_FOR_FRESH_CANDIDATE_REGENERATION",
    "ABSTAIN_MISSING_SHARED_FACTOR_BRIDGE",
    "ABSTAIN_MISSING_SUPPLEMENTAL_FACTOR_BRIDGE",
    "DEFER_TARGET_RELATION_ALREADY_GROUNDED",
]


class N11RelativeContributionBranchDecision(
    StrictModel
):
    schema_version: Literal[
        "n11-relative-contribution-branch-decision-v1"
    ] = (
        "n11-relative-contribution-branch-decision-v1"
    )

    decision_id: str = Field(
        min_length=1
    )

    source_missing_bridge_opportunity_id: str = Field(
        min_length=1
    )

    source_d2_search_id: str = Field(
        min_length=1
    )

    operator: Literal[
        "RELATIVE_CONTRIBUTION_SHIFT"
    ] = "RELATIVE_CONTRIBUTION_SHIFT"

    status: N11RelativeContributionDecisionStatus

    shared_factor_bridge_attachment_ids: list[str]
    supplemental_factor_bridge_attachment_ids: list[str]
    target_relation_attachment_ids: list[str]

    electromagnetic_factor_bridge_grounded: bool
    chemical_factor_bridge_grounded: bool
    relative_contribution_relation_grounded: bool

    eligible_for_fresh_regeneration: bool

    old_candidate_approved: Literal[
        False
    ] = False

    old_candidate_may_be_reused: Literal[
        False
    ] = False

    fresh_candidate_required_if_continued: Literal[
        True
    ] = True

    fresh_external_novelty_required_if_regenerated: Literal[
        True
    ] = True

    fresh_n10_required_if_regenerated: Literal[
        True
    ] = True

    next_action: Literal[
        "REGENERATE_FRESH_CANDIDATE",
        "SEARCH_ALTERNATE_SUPPLEMENTAL_MECHANISM_OR_GAP",
        "REASSESS_TARGET_RELATION_PRIOR_ART",
    ]

    reason_codes: list[str] = Field(
        min_length=1
    )

    production_authority: Literal[
        False
    ] = False

    @model_validator(
        mode="after"
    )
    def _status_consistency(
        self,
    ) -> "N11RelativeContributionBranchDecision":
        eligible = (
            self.status
            == "ELIGIBLE_FOR_FRESH_CANDIDATE_REGENERATION"
        )

        if (
            self.eligible_for_fresh_regeneration
            != eligible
        ):
            raise ValueError(
                "eligible_for_fresh_regeneration "
                "must agree with status"
            )

        if (
            eligible
            and self.next_action
            != "REGENERATE_FRESH_CANDIDATE"
        ):
            raise ValueError(
                "eligible branch must regenerate "
                "a fresh candidate"
            )

        if (
            self.status
            == "DEFER_TARGET_RELATION_ALREADY_GROUNDED"
            and self.next_action
            != "REASSESS_TARGET_RELATION_PRIOR_ART"
        ):
            raise ValueError(
                "grounded target relation requires "
                "prior-art reassessment"
            )

        return self


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    payload = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    return (
        f"{prefix}:"
        f"{hashlib.sha256(payload).hexdigest()[:length]}"
    )


def decide_relative_contribution_shift_reeligibility(
    *,
    opportunity: N11MissingBridgeOpportunity,
    d2_result: N11GroundedClaimAttachmentResult,
) -> N11RelativeContributionBranchDecision:
    """Deterministic N11-D3 decision for RELATIVE_CONTRIBUTION_SHIFT.

    Required scientific geometry:

        factor -> electromagnetic/shared mechanism
        factor -> chemical/supplemental mechanism

    Only then may the operator generate a NEW candidate.

    A grounded factor -> relative-contribution relation is not treated
    as permission to regenerate. It is routed to prior-art reassessment
    because it may cover the proposed target relation itself.
    """

    if (
        d2_result
        .source_missing_bridge_opportunity_id
        != opportunity.opportunity_id
    ):
        raise ValueError(
            "D2 result does not belong to "
            "the supplied D1 opportunity"
        )

    shared_ids: list[str] = []
    supplemental_ids: list[str] = []
    target_ids: list[str] = []

    for candidate in (
        d2_result.candidates
    ):
        terms = set(
            candidate
            .matched_base_context_terms
        )

        if (
            "electromagnetic enhancement"
            in terms
        ):
            shared_ids.append(
                candidate.attachment_id
            )

        if (
            "chemical enhancement"
            in terms
        ):
            supplemental_ids.append(
                candidate.attachment_id
            )

        if (
            "relative contribution"
            in terms
        ):
            target_ids.append(
                candidate.attachment_id
            )

    shared_ids = sorted(
        set(shared_ids)
    )

    supplemental_ids = sorted(
        set(supplemental_ids)
    )

    target_ids = sorted(
        set(target_ids)
    )

    em_grounded = bool(
        shared_ids
    )

    chemical_grounded = bool(
        supplemental_ids
    )

    target_grounded = bool(
        target_ids
    )

    if target_grounded:
        status = (
            "DEFER_TARGET_RELATION_ALREADY_GROUNDED"
        )

        eligible = False

        next_action = (
            "REASSESS_TARGET_RELATION_PRIOR_ART"
        )

        reasons = [
            "FACTOR_TO_RELATIVE_CONTRIBUTION_RELATION_GROUNDED",
            "DO_NOT_TREAT_REPORTED_TARGET_RELATION_AS_GENERATION_LICENSE",
        ]

    elif not em_grounded:
        status = (
            "ABSTAIN_MISSING_SHARED_FACTOR_BRIDGE"
        )

        eligible = False

        next_action = (
            "SEARCH_ALTERNATE_SUPPLEMENTAL_MECHANISM_OR_GAP"
        )

        reasons = [
            "NO_FACTOR_TO_ELECTROMAGNETIC_SHARED_BRIDGE",
            "RELATIVE_CONTRIBUTION_OPERATOR_NOT_REELIGIBLE",
        ]

    elif not chemical_grounded:
        status = (
            "ABSTAIN_MISSING_SUPPLEMENTAL_FACTOR_BRIDGE"
        )

        eligible = False

        next_action = (
            "SEARCH_ALTERNATE_SUPPLEMENTAL_MECHANISM_OR_GAP"
        )

        reasons = [
            "FACTOR_TO_ELECTROMAGNETIC_SHARED_BRIDGE_GROUNDED",
            "NO_FACTOR_TO_CHEMICAL_SUPPLEMENTAL_BRIDGE",
            "RELATIVE_CONTRIBUTION_OPERATOR_NOT_REELIGIBLE",
        ]

    else:
        status = (
            "ELIGIBLE_FOR_FRESH_CANDIDATE_REGENERATION"
        )

        eligible = True

        next_action = (
            "REGENERATE_FRESH_CANDIDATE"
        )

        reasons = [
            "FACTOR_TO_ELECTROMAGNETIC_SHARED_BRIDGE_GROUNDED",
            "FACTOR_TO_CHEMICAL_SUPPLEMENTAL_BRIDGE_GROUNDED",
            "OPERATOR_REELIGIBLE_FOR_FRESH_GENERATION_ONLY",
        ]

    return (
        N11RelativeContributionBranchDecision(
            decision_id=_stable_id(
                "n11_d3_relative_contribution",
                opportunity.opportunity_id,
                d2_result.search_id,
                status,
                *shared_ids,
                *supplemental_ids,
                *target_ids,
            ),
            source_missing_bridge_opportunity_id=(
                opportunity.opportunity_id
            ),
            source_d2_search_id=(
                d2_result.search_id
            ),
            status=status,
            shared_factor_bridge_attachment_ids=(
                shared_ids
            ),
            supplemental_factor_bridge_attachment_ids=(
                supplemental_ids
            ),
            target_relation_attachment_ids=(
                target_ids
            ),
            electromagnetic_factor_bridge_grounded=(
                em_grounded
            ),
            chemical_factor_bridge_grounded=(
                chemical_grounded
            ),
            relative_contribution_relation_grounded=(
                target_grounded
            ),
            eligible_for_fresh_regeneration=(
                eligible
            ),
            next_action=next_action,
            reason_codes=reasons,
        )
    )
