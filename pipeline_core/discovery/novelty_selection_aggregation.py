from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltySelectionRole,
)


NonobviousnessOutcome = Literal[
    "POTENTIALLY_NON_OBVIOUS",
    "SATURATED_PRIOR_ART",
    "ROUTINE_FROM_PRIOR_ART",
    "INSUFFICIENT_FOR_JUDGMENT",
    "NEEDS_REFINEMENT",
]

_ALLOWED_SELECTION_ROLES = frozenset(
    get_args(NoveltySelectionRole)
)

_ALLOWED_OUTCOMES = frozenset(
    get_args(NonobviousnessOutcome)
)

RoleAwareSelectionClass = Literal[
    "ELIGIBLE",
    "CONDITIONAL",
    "INELIGIBLE",
]

RoleAwareSelectionAction = Literal[
    "KEEP_ROLE_AWARE_NONOBVIOUS_CANDIDATE",
    "RESOLVE_NOVELTY_BEARING_EVIDENCE",
    "REFINE_NOVELTY_BEARING_SPECIFICATION",
    "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH",
    "RESOLVE_REQUIRED_ENABLING_RELATION",
    "REFINE_NOVELTY_SELECTION_ROLE_SPECIFICATION",
]


_KNOWN_OR_POSITIVE = {
    "POTENTIALLY_NON_OBVIOUS",
    "SATURATED_PRIOR_ART",
    "ROUTINE_FROM_PRIOR_ART",
}

_UNRESOLVED = {
    "INSUFFICIENT_FOR_JUDGMENT",
    "NEEDS_REFINEMENT",
}

_ROUTINE = {
    "SATURATED_PRIOR_ART",
    "ROUTINE_FROM_PRIOR_ART",
}


@dataclass(frozen=True)
class RoleAwareAtomicClaim:
    """Outcome-blind-role + N9 outcome used for shadow aggregation.

    novelty_selection_role must have been assigned independently from
    prior-art/non-obviousness outcomes. This object does not infer a role
    from importance, claim kind, prior-art status, or scientific wording.
    """

    claim_id: str
    novelty_selection_role: NoveltySelectionRole
    nonobviousness_outcome: NonobviousnessOutcome


@dataclass(frozen=True)
class RoleAwareAggregation:
    """Pure hypothesis-level selection result.

    CONDITIONAL is not positive non-obviousness authority. It means that
    a novelty-bearing hypothesis structure remains scientifically
    candidate-worthy, but the evidence/specification is insufficient for
    ELIGIBLE authority.

    This contract has no production fallback authority in N10-A2.
    """

    selection_class: RoleAwareSelectionClass
    action: RoleAwareSelectionAction

    novelty_bearing_claim_ids: tuple[str, ...]
    required_enabling_claim_ids: tuple[str, ...]
    testing_prediction_claim_ids: tuple[str, ...]
    auxiliary_claim_ids: tuple[str, ...]

    blocking_claim_ids: tuple[str, ...]
    unresolved_claim_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    @property
    def positive_nonobviousness_authority(self) -> bool:
        return self.selection_class == "ELIGIBLE"


def _ids_for_role(
    claims: tuple[RoleAwareAtomicClaim, ...],
    role: NoveltySelectionRole,
) -> tuple[str, ...]:
    return tuple(
        claim.claim_id
        for claim in claims
        if claim.novelty_selection_role == role
    )


def aggregate_role_aware_nonobviousness(
    claims: tuple[RoleAwareAtomicClaim, ...],
) -> RoleAwareAggregation:
    """Aggregate atomic N9 outcomes without conflating scientific roles.

    Policy:
    - NOVELTY_BEARING claims carry hypothesis-level distinctiveness.
    - Every declared NOVELTY_BEARING branch must independently avoid
      routine/saturated prior art.
    - Positive authority requires every NOVELTY_BEARING branch to be
      POTENTIALLY_NON_OBVIOUS.
    - REQUIRED_ENABLING_RELATION may be established/routine without
      destroying higher-order novelty, but unresolved enabling relations
      prevent ELIGIBLE authority.
    - TESTING_PREDICTION and AUXILIARY prior-art outcomes do not by
      themselves destroy novelty carried by a separate novelty-bearing
      relation.
    - Absence of a NOVELTY_BEARING role fails closed to role refinement.
    - CONDITIONAL never means novel; it means unresolved candidate status.
    """

    if not claims:
        return RoleAwareAggregation(
            selection_class="INELIGIBLE",
            action=(
                "REFINE_NOVELTY_SELECTION_ROLE_SPECIFICATION"
            ),
            novelty_bearing_claim_ids=(),
            required_enabling_claim_ids=(),
            testing_prediction_claim_ids=(),
            auxiliary_claim_ids=(),
            blocking_claim_ids=(),
            unresolved_claim_ids=(),
            reason_codes=(
                "no_atomic_claims",
                "no_novelty_bearing_claims",
            ),
        )

    seen: set[str] = set()

    for claim in claims:
        claim_id = str(claim.claim_id).strip()

        if not claim_id:
            raise ValueError(
                "role-aware atomic claim requires claim_id"
            )

        if claim_id in seen:
            raise ValueError(
                "duplicate role-aware atomic claim_id: "
                + claim_id
            )

        seen.add(claim_id)

        if (
            claim.novelty_selection_role
            not in _ALLOWED_SELECTION_ROLES
        ):
            raise ValueError(
                "unsupported novelty selection role: "
                + str(
                    claim.novelty_selection_role
                )
            )

        if (
            claim.nonobviousness_outcome
            not in _ALLOWED_OUTCOMES
        ):
            raise ValueError(
                "unsupported nonobviousness outcome: "
                + str(
                    claim.nonobviousness_outcome
                )
            )

    novelty = tuple(
        claim
        for claim in claims
        if claim.novelty_selection_role
        == "NOVELTY_BEARING"
    )

    enabling = tuple(
        claim
        for claim in claims
        if claim.novelty_selection_role
        == "REQUIRED_ENABLING_RELATION"
    )

    testing = tuple(
        claim
        for claim in claims
        if claim.novelty_selection_role
        == "TESTING_PREDICTION"
    )

    auxiliary = tuple(
        claim
        for claim in claims
        if claim.novelty_selection_role
        == "AUXILIARY"
    )

    novelty_ids = tuple(
        claim.claim_id
        for claim in novelty
    )

    enabling_ids = tuple(
        claim.claim_id
        for claim in enabling
    )

    testing_ids = tuple(
        claim.claim_id
        for claim in testing
    )

    auxiliary_ids = tuple(
        claim.claim_id
        for claim in auxiliary
    )

    if not novelty:
        return RoleAwareAggregation(
            selection_class="INELIGIBLE",
            action=(
                "REFINE_NOVELTY_SELECTION_ROLE_SPECIFICATION"
            ),
            novelty_bearing_claim_ids=(),
            required_enabling_claim_ids=enabling_ids,
            testing_prediction_claim_ids=testing_ids,
            auxiliary_claim_ids=auxiliary_ids,
            blocking_claim_ids=(),
            unresolved_claim_ids=(),
            reason_codes=(
                "no_novelty_bearing_claims",
            ),
        )

    routine_novelty = tuple(
        claim
        for claim in novelty
        if claim.nonobviousness_outcome
        in _ROUTINE
    )

    if routine_novelty:
        blocking_ids = tuple(
            claim.claim_id
            for claim in routine_novelty
        )

        return RoleAwareAggregation(
            selection_class="INELIGIBLE",
            action=(
                "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH"
            ),
            novelty_bearing_claim_ids=novelty_ids,
            required_enabling_claim_ids=enabling_ids,
            testing_prediction_claim_ids=testing_ids,
            auxiliary_claim_ids=auxiliary_ids,
            blocking_claim_ids=blocking_ids,
            unresolved_claim_ids=(),
            reason_codes=tuple(
                [
                    "novelty_bearing_branch_routine_or_saturated",
                    *[
                        "novelty_bearing_claim_blocked:"
                        + claim.claim_id
                        + ":"
                        + claim.nonobviousness_outcome
                        for claim in routine_novelty
                    ],
                ]
            ),
        )

    unresolved_novelty = tuple(
        claim
        for claim in novelty
        if claim.nonobviousness_outcome
        in _UNRESOLVED
    )

    if unresolved_novelty:
        unresolved_ids = tuple(
            claim.claim_id
            for claim in unresolved_novelty
        )

        if any(
            claim.nonobviousness_outcome
            == "NEEDS_REFINEMENT"
            for claim in unresolved_novelty
        ):
            action: RoleAwareSelectionAction = (
                "REFINE_NOVELTY_BEARING_SPECIFICATION"
            )
        else:
            action = (
                "RESOLVE_NOVELTY_BEARING_EVIDENCE"
            )

        return RoleAwareAggregation(
            selection_class="CONDITIONAL",
            action=action,
            novelty_bearing_claim_ids=novelty_ids,
            required_enabling_claim_ids=enabling_ids,
            testing_prediction_claim_ids=testing_ids,
            auxiliary_claim_ids=auxiliary_ids,
            blocking_claim_ids=(),
            unresolved_claim_ids=unresolved_ids,
            reason_codes=tuple(
                [
                    "novelty_bearing_nonobviousness_unresolved",
                    *[
                        "novelty_bearing_claim_unresolved:"
                        + claim.claim_id
                        + ":"
                        + claim.nonobviousness_outcome
                        for claim in unresolved_novelty
                    ],
                ]
            ),
        )

    # If execution reaches here, every novelty-bearing branch is
    # POTENTIALLY_NON_OBVIOUS because routine and unresolved outcomes
    # have already been handled exhaustively.
    if not all(
        claim.nonobviousness_outcome
        == "POTENTIALLY_NON_OBVIOUS"
        for claim in novelty
    ):
        raise ValueError(
            "unsupported novelty-bearing outcome"
        )

    unresolved_enabling = tuple(
        claim
        for claim in enabling
        if claim.nonobviousness_outcome
        in _UNRESOLVED
    )

    if unresolved_enabling:
        unresolved_ids = tuple(
            claim.claim_id
            for claim in unresolved_enabling
        )

        return RoleAwareAggregation(
            selection_class="CONDITIONAL",
            action="RESOLVE_REQUIRED_ENABLING_RELATION",
            novelty_bearing_claim_ids=novelty_ids,
            required_enabling_claim_ids=enabling_ids,
            testing_prediction_claim_ids=testing_ids,
            auxiliary_claim_ids=auxiliary_ids,
            blocking_claim_ids=(),
            unresolved_claim_ids=unresolved_ids,
            reason_codes=tuple(
                [
                    "required_enabling_relation_unresolved",
                    *[
                        "required_enabling_claim_unresolved:"
                        + claim.claim_id
                        + ":"
                        + claim.nonobviousness_outcome
                        for claim in unresolved_enabling
                    ],
                ]
            ),
        )

    if not all(
        claim.nonobviousness_outcome
        in _KNOWN_OR_POSITIVE
        for claim in enabling
    ):
        raise ValueError(
            "unsupported required-enabling outcome"
        )

    return RoleAwareAggregation(
        selection_class="ELIGIBLE",
        action="KEEP_ROLE_AWARE_NONOBVIOUS_CANDIDATE",
        novelty_bearing_claim_ids=novelty_ids,
        required_enabling_claim_ids=enabling_ids,
        testing_prediction_claim_ids=testing_ids,
        auxiliary_claim_ids=auxiliary_ids,
        blocking_claim_ids=(),
        unresolved_claim_ids=(),
        reason_codes=(
            "all_novelty_bearing_claims_potentially_nonobvious",
            "required_enabling_relations_nonblocking",
        ),
    )
