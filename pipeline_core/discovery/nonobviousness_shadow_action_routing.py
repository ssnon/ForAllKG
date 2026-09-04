from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AtomicResolutionRequirement:
    claim_id: str
    novelty_selection_role: str
    nonobviousness_outcome: str
    action: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class ShadowActionRouting:
    primary_action: str
    resolution_requirements: tuple[
        AtomicResolutionRequirement,
        ...
    ]


def _atomic_action(
    *,
    role: str,
    outcome: str,
    reason_codes: tuple[str, ...],
) -> str | None:
    reasons = set(reason_codes)

    if role == "NOVELTY_BEARING":
        if outcome in {
            "SATURATED_PRIOR_ART",
            "ROUTINE_FROM_PRIOR_ART",
        }:
            return (
                "REMOVE_OR_REAXIS_ROUTINE_"
                "NOVELTY_BRANCH"
            )

        if outcome == "NEEDS_REFINEMENT":
            if (
                "partial_prior_art_requires_resolution"
                in reasons
            ):
                return (
                    "RESOLVE_NOVELTY_BEARING_"
                    "PRIOR_ART_RELATION"
                )

            if (
                "atomic_specification_incomplete"
                in reasons
            ):
                return (
                    "REFINE_NOVELTY_BEARING_"
                    "SPECIFICATION"
                )

            return (
                "RESOLVE_NOVELTY_BEARING_"
                "REFINEMENT_STATE"
            )

        if outcome == "INSUFFICIENT_FOR_JUDGMENT":
            return (
                "RESOLVE_NOVELTY_BEARING_EVIDENCE"
            )

        return None

    if role == "REQUIRED_ENABLING_RELATION":
        # Known/routine enabling relations are allowed.
        if outcome in {
            "POTENTIALLY_NON_OBVIOUS",
            "SATURATED_PRIOR_ART",
            "ROUTINE_FROM_PRIOR_ART",
        }:
            return None

        if outcome == "NEEDS_REFINEMENT":
            if (
                "partial_prior_art_requires_resolution"
                in reasons
            ):
                return (
                    "RESOLVE_REQUIRED_ENABLING_"
                    "PRIOR_ART_RELATION"
                )

            if (
                "atomic_specification_incomplete"
                in reasons
            ):
                return (
                    "REFINE_REQUIRED_ENABLING_"
                    "RELATION_SPECIFICATION"
                )

            return (
                "RESOLVE_REQUIRED_ENABLING_"
                "REFINEMENT_STATE"
            )

        if outcome == "INSUFFICIENT_FOR_JUDGMENT":
            return (
                "RESOLVE_REQUIRED_ENABLING_RELATION"
            )

        return None

    # TESTING_PREDICTION and AUXILIARY are nonblocking under the
    # selection contract. Their uncertainty may remain diagnostic,
    # but does not generate a hypothesis-selection resolution action.
    return None


_PRIORITY = {
    "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH": 0,
    "REFINE_NOVELTY_BEARING_SPECIFICATION": 1,
    "RESOLVE_NOVELTY_BEARING_PRIOR_ART_RELATION": 2,
    "RESOLVE_NOVELTY_BEARING_REFINEMENT_STATE": 3,
    "RESOLVE_NOVELTY_BEARING_EVIDENCE": 4,
    "REFINE_REQUIRED_ENABLING_RELATION_SPECIFICATION": 5,
    "RESOLVE_REQUIRED_ENABLING_PRIOR_ART_RELATION": 6,
    "RESOLVE_REQUIRED_ENABLING_REFINEMENT_STATE": 7,
    "RESOLVE_REQUIRED_ENABLING_RELATION": 8,
}


def route_shadow_resolution_actions(
    *,
    selection_class: str,
    atomic_claims: list[dict[str, object]],
    fallback_action: str,
) -> ShadowActionRouting:
    """Refine action semantics without changing selection authority.

    The caller remains authoritative for ELIGIBLE / CONDITIONAL /
    INELIGIBLE. This function only explains what unresolved work is
    actually required.

    Critically:
      partial prior-art resolution != scientific under-specification.
    """

    requirements: list[
        AtomicResolutionRequirement
    ] = []

    for row in atomic_claims:
        claim_id = str(
            row.get("claim_id") or ""
        ).strip()

        role = str(
            row.get("novelty_selection_role") or ""
        ).strip()

        outcome = str(
            row.get("nonobviousness_outcome") or ""
        ).strip()

        raw_reasons = row.get(
            "outcome_reason_codes",
            [],
        )

        if not isinstance(
            raw_reasons,
            (list, tuple),
        ):
            raise ValueError(
                "outcome_reason_codes must be list/tuple"
            )

        reason_codes = tuple(
            str(value)
            for value in raw_reasons
            if str(value).strip()
        )

        action = _atomic_action(
            role=role,
            outcome=outcome,
            reason_codes=reason_codes,
        )

        if action is None:
            continue

        requirements.append(
            AtomicResolutionRequirement(
                claim_id=claim_id,
                novelty_selection_role=role,
                nonobviousness_outcome=outcome,
                action=action,
                reason_codes=reason_codes,
            )
        )

    if not requirements:
        return ShadowActionRouting(
            primary_action=fallback_action,
            resolution_requirements=(),
        )

    ordered = sorted(
        requirements,
        key=lambda row: (
            _PRIORITY.get(
                row.action,
                999,
            ),
            row.claim_id,
        ),
    )

    primary = ordered[0].action

    # An already decisive INELIGIBLE base state must not be softened
    # merely because another claim also has an unresolved requirement.
    if (
        selection_class == "INELIGIBLE"
        and any(
            row.action
            == "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH"
            for row in ordered
        )
    ):
        primary = (
            "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH"
        )

    return ShadowActionRouting(
        primary_action=primary,
        resolution_requirements=tuple(ordered),
    )
