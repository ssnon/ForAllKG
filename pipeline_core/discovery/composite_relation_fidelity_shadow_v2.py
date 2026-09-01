from __future__ import annotations

from dataclasses import replace

from pipeline_core.discovery.composite_relation_fidelity_shadow import (
    CompositeRelationFidelityShadowCritic,
    CompositeRelationFidelityShadowReview,
    _has_role_local_signal,
)


# Frozen from N8-A15Q.
#
# Development-contaminated cases:
#   Q36, Q38, Q40
#
# Do not tune these values further against those cases.
#
# Scientific criterion is unchanged:
#
#   matched source state
#   + mediator contrast
#   + outcome contrast
#
# must occur in the SAME predicted observable.
#
# V2 only expands ordinary linguistic realizations of
# mediator/outcome contrast. Existing V1 signals are preserved
# additively and matched_source_state is never changed.

_GENERAL_RELATION_PATTERN = (
    r"\b(?:"
    r"variation|variability|"
    r"vary|varies|varied|varying|"
    r"different|differ|differs|differing|"
    r"relative|"
    r"relationship|association|associated|dependence|"
    r"contrast|contrasting|"
    r"change|changes|changed|changing"
    r")\b"
)

_GENERAL_RELATION_MAX_GAP_TOKENS = 3


class CompositeRelationFidelityShadowCriticV2:
    """
    Additive shadow-only linguistic generalization of A15F.

    This detector does not alter the scientific structural rule.
    """

    def __init__(self) -> None:
        self._base = (
            CompositeRelationFidelityShadowCritic()
        )

    def review(
        self,
        axis,
        hypothesis,
    ) -> CompositeRelationFidelityShadowReview:

        base = self._base.review(
            axis,
            hypothesis,
        )

        if not base.applicable:
            return base

        raw_observations = tuple(
            getattr(
                hypothesis,
                "predicted_observations",
                (),
            )
            or ()
        )

        if (
            len(raw_observations)
            != len(base.observations)
        ):
            # Preserve V1 rather than guessing observation alignment.
            return base

        diagnostics = []

        for base_diag, raw_obs in zip(
            base.observations,
            raw_observations,
        ):

            text = str(
                getattr(
                    raw_obs,
                    "observable",
                    "",
                )
                or ""
            )

            extra_mediator = (
                False
                if base_diag.mediator_contrast
                else _has_role_local_signal(
                    text,
                    base_diag.mediator_matches,
                    (
                        _GENERAL_RELATION_PATTERN,
                    ),
                    max_gap_tokens=(
                        _GENERAL_RELATION_MAX_GAP_TOKENS
                    ),
                )
            )

            extra_outcome = (
                False
                if base_diag.outcome_contrast
                else _has_role_local_signal(
                    text,
                    base_diag.outcome_matches,
                    (
                        _GENERAL_RELATION_PATTERN,
                    ),
                    max_gap_tokens=(
                        _GENERAL_RELATION_MAX_GAP_TOKENS
                    ),
                )
            )

            mediator_contrast = (
                base_diag.mediator_contrast
                or extra_mediator
            )

            outcome_contrast = (
                base_diag.outcome_contrast
                or extra_outcome
            )

            complete = (
                base_diag.matched_source_state
                and mediator_contrast
                and outcome_contrast
            )

            diagnostics.append(
                replace(
                    base_diag,
                    mediator_contrast=(
                        mediator_contrast
                    ),
                    outcome_contrast=(
                        outcome_contrast
                    ),
                    complete_conditional_consequence=(
                        complete
                    ),
                )
            )

        status = (
            "pass"
            if any(
                row.complete_conditional_consequence
                for row in diagnostics
            )
            else "fail"
        )

        # Additive detector is forbidden from converting any V1
        # positive into a negative.
        if (
            base.status == "pass"
            and status != "pass"
        ):
            raise RuntimeError(
                "V2 violated additive monotonicity: "
                "a V1 PASS became non-PASS."
            )

        if status == base.status:
            reason_codes = tuple(
                base.reason_codes
            )

        elif (
            base.status == "fail"
            and status == "pass"
        ):
            reason_codes = (
                "complete_conditional_consequence",
                "additive_general_relational_signal",
                "matched_source_state_unchanged",
                "v1_signals_preserved",
            )

        else:
            raise RuntimeError(
                "Unexpected V1/V2 status transition: "
                f"{base.status!r} -> {status!r}"
            )

        return replace(
            base,
            status=status,
            observations=tuple(
                diagnostics
            ),
            reason_codes=reason_codes,
        )
