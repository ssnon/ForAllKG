from __future__ import annotations

from pipeline_core.discovery.nonobviousness_mechanism_semantics_contracts import (
    MechanismOperatorPolicyResult,
    MechanismSearchOperator,
    MechanismSemanticDraft,
    MechanismSupplyGeometry,
)


_OPERATORS: tuple[
    MechanismSearchOperator,
    ...,
] = (
    "MECHANISM_AUGMENTATION",
    "RELATIVE_CONTRIBUTION_SHIFT",
    "PATHWAY_COMPETITION",
    "MECHANISM_SWITCH",
)


def derive_mechanism_operator_policy(
    *,
    draft: MechanismSemanticDraft,
    supply_geometry:
        MechanismSupplyGeometry,
    hypothesis_bound_gap_available:
        bool,
    grounded_design_lever_available:
        bool = False,
    explicit_competition_signal:
        bool = False,
    explicit_switch_signal:
        bool = False,
) -> MechanismOperatorPolicyResult:
    """Translate semantic relation into conservative N11 search authority.

    The LLM classifies mechanism semantics only.

    Operator eligibility is deterministic and fail-closed.

    Grounded components may justify searching an unresolved relation,
    but the unresolved task→mechanism relation is never converted into
    positive scientific evidence.
    """

    eligible: list[
        MechanismSearchOperator
    ] = []

    blocked: dict[
        str,
        list[str],
    ] = {}

    def block(
        operator:
            MechanismSearchOperator,
        *reason_codes: str,
    ) -> None:
        blocked[
            operator
        ] = list(
            dict.fromkeys(
                reason_codes
            )
        )

    # ------------------------------------------------------------
    # Global fail-closed boundaries.
    # ------------------------------------------------------------

    if draft.task_relation_grounded:
        for operator in _OPERATORS:
            block(
                operator,
                "TASK_RELATION_ALREADY_GROUNDED",
                "NO_UNRESOLVED_RELATION_SEARCH_NEEDED",
            )

        return (
            MechanismOperatorPolicyResult(
                supply_geometry=
                    supply_geometry,
                semantic_classification=
                    draft.classification,
                task_relation_grounded=
                    True,
                hypothesis_bound_gap_available=
                    hypothesis_bound_gap_available,
                grounded_design_lever_available=
                    grounded_design_lever_available,
                explicit_competition_signal=
                    explicit_competition_signal,
                explicit_switch_signal=
                    explicit_switch_signal,
                eligible_operators=[],
                blocked_operators=
                    blocked,
            )
        )

    if not hypothesis_bound_gap_available:
        for operator in _OPERATORS:
            block(
                operator,
                "NO_HYPOTHESIS_BOUND_UNRESOLVED_GAP",
            )

        return (
            MechanismOperatorPolicyResult(
                supply_geometry=
                    supply_geometry,
                semantic_classification=
                    draft.classification,
                task_relation_grounded=
                    False,
                hypothesis_bound_gap_available=
                    False,
                grounded_design_lever_available=
                    grounded_design_lever_available,
                explicit_competition_signal=
                    explicit_competition_signal,
                explicit_switch_signal=
                    explicit_switch_signal,
                eligible_operators=[],
                blocked_operators=
                    blocked,
            )
        )

    # ------------------------------------------------------------
    # MECHANISM_AUGMENTATION
    #
    # Baseline-like mechanism plus a positively grounded additional
    # mechanism component can justify searching whether the task
    # variable changes or recruits that extra component.
    # ------------------------------------------------------------

    augmentation_relation = (
        draft.classification
        in {
            "SUPPLEMENTAL_SUBSUMES_BASELINE",
            "PARTIAL_OVERLAP_WITH_DISTINCT_COMPONENT",
        }
    )

    if (
        augmentation_relation
        and draft.supplemental_only_components
    ):
        eligible.append(
            "MECHANISM_AUGMENTATION"
        )
    else:
        reasons = []

        if not augmentation_relation:
            reasons.append(
                "SEMANTIC_RELATION_NOT_AUGMENTATIVE"
            )

        if (
            not draft
            .supplemental_only_components
        ):
            reasons.append(
                "NO_SUPPLEMENTAL_ONLY_MECHANISTIC_COMPONENT"
            )

        block(
            "MECHANISM_AUGMENTATION",
            *reasons,
        )

    # ------------------------------------------------------------
    # RELATIVE_CONTRIBUTION_SHIFT
    #
    # Requires a shared baseline-like component plus an explicit
    # additional component. This licenses search over relative
    # contributions, not a claim that such a shift already exists.
    # ------------------------------------------------------------

    relative_shift_relation = (
        draft.classification
        == (
            "PARTIAL_OVERLAP_"
            "WITH_DISTINCT_COMPONENT"
        )
    )

    if (
        relative_shift_relation
        and draft.shared_mechanistic_components
        and draft.supplemental_only_components
    ):
        eligible.append(
            "RELATIVE_CONTRIBUTION_SHIFT"
        )
    else:
        reasons = []

        if not relative_shift_relation:
            reasons.append(
                "SEMANTIC_RELATION_NOT_PARTIAL_OVERLAP"
            )

        if (
            not draft
            .shared_mechanistic_components
        ):
            reasons.append(
                "NO_SHARED_MECHANISTIC_COMPONENT"
            )

        if (
            not draft
            .supplemental_only_components
        ):
            reasons.append(
                "NO_DISTINCT_SUPPLEMENTAL_COMPONENT"
            )

        block(
            "RELATIVE_CONTRIBUTION_SHIFT",
            *reasons,
        )

    # ------------------------------------------------------------
    # PATHWAY_COMPETITION
    #
    # Distinct mechanisms are insufficient by themselves.
    # Co-occurrence and synergy are explicitly not competition.
    # ------------------------------------------------------------

    competition_semantics = (
        draft.classification
        in {
            "DISTINCT_MECHANISMS",
            "PARTIAL_OVERLAP_WITH_DISTINCT_COMPONENT",
        }
    )

    if (
        competition_semantics
        and explicit_competition_signal
    ):
        eligible.append(
            "PATHWAY_COMPETITION"
        )
    else:
        reasons = []

        if not competition_semantics:
            reasons.append(
                "NO_MULTIPLE_DISTINCT_MECHANISTIC_COMPONENTS"
            )

        if not explicit_competition_signal:
            reasons.append(
                "NO_EXPLICIT_COMPETITION_SIGNAL"
            )

        block(
            "PATHWAY_COMPETITION",
            *reasons,
        )

    # ------------------------------------------------------------
    # MECHANISM_SWITCH
    #
    # Strongest operator. Requires:
    # - distinct mechanistic content
    # - explicit switch/transition signal
    # - grounded lever/regime variable
    #
    # Two mechanisms alone can never imply a switch.
    # ------------------------------------------------------------

    switch_semantics = (
        draft.classification
        in {
            "DISTINCT_MECHANISMS",
            "PARTIAL_OVERLAP_WITH_DISTINCT_COMPONENT",
        }
    )

    if (
        switch_semantics
        and explicit_switch_signal
        and grounded_design_lever_available
    ):
        eligible.append(
            "MECHANISM_SWITCH"
        )
    else:
        reasons = []

        if not switch_semantics:
            reasons.append(
                "NO_MULTIPLE_DISTINCT_MECHANISTIC_COMPONENTS"
            )

        if not explicit_switch_signal:
            reasons.append(
                "NO_EXPLICIT_SWITCH_OR_TRANSITION_SIGNAL"
            )

        if (
            not grounded_design_lever_available
        ):
            reasons.append(
                "NO_GROUNDED_SWITCH_LEVER"
            )

        block(
            "MECHANISM_SWITCH",
            *reasons,
        )

    return (
        MechanismOperatorPolicyResult(
            supply_geometry=
                supply_geometry,
            semantic_classification=
                draft.classification,
            task_relation_grounded=
                draft.task_relation_grounded,
            hypothesis_bound_gap_available=
                hypothesis_bound_gap_available,
            grounded_design_lever_available=
                grounded_design_lever_available,
            explicit_competition_signal=
                explicit_competition_signal,
            explicit_switch_signal=
                explicit_switch_signal,
            eligible_operators=
                sorted(
                    set(eligible)
                ),
            blocked_operators=
                blocked,
        )
    )
