from pipeline_core.discovery.nonobviousness_mechanism_operator_policy import (
    derive_mechanism_operator_policy,
)
from pipeline_core.discovery.nonobviousness_mechanism_semantics_contracts import (
    MechanismSemanticDraft,
)


def draft(
    classification,
    *,
    shared=None,
    baseline_only=None,
    supplemental_only=None,
    grounded=False,
):
    return MechanismSemanticDraft(
        classification=classification,
        shared_mechanistic_components=(
            shared or []
        ),
        baseline_only_components=(
            baseline_only or []
        ),
        supplemental_only_components=(
            supplemental_only or []
        ),
        task_relation_grounded=grounded,
        reason_summary="test",
        epistemic_cautions=[],
        confidence="HIGH",
    )


def test_real_b1_partial_overlap_enables_only_safe_search_family():
    row = derive_mechanism_operator_policy(
        draft=draft(
            "PARTIAL_OVERLAP_WITH_DISTINCT_COMPONENT",
            shared=[
                "electromagnetic enhancement",
            ],
            baseline_only=[
                "plasmon hybridization",
            ],
            supplemental_only=[
                "chemical enhancement",
            ],
        ),
        supply_geometry=
            "COMMON_ANCHOR_CONTEXT",
        hypothesis_bound_gap_available=
            True,
        grounded_design_lever_available=
            True,
        explicit_competition_signal=
            False,
        explicit_switch_signal=
            False,
    )

    assert set(
        row.eligible_operators
    ) == {
        "MECHANISM_AUGMENTATION",
        "RELATIVE_CONTRIBUTION_SHIFT",
    }

    assert (
        "PATHWAY_COMPETITION"
        in row.blocked_operators
    )

    assert (
        "NO_EXPLICIT_COMPETITION_SIGNAL"
        in row.blocked_operators[
            "PATHWAY_COMPETITION"
        ]
    )

    assert (
        "MECHANISM_SWITCH"
        in row.blocked_operators
    )

    assert (
        "NO_EXPLICIT_SWITCH_OR_TRANSITION_SIGNAL"
        in row.blocked_operators[
            "MECHANISM_SWITCH"
        ]
    )


def test_same_mechanism_enables_no_operator():
    row = derive_mechanism_operator_policy(
        draft=draft(
            "SAME_MECHANISM",
            shared=["plasmonic EM"],
        ),
        supply_geometry=
            "COMMON_ANCHOR_CONTEXT",
        hypothesis_bound_gap_available=
            True,
    )

    assert row.eligible_operators == []


def test_partial_overlap_requires_bound_gap():
    row = derive_mechanism_operator_policy(
        draft=draft(
            "PARTIAL_OVERLAP_WITH_DISTINCT_COMPONENT",
            shared=["EM"],
            supplemental_only=["chemical"],
        ),
        supply_geometry=
            "COMMON_ANCHOR_CONTEXT",
        hypothesis_bound_gap_available=
            False,
    )

    assert row.eligible_operators == []

    for operator in (
        "MECHANISM_AUGMENTATION",
        "RELATIVE_CONTRIBUTION_SHIFT",
        "PATHWAY_COMPETITION",
        "MECHANISM_SWITCH",
    ):
        assert (
            "NO_HYPOTHESIS_BOUND_UNRESOLVED_GAP"
            in row.blocked_operators[
                operator
            ]
        )


def test_grounded_task_relation_disables_unresolved_relation_search():
    row = derive_mechanism_operator_policy(
        draft=draft(
            "PARTIAL_OVERLAP_WITH_DISTINCT_COMPONENT",
            shared=["EM"],
            supplemental_only=["chemical"],
            grounded=True,
        ),
        supply_geometry=
            "DIRECT_SCIENTIFIC_CHAIN",
        hypothesis_bound_gap_available=
            True,
    )

    assert row.eligible_operators == []

    for operator in (
        "MECHANISM_AUGMENTATION",
        "RELATIVE_CONTRIBUTION_SHIFT",
        "PATHWAY_COMPETITION",
        "MECHANISM_SWITCH",
    ):
        assert (
            "TASK_RELATION_ALREADY_GROUNDED"
            in row.blocked_operators[
                operator
            ]
        )


def test_supplemental_subsumption_allows_augmentation_not_shift():
    row = derive_mechanism_operator_policy(
        draft=draft(
            "SUPPLEMENTAL_SUBSUMES_BASELINE",
            shared=["EM"],
            supplemental_only=["chemical"],
        ),
        supply_geometry=
            "COMMON_ANCHOR_CONTEXT",
        hypothesis_bound_gap_available=
            True,
    )

    assert row.eligible_operators == [
        "MECHANISM_AUGMENTATION"
    ]


def test_distinct_mechanisms_do_not_imply_competition():
    row = derive_mechanism_operator_policy(
        draft=draft(
            "DISTINCT_MECHANISMS",
            baseline_only=["mechanism A"],
            supplemental_only=["mechanism B"],
        ),
        supply_geometry=
            "COMMON_ANCHOR_CONTEXT",
        hypothesis_bound_gap_available=
            True,
        explicit_competition_signal=
            False,
    )

    assert (
        "PATHWAY_COMPETITION"
        not in row.eligible_operators
    )

    assert (
        "NO_EXPLICIT_COMPETITION_SIGNAL"
        in row.blocked_operators[
            "PATHWAY_COMPETITION"
        ]
    )


def test_explicit_competition_signal_can_enable_competition():
    row = derive_mechanism_operator_policy(
        draft=draft(
            "DISTINCT_MECHANISMS",
            baseline_only=["mechanism A"],
            supplemental_only=["mechanism B"],
        ),
        supply_geometry=
            "COMMON_ANCHOR_CONTEXT",
        hypothesis_bound_gap_available=
            True,
        explicit_competition_signal=
            True,
    )

    assert (
        "PATHWAY_COMPETITION"
        in row.eligible_operators
    )


def test_switch_requires_signal_and_grounded_lever():
    base = draft(
        "DISTINCT_MECHANISMS",
        baseline_only=["mechanism A"],
        supplemental_only=["mechanism B"],
    )

    no_lever = (
        derive_mechanism_operator_policy(
            draft=base,
            supply_geometry=
                "COMMON_ANCHOR_CONTEXT",
            hypothesis_bound_gap_available=
                True,
            explicit_switch_signal=
                True,
            grounded_design_lever_available=
                False,
        )
    )

    assert (
        "MECHANISM_SWITCH"
        not in no_lever.eligible_operators
    )

    with_lever = (
        derive_mechanism_operator_policy(
            draft=base,
            supply_geometry=
                "COMMON_ANCHOR_CONTEXT",
            hypothesis_bound_gap_available=
                True,
            explicit_switch_signal=
                True,
            grounded_design_lever_available=
                True,
        )
    )

    assert (
        "MECHANISM_SWITCH"
        in with_lever.eligible_operators
    )


def test_policy_never_promotes_unresolved_relation_to_evidence():
    row = derive_mechanism_operator_policy(
        draft=draft(
            "PARTIAL_OVERLAP_WITH_DISTINCT_COMPONENT",
            shared=["EM"],
            supplemental_only=["chemical"],
        ),
        supply_geometry=
            "COMMON_ANCHOR_CONTEXT",
        hypothesis_bound_gap_available=
            True,
    )

    assert (
        row.unresolved_relation_promoted_to_positive_evidence
        is False
    )

    assert (
        row.llm_has_operator_authority
        is False
    )

    assert (
        row.scientific_selection_changed
        is False
    )
