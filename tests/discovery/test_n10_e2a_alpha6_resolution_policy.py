from pipeline_core.discovery.n10_alpha6_resolution_policy import (
    REFINE_NOVELTY_BEARING_SPECIFICATION,
    alpha6_resolution_directive_from_gate_row,
)


def _row(**overrides):
    row = {
        "selection_class": "CONDITIONAL",
        "fallback_allowed": False,
        "positive_nonobviousness_authority": False,
        "base_aggregation_action": (
            REFINE_NOVELTY_BEARING_SPECIFICATION
        ),
    }
    row.update(overrides)
    return row


def test_specification_refinement_gets_bounded_directive():
    directive = alpha6_resolution_directive_from_gate_row(
        _row()
    )

    assert directive.force_bounded_refinement is True
    assert (
        directive.use_source_external_without_targeted_search
        is True
    )
    assert (
        directive.bypass_resolved_candidate_external_exit
        is True
    )
    assert (
        directive.reason_code
        == "n10_refine_novelty_bearing_specification"
    )


def test_missing_row_is_inert():
    directive = alpha6_resolution_directive_from_gate_row(
        None
    )

    assert directive.force_bounded_refinement is False


def test_non_conditional_is_inert():
    for selection in [
        "ELIGIBLE",
        "INELIGIBLE",
        "",
        None,
    ]:
        directive = (
            alpha6_resolution_directive_from_gate_row(
                _row(selection_class=selection)
            )
        )

        assert directive.force_bounded_refinement is False


def test_fallback_positive_is_inert():
    directive = alpha6_resolution_directive_from_gate_row(
        _row(fallback_allowed=True)
    )

    assert directive.force_bounded_refinement is False


def test_missing_or_positive_authority_is_inert():
    positive = alpha6_resolution_directive_from_gate_row(
        _row(
            positive_nonobviousness_authority=True
        )
    )

    missing_row = _row()
    del missing_row[
        "positive_nonobviousness_authority"
    ]

    missing = alpha6_resolution_directive_from_gate_row(
        missing_row
    )

    assert positive.force_bounded_refinement is False
    assert missing.force_bounded_refinement is False


def test_other_n10_actions_do_not_gain_refinement_authority():
    other_actions = [
        "RESOLVE_NOVELTY_BEARING_PRIOR_ART_RELATION",
        "RESOLVE_NOVELTY_BEARING_REFINEMENT_STATE",
        "RESOLVE_NOVELTY_BEARING_EVIDENCE",
        "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH",
        "",
        None,
    ]

    for action in other_actions:
        directive = (
            alpha6_resolution_directive_from_gate_row(
                _row(
                    base_aggregation_action=action
                )
            )
        )

        assert directive.force_bounded_refinement is False


def test_missing_action_is_inert():
    row = _row()
    del row["base_aggregation_action"]

    directive = alpha6_resolution_directive_from_gate_row(
        row
    )

    assert directive.force_bounded_refinement is False


def test_malformed_non_mapping_is_inert():
    for value in [
        [],
        "CONDITIONAL",
        False,
        1,
    ]:
        directive = (
            alpha6_resolution_directive_from_gate_row(
                value  # type: ignore[arg-type]
            )
        )

        assert directive.force_bounded_refinement is False
