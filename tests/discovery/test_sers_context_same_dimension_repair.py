from domains.sers.hypothesis_context_interpreter import (
    _is_bounded_context_validation_issue,
    _is_same_dimension_contract_issue,
)


def test_generalize_dimension_violation_is_repairable() -> None:
    issue = (
        "bridge:hypothesis:test:mention:1: "
        "generalize cannot silently change context dimension"
    )

    assert (
        _is_same_dimension_contract_issue(
            issue
        )
    )

    assert (
        _is_bounded_context_validation_issue(
            issue
        )
    )


def test_preserve_dimension_violation_is_repairable() -> None:
    issue = (
        "prediction:test:mention:0: "
        "preserve cannot silently change context dimension"
    )

    assert (
        _is_same_dimension_contract_issue(
            issue
        )
    )


def test_intentionally_vary_dimension_violation_is_repairable() -> None:
    issue = (
        "prediction:test:mention:0: "
        "intentionally_vary cannot silently change context dimension"
    )

    assert (
        _is_same_dimension_contract_issue(
            issue
        )
    )


def test_other_semantic_failures_remain_outside_repair_lane() -> None:
    rows = [
        (
            "bridge:test:mention:0: "
            "reattach requires attachment-bound source fact"
        ),
        (
            "bridge:test:mention:0: "
            "context conflation"
        ),
        (
            "bridge:test:mention:0: "
            "combine treatment misuse"
        ),
    ]

    for issue in rows:
        assert not (
            _is_bounded_context_validation_issue(
                issue
            )
        )


def test_generic_dimension_wording_is_not_enough() -> None:
    # The lane is intentionally restricted to known same-dimension
    # treatments rather than every error containing "dimension".
    issue = (
        "bridge:test:mention:0: "
        "reattach cannot silently change context dimension"
    )

    assert not (
        _is_same_dimension_contract_issue(
            issue
        )
    )
