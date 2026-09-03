from pipeline_core.discovery.novelty_closure_compiler import (
    compile_nonobviousness_evidence_closure,
)


def _external_review(
    slot: str,
    state: str = "UNASSESSED",
):
    return {
        "slot": slot,
        "evidence_state": state,
        "positive_work_ids": [],
        "negative_coverage_sufficient": False,
    }


def _reviews():
    return [
        _external_review(
            "BASE_RELATION"
        ),
        _external_review(
            "DISTINGUISHING_FACTOR_EFFECT"
        ),
        _external_review(
            "BRIDGE_RELATION"
        ),
        _external_review(
            "FULL_RELATION"
        ),
    ]


def test_internal_positive_grounding_can_establish_base():
    compilation = (
        compile_nonobviousness_evidence_closure(
            reviews=_reviews(),
            internal_reviews=[
                {
                    "slot":
                        "BASE_RELATION",
                    "positive_statement_ids": [
                        "stmt:grounded-base"
                    ],
                }
            ],
        )
    )

    closure = compilation.closure

    assert (
        closure.base_relation
        == "ESTABLISHED"
    )

    assert (
        closure.base_internal_statement_ids
        == (
            "stmt:grounded-base",
        )
    )

    assert closure.base_work_ids == ()

    assert (
        "base_relation:"
        "established_from_internal_grounding"
        in compilation.reason_codes
    )


def test_internal_grounding_never_creates_negative_closure():
    compilation = (
        compile_nonobviousness_evidence_closure(
            reviews=_reviews(),
            internal_reviews=[
                {
                    "slot":
                        "BASE_RELATION",
                    "positive_statement_ids": [],
                }
            ],
        )
    )

    assert (
        compilation.closure.base_relation
        == "UNASSESSED"
    )

    assert (
        compilation
        .closure
        .base_internal_statement_ids
        == ()
    )


def test_external_established_still_works_without_internal_grounding():
    reviews = _reviews()

    reviews[0] = {
        "slot": "BASE_RELATION",
        "evidence_state": "ESTABLISHED",
        "positive_work_ids": [
            "work:external"
        ],
        "negative_coverage_sufficient": False,
    }

    compilation = (
        compile_nonobviousness_evidence_closure(
            reviews=reviews,
        )
    )

    assert (
        compilation.closure.base_relation
        == "ESTABLISHED"
    )

    assert (
        compilation.closure.base_work_ids
        == (
            "work:external",
        )
    )

    assert (
        compilation
        .closure
        .base_internal_statement_ids
        == ()
    )


def test_illegal_external_negative_still_fails_closed():
    reviews = _reviews()

    reviews[0] = {
        "slot": "BASE_RELATION",
        "evidence_state": "NOT_FOUND",
        "positive_work_ids": [],
        "negative_coverage_sufficient": False,
    }

    compilation = (
        compile_nonobviousness_evidence_closure(
            reviews=reviews,
        )
    )

    assert (
        compilation.closure.base_relation
        == "UNASSESSED"
    )

    assert (
        "base_relation:"
        "illegal_negative_closure_fail_closed"
        in compilation.reason_codes
    )

def test_internal_positive_overrides_search_bounded_external_negative():
    reviews = _reviews()

    reviews[0] = {
        "slot":
            "BASE_RELATION",
        "evidence_state":
            "NOT_FOUND",
        "positive_work_ids": [],
        "negative_coverage_sufficient":
            True,
    }

    compilation = (
        compile_nonobviousness_evidence_closure(
            reviews=reviews,
            internal_reviews=[
                {
                    "slot":
                        "BASE_RELATION",
                    "positive_statement_ids": [
                        "stmt:grounded-base"
                    ],
                }
            ],
        )
    )

    assert (
        compilation.closure.base_relation
        == "ESTABLISHED"
    )

    assert (
        compilation
        .closure
        .base_internal_statement_ids
        == (
            "stmt:grounded-base",
        )
    )

    assert (
        "base_relation:"
        "internal_positive_overrides_"
        "search_bounded_negative"
        in compilation.reason_codes
    )
