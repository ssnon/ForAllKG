from types import SimpleNamespace

import pytest

from pipeline_core.discovery.hypothesis_semantic_action_adapter import (
    HypothesisSemanticActionBindingError,
    HypothesisSemanticFindingActionAdapter,
    _authority_for_verdict,
)
from pipeline_core.discovery.hypothesis_semantic_contracts import (
    SEMANTIC_DIMENSIONS,
)


def test_pass_is_informational():
    assert (
        _authority_for_verdict(
            "pass"
        )
        == "informational"
    )


def test_not_applicable_is_informational():
    assert (
        _authority_for_verdict(
            "not_applicable"
        )
        == "informational"
    )


def test_warning_is_advisory():
    assert (
        _authority_for_verdict(
            "warning"
        )
        == "advisory"
    )


def test_fail_is_actionable_but_not_terminal():
    authority = (
        _authority_for_verdict(
            "fail"
        )
    )

    assert authority == "actionable"
    assert authority != "terminal_candidate"


def _dimension(
    name,
    *,
    verdict="pass",
    hypothesis_ids=None,
):
    return SimpleNamespace(
        dimension=name,
        verdict=verdict,
        rationale=(
            f"synthetic rationale for {name}"
        ),
        hypothesis_ids=(
            ["hypothesis:final"]
            if hypothesis_ids is None
            else list(hypothesis_ids)
        ),
        statement_ids=[],
    )


def _review(
    *,
    warning_dimension=None,
    portfolio_verdict="pass",
):
    dimensions = []

    for name in SEMANTIC_DIMENSIONS:
        if (
            name
            == "abstention_appropriateness"
        ):
            dimensions.append(
                _dimension(
                    name,
                    verdict=
                        portfolio_verdict,
                    hypothesis_ids=[],
                )
            )
            continue

        verdict = (
            "warning"
            if name
            == warning_dimension
            else "pass"
        )

        dimensions.append(
            _dimension(
                name,
                verdict=verdict,
            )
        )

    return SimpleNamespace(
        review_id="semantic:test",
        source_portfolio_id=
            "portfolio:final",
        critic_prompt_version=
            "semantic-prompt:test",
        dimensions=dimensions,
    )


def _portfolio():
    return SimpleNamespace(
        portfolio_id=
            "portfolio:final",

        hypotheses=[
            SimpleNamespace(
                hypothesis_id=
                    "hypothesis:final"
            )
        ],
    )


def test_direct_final_semantic_warning_normalizes_to_advisory():
    review = _review(
        warning_dimension=
            "directional_specificity"
    )

    rows = (
        HypothesisSemanticFindingActionAdapter()
        .normalize(
            review=review,
            final_portfolio=
                _portfolio(),
        )
    )

    warning = [
        row
        for row in rows
        if (
            row.source_attributes[
                "semantic_dimension"
            ]
            ==
            "directional_specificity"
        )
    ]

    assert len(warning) == 1

    assert (
        warning[0].authority
        == "advisory"
    )

    assert (
        warning[0].lineage_ref_ids
        == []
    )


def test_informational_portfolio_dimension_is_not_broadcast():
    rows = (
        HypothesisSemanticFindingActionAdapter()
        .normalize(
            review=_review(),
            final_portfolio=
                _portfolio(),
        )
    )

    dimensions = {
        row.source_attributes[
            "semantic_dimension"
        ]
        for row in rows
    }

    assert (
        "abstention_appropriateness"
        not in dimensions
    )

    assert len(rows) == (
        len(SEMANTIC_DIMENSIONS)
        - 1
    )


def test_noninformational_portfolio_dimension_requires_separate_policy():
    with pytest.raises(
        HypothesisSemanticActionBindingError,
        match="separate portfolio action policy",
    ):
        (
            HypothesisSemanticFindingActionAdapter()
            .normalize(
                review=_review(
                    portfolio_verdict=
                        "warning"
                ),
                final_portfolio=
                    _portfolio(),
            )
        )


def test_semantic_review_must_directly_bind_final_portfolio():
    review = _review()

    review.source_portfolio_id = (
        "portfolio:older"
    )

    with pytest.raises(
        HypothesisSemanticActionBindingError,
        match="directly bound",
    ):
        (
            HypothesisSemanticFindingActionAdapter()
            .normalize(
                review=review,
                final_portfolio=
                    _portfolio(),
            )
        )


def test_foreign_hypothesis_reference_is_rejected():
    review = _review()

    review.dimensions[0].hypothesis_ids = [
        "hypothesis:foreign"
    ]

    with pytest.raises(
        HypothesisSemanticActionBindingError,
        match="non-final hypothesis IDs",
    ):
        (
            HypothesisSemanticFindingActionAdapter()
            .normalize(
                review=review,
                final_portfolio=
                    _portfolio(),
            )
        )
