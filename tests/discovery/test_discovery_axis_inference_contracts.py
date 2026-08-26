import pytest
from pydantic import ValidationError

from pipeline_core.discovery.discovery_axis_inference_contracts import (
    AxisInferenceAssertionDraft,
    AxisInferenceReview,
    AxisInferenceReviewDraft,
    inference_review_status,
)


def assertion(
    *,
    assertion_id: str = "central",
    assertion_kind: str = "central_hypothesis",
    source_class: str,
    action: str,
) -> AxisInferenceAssertionDraft:
    return AxisInferenceAssertionDraft(
        assertion_id=assertion_id,
        assertion_kind=assertion_kind,
        assertion_text="Synthetic assertion.",
        source_class=source_class,
        action=action,
        grounded_statement_ids=[],
        axis_basis=[],
        specificity_tags=[],
        rationale="Synthetic contract test.",
    )


@pytest.mark.parametrize(
    ("source_class", "action"),
    [
        ("G_GROUNDED", "KEEP"),
        ("A_AXIS", "KEEP_HYPOTHETICAL"),
        ("S_BOUNDED_SYNTHESIS", "KEEP"),
        ("S_BOUNDED_SYNTHESIS", "OPEN_DIRECTION"),
        ("X_UNSUPPORTED_SPECIFICITY", "REFRAME"),
        ("X_UNSUPPORTED_SPECIFICITY", "REMOVE"),
    ],
)
def test_valid_source_action_pairs(
    source_class: str,
    action: str,
) -> None:
    row = assertion(
        source_class=source_class,
        action=action,
    )

    assert row.source_class == source_class
    assert row.action == action


@pytest.mark.parametrize(
    ("source_class", "action"),
    [
        ("G_GROUNDED", "REFRAME"),
        ("A_AXIS", "KEEP"),
        ("A_AXIS", "REFRAME"),
        ("S_BOUNDED_SYNTHESIS", "REMOVE"),
        ("X_UNSUPPORTED_SPECIFICITY", "KEEP"),
        ("X_UNSUPPORTED_SPECIFICITY", "OPEN_DIRECTION"),
    ],
)
def test_invalid_source_action_pairs_are_rejected_at_compiled_boundary(
    source_class: str,
    action: str,
) -> None:
    # Raw LLM-facing assertions intentionally remain parseable so a
    # bounded contract-repair pass can see the invalid serialization.
    row = assertion(
        source_class=source_class,
        action=action,
    )

    assert row.source_class == source_class
    assert row.action == action

    # Final compiled review remains strict.
    with pytest.raises(
        ValidationError,
        match="inference source/action mismatch",
    ):
        compiled_review(
            assertions=[row],
            status="pass",
        )


def test_review_draft_requires_exactly_one_central_assertion() -> None:
    with pytest.raises(
        ValidationError,
        match="exactly one central_hypothesis",
    ):
        AxisInferenceReviewDraft(
            assertions=[
                assertion(
                    assertion_id="p1",
                    assertion_kind="prediction",
                    source_class="S_BOUNDED_SYNTHESIS",
                    action="KEEP",
                )
            ],
            overall_risk="low",
            interpretation="No central assertion.",
        )


def test_review_draft_rejects_duplicate_assertion_ids() -> None:
    with pytest.raises(
        ValidationError,
        match="duplicate inference assertion_id",
    ):
        AxisInferenceReviewDraft(
            assertions=[
                assertion(
                    assertion_id="same",
                    source_class="S_BOUNDED_SYNTHESIS",
                    action="KEEP",
                ),
                assertion(
                    assertion_id="same",
                    assertion_kind="prediction",
                    source_class="S_BOUNDED_SYNTHESIS",
                    action="KEEP",
                ),
            ],
            overall_risk="low",
            interpretation="Duplicate IDs.",
        )


def compiled_review(
    *,
    assertions: list[
        AxisInferenceAssertionDraft
    ],
    status: str,
) -> AxisInferenceReview:
    return AxisInferenceReview(
        review_id="axis_inference_review:test",
        axis_id="axis:test",
        hypothesis_id="hypothesis:test",
        source_context_id="context:test",
        source_context_sha256="abc123",
        critic_prompt_version="axis-inference-prompt-v1",
        critic_prompt_sha256="def456",
        status=status,
        assertions=assertions,
        overall_risk="moderate",
        reason_codes=[],
        interpretation="Synthetic compiled review.",
    )


def test_pass_review_accepts_keep_actions_only() -> None:
    rows = [
        assertion(
            source_class="S_BOUNDED_SYNTHESIS",
            action="KEEP",
        ),
        assertion(
            assertion_id="p1",
            assertion_kind="prediction",
            source_class="A_AXIS",
            action="KEEP_HYPOTHETICAL",
        ),
    ]

    review = compiled_review(
        assertions=rows,
        status="pass",
    )

    assert review.status == "pass"
    assert inference_review_status(rows) == "pass"


@pytest.mark.parametrize(
    ("source_class", "action"),
    [
        ("S_BOUNDED_SYNTHESIS", "OPEN_DIRECTION"),
        ("X_UNSUPPORTED_SPECIFICITY", "REFRAME"),
        ("X_UNSUPPORTED_SPECIFICITY", "REMOVE"),
    ],
)
def test_repair_actions_require_reframe_required_status(
    source_class: str,
    action: str,
) -> None:
    rows = [
        assertion(
            source_class="S_BOUNDED_SYNTHESIS",
            action="KEEP",
        ),
        assertion(
            assertion_id="p1",
            assertion_kind="prediction",
            source_class=source_class,
            action=action,
        ),
    ]

    review = compiled_review(
        assertions=rows,
        status="reframe_required",
    )

    assert review.status == "reframe_required"
    assert (
        inference_review_status(rows)
        == "reframe_required"
    )


def test_pass_status_rejects_hidden_repair_action() -> None:
    rows = [
        assertion(
            source_class="S_BOUNDED_SYNTHESIS",
            action="KEEP",
        ),
        assertion(
            assertion_id="p1",
            assertion_kind="prediction",
            source_class="X_UNSUPPORTED_SPECIFICITY",
            action="REFRAME",
        ),
    ]

    with pytest.raises(
        ValidationError,
        match="status/action mismatch",
    ):
        compiled_review(
            assertions=rows,
            status="pass",
        )


def test_reframe_required_rejects_keep_only_review() -> None:
    rows = [
        assertion(
            source_class="S_BOUNDED_SYNTHESIS",
            action="KEEP",
        )
    ]

    with pytest.raises(
        ValidationError,
        match="status/action mismatch",
    ):
        compiled_review(
            assertions=rows,
            status="reframe_required",
        )
