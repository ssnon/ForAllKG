import pytest

from domains.sers.hypothesis_action_policy import (
    SERSContextActionPolicyError,
    SERSContextLifecycleActionPolicy,
)
from pipeline_core.discovery.hypothesis_action_contracts import (
    G1FindingRef,
    G1FindingScope,
)


PORTFOLIO = "portfolio:final"
HYPOTHESIS = "hypothesis:final"


def _scope(
    *,
    kind="central",
    assertion_ids=None,
):
    return G1FindingScope(
        kind=kind,
        hypothesis_ids=[
            HYPOTHESIS
        ],
        assertion_ids=(
            list(
                assertion_ids
                or [
                    f"{kind}:test"
                ]
            )
        ),
    )


def _finding(
    *,
    ref_id,
    status,
    authority,
    scope=None,
):
    target_scope = (
        scope
        or _scope()
    )

    return G1FindingRef(
        finding_ref_id=
            ref_id,

        source_kind=
            "context_review",

        source_artifact_id=
            "review:test",

        source_finding_id=
            "source:"
            + ref_id,

        source_status=
            status,

        authority=
            authority,

        source_portfolio_id=
            PORTFOLIO,

        source_hypothesis_ids=[
            HYPOTHESIS
        ],

        source_scope=
            target_scope,

        target_portfolio_id=
            PORTFOLIO,

        target_hypothesis_id=
            HYPOTHESIS,

        target_scope=
            target_scope,

        rationale=
            "test finding",
    )


def _decide(
    findings,
):
    return (
        SERSContextLifecycleActionPolicy()
        .decide(
            target_portfolio_id=
                PORTFOLIO,

            target_hypothesis_id=
                HYPOTHESIS,

            findings=
                list(findings),
        )
    )


def test_informational_only_is_clean_keep():
    decision = _decide([
        _finding(
            ref_id="f:info",
            status=
                "compatible_extension",
            authority=
                "informational",
        )
    ])

    assert (
        decision.disposition
        == "keep"
    )

    assert (
        decision.directives
        == []
    )


def test_unknown_only_is_keep_with_warning():
    decision = _decide([
        _finding(
            ref_id="f:unknown",
            status="unknown",
            authority="advisory",
        )
    ])

    assert (
        decision.disposition
        == "keep_with_warning"
    )

    assert (
        decision.directives
        == []
    )

    assert (
        "S1_CONTEXT_UNKNOWN_WARNING"
        in decision.reason_codes
    )


@pytest.mark.parametrize(
    "status",
    [
        "role_mismatch",
        "context_conflation",
        "conflict",
    ],
)
def test_actionable_context_status_requires_local_reframe(
    status,
):
    decision = _decide([
        _finding(
            ref_id=
                "f:" + status,

            status=
                status,

            authority=
                "actionable",
        )
    ])

    assert (
        decision.disposition
        == "repair_required"
    )

    assert len(
        decision.directives
    ) == 1

    assert (
        decision.directives[0]
        .action
        == "reframe"
    )


def test_unknown_does_not_escalate_actionable_context_to_reject():
    decision = _decide([
        _finding(
            ref_id=
                "f:role",

            status=
                "role_mismatch",

            authority=
                "actionable",
        ),
        _finding(
            ref_id=
                "f:unknown",

            status=
                "unknown",

            authority=
                "advisory",
        ),
    ])

    assert (
        decision.disposition
        == "repair_required"
    )

    assert (
        decision.disposition
        != "reject"
    )

    assert (
        "S1_CONTEXT_UNKNOWN_WARNING_PRESENT"
        in decision.reason_codes
    )


def test_same_target_scope_folds_into_one_reframe_directive():
    scope = _scope(
        kind="prediction",
        assertion_ids=[
            "prediction:final"
        ],
    )

    decision = _decide([
        _finding(
            ref_id="f:one",
            status="role_mismatch",
            authority="actionable",
            scope=scope,
        ),
        _finding(
            ref_id="f:two",
            status="context_conflation",
            authority="actionable",
            scope=scope,
        ),
    ])

    assert (
        decision.disposition
        == "repair_required"
    )

    assert len(
        decision.directives
    ) == 1

    assert set(
        decision.directives[0]
        .finding_ref_ids
    ) == {
        "f:one",
        "f:two",
    }


def test_distinct_local_scopes_remain_distinct_directives():
    decision = _decide([
        _finding(
            ref_id="f:central",
            status="role_mismatch",
            authority="actionable",
            scope=_scope(
                kind="central",
                assertion_ids=[
                    "central:final"
                ],
            ),
        ),
        _finding(
            ref_id="f:bridge",
            status="role_mismatch",
            authority="actionable",
            scope=_scope(
                kind="bridge",
                assertion_ids=[
                    "bridge:final"
                ],
            ),
        ),
    ])

    assert len(
        decision.directives
    ) == 2

    assert {
        row.target_scope.kind
        for row in decision.directives
    } == {
        "central",
        "bridge",
    }


def test_s1_policy_rejects_status_authority_corruption():
    corrupted = _finding(
        ref_id="f:corrupt",
        status="unknown",
        authority="actionable",
    )

    with pytest.raises(
        SERSContextActionPolicyError,
        match="status/authority mismatch",
    ):
        _decide([
            corrupted
        ])


def test_s1_policy_never_accepts_terminal_authority():
    # Construct then corrupt without invoking Pydantic validation
    # so the policy boundary itself is tested.
    row = _finding(
        ref_id="f:terminal",
        status="role_mismatch",
        authority="actionable",
    )

    row = row.model_copy(
        update={
            "authority":
                "terminal_candidate"
        }
    )

    with pytest.raises(
        SERSContextActionPolicyError,
    ):
        _decide([
            row
        ])


def test_empty_context_findings_is_clean_keep():
    decision = _decide([])

    assert (
        decision.disposition
        == "keep"
    )

    assert (
        decision.mutation_applied
        is False
    )
