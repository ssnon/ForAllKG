import pytest
from pydantic import ValidationError

from pipeline_core.discovery.hypothesis_action_contracts import (
    G1ActionDecision,
    G1ActionDirective,
    G1FindingRef,
    G1FindingScope,
)


def _scope(
    kind="hypothesis",
    *,
    hypothesis_id="hypothesis:final",
    assertion_ids=None,
):
    return G1FindingScope(
        kind=kind,
        hypothesis_ids=[
            hypothesis_id
        ],
        assertion_ids=list(
            assertion_ids or []
        ),
    )


def _finding(
    *,
    ref_id="finding-ref:1",
    authority="informational",
    source_hypothesis_id="hypothesis:final",
    target_hypothesis_id="hypothesis:final",
    source_portfolio_id="portfolio:final",
    target_portfolio_id="portfolio:final",
    source_scope=None,
    target_scope=None,
    lineage=None,
):
    return G1FindingRef(
        finding_ref_id=ref_id,
        source_kind="context_review",
        source_artifact_id="context-review:test",
        source_finding_id="source-finding:test",
        source_status="test_status",
        authority=authority,
        source_portfolio_id=source_portfolio_id,
        source_hypothesis_ids=[
            source_hypothesis_id
        ],
        source_scope=(
            source_scope
            or _scope(
                hypothesis_id=
                    source_hypothesis_id
            )
        ),
        target_portfolio_id=
            target_portfolio_id,
        target_hypothesis_id=
            target_hypothesis_id,
        target_scope=(
            target_scope
            or _scope(
                hypothesis_id=
                    target_hypothesis_id
            )
        ),
        lineage_ref_ids=list(
            lineage or []
        ),
        rationale="test finding",
    )


def test_cross_generation_finding_requires_explicit_lineage():
    with pytest.raises(
        ValidationError,
        match="requires explicit lineage",
    ):
        _finding(
            source_hypothesis_id=
                "hypothesis:source",
            target_hypothesis_id=
                "hypothesis:final",
            source_portfolio_id=
                "portfolio:source",
            target_portfolio_id=
                "portfolio:final",
        )


def test_cross_generation_finding_preserves_source_and_target_scope():
    finding = _finding(
        source_hypothesis_id=
            "hypothesis:source",
        target_hypothesis_id=
            "hypothesis:final",
        source_portfolio_id=
            "portfolio:source",
        target_portfolio_id=
            "portfolio:final",
        source_scope=_scope(
            "prediction",
            hypothesis_id=
                "hypothesis:source",
            assertion_ids=[
                "prediction:source"
            ],
        ),
        target_scope=_scope(
            "prediction",
            hypothesis_id=
                "hypothesis:final",
            assertion_ids=[
                "prediction:final"
            ],
        ),
        lineage=[
            "novelty-refinement-report:test"
        ],
    )

    assert (
        finding.source_scope.assertion_ids
        == ["prediction:source"]
    )

    assert (
        finding.target_scope.assertion_ids
        == ["prediction:final"]
    )


def test_unknown_style_advisory_can_be_kept_with_warning():
    finding = _finding(
        authority="advisory"
    )

    decision = G1ActionDecision(
        decision_id="g1-decision:warning",
        target_portfolio_id="portfolio:final",
        target_hypothesis_id="hypothesis:final",
        findings=[finding],
        disposition="keep_with_warning",
        interpretation="warning only",
    )

    assert (
        decision.disposition
        == "keep_with_warning"
    )


def test_advisory_finding_cannot_directly_reject():
    finding = _finding(
        authority="advisory"
    )

    with pytest.raises(
        ValidationError,
        match="terminal_candidate",
    ):
        G1ActionDecision(
            decision_id="g1-decision:bad-reject",
            target_portfolio_id="portfolio:final",
            target_hypothesis_id="hypothesis:final",
            findings=[finding],
            disposition="reject",
            interpretation="invalid reject",
        )


def test_actionable_finding_requires_local_repair_directive():
    finding = _finding(
        authority="actionable"
    )

    with pytest.raises(
        ValidationError,
        match="requires at least one local directive",
    ):
        G1ActionDecision(
            decision_id="g1-decision:no-directive",
            target_portfolio_id="portfolio:final",
            target_hypothesis_id="hypothesis:final",
            findings=[finding],
            disposition="repair_required",
            interpretation="repair required",
        )


def test_actionable_context_finding_can_request_bounded_reframe():
    source_scope = _scope(
        "bridge",
        hypothesis_id="hypothesis:source",
        assertion_ids=[
            "bridge:hypothesis:source"
        ],
    )

    target_scope = _scope(
        "bridge",
        hypothesis_id="hypothesis:final",
        assertion_ids=[
            "bridge:hypothesis:final"
        ],
    )

    finding = _finding(
        authority="actionable",
        source_hypothesis_id=
            "hypothesis:source",
        target_hypothesis_id=
            "hypothesis:final",
        source_portfolio_id=
            "portfolio:source",
        target_portfolio_id=
            "portfolio:final",
        source_scope=source_scope,
        target_scope=target_scope,
        lineage=[
            "novelty-refinement-report:test"
        ],
    )

    directive = G1ActionDirective(
        directive_id="directive:reframe",
        action="reframe",
        target_scope=target_scope,
        finding_ref_ids=[
            finding.finding_ref_id
        ],
        rationale=(
            "repair context attachment"
        ),
    )

    decision = G1ActionDecision(
        decision_id="g1-decision:repair",
        target_portfolio_id=
            "portfolio:final",
        target_hypothesis_id=
            "hypothesis:final",
        findings=[finding],
        directives=[directive],
        disposition="repair_required",
        interpretation=(
            "local reframe before acceptance"
        ),
    )

    assert (
        decision.directives[0].action
        == "reframe"
    )


def test_remove_assertion_is_not_allowed_on_central_scope():
    with pytest.raises(
        ValidationError,
        match="prediction/assumption",
    ):
        G1ActionDirective(
            directive_id="directive:bad-remove",
            action="remove_assertion",
            target_scope=_scope(
                "central",
                assertion_ids=[
                    "central:hypothesis:final"
                ],
            ),
            finding_ref_ids=[
                "finding-ref:1"
            ],
            rationale="invalid",
        )


def test_reject_requires_terminal_candidate_authority():
    finding = _finding(
        authority="terminal_candidate"
    )

    decision = G1ActionDecision(
        decision_id="g1-decision:reject",
        target_portfolio_id=
            "portfolio:final",
        target_hypothesis_id=
            "hypothesis:final",
        findings=[finding],
        disposition="reject",
        interpretation="terminal prior-art state",
    )

    assert (
        decision.disposition
        == "reject"
    )


def test_unresolved_does_not_automatically_reject():
    finding = _finding(
        authority="advisory"
    )

    decision = G1ActionDecision(
        decision_id="g1-decision:unresolved",
        target_portfolio_id=
            "portfolio:final",
        target_hypothesis_id=
            "hypothesis:final",
        findings=[finding],
        disposition="unresolved",
        interpretation="insufficient basis",
    )

    assert (
        decision.disposition
        == "unresolved"
    )


def test_clean_keep_accepts_only_informational_findings():
    decision = G1ActionDecision(
        decision_id="g1-decision:keep",
        target_portfolio_id=
            "portfolio:final",
        target_hypothesis_id=
            "hypothesis:final",
        findings=[
            _finding(
                authority="informational"
            )
        ],
        disposition="keep",
        interpretation="no unresolved issue",
    )

    assert decision.disposition == "keep"


def test_finding_target_scope_cannot_span_multiple_hypotheses():
    with pytest.raises(
        ValidationError,
        match="exactly one target_hypothesis_id",
    ):
        G1FindingRef(
            finding_ref_id="finding-ref:multi-target",
            source_kind="context_review",
            source_artifact_id="context-review:test",
            source_finding_id="source-finding:test",
            source_status="role_mismatch",
            authority="actionable",
            source_portfolio_id="portfolio:final",
            source_hypothesis_ids=[
                "hypothesis:final",
            ],
            source_scope=_scope(
                hypothesis_id="hypothesis:final",
            ),
            target_portfolio_id="portfolio:final",
            target_hypothesis_id="hypothesis:final",
            target_scope=G1FindingScope(
                kind="hypothesis",
                hypothesis_ids=[
                    "hypothesis:final",
                    "hypothesis:other",
                ],
            ),
            rationale="invalid multi-target binding",
        )


def test_directive_scope_cannot_span_multiple_hypotheses():
    finding = _finding(
        authority="actionable",
    )

    directive = G1ActionDirective(
        directive_id="directive:multi-target",
        action="reframe",
        target_scope=G1FindingScope(
            kind="hypothesis",
            hypothesis_ids=[
                "hypothesis:final",
                "hypothesis:other",
            ],
        ),
        finding_ref_ids=[
            finding.finding_ref_id,
        ],
        rationale="invalid cross-hypothesis mutation",
    )

    with pytest.raises(
        ValidationError,
        match=(
            "exactly the decision target hypothesis"
        ),
    ):
        G1ActionDecision(
            decision_id="g1-decision:multi-target",
            target_portfolio_id="portfolio:final",
            target_hypothesis_id="hypothesis:final",
            findings=[finding],
            directives=[directive],
            disposition="repair_required",
            interpretation="invalid",
        )
