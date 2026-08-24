import pytest

from domains.sers.hypothesis_combined_action_policy import (
    SERSCombinedActionPolicyError,
    SERSCombinedLifecycleActionPolicy,
)
from pipeline_core.discovery.hypothesis_action_contracts import (
    G1ActionDecision,
    G1ActionDirective,
    G1FindingRef,
    G1FindingScope,
)


PORTFOLIO = "portfolio:final"
HYPOTHESIS = "hypothesis:final"


def _hyp_scope():
    return G1FindingScope(
        kind="hypothesis",
        hypothesis_ids=[
            HYPOTHESIS
        ],
    )


def _local_scope(
    kind="central",
    assertion_id="central:final",
):
    return G1FindingScope(
        kind=kind,
        hypothesis_ids=[
            HYPOTHESIS
        ],
        assertion_ids=[
            assertion_id
        ],
    )


def _finding(
    *,
    ref_id,
    source_kind,
    authority,
    status,
    scope=None,
):
    target_scope = (
        scope
        or _hyp_scope()
    )

    return G1FindingRef(
        finding_ref_id=
            ref_id,

        source_kind=
            source_kind,

        source_artifact_id=
            "artifact:" + source_kind,

        source_finding_id=
            "source:" + ref_id,

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
            "synthetic finding",
    )


def _context_decision(
    *,
    actionable=True,
):
    if actionable:
        finding = _finding(
            ref_id="context:actionable",
            source_kind="context_review",
            authority="actionable",
            status="role_mismatch",
            scope=_local_scope(),
        )

        directive = G1ActionDirective(
            directive_id=
                "directive:context",

            action=
                "reframe",

            target_scope=
                finding.target_scope,

            finding_ref_ids=[
                finding.finding_ref_id
            ],

            rationale=
                "context reframe",
        )

        return G1ActionDecision(
            decision_id=
                "decision:context",

            target_portfolio_id=
                PORTFOLIO,

            target_hypothesis_id=
                HYPOTHESIS,

            findings=[
                finding
            ],

            directives=[
                directive
            ],

            disposition=
                "repair_required",

            reason_codes=[
                "CONTEXT_REPAIR"
            ],

            interpretation=
                "context repair required",

            mutation_applied=
                False,
        )

    finding = _finding(
        ref_id="context:info",
        source_kind="context_review",
        authority="informational",
        status="compatible_extension",
        scope=_local_scope(),
    )

    return G1ActionDecision(
        decision_id=
            "decision:context-clear",

        target_portfolio_id=
            PORTFOLIO,

        target_hypothesis_id=
            HYPOTHESIS,

        findings=[
            finding
        ],

        directives=[],

        disposition="keep",

        reason_codes=[
            "CONTEXT_CLEAR"
        ],

        interpretation=
            "context clear",

        mutation_applied=False,
    )


def _decide(
    *,
    context=None,
    external=None,
    semantic=None,
):
    return (
        SERSCombinedLifecycleActionPolicy()
        .decide(
            target_portfolio_id=
                PORTFOLIO,

            target_hypothesis_id=
                HYPOTHESIS,

            context_decision=(
                context
                or _context_decision()
            ),

            external_findings=
                list(external or []),

            semantic_findings=
                list(semantic or []),
        )
    )


def test_context_reframe_and_novelty_pressure_are_combined():
    external = [
        _finding(
            ref_id="external:aggregate",
            source_kind="external_novelty",
            authority="actionable",
            status="LITERATURE_SUPPORTED_EXTENSION",
        ),
        _finding(
            ref_id="external:claim1",
            source_kind="external_novelty",
            authority="actionable",
            status="PARTIAL_PRIOR_ART",
            scope=G1FindingScope(
                kind="external_novelty_claim",
                hypothesis_ids=[
                    HYPOTHESIS
                ],
                assertion_ids=[
                    "claim:1"
                ],
            ),
        ),
    ]

    semantic = [
        _finding(
            ref_id="semantic:warning",
            source_kind="semantic_review",
            authority="advisory",
            status="warning",
        )
    ]

    decision = _decide(
        external=external,
        semantic=semantic,
    )

    assert (
        decision.disposition
        == "repair_required"
    )

    actions = [
        row.action
        for row in decision.directives
    ]

    assert actions.count(
        "reframe"
    ) == 1

    assert actions.count(
        "downgrade"
    ) == 1

    downgrade = next(
        row
        for row in decision.directives
        if row.action
        == "downgrade"
    )

    assert set(
        downgrade.finding_ref_ids
    ) == {
        "external:aggregate",
        "external:claim1",
    }


def test_external_actionables_fold_into_one_downgrade():
    external = [
        _finding(
            ref_id=f"external:{index}",
            source_kind="external_novelty",
            authority="actionable",
            status="PARTIAL_PRIOR_ART",
            scope=G1FindingScope(
                kind="external_novelty_claim",
                hypothesis_ids=[
                    HYPOTHESIS
                ],
                assertion_ids=[
                    f"claim:{index}"
                ],
            ),
        )
        for index in range(3)
    ]

    decision = _decide(
        external=external,
    )

    downgrades = [
        row
        for row in decision.directives
        if row.action
        == "downgrade"
    ]

    assert len(downgrades) == 1

    assert len(
        downgrades[0]
        .finding_ref_ids
    ) == 3

    assert (
        downgrades[0]
        .target_scope.kind
        == "hypothesis"
    )


def test_no_novelty_pressure_means_no_downgrade():
    external = [
        _finding(
            ref_id="external:info",
            source_kind="external_novelty",
            authority="informational",
            status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        )
    ]

    semantic = [
        _finding(
            ref_id="semantic:warning",
            source_kind="semantic_review",
            authority="advisory",
            status="warning",
        )
    ]

    decision = _decide(
        external=external,
        semantic=semantic,
    )

    assert (
        decision.disposition
        == "repair_required"
    )

    assert all(
        row.action
        != "downgrade"
        for row in decision.directives
    )


def test_advisory_only_combines_to_keep_with_warning():
    context = _context_decision(
        actionable=False,
    )

    semantic = [
        _finding(
            ref_id="semantic:warning",
            source_kind="semantic_review",
            authority="advisory",
            status="warning",
        )
    ]

    decision = _decide(
        context=context,
        semantic=semantic,
    )

    assert (
        decision.disposition
        == "keep_with_warning"
    )

    assert (
        decision.directives
        == []
    )


def test_all_informational_combines_to_keep():
    context = _context_decision(
        actionable=False,
    )

    external = [
        _finding(
            ref_id="external:info",
            source_kind="external_novelty",
            authority="informational",
            status="COMPONENTS_ONLY",
        )
    ]

    decision = _decide(
        context=context,
        external=external,
    )

    assert (
        decision.disposition
        == "keep"
    )

    assert (
        decision.directives
        == []
    )


def test_semantic_fail_is_nonterminal_reframe():
    context = _context_decision(
        actionable=False,
    )

    semantic = [
        _finding(
            ref_id="semantic:fail",
            source_kind="semantic_review",
            authority="actionable",
            status="fail",
        )
    ]

    decision = _decide(
        context=context,
        semantic=semantic,
    )

    assert (
        decision.disposition
        == "repair_required"
    )

    semantic_directives = [
        row
        for row in decision.directives
        if (
            "semantic:fail"
            in row.finding_ref_ids
        )
    ]

    assert len(
        semantic_directives
    ) == 1

    assert (
        semantic_directives[0]
        .action
        == "reframe"
    )


def test_terminal_candidate_is_not_silently_rejected():
    context = _context_decision(
        actionable=False,
    )

    external = [
        _finding(
            ref_id="external:terminal",
            source_kind="external_novelty",
            authority="terminal_candidate",
            status="DIRECT_PRIOR_ART",
        )
    ]

    with pytest.raises(
        SERSCombinedActionPolicyError,
        match="terminal adjudication",
    ):
        _decide(
            context=context,
            external=external,
        )


def test_duplicate_finding_id_across_lanes_is_rejected():
    duplicate = (
        "duplicate:finding"
    )

    context_finding = _finding(
        ref_id=duplicate,
        source_kind="context_review",
        authority="informational",
        status="match",
        scope=_local_scope(),
    )

    context = G1ActionDecision(
        decision_id=
            "decision:dup",

        target_portfolio_id=
            PORTFOLIO,

        target_hypothesis_id=
            HYPOTHESIS,

        findings=[
            context_finding
        ],

        directives=[],

        disposition="keep",

        reason_codes=[],

        interpretation=
            "duplicate test",

        mutation_applied=False,
    )

    external = [
        _finding(
            ref_id=duplicate,
            source_kind="external_novelty",
            authority="informational",
            status="COMPONENTS_ONLY",
        )
    ]

    with pytest.raises(
        SERSCombinedActionPolicyError,
        match="duplicate finding_ref_id",
    ):
        _decide(
            context=context,
            external=external,
        )
