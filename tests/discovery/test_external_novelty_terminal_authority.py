from types import SimpleNamespace

import pytest

from pipeline_core.discovery.external_novelty_terminal_authority import (
    ExternalNoveltyTerminalAuthorityError,
    ExternalNoveltyTerminalAuthorityResolver,
)
from pipeline_core.discovery.hypothesis_action_contracts import (
    G1FindingRef,
    G1FindingScope,
)


PORTFOLIO = "portfolio:final"
HYPOTHESIS = "hypothesis:final"


def _finding(
    *,
    ref_id="finding:external",
    authority="informational",
    status="COMPONENTS_ONLY",
    source_kind="external_novelty",
):
    scope = G1FindingScope(
        kind="hypothesis",
        hypothesis_ids=[
            HYPOTHESIS
        ],
    )

    return G1FindingRef(
        finding_ref_id=
            ref_id,

        source_kind=
            source_kind,

        source_artifact_id=
            "external:test",

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
            scope,

        target_portfolio_id=
            PORTFOLIO,

        target_hypothesis_id=
            HYPOTHESIS,

        target_scope=
            scope,

        rationale=
            "synthetic external finding",
    )


def _report(
    *,
    decision="kept_original",
    final_status=
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
):
    return SimpleNamespace(
        schema_version=
            "novelty-refinement-report-v2",

        report_id=
            "refinement:test",

        final_portfolio_id=
            PORTFOLIO,

        attempts=[
            SimpleNamespace(
                decision=
                    decision,

                final_hypothesis_id=
                    HYPOTHESIS,

                final_external_status=
                    final_status,
            )
        ],
    )


def _resolve(
    findings,
    *,
    report=None,
):
    return (
        ExternalNoveltyTerminalAuthorityResolver()
        .resolve(
            target_portfolio_id=
                PORTFOLIO,

            target_hypothesis_id=
                HYPOTHESIS,

            findings=
                list(findings),

            refinement_report=(
                report
                or _report()
            ),
        )
    )


def test_nonterminal_external_finding_passes_through():
    finding = _finding()

    bundle = _resolve([
        finding
    ])

    assert (
        bundle.original_terminal_count
        == 0
    )

    assert (
        bundle.resolved_terminal_count
        == 0
    )

    assert (
        bundle.findings[0]
        .finding_ref_id
        == finding.finding_ref_id
    )

    assert (
        bundle.findings[0]
        .authority
        == "informational"
    )


@pytest.mark.parametrize(
    "status",
    [
        "WELL_ESTABLISHED",
        "CONFLICTING_PRIOR_ART",
        "DIRECT_PRIOR_ART",
    ],
)
def test_stale_terminal_candidate_is_superseded_by_non_destructive_r6_final(
    status,
):
    finding = _finding(
        authority=
            "terminal_candidate",

        status=
            status,
    )

    bundle = _resolve([
        finding
    ])

    assert (
        bundle.original_terminal_count
        == 1
    )

    assert (
        bundle.resolved_terminal_count
        == 1
    )

    assert (
        bundle.reject_authorized
        is False
    )

    resolved = bundle.findings[0]

    assert (
        resolved.authority
        == "informational"
    )

    assert (
        resolved.finding_ref_id
        != finding.finding_ref_id
    )

    assert (
        resolved.source_status
        == status
    )

    assert (
        resolved.source_attributes[
            "original_authority"
        ]
        == "terminal_candidate"
    )

    assert (
        resolved.source_attributes[
            "terminal_resolution"
        ]
        ==
        "superseded_by_final_reassessment"
    )

    assert (
        resolved.source_attributes[
            "r6_final_external_status"
        ]
        ==
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
    )

    assert (
        resolved.source_attributes[
            "reject_authorized"
        ]
        == "false"
    )


@pytest.mark.parametrize(
    "status",
    [
        "WELL_ESTABLISHED",
        "CONFLICTING_PRIOR_ART",
    ],
)
def test_destructive_status_on_final_survivor_is_invariant_violation(
    status,
):
    with pytest.raises(
        ExternalNoveltyTerminalAuthorityError,
        match="invariant is violated",
    ):
        _resolve(
            [
                _finding(
                    authority=
                        "terminal_candidate",

                    status=
                        status,
                )
            ],
            report=_report(
                final_status=
                    status
            ),
        )


def test_accepted_refinement_requires_fresh_final_novelty():
    with pytest.raises(
        ExternalNoveltyTerminalAuthorityError,
        match="fresh final novelty assessment",
    ):
        _resolve(
            [
                _finding(
                    authority=
                        "terminal_candidate",

                    status=
                        "WELL_ESTABLISHED",
                )
            ],
            report=_report(
                decision=
                    "accepted_refinement"
            ),
        )


def test_missing_final_survivor_is_rejected():
    report = _report()

    report.attempts[0].final_hypothesis_id = (
        "hypothesis:other"
    )

    with pytest.raises(
        ExternalNoveltyTerminalAuthorityError,
        match="exactly one R6 surviving attempt",
    ):
        _resolve(
            [
                _finding()
            ],
            report=report,
        )


def test_non_external_finding_is_rejected():
    with pytest.raises(
        ExternalNoveltyTerminalAuthorityError,
        match="external_novelty",
    ):
        _resolve([
            _finding(
                source_kind=
                    "semantic_review"
            )
        ])
