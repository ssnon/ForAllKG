from __future__ import annotations

from domains.registry import get_domain_profile
from pipeline_core.discovery.explorer_validation import ExplorationReportValidator
from tests._hypothesis_v260_fixtures import make_context, make_packet_and_report


def test_context_builder_separates_positive_premises_and_gaps():
    context = make_context()
    index = {x.statement_id: x for x in context.evidence_statements}

    assert index["s:reported"].eligible_as_premise is True
    assert index["s:candidate"].eligible_as_premise is True
    assert index["s:candidate"].requires_verification is True
    assert "candidate_requires_verification" in index["s:candidate"].premise_restrictions

    assert index["s:gap"].eligible_as_premise is False
    assert index["s:gap"].eligible_as_gap is True
    assert "unresolved_not_positive_premise" in index["s:gap"].premise_restrictions

    assert index["s:k10"].eligible_as_premise is True
    assert "partial_paper_absence_not_allowed" in index["s:k10"].premise_restrictions
    assert context.partial_absence_blocked_paper_ids == ["Kiwook_10"]


def test_context_hash_is_deterministic():
    packet, report = make_packet_and_report()
    a = make_context()
    from pipeline_core.discovery.hypothesis_context import HypothesisContextBuilder

    b = HypothesisContextBuilder(
        validator=ExplorationReportValidator(
            semantics=get_domain_profile(
                packet.domain_profile_id
            ).discovery,
        ),
    ).build(packet, report)
    assert a.context_id == b.context_id
    assert a.context_sha256 == b.context_sha256

def test_scope_limit_synthesis_is_not_positive_premise():
    from pipeline_core.discovery.explorer_contracts import ExplorerStatement
    from pipeline_core.discovery.hypothesis_context import HypothesisContextBuilder

    packet, report = make_packet_and_report()

    scope_statement = ExplorerStatement(
        statement_id="s:scope_synthesis",
        text=(
            "The packet supports a coordination-to-adsorption connection, "
            "but not a charge-transfer-mediated explanation."
        ),
        epistemic_role="evidence_synthesis",
        claim_kind="scope_limit",
        support_node_ids=["n:reported"],
        paper_ids=["Kiwook_1"],
    )

    modified_report = report.model_copy(
        update={
            "statements": report.statements + [scope_statement]
        }
    )

    context = HypothesisContextBuilder(
        validator=ExplorationReportValidator(
            semantics=get_domain_profile(
                packet.domain_profile_id
            ).discovery,
        ),
    ).build(
        packet,
        modified_report,
    )

    index = {
        statement.statement_id: statement
        for statement in context.evidence_statements
    }

    row = index["s:scope_synthesis"]

    assert row.eligible_as_premise is False
    assert row.eligible_as_gap is False
    assert (
        "scope_limit_not_positive_premise"
        in row.premise_restrictions
    )