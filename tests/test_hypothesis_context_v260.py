from __future__ import annotations

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
    from dac_her.hypothesis_context import HypothesisContextBuilder

    b = HypothesisContextBuilder().build(packet, report)
    assert a.context_id == b.context_id
    assert a.context_sha256 == b.context_sha256
