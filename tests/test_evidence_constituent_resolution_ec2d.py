from __future__ import annotations

from dac_her.evidence_constituent_resolution import (
    ExistingConstituentResolutionPolicy,
    find_existing_constituent,
)
from dac_her.evidence_family_diagnostics import EvidenceFamilyProfile
from dac_her.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
)


def _statement(
    statement_id: str,
    *,
    claim_kind: str,
    papers: list[str],
    nodes: list[str],
    edges: list[str],
    eligible: bool = True,
) -> HypothesisEvidenceStatement:
    return HypothesisEvidenceStatement(
        statement_id=statement_id,
        text=statement_id,
        epistemic_role="reported",
        claim_kind=claim_kind,
        paper_ids=papers,
        scientific_support_node_ids=nodes,
        scientific_support_edge_ids=edges,
        support_path_ids=[],
        alignment_path_ids=[],
        eligible_as_premise=eligible,
    )


def _context(statements: list[HypothesisEvidenceStatement]) -> HypothesisContext:
    return HypothesisContext(
        context_id="context:test",
        context_sha256="a" * 64,
        source_packet_id="packet:test",
        source_packet_sha256="b" * 64,
        source_report_id="report:test",
        source_report_sha256="c" * 64,
        task_id="task:test",
        question="test",
        corpus_id="corpus:test",
        domain_profile_id="dac_her",
        evidence_statements=statements,
    )


def test_ec2d_reuses_smallest_full_support_container():
    parent = _statement(
        "stmt:parent",
        claim_kind="mechanism",
        papers=["P1", "P2", "P3"],
        nodes=["n1", "n2", "n3"],
        edges=["e1", "e2", "e3"],
    )
    broader = _statement(
        "stmt:broader",
        claim_kind="observation",
        papers=["P1", "P2", "P3"],
        nodes=["n1", "n2", "n3"],
        edges=["e1", "e2", "e3"],
    )
    tighter = _statement(
        "stmt:tighter",
        claim_kind="observation",
        papers=["P1", "P2"],
        nodes=["n1", "n2"],
        edges=["e1", "e2"],
    )

    family = EvidenceFamilyProfile(
        family_id="family:test",
        paper_ids=["P1", "P2"],
        paper_count=2,
        node_types=["CoordinationMotif"],
        edge_relations=["HAS_MOTIF"],
        direct_support_node_ids=["n1", "n2"],
        direct_support_edge_ids=["e1", "e2"],
    )

    best, candidates = find_existing_constituent(
        family=family,
        family_claim_kind="observation",
        context=_context([parent, broader, tighter]),
        parent_statement_id="stmt:parent",
    )

    assert best is not None
    assert best.statement_id == "stmt:tighter"
    assert best.exact_support_match is True
    assert [row.statement_id for row in candidates] == [
        "stmt:tighter",
        "stmt:broader",
    ]


def test_ec2d_requires_claim_kind_and_full_support_containment():
    parent = _statement(
        "stmt:parent",
        claim_kind="mechanism",
        papers=["P1", "P2"],
        nodes=["n1", "n2"],
        edges=["e1", "e2"],
    )
    wrong_kind = _statement(
        "stmt:wrong-kind",
        claim_kind="mechanism",
        papers=["P1", "P2"],
        nodes=["n1", "n2"],
        edges=["e1", "e2"],
    )
    partial = _statement(
        "stmt:partial",
        claim_kind="observation",
        papers=["P1", "P2"],
        nodes=["n1"],
        edges=["e1"],
    )

    family = EvidenceFamilyProfile(
        family_id="family:test",
        paper_ids=["P1", "P2"],
        paper_count=2,
        node_types=["CoordinationMotif"],
        edge_relations=["HAS_MOTIF"],
        direct_support_node_ids=["n1", "n2"],
        direct_support_edge_ids=["e1", "e2"],
    )

    best, candidates = find_existing_constituent(
        family=family,
        family_claim_kind="observation",
        context=_context([parent, wrong_kind, partial]),
        parent_statement_id="stmt:parent",
    )

    assert best is None
    assert candidates == []


def test_ec2d_policy_is_existing_first_and_conditional():
    policy = ExistingConstituentResolutionPolicy()

    assert policy.require_full_scientific_support_containment is True
    assert policy.generated_family_children_can_resolve_existing is False
    assert policy.unresolved_family_is_materialized is True
    assert policy.parent_statement_retained is True
    assert policy.parent_statement_modified is False
    assert policy.scientific_support_invented is False
