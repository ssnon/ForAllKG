from __future__ import annotations

from dac_her.evidence_family_decomposition import (
    _child_statement_id,
    _family_claim_kind,
    EvidenceFamilyDecompositionPolicy,
)
from dac_her.evidence_family_diagnostics import (
    EvidenceFamilyProfile,
)


def test_ec2b_family_claim_kind_distinguishes_observation_and_mechanism():
    observation = EvidenceFamilyProfile(
        family_id="F1",
        paper_ids=["P1", "P2"],
        paper_count=2,
        node_types=["CoordinationMotif"],
        edge_relations=["HAS_MOTIF"],
    )
    mechanism = EvidenceFamilyProfile(
        family_id="F2",
        paper_ids=["P3"],
        paper_count=1,
        node_types=[
            "MechanismClaim",
            "ObservationClaim",
        ],
        edge_relations=[
            "SUPPORTED_MECHANISM_INTERPRETATION",
            "SUPPORTED_OBSERVATION",
        ],
    )

    assert (
        _family_claim_kind(
            observation,
            parent_claim_kind="mechanism",
        )
        == "observation"
    )
    assert (
        _family_claim_kind(
            mechanism,
            parent_claim_kind="mechanism",
        )
        == "mechanism"
    )


def test_ec2b_child_ids_are_deterministic_and_family_specific():
    a1 = _child_statement_id(
        parent_statement_id="stmt:parent",
        family_id="family:A",
    )
    a2 = _child_statement_id(
        parent_statement_id="stmt:parent",
        family_id="family:A",
    )
    b = _child_statement_id(
        parent_statement_id="stmt:parent",
        family_id="family:B",
    )

    assert a1 == a2
    assert a1 != b
    assert a1.startswith("stmtfam:")


def test_ec2b_policy_is_additive_opt_in_and_parent_preserving():
    policy = EvidenceFamilyDecompositionPolicy()

    assert policy.opt_in_only is True
    assert policy.parent_statement_retained is True
    assert policy.parent_statement_modified is False
    assert policy.only_ec2a_candidates is True
    assert policy.require_evidence_synthesis_parent is True
    assert policy.external_evidence_added is False
    assert policy.scientific_support_invented is False
