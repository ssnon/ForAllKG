from __future__ import annotations

from dac_her.evidence_family_selection import (
    EvidenceFamilyHierarchy,
    FamilyHierarchyChild,
    FamilyHierarchyGroup,
    FamilyPremiseSelectionPolicy,
    render_family_hierarchy_guidance,
)


def _hierarchy() -> EvidenceFamilyHierarchy:
    return EvidenceFamilyHierarchy(
        hierarchy_id="hier:test",
        hierarchy_sha256="a" * 64,
        source_decomposition_report_id="decomp:test",
        source_decomposition_report_sha256="b" * 64,
        source_context_id="context:test",
        source_context_sha256="c" * 64,
        domain_profile_id="dac_her",
        groups=[
            FamilyHierarchyGroup(
                parent_statement_id="stmt:parent",
                child_statement_ids=[
                    "stmtfam:obs",
                    "stmtfam:mech",
                ],
                children=[
                    FamilyHierarchyChild(
                        child_statement_id="stmtfam:obs",
                        family_id="family:obs",
                        paper_ids=["P1", "P2"],
                        claim_kind="observation",
                        node_types=["CoordinationMotif"],
                        edge_relations=["HAS_MOTIF"],
                    ),
                    FamilyHierarchyChild(
                        child_statement_id="stmtfam:mech",
                        family_id="family:mech",
                        paper_ids=["P3"],
                        claim_kind="mechanism",
                        node_types=["MechanismClaim"],
                        edge_relations=[
                            "SUPPORTED_MECHANISM_INTERPRETATION"
                        ],
                    ),
                ],
            )
        ],
    )


def test_ec2c_guidance_is_specificity_based_not_child_forcing():
    text = render_family_hierarchy_guidance(_hierarchy())
    assert "MINIMALLY-SUFFICIENT PREMISE PRINCIPLE" in text
    assert "Prefer the most specific family child" in text
    assert "Use the broader parent synthesis" in text
    assert "Do NOT select a child merely to increase premise diversity" in text
    assert "stmt:parent" in text
    assert "stmtfam:obs" in text
    assert "stmtfam:mech" in text


def test_ec2c_policy_does_not_force_child_or_forbid_parent():
    policy = FamilyPremiseSelectionPolicy()
    assert policy.child_use_forced is False
    assert policy.parent_use_forbidden is False
    assert policy.prefer_specific_sufficient_child is True
    assert policy.parent_allowed_for_cross_family_synthesis is True
    assert policy.avoid_parent_plus_all_children_redundancy is True
