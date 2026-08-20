from __future__ import annotations

from dac_her.explorer_compiler import ExplorationReportCompiler
from dac_her.explorer_contracts import GraphExplorerPacket
from dac_her.explorer_draft import ExplorationDraft
from dac_her.explorer_normalization import ExplorerDraftNormalizer
from dac_her.explorer_validation import ExplorationReportValidator
from pipeline_core.explorer_text_safety import contains_absence_language


def _packet(*, absence_allowed: bool) -> GraphExplorerPacket:
    return GraphExplorerPacket.model_validate(
        {
            "packet_id": "packet:absence",
            "packet_sha256": "sha-absence",
            "domain_profile_id": "dac_her",
            "task": {
                "task_id": "task:absence",
                "question": "test absence discipline",
                "traversal_mode": "mechanism",
                "objective": "map_evidence",
            },
            "corpus": {
                "corpus_id": "c",
                "projection_mode": "mechanism",
                "papers": [
                    {
                        "paper_id": "paper:partial",
                        "quality_status": "partial_acceptable",
                        "absence_claims_allowed": absence_allowed,
                    }
                ],
                "substrate_version": "test",
            },
            "retrieval_summary": {"algorithm": "top_n"},
            "direct_concept_hits": [],
            "paths": [],
            "evidence_catalog": {
                "nodes": {
                    "node:reported": {
                        "node_id": "node:reported",
                        "node_type": "Observation",
                        "label": "reported HER observation",
                        "node_text": "The paper reports a HER observation.",
                        "source_paper_id": "paper:partial",
                        "source_paper_ids": ["paper:partial"],
                        "absence_claims_allowed": absence_allowed,
                    }
                },
                "edges": {},
            },
            "provenance_summary": {
                "strict_provenance": True,
                "edge_count": 0,
                "pointer_grounded_edge_count": 0,
                "pointer_recovered_from_traversal_count": 0,
                "derived_alignment_edge_count": 0,
                "missing_pointer_edge_count": 0,
                "materialized_node_count": 1,
                "suppressed_alignment_member_node_count": 0,
            },
        }
    )


def _draft() -> ExplorationDraft:
    return ExplorationDraft.model_validate(
        {
            "statements": [
                {
                    "local_id": "s_absent",
                    "text": "No evidence of the relation was reported in this paper.",
                    "epistemic_role": "reported",
                    "claim_kind": "scope_limit",
                    "support_node_ids": ["node:reported"],
                },
                {
                    "local_id": "s_keep",
                    "text": "The supplied evidence contains a reported HER observation.",
                    "epistemic_role": "reported",
                    "claim_kind": "observation",
                    "support_node_ids": ["node:reported"],
                },
            ],
            "direct_finding_local_ids": ["s_absent", "s_keep"],
            "reported_design_levers": [
                {
                    "local_id": "lever_drop",
                    "label": "absence-derived lever",
                    "statement_local_ids": ["s_absent"],
                }
            ],
        }
    )


def test_shared_absence_detector_matches_validator_policy_language():
    assert contains_absence_language("not reported")
    assert contains_absence_language("no evidence")
    assert contains_absence_language("not observed")
    assert not contains_absence_language("the packet is insufficient to determine whether it was reported")


def test_partial_paper_absence_statement_is_dropped_and_cascade_pruned():
    result = ExplorerDraftNormalizer().normalize(
        _packet(absence_allowed=False),
        _draft(),
    )
    assert [row.local_id for row in result.draft.statements] == ["s_keep"]
    assert result.draft.direct_finding_local_ids == ["s_keep"]
    assert result.draft.reported_design_levers == []
    action = next(
        row
        for row in result.audit.actions
        if row.action == "drop_unverifiable_paper_absence_statement"
    )
    assert "paper:partial" in action.reason
    assert result.audit.applied is True


def test_absence_allowed_paper_keeps_statement():
    result = ExplorerDraftNormalizer().normalize(
        _packet(absence_allowed=True),
        _draft(),
    )
    assert {row.local_id for row in result.draft.statements} == {"s_absent", "s_keep"}
    assert not any(
        row.action == "drop_unverifiable_paper_absence_statement"
        for row in result.audit.actions
    )


def test_nonabsence_statement_on_partial_paper_is_not_dropped():
    draft = ExplorationDraft.model_validate(
        {
            "statements": [
                {
                    "local_id": "s1",
                    "text": "The paper reports a HER observation.",
                    "epistemic_role": "reported",
                    "claim_kind": "observation",
                    "support_node_ids": ["node:reported"],
                }
            ]
        }
    )
    result = ExplorerDraftNormalizer().normalize(
        _packet(absence_allowed=False),
        draft,
    )
    assert [row.local_id for row in result.draft.statements] == ["s1"]


def test_normalized_draft_compiles_and_validates_without_absence_error():
    packet = _packet(absence_allowed=False)
    normalized = ExplorerDraftNormalizer().normalize(packet, _draft()).draft
    report = ExplorationReportCompiler().compile(packet, normalized)
    validation = ExplorationReportValidator().validate(packet, report)
    assert validation.passes is True, [
        (row.code, row.location, row.message)
        for row in validation.issues
    ]
    assert not any(
        row.code == "PAPER_ABSENCE_CLAIM_NOT_ALLOWED"
        for row in validation.issues
    )
