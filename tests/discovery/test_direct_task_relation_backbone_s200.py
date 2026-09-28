from __future__ import annotations

from pipeline_core.discovery.direct_relationpattern_task_shadow import (
    DirectRelationPatternAssessment,
    candidate_from_relationpattern_mapping,
)
from pipeline_core.discovery.direct_task_relation_backbone import (
    materialize_direct_task_relation_backbone,
)


def _pattern_row(
    *,
    subject: str,
    relation: str,
    object_: str,
    concept_id: str = "bridge::b",
):
    phrase = (
        subject
        + " "
        + relation.lower().replace("_", " ")
        + " "
        + object_
    )

    return {
        "node_id": "paper::p::bridge::" + concept_id,
        "id": concept_id,
        "concept_type": "RelationPattern",
        "label": phrase,
        "source_phrase": phrase,
        "description": None,
        "retention_lane": "accepted_pattern",
        "evidence_scope": "paper_result",
        "pattern_subject": subject,
        "pattern_relation": relation,
        "pattern_object": object_,
        "relation_strength": "correlational",
        "qualifiers_json": "[]",
        "pattern_support_mode": "explicit_single_span",
        "supporting_phrases_json": '["' + phrase + '"]',
        "subject_evidence_phrase": subject,
        "relation_evidence_phrase": relation.lower().replace("_", " "),
        "object_evidence_phrase": object_,
        "comparison_items_json": "[]",
        "paper_id": "p",
        "chunk_id": "chunk:p",
        "document_id": "document:p",
    }


def _assessment(
    candidate_id: str,
    *,
    task_class: str = "DIRECT",
    stable: bool = True,
    status: str | None = "PASS",
    role: str | None = "DIRECT_ANSWER",
):
    pass_status = "PASS" if task_class == "DIRECT" else "FAIL"
    pass_role = (
        "DIRECT_ANSWER"
        if task_class == "DIRECT"
        else "TASK_REPLACEMENT"
    )

    return DirectRelationPatternAssessment(
        candidate_id=candidate_id,
        retrieval_rank=1,
        pass_1_status=pass_status,
        pass_2_status=pass_status,
        pass_1_role=pass_role,
        pass_2_role=pass_role,
        decision_stable=stable,
        stable_status=status,
        stable_role=role,
        task_class=task_class,
        task_responsive=(
            stable
            and task_class
            in {
                "DIRECT",
                "SUBORDINATE",
            }
        ),
    )


def test_strict_source_anchor_projects_opposite_target_role_without_equivalence():
    candidate = candidate_from_relationpattern_mapping(
        row=_pattern_row(
            subject="mode-intensity ratio",
            relation="CORRELATES_WITH",
            object_="molecular orientation",
        ),
        joint_query="molecular orientation Raman intensity",
        retrieval_rank=3,
        semantic_similarity=0.77,
    )

    backbone = materialize_direct_task_relation_backbone(
        candidate=candidate,
        assessment=_assessment(candidate.candidate_id),
        requested_source="molecular orientation",
        requested_target="Raman intensity",
    )

    assert backbone is not None
    assert (
        backbone.role_projection.projection_mode
        == "strict_source_anchor_opposite_slot"
    )
    assert backbone.role_projection.source_role_slot == "object"
    assert backbone.role_projection.target_role_slot == "subject"
    assert backbone.role_projection.target_role_text == "mode-intensity ratio"
    assert backbone.endpoint_equivalence_authorized is False
    assert backbone.scientific_identity_asserted is False


def test_distinct_lexical_slots_can_project_roles_only_after_direct_gate():
    candidate = candidate_from_relationpattern_mapping(
        row=_pattern_row(
            subject="localized surface plasmon resonance",
            relation="VARIES_WITH",
            object_="Ag-Au alloy metal composition",
        ),
        joint_query=(
            "Au/Ag composition ratio "
            "localized surface plasmon resonance wavelength"
        ),
        retrieval_rank=3,
        semantic_similarity=0.81,
    )

    backbone = materialize_direct_task_relation_backbone(
        candidate=candidate,
        assessment=_assessment(candidate.candidate_id),
        requested_source="Au/Ag composition ratio",
        requested_target="localized surface plasmon resonance wavelength",
    )

    assert backbone is not None
    assert (
        backbone.role_projection.projection_mode
        == "distinct_lexical_opposite_slots"
    )
    assert backbone.role_projection.source_role_slot == "object"
    assert backbone.role_projection.target_role_slot == "subject"
    assert (
        backbone.role_projection.endpoint_equivalence_authorized
        is False
    )


def test_task_replacing_relation_is_blocked_even_with_structural_signal():
    candidate = candidate_from_relationpattern_mapping(
        row=_pattern_row(
            subject="SERS enhancement",
            relation="VARIES_WITH",
            object_="interparticle spacing",
        ),
        joint_query=(
            "interparticle spacing disorder "
            "spatial SERS uniformity"
        ),
        retrieval_rank=8,
        semantic_similarity=0.75,
    )

    assessment = _assessment(
        candidate.candidate_id,
        task_class="TASK_REPLACING",
        stable=True,
        status="FAIL",
        role="TASK_REPLACEMENT",
    )

    backbone = materialize_direct_task_relation_backbone(
        candidate=candidate,
        assessment=assessment,
        requested_source="interparticle spacing disorder",
        requested_target="spatial SERS uniformity",
    )

    assert backbone is None


def test_subordinate_is_not_direct_backbone_authority_in_v1():
    candidate = candidate_from_relationpattern_mapping(
        row=_pattern_row(
            subject="surface chemistry",
            relation="MODULATES",
            object_="adsorption behavior",
        ),
        joint_query="surface functionalization adsorption selectivity",
        retrieval_rank=2,
        semantic_similarity=0.74,
    )

    assessment = DirectRelationPatternAssessment(
        candidate_id=candidate.candidate_id,
        retrieval_rank=2,
        pass_1_status="WARNING",
        pass_2_status="WARNING",
        pass_1_role="SUBORDINATE_EXTENSION",
        pass_2_role="SUBORDINATE_EXTENSION",
        decision_stable=True,
        stable_status="WARNING",
        stable_role="SUBORDINATE_EXTENSION",
        task_class="SUBORDINATE",
        task_responsive=True,
    )

    assert (
        materialize_direct_task_relation_backbone(
            candidate=candidate,
            assessment=assessment,
            requested_source="surface functionalization",
            requested_target="adsorption selectivity",
        )
        is None
    )


def test_direct_backbone_remains_shadow_only_and_non_novelty_authoritative():
    candidate = candidate_from_relationpattern_mapping(
        row=_pattern_row(
            subject="hotspot intensity",
            relation="VARIES_WITH",
            object_="light polarization",
        ),
        joint_query="incident light polarization hotspot intensity",
        retrieval_rank=1,
        semantic_similarity=0.79,
    )

    backbone = materialize_direct_task_relation_backbone(
        candidate=candidate,
        assessment=_assessment(candidate.candidate_id),
        requested_source="incident light polarization",
        requested_target="hotspot intensity",
    )

    assert backbone is not None
    assert backbone.shadow_only is True
    assert backbone.production_selection_authority is False
    assert backbone.novelty_authority_created is False
    assert backbone.positive_premise_authority_created is False
