from __future__ import annotations

from dataclasses import dataclass

from pipeline_core.discovery.direct_relationpattern_task_shadow import (
    build_report,
    candidate_from_relationpattern_mapping,
    evaluate_candidates,
)
from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisResponsivenessDraft,
)


def _pattern_row():
    phrase = (
        "mode-intensity ratio correlates "
        "with molecular orientation"
    )
    return {
        "node_id": "paper::p::bridge::b",
        "id": "bridge::b",
        "concept_type": "RelationPattern",
        "label": phrase,
        "source_phrase": phrase,
        "description": None,
        "retention_lane": "accepted_pattern",
        "evidence_scope": "paper_result",
        "pattern_subject": "mode-intensity ratio",
        "pattern_relation": "CORRELATES_WITH",
        "pattern_object": "molecular orientation",
        "relation_strength": "correlational",
        "qualifiers_json": "[]",
        "pattern_support_mode": "explicit_single_span",
        "supporting_phrases_json": '["' + phrase + '"]',
        "subject_evidence_phrase": "mode-intensity ratio",
        "relation_evidence_phrase": "correlates with",
        "object_evidence_phrase": "molecular orientation",
        "comparison_items_json": "[]",
        "paper_id": "p",
        "chunk_id": "chunk:p",
        "document_id": "document:p",
    }


@dataclass
class _Generation:
    draft: QuestionAxisResponsivenessDraft


class _Backend:
    def __init__(self, drafts):
        self._drafts = list(drafts)

    def review(
        self,
        prompt,
        *,
        review_pass_index,
        debug_path=None,
    ):
        return _Generation(
            draft=self._drafts.pop(0)
        )


def _direct_draft():
    return QuestionAxisResponsivenessDraft(
        requested_variable_preservation="YES",
        requested_outcome_preservation="YES",
        relation_nucleus_preservation="YES",
        axis_role="DIRECT_ANSWER",
        overall_status="PASS",
        rationale="Directly addresses the requested relation.",
    )


def _replacement_draft():
    return QuestionAxisResponsivenessDraft(
        requested_variable_preservation="PARTIAL",
        requested_outcome_preservation="NO",
        relation_nucleus_preservation="NO",
        axis_role="TASK_REPLACEMENT",
        overall_status="FAIL",
        rationale="Replaces the requested primary relation.",
    )


def test_accepted_pattern_becomes_confirmed_known_shadow_candidate():
    candidate = candidate_from_relationpattern_mapping(
        row=_pattern_row(),
        joint_query="molecular orientation Raman intensity",
        retrieval_rank=3,
        semantic_similarity=0.77,
    )

    assert candidate.relation_component.authority.value == "confirmed_known"
    assert candidate.axis.inspiration_role == "KNOWN_RELATION_COMPONENT"
    assert (
        candidate.axis.source_mode
        == "direct_accepted_relationpattern_shadow"
    )
    assert candidate.axis.requires_verification is False
    assert candidate.retrieval_is_selection_authority is False


def test_two_pass_direct_becomes_shadow_responsive():
    candidate = candidate_from_relationpattern_mapping(
        row=_pattern_row(),
        joint_query="molecular orientation Raman intensity",
        retrieval_rank=3,
        semantic_similarity=0.77,
    )

    assessments = evaluate_candidates(
        question=(
            "How does molecular orientation alter "
            "Raman mode intensity?"
        ),
        candidates=[candidate],
        backend=_Backend(
            [_direct_draft(), _direct_draft()]
        ),
    )

    row = assessments[0]
    assert row.decision_stable is True
    assert row.task_class == "DIRECT"
    assert row.task_responsive is True
    assert row.critic_is_selection_authority is False


def test_task_replacement_fails_shadow_responsiveness():
    candidate = candidate_from_relationpattern_mapping(
        row=_pattern_row(),
        joint_query="molecular orientation Raman intensity",
        retrieval_rank=3,
        semantic_similarity=0.77,
    )

    assessments = evaluate_candidates(
        question=(
            "How does nanogap disorder change "
            "calibration transferability?"
        ),
        candidates=[candidate],
        backend=_Backend(
            [_replacement_draft(), _replacement_draft()]
        ),
    )

    row = assessments[0]
    assert row.decision_stable is True
    assert row.task_class == "TASK_REPLACING"
    assert row.task_responsive is False


def test_report_has_no_production_or_novelty_authority():
    candidate = candidate_from_relationpattern_mapping(
        row=_pattern_row(),
        joint_query="molecular orientation Raman intensity",
        retrieval_rank=3,
        semantic_similarity=0.77,
    )

    assessments = evaluate_candidates(
        question=(
            "How does molecular orientation alter "
            "Raman mode intensity?"
        ),
        candidates=[candidate],
        backend=_Backend(
            [_direct_draft(), _direct_draft()]
        ),
    )

    report = build_report(
        question=(
            "How does molecular orientation alter "
            "Raman mode intensity?"
        ),
        retrieval_source="molecular orientation",
        retrieval_target="Raman intensity",
        joint_query="molecular orientation Raman intensity",
        retrieval_top_k=20,
        retrieved_node_count=20,
        validation_rejection_count=0,
        candidates=[candidate],
        assessments=assessments,
    )

    assert report.stable_direct_count == 1
    assert report.responsive_candidate_ids == [
        candidate.candidate_id
    ]
    assert report.shadow_only is True
    assert report.production_selection_changed is False
    assert report.novelty_authority_created is False
    assert report.positive_premise_authority_created is False
