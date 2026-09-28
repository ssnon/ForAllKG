from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxis
from pipeline_core.discovery.question_axis_responsiveness import summarize_question_axis_two_pass
from pipeline_core.discovery.question_axis_responsiveness_prompt import QuestionAxisResponsivenessPromptAssembler
from pipeline_core.discovery.question_task_preservation_policy import classify_task_preservation
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentView,
    confirmed_known_component_from_mapping,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _sha256_json(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


class DirectRelationPatternCandidate(StrictModel):
    candidate_id: str
    joint_query: str
    retrieval_rank: int = Field(ge=1)
    semantic_similarity: float
    relation_component: RelationComponentView
    axis: DiscoveryAxis

    confirmed_known_relation_authority: bool = True
    retrieval_is_scientific_truth_authority: bool = False
    retrieval_is_task_responsiveness_authority: bool = False
    retrieval_is_novelty_authority: bool = False
    retrieval_is_selection_authority: bool = False


class DirectRelationPatternAssessment(StrictModel):
    candidate_id: str
    retrieval_rank: int
    pass_1_status: str
    pass_2_status: str
    pass_1_role: str
    pass_2_role: str
    decision_stable: bool
    stable_status: str | None = None
    stable_role: str | None = None
    task_class: str
    task_responsive: bool

    critic_is_scientific_truth_authority: bool = False
    critic_is_novelty_authority: bool = False
    critic_is_selection_authority: bool = False


class DirectRelationPatternTaskShadowReport(StrictModel):
    schema_version: str = "direct-relationpattern-task-shadow-v1"
    report_id: str
    report_sha256: str

    question: str
    retrieval_source: str
    retrieval_target: str
    joint_query: str

    retrieval_top_k: int = Field(ge=1)
    retrieved_node_count: int = Field(ge=0)
    accepted_relationpattern_count: int = Field(ge=0)
    validation_rejection_count: int = Field(ge=0)

    candidates: list[DirectRelationPatternCandidate] = Field(default_factory=list)
    assessments: list[DirectRelationPatternAssessment] = Field(default_factory=list)
    responsive_candidate_ids: list[str] = Field(default_factory=list)

    stable_direct_count: int = Field(ge=0)
    stable_subordinate_count: int = Field(ge=0)
    stable_task_replacing_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)

    shadow_only: bool = True
    production_selection_changed: bool = False
    canonical_graph_mutated: bool = False
    novelty_authority_created: bool = False
    positive_premise_authority_created: bool = False


class ResponsivenessBackendProtocol(Protocol):
    def review(
        self,
        prompt: Any,
        *,
        review_pass_index: int,
        debug_path: str | None = None,
    ) -> Any:
        ...


def relationpattern_mapping_from_graph_node(
    *,
    node_id: str,
    attrs: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(attrs)
    row["node_id"] = str(node_id)
    return row


def candidate_from_relationpattern_mapping(
    *,
    row: Mapping[str, Any],
    joint_query: str,
    retrieval_rank: int,
    semantic_similarity: float,
) -> DirectRelationPatternCandidate:
    component = confirmed_known_component_from_mapping(row)

    candidate_id = _stable_id(
        "direct_relationpattern_candidate",
        component.component_id,
        joint_query,
        retrieval_rank,
    )

    axis = DiscoveryAxis(
        axis_id=_stable_id(
            "direct_relationpattern_axis",
            component.component_id,
            joint_query,
        ),
        axis_rank=retrieval_rank,
        inspiration_id=component.accepted_pattern_id or "",
        source_path_id="",
        candidate_unit_id="",
        label=f"{component.subject} {component.relation} {component.object}",
        entry_anchor_id=component.accepted_pattern_id or "",
        entry_anchor_label=component.subject,
        exit_anchor_id=component.accepted_pattern_id or "",
        exit_anchor_label=component.object,
        proposed_subject=component.subject,
        proposed_relation=component.relation,
        proposed_object=component.object,
        rendered_path=f"{component.subject} --{component.relation}--> {component.object}",
        source_mode="direct_accepted_relationpattern_shadow",
        inspiration_role="KNOWN_RELATION_COMPONENT",
        external_relation_source_mode="NOT_APPLICABLE",
        second_order_gap_required=False,
        exploration_score=0.0,
        candidate_unit_score=0.0,
        planner_score=float(semantic_similarity),
        mechanistic_continuity_band="not_applicable",
        generic_entity_fraction=0.0,
        registry_hop_fraction=0.0,
        grounding_semantic_overlap=0.0,
        reaction_domain_switch_penalty=0.0,
        requires_verification=False,
        reason_codes=[
            "ACCEPTED_RELATIONPATTERN",
            "JOINT_SOURCE_TARGET_RETRIEVAL",
            "CONFIRMED_KNOWN_RELATION_COMPONENT",
            "SHADOW_ONLY",
        ],
    )

    return DirectRelationPatternCandidate(
        candidate_id=candidate_id,
        joint_query=joint_query,
        retrieval_rank=retrieval_rank,
        semantic_similarity=float(semantic_similarity),
        relation_component=component,
        axis=axis,
    )


def evaluate_candidates(
    *,
    question: str,
    candidates: Sequence[DirectRelationPatternCandidate],
    backend: ResponsivenessBackendProtocol,
    debug_dir: str | None = None,
) -> tuple[DirectRelationPatternAssessment, ...]:
    assembler = QuestionAxisResponsivenessPromptAssembler()
    assessments = []

    for index, candidate in enumerate(candidates, start=1):
        prompt = assembler.build(
            question=question,
            axis=candidate.axis,
        )

        debug_1 = (
            f"{debug_dir}/candidate_{index:02d}.pass1.json"
            if debug_dir
            else None
        )
        debug_2 = (
            f"{debug_dir}/candidate_{index:02d}.pass2.json"
            if debug_dir
            else None
        )

        generation_1 = backend.review(
            prompt,
            review_pass_index=1,
            debug_path=debug_1,
        )
        generation_2 = backend.review(
            prompt,
            review_pass_index=2,
            debug_path=debug_2,
        )

        stability = summarize_question_axis_two_pass(
            generation_1.draft,
            generation_2.draft,
        )

        preservation = classify_task_preservation(
            candidate_id=candidate.candidate_id,
            quality_eligible=True,
            stability=stability,
        )

        responsive = (
            preservation.decision_stable
            and preservation.task_class in {"DIRECT", "SUBORDINATE"}
        )

        assessments.append(
            DirectRelationPatternAssessment(
                candidate_id=candidate.candidate_id,
                retrieval_rank=candidate.retrieval_rank,
                pass_1_status=generation_1.draft.overall_status,
                pass_2_status=generation_2.draft.overall_status,
                pass_1_role=generation_1.draft.axis_role,
                pass_2_role=generation_2.draft.axis_role,
                decision_stable=stability.decision_stable,
                stable_status=stability.stable_status,
                stable_role=stability.stable_role,
                task_class=preservation.task_class,
                task_responsive=responsive,
            )
        )

    return tuple(assessments)


def build_report(
    *,
    question: str,
    retrieval_source: str,
    retrieval_target: str,
    joint_query: str,
    retrieval_top_k: int,
    retrieved_node_count: int,
    validation_rejection_count: int,
    candidates: Sequence[DirectRelationPatternCandidate],
    assessments: Sequence[DirectRelationPatternAssessment],
) -> DirectRelationPatternTaskShadowReport:
    by_id = {row.candidate_id: row for row in assessments}

    if len(by_id) != len(assessments):
        raise ValueError("duplicate direct RelationPattern assessment")

    for candidate in candidates:
        if candidate.candidate_id not in by_id:
            raise ValueError(
                "missing assessment for direct RelationPattern candidate "
                + candidate.candidate_id
            )

    responsive = [
        row.candidate_id
        for row in assessments
        if row.task_responsive
    ]

    stable_direct = sum(
        row.decision_stable and row.task_class == "DIRECT"
        for row in assessments
    )
    stable_subordinate = sum(
        row.decision_stable and row.task_class == "SUBORDINATE"
        for row in assessments
    )
    stable_replacing = sum(
        row.decision_stable and row.task_class == "TASK_REPLACING"
        for row in assessments
    )
    unresolved = len(assessments) - (
        stable_direct + stable_subordinate + stable_replacing
    )

    base = {
        "schema_version": "direct-relationpattern-task-shadow-v1",
        "question": question,
        "retrieval_source": retrieval_source,
        "retrieval_target": retrieval_target,
        "joint_query": joint_query,
        "retrieval_top_k": retrieval_top_k,
        "retrieved_node_count": retrieved_node_count,
        "accepted_relationpattern_count": len(candidates),
        "validation_rejection_count": validation_rejection_count,
        "candidates": [
            row.model_dump(mode="json")
            for row in candidates
        ],
        "assessments": [
            row.model_dump(mode="json")
            for row in assessments
        ],
        "responsive_candidate_ids": responsive,
        "stable_direct_count": stable_direct,
        "stable_subordinate_count": stable_subordinate,
        "stable_task_replacing_count": stable_replacing,
        "unresolved_count": unresolved,
        "shadow_only": True,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
        "novelty_authority_created": False,
        "positive_premise_authority_created": False,
    }

    report_id = _stable_id(
        "direct_relationpattern_task_shadow",
        question,
        retrieval_source,
        retrieval_target,
        joint_query,
        *responsive,
    )

    payload_for_hash = {
        **base,
        "report_id": report_id,
    }

    return DirectRelationPatternTaskShadowReport(
        **payload_for_hash,
        report_sha256=_sha256_json(payload_for_hash),
    )


__all__ = [
    "DirectRelationPatternCandidate",
    "DirectRelationPatternAssessment",
    "DirectRelationPatternTaskShadowReport",
    "relationpattern_mapping_from_graph_node",
    "candidate_from_relationpattern_mapping",
    "evaluate_candidates",
    "build_report",
]
