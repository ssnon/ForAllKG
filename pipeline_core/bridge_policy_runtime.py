from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from pipeline_core.bridge_schemas import (
    BridgeChunkGraph,
    BridgeConcept,
    BridgeLink,
)


@dataclass(frozen=True)
class PluginBridgePolicyIssue:
    code: str
    field: str
    detail: str
    repairable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PluginBridgeRejection:
    paper_id: str
    chunk_id: str
    concept_id: str
    label: str
    retention_lane: str
    pattern_subject: str
    pattern_relation: str
    pattern_object: str
    pattern_support_mode: str
    subject_evidence_phrase: str
    relation_evidence_phrase: str
    object_evidence_phrase: str
    source_phrase: str
    reason_codes: tuple[str, ...]
    reason_details: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row['reason_codes'] = list(self.reason_codes)
        row['reason_details'] = list(self.reason_details)
        return row


@dataclass(frozen=True)
class PluginBridgePolicyPartition:
    accepted: BridgeChunkGraph
    candidates: BridgeChunkGraph
    candidate_records: tuple[PluginBridgeRejection, ...]
    rejections: tuple[PluginBridgeRejection, ...]


PluginIssueBuilder = Callable[..., list[PluginBridgePolicyIssue]]
PluginNormalizer = Callable[[Any], str]


def dedupe_policy_issues(
    issues: Iterable[PluginBridgePolicyIssue],
) -> list[PluginBridgePolicyIssue]:
    rows: list[PluginBridgePolicyIssue] = []
    seen: set[tuple[str, str, str, bool]] = set()
    for issue in issues:
        signature = (
            issue.code,
            issue.field,
            issue.detail,
            issue.repairable,
        )
        if signature in seen:
            continue
        seen.add(signature)
        rows.append(issue)
    return rows


def partition_with_policy(
    result: BridgeChunkGraph,
    *,
    strict_nodes: list[dict[str, Any]],
    core_text: str | None,
    issue_builder: PluginIssueBuilder,
    normalizer: PluginNormalizer,
    candidate_only_codes: frozenset[str],
) -> PluginBridgePolicyPartition:
    """Generic accepted/candidate/rejected lane materialization.

    Scientific policy remains domain-owned in ``issue_builder``. This function
    only handles deterministic deduplication and lane partitioning.
    """
    accepted_ids: set[str] = set()
    candidate_ids: set[str] = set()
    accepted_concepts: list[BridgeConcept] = []
    candidate_concepts: list[BridgeConcept] = []
    candidate_records: list[PluginBridgeRejection] = []
    rejections: list[PluginBridgeRejection] = []
    seen_signatures: set[tuple[str, ...]] = set()

    links_by_concept: dict[str, list[BridgeLink]] = {}
    for link in result.links:
        links_by_concept.setdefault(link.concept_id, []).append(link)

    for concept in result.concepts:
        issues = issue_builder(
            concept,
            strict_nodes=strict_nodes,
            core_text=core_text,
            linked_links=links_by_concept.get(concept.id, []),
        )

        signature = (
            concept.retention_lane,
            normalizer(concept.label),
            normalizer(concept.pattern_subject or ''),
            str(concept.pattern_relation or ''),
            normalizer(concept.pattern_object or ''),
        )
        if signature in seen_signatures:
            issues.append(
                PluginBridgePolicyIssue(
                    code='DUPLICATE_MENTION',
                    field='label',
                    detail=(
                        'An equivalent Bridge mention already appeared in '
                        'this chunk.'
                    ),
                )
            )
        seen_signatures.add(signature)
        issues = dedupe_policy_issues(issues)

        if not issues:
            accepted_concepts.append(concept)
            accepted_ids.add(concept.id)
            continue

        reason_codes = tuple(dict.fromkeys(issue.code for issue in issues))
        record = PluginBridgeRejection(
            paper_id=result.paper_id,
            chunk_id=result.chunk_id,
            concept_id=concept.id,
            label=concept.label,
            retention_lane=concept.retention_lane,
            pattern_subject=concept.pattern_subject or '',
            pattern_relation=concept.pattern_relation or '',
            pattern_object=concept.pattern_object or '',
            pattern_support_mode=concept.pattern_support_mode or '',
            subject_evidence_phrase=concept.subject_evidence_phrase or '',
            relation_evidence_phrase=concept.relation_evidence_phrase or '',
            object_evidence_phrase=concept.object_evidence_phrase or '',
            source_phrase=concept.source_phrase,
            reason_codes=reason_codes,
            reason_details=tuple(issue.to_dict() for issue in issues),
        )

        if set(reason_codes).issubset(candidate_only_codes):
            candidate_concepts.append(concept)
            candidate_ids.add(concept.id)
            candidate_records.append(record)
        else:
            rejections.append(record)

    accepted = result.model_copy(
        update={
            'concepts': accepted_concepts,
            'links': [
                link for link in result.links if link.concept_id in accepted_ids
            ],
        }
    )
    candidates = result.model_copy(
        update={
            'concepts': candidate_concepts,
            'links': [
                link for link in result.links if link.concept_id in candidate_ids
            ],
        }
    )

    return PluginBridgePolicyPartition(
        accepted=BridgeChunkGraph.model_validate(accepted.model_dump()),
        candidates=BridgeChunkGraph.model_validate(candidates.model_dump()),
        candidate_records=tuple(candidate_records),
        rejections=tuple(rejections),
    )
