from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass

import networkx as nx

from dac_her.reproducibility_domain import (
    ReproducibilityDomainAdapter,
    ReproducibilityEvidence,
)


_ALLOWED_SOURCE_TYPES = frozenset({
    "Measurement",
    "MeasurementGroup",
    "Experiment",
})


def stable_reproducibility_id(
    *,
    paper_id: str,
    evidence_kind: str,
    primary_source_node_id: str,
) -> str:
    payload = "|".join((paper_id, evidence_kind, primary_source_node_id))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"repro:{digest}"


def stable_reproducibility_result_id(
    *,
    paper_id: str,
    evidence_kind: str,
    source_mention_node_ids: tuple[str, ...],
) -> str:
    mentions = tuple(sorted(set(map(str, source_mention_node_ids))))
    if not mentions:
        raise ValueError(
            "stable reproducibility result IDs require source mention nodes."
        )
    if len(mentions) == 1:
        return stable_reproducibility_id(
            paper_id=paper_id,
            evidence_kind=evidence_kind,
            primary_source_node_id=mentions[0],
        )
    payload = "|".join((paper_id, evidence_kind, *mentions))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"repro-result:{digest}"


@dataclass(frozen=True)
class ReproducibilityDuplicateCandidate:
    left_evidence_id: str
    right_evidence_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ReproducibilityAudit:
    domain_profile_id: str
    reproducibility_semantics_id: str
    evidence_count: int
    quantitative_evidence_count: int
    evidence_kind_counts: dict[str, int]
    scope_counts: dict[str, int]
    source_mention_count: int
    consolidated_result_count: int
    possible_duplicate_result_pair_count: int
    possible_duplicate_result_cluster_count: int
    possible_duplicate_results: tuple[ReproducibilityDuplicateCandidate, ...]
    issues: tuple[str, ...]
    structural_gate: bool

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["possible_duplicate_results"] = [
            item.to_dict() for item in self.possible_duplicate_results
        ]
        row["issues"] = list(self.issues)
        return row



def _relation(attrs: dict[str, object]) -> str:
    return str(attrs.get("relation", "")).strip()


def _outgoing_ids(
    graph: nx.Graph,
    node_id: str,
    relations: frozenset[str],
) -> set[str]:
    if node_id not in graph or not graph.is_directed():
        return set()
    if graph.is_multigraph():
        iterator = graph.out_edges(node_id, keys=True, data=True)
        return {
            str(right)
            for _left, right, _key, attrs in iterator
            if _relation(dict(attrs)) in relations
        }
    return {
        str(right)
        for _left, right, attrs in graph.out_edges(node_id, data=True)
        if _relation(dict(attrs)) in relations
    }


def _incoming_ids(
    graph: nx.Graph,
    node_id: str,
    relations: frozenset[str],
) -> set[str]:
    if node_id not in graph or not graph.is_directed():
        return set()
    if graph.is_multigraph():
        iterator = graph.in_edges(node_id, keys=True, data=True)
        return {
            str(left)
            for left, _right, _key, attrs in iterator
            if _relation(dict(attrs)) in relations
        }
    return {
        str(left)
        for left, _right, attrs in graph.in_edges(node_id, data=True)
        if _relation(dict(attrs)) in relations
    }


def _evidence_subject_ids(
    graph: nx.Graph,
    item: ReproducibilityEvidence,
) -> set[str]:
    subjects: set[str] = set()
    for measurement_id in item.source_measurement_ids:
        if measurement_id not in graph:
            continue
        subject_id = str(
            graph.nodes[measurement_id].get("subject_id", "")
        ).strip()
        if subject_id:
            subjects.add(subject_id)
        subjects.update(
            _outgoing_ids(
                graph,
                measurement_id,
                frozenset({"MEASURED_FOR"}),
            )
        )
    for experiment_id in item.source_experiment_ids:
        subjects.update(
            _incoming_ids(
                graph,
                experiment_id,
                frozenset({"TESTED_IN", "CHARACTERIZED_IN"}),
            )
        )
    return subjects


def _evidence_context_anchor_ids(
    graph: nx.Graph,
    item: ReproducibilityEvidence,
) -> set[str]:
    anchors: set[str] = set()
    for experiment_id in item.source_experiment_ids:
        anchors.update(
            _outgoing_ids(
                graph,
                experiment_id,
                frozenset({"USES_ANALYTE", "USES_REPORTER"}),
            )
        )
    return anchors


def _same_numeric_result(
    left: ReproducibilityEvidence,
    right: ReproducibilityEvidence,
) -> bool:
    if left.value_numeric is None or right.value_numeric is None:
        return False
    return (
        left.value_numeric == right.value_numeric
        and left.unit.strip().lower() == right.unit.strip().lower()
    )


def _same_lineage(
    left: ReproducibilityEvidence,
    right: ReproducibilityEvidence,
) -> bool:
    return bool(
        set(left.source_measurement_group_ids)
        & set(right.source_measurement_group_ids)
    ) or bool(
        set(left.source_experiment_ids)
        & set(right.source_experiment_ids)
    )


def _possible_duplicate_candidate(
    graph: nx.Graph,
    left: ReproducibilityEvidence,
    right: ReproducibilityEvidence,
) -> str | None:
    if left.paper_id != right.paper_id:
        return None
    if left.reproducibility_scope == "unknown":
        return None
    if left.reproducibility_scope != right.reproducibility_scope:
        return None

    compatible_kinds = {left.evidence_kind, right.evidence_kind}
    if not (
        left.evidence_kind == right.evidence_kind
        or compatible_kinds
        <= {"relative_standard_deviation", "repeatability_statement"}
    ):
        return None

    left_subjects = _evidence_subject_ids(graph, left)
    right_subjects = _evidence_subject_ids(graph, right)
    if not left_subjects or not right_subjects:
        return None
    if not (left_subjects & right_subjects):
        return None

    left_anchors = _evidence_context_anchor_ids(graph, left)
    right_anchors = _evidence_context_anchor_ids(graph, right)
    if left_anchors and right_anchors and not (left_anchors & right_anchors):
        return None

    if (
        left.evidence_kind == "relative_standard_deviation"
        and right.evidence_kind == "relative_standard_deviation"
    ):
        if not _same_numeric_result(left, right):
            return None
        if _same_lineage(left, right):
            return None
        return "same_quantitative_result_without_shared_lineage"

    if (
        "relative_standard_deviation" in compatible_kinds
        and "repeatability_statement" in compatible_kinds
    ):
        if _same_lineage(left, right):
            return None
        return "qualitative_and_quantitative_same_scope_result"

    return None


def _candidate_cluster_count(
    candidates: list[ReproducibilityDuplicateCandidate],
) -> int:
    if not candidates:
        return 0
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for candidate in candidates:
        union(candidate.left_evidence_id, candidate.right_evidence_id)
    return len({find(value) for value in parent})


def audit_reproducibility_evidence(
    *,
    evidence: list[ReproducibilityEvidence],
    source_graphs: dict[str, nx.Graph],
    adapter: ReproducibilityDomainAdapter,
) -> ReproducibilityAudit:
    issues: list[str] = []
    seen: set[str] = set()

    for item in evidence:
        if item.evidence_id in seen:
            issues.append(f"duplicate_evidence_id:{item.evidence_id}")
        seen.add(item.evidence_id)

        graph = source_graphs.get(item.paper_id)
        if graph is None:
            issues.append(f"missing_source_graph:{item.paper_id}")
            continue

        for node_id in item.source_node_ids:
            if node_id not in graph:
                issues.append(
                    f"missing_source_node:{item.evidence_id}:{node_id}"
                )
                continue
            node_type = str(graph.nodes[node_id].get("type", ""))
            if node_type not in _ALLOWED_SOURCE_TYPES:
                issues.append(
                    "unsupported_source_type:"
                    f"{item.evidence_id}:{node_id}:{node_type}"
                )

        for mention_id in item.source_mention_node_ids:
            if mention_id not in graph:
                issues.append(
                    f"missing_source_mention:{item.evidence_id}:{mention_id}"
                )

    duplicate_candidates: list[ReproducibilityDuplicateCandidate] = []
    for left_index, left in enumerate(evidence):
        graph = source_graphs.get(left.paper_id)
        if graph is None:
            continue
        for right in evidence[left_index + 1:]:
            if right.paper_id != left.paper_id:
                continue
            reason = _possible_duplicate_candidate(
                graph,
                left,
                right,
            )
            if reason is None:
                continue
            duplicate_candidates.append(
                ReproducibilityDuplicateCandidate(
                    left_evidence_id=left.evidence_id,
                    right_evidence_id=right.evidence_id,
                    reason=reason,
                )
            )

    duplicate_candidates.sort(
        key=lambda item: (
            item.left_evidence_id,
            item.right_evidence_id,
            item.reason,
        )
    )

    kinds = Counter(item.evidence_kind for item in evidence)
    scopes = Counter(item.reproducibility_scope for item in evidence)
    quantitative = sum(item.value_numeric is not None for item in evidence)
    mention_count = sum(
        len(item.source_mention_node_ids) for item in evidence
    )
    consolidated = sum(
        item.result_identity_status == "consolidated_exact"
        for item in evidence
    )
    return ReproducibilityAudit(
        domain_profile_id=adapter.domain_profile_id,
        reproducibility_semantics_id=adapter.semantics_id,
        evidence_count=len(evidence),
        quantitative_evidence_count=quantitative,
        evidence_kind_counts=dict(sorted(kinds.items())),
        scope_counts=dict(sorted(scopes.items())),
        source_mention_count=mention_count,
        consolidated_result_count=consolidated,
        possible_duplicate_result_pair_count=len(duplicate_candidates),
        possible_duplicate_result_cluster_count=_candidate_cluster_count(
            duplicate_candidates
        ),
        possible_duplicate_results=tuple(duplicate_candidates),
        issues=tuple(sorted(set(issues))),
        structural_gate=not issues,
    )
