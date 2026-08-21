from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx

from pipeline_core.corpus.extraction.chemistry_signatures import (
    METAL_NAMES,
    metal_signature,
)
from pipeline_core.domain.domain_profile import ScientificDomainProfile

_ACTIVE_DOMAIN_PROFILE: ContextVar[ScientificDomainProfile | None] = ContextVar(
    "resolution_domain_profile",
    default=None,
)

CLAIM_NODE_TYPES: frozenset[str] = frozenset(
    {"ObservationClaim", "MechanismClaim"}
)


def _active_domain_profile() -> ScientificDomainProfile:
    profile = _ACTIVE_DOMAIN_PROFILE.get()

    if profile is None:
        raise RuntimeError(
            "Resolution domain policy is not active. "
            "Supply domain_profile at the public resolution boundary."
        )

    return profile

_ELEMENT_NAMES = {
    name: symbol.lower()
    for name, symbol in METAL_NAMES.items()
}

_STATE_RE = re.compile(r"\b(?:[a-z]{1,2}\()?\d+h\)?\b", re.I)


def normalize_scientific_text(
    value: Any,
    *,
    domain_profile: ScientificDomainProfile | None = None,
) -> str:
    profile = (
        _active_domain_profile()
        if domain_profile is None
        else domain_profile
    )
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.lower().strip()
    for name, symbol in _ELEMENT_NAMES.items():
        text = re.sub(rf"\b{re.escape(name)}\b", symbol, text)
    replacements = dict(profile.resolution.text_replacements)
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9+.%\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_tokens(
    value: Any,
    *,
    domain_profile: ScientificDomainProfile | None = None,
) -> frozenset[str]:
    return frozenset(
        normalize_scientific_text(
            value,
            domain_profile=domain_profile,
        ).split()
    )


def _jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _stable_id(kind: str, left_id: str, right_id: str) -> str:
    left, right = sorted((str(left_id), str(right_id)))
    digest = hashlib.sha256(f"{kind}|{left}|{right}".encode()).hexdigest()[:20]
    return f"resolution:{kind}:{digest}"


def _node_label(graph: nx.Graph, node_id: str) -> str:
    data = graph.nodes[node_id]
    return str(data.get("label") or data.get("statement") or data.get("metric") or node_id)


def _node_type(graph: nx.Graph, node_id: str) -> str:
    return str(graph.nodes[node_id].get("type", ""))


def _component_index(graph: nx.Graph) -> dict[str, int]:
    components = (
        nx.weakly_connected_components(graph)
        if graph.is_directed() else nx.connected_components(graph)
    )
    ordered = sorted(components, key=lambda item: (-len(item), tuple(sorted(map(str, item)))))
    return {
        str(node_id): index
        for index, component in enumerate(ordered, start=1)
        for node_id in component
    }


def _incident(graph: nx.Graph, node_id: str):
    if graph.is_directed():
        for source, _, data in graph.in_edges(node_id, data=True):
            yield "in", str(source), data
        for _, target, data in graph.out_edges(node_id, data=True):
            yield "out", str(target), data
    else:
        for source, target, data in graph.edges(node_id, data=True):
            neighbor = target if str(source) == node_id else source
            yield "undirected", str(neighbor), data


def _chunk_ids(graph: nx.Graph, node_id: str) -> tuple[str, ...]:
    return tuple(sorted({
        str(data.get("chunk_id"))
        for _, _, data in _incident(graph, node_id)
        if data.get("chunk_id")
    }))

def _document_ids(
    graph: nx.Graph,
    node_id: str,
) -> tuple[str, ...]:
    return tuple(sorted({
        str(data.get("document_id"))
        for _, _, data in _incident(graph, node_id)
        if data.get("document_id")
    }))

def _neighborhood_signature(graph: nx.Graph, node_id: str) -> frozenset[str]:
    rows: set[str] = set()
    for direction, neighbor, data in _incident(graph, node_id):
        rows.add("|".join((
            direction,
            str(data.get("relation", "")),
            _node_type(graph, neighbor),
            normalize_scientific_text(_node_label(graph, neighbor)),
        )))
    return frozenset(rows)


def _metals(
    value: Any,
) -> frozenset[str]:
    return frozenset(
        metal_signature(value)
    )


def _state_tokens(value: Any) -> frozenset[str]:
    return frozenset(match.lower() for match in _STATE_RE.findall(str(value or "")))


def _reaction_signature(label: str) -> str:
    normalized = normalize_scientific_text(label)
    aliases = dict(
        _active_domain_profile().resolution.reaction_aliases
    )
    return aliases.get(normalized, normalized.replace(" ", "_"))


def _type_signature(graph: nx.Graph, node_id: str) -> tuple[Any, ...]:
    data = graph.nodes[node_id]
    node_type = _node_type(graph, node_id)
    label = _node_label(graph, node_id)
    if node_type == "Metal":
        metals = _metals(label)
        return (node_type, tuple(sorted(metals)) or (normalize_scientific_text(label),))
    if node_type == "Reaction":
        return (node_type, _reaction_signature(label))
    if node_type in {"Catalyst", "CatalystModel"}:
        tokens = normalized_tokens(label)
        profile = _active_domain_profile()
        nuclearity = profile.resolution.catalyst_nuclearity(tokens)
        support_tokens = tuple(sorted(
            tokens & profile.resolution.support_signature_tokens
        ))
        return (
            node_type,
            tuple(sorted(_metals(label))),
            nuclearity,
            support_tokens,
            tuple(sorted(_state_tokens(label))),
        )
    if node_type == "Measurement":
        return (
            node_type,
            str(data.get("metric_id") or normalize_scientific_text(data.get("metric"))),
            str(data.get("subject_id", "")),
            str(data.get("value_numeric", "")),
            normalize_scientific_text(data.get("value_text")),
            normalize_scientific_text(data.get("unit")),
            str(data.get("conditions_json", "[]")),
        )
    return (node_type, normalize_scientific_text(label))


def _hard_conflicts(graph: nx.Graph, left_id: str, right_id: str) -> tuple[str, ...]:
    left_type, right_type = _node_type(graph, left_id), _node_type(graph, right_id)
    conflicts: list[str] = []
    if left_type != right_type:
        conflicts.append("different node types")
        return tuple(conflicts)
    left_label, right_label = _node_label(graph, left_id), _node_label(graph, right_id)
    if left_type in {"Catalyst", "CatalystModel", "Metal"}:
        left_metals, right_metals = _metals(left_label), _metals(right_label)
        if left_metals and right_metals and left_metals != right_metals:
            conflicts.append("conflicting metal compositions")
    if left_type in {"CatalystModel", "Intermediate"}:
        left_states, right_states = _state_tokens(left_label), _state_tokens(right_label)
        if left_states and right_states and left_states != right_states:
            conflicts.append("conflicting adsorbate/coverage states")
    if left_type == "Measurement":
        left, right = graph.nodes[left_id], graph.nodes[right_id]
        if str(left.get("metric_id", "")) != str(right.get("metric_id", "")):
            conflicts.append("different metric IDs")
        if str(left.get("subject_id", "")) != str(right.get("subject_id", "")):
            conflicts.append("different measurement subjects")
        if str(left.get("value_numeric", "")) != str(right.get("value_numeric", "")):
            conflicts.append("different numeric values")
        if normalize_scientific_text(left.get("value_text")) != normalize_scientific_text(right.get("value_text")):
            conflicts.append("different textual values")
        if normalize_scientific_text(left.get("unit")) != normalize_scientific_text(right.get("unit")):
            conflicts.append("different units")
    return tuple(conflicts)


@dataclass(frozen=True)
class ResolutionCandidate:
    candidate_id: str
    candidate_kind: str
    left_id: str
    right_id: str
    node_type: str
    left_label: str
    right_label: str
    left_component: int
    right_component: int
    signature_equal: bool
    label_similarity: float
    token_jaccard: float
    neighborhood_similarity: float
    total_score: float
    recommendation: str
    merge_safety: str
    auto_approve: bool
    reasons: tuple[str, ...]
    conflicts: tuple[str, ...]
    left_degree: int
    right_degree: int
    left_chunk_ids: tuple[str, ...]
    right_chunk_ids: tuple[str, ...]
    left_documents: tuple[str, ...]
    right_documents: tuple[str, ...]
    cross_document: bool
    normalized_label_equal: bool
    review_priority: str

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        for key in ("reasons", "conflicts", "left_chunk_ids", "right_chunk_ids", "left_documents", "right_documents"):
            row[f"{key}_json"] = json.dumps(list(row.pop(key)), ensure_ascii=False)
        return row


@dataclass(frozen=True)
class CandidateSummary:
    total_candidates: int
    exact_entity_candidates: int
    fuzzy_cross_component_candidates: int
    fuzzy_intra_component_candidates: int
    measurement_duplicate_candidates: int
    auto_approved_candidates: int
    weak_components: int
    nodes: int
    edges: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate(
    graph: nx.Graph,
    *,
    left_id: str,
    right_id: str,
    kind: str,
    component_index: Mapping[str, int],
) -> ResolutionCandidate | None:
    conflicts = _hard_conflicts(graph, left_id, right_id)
    if conflicts:
        return None
    left_label, right_label = _node_label(graph, left_id), _node_label(graph, right_id)
    signature_equal = _type_signature(graph, left_id) == _type_signature(graph, right_id)
    normalized_left = normalize_scientific_text(
        left_label
    )
    normalized_right = normalize_scientific_text(
        right_label
    )

    normalized_label_equal = (
        normalized_left == normalized_right
    )

    left_documents = _document_ids(
        graph,
        left_id,
    )
    right_documents = _document_ids(
        graph,
        right_id,
    )

    cross_document = bool(
        left_documents
        and right_documents
        and set(left_documents) != set(right_documents)
    )
    label_similarity = SequenceMatcher(
        None, normalized_left, normalized_right
    ).ratio()
    token_jaccard = _jaccard(normalized_tokens(left_label), normalized_tokens(right_label))
    neighborhood_similarity = _jaccard(
        _neighborhood_signature(graph, left_id), _neighborhood_signature(graph, right_id)
    )
    total_score = min(1.0, (
        0.42 * label_similarity
        + 0.28 * token_jaccard
        + 0.20 * neighborhood_similarity
        + 0.10 * float(signature_equal)
    ))
    node_type = _node_type(graph, left_id)
    profile = _active_domain_profile()
    auto_approve = bool(
        node_type in profile.resolution.auto_merge_types
        and signature_equal
        and label_similarity >= 0.80
    )

    high_priority_review = bool(
        node_type in profile.resolution.high_priority_review_types
        and normalized_label_equal
        and signature_equal
        and cross_document
    )

    if auto_approve:
        recommendation = "same_entity"
        merge_safety = "safe_exact_registry_entity"
        review_priority = "automatic"

    elif high_priority_review:
        # 같은 entity일 가능성은 높지만,
        # Catalyst 계열은 자동 병합하지 않는다.
        recommendation = "same_entity"
        merge_safety = "review_required"
        review_priority = "high"

    else:
        recommendation = "needs_review"
        merge_safety = "review_required"
        review_priority = "normal"

    reasons = []
    if signature_equal:
        reasons.append("type-specific structured signatures match")
    if component_index[left_id] != component_index[right_id]:
        reasons.append("nodes occur in different weak components")
    if label_similarity >= 0.8:
        reasons.append("high normalized label similarity")
    if neighborhood_similarity >= 0.5:
        reasons.append("compatible graph neighborhoods")
    return ResolutionCandidate(
        candidate_id=_stable_id(kind, left_id, right_id),
        candidate_kind=kind,
        left_id=left_id,
        right_id=right_id,
        node_type=node_type,
        left_label=left_label,
        right_label=right_label,
        left_component=component_index[left_id],
        right_component=component_index[right_id],
        signature_equal=signature_equal,
        label_similarity=round(label_similarity, 6),
        token_jaccard=round(token_jaccard, 6),
        neighborhood_similarity=round(neighborhood_similarity, 6),
        total_score=round(total_score, 6),
        recommendation=recommendation,
        merge_safety=merge_safety,
        auto_approve=auto_approve,
        reasons=tuple(reasons),
        conflicts=conflicts,
        left_degree=int(graph.degree(left_id)),
        right_degree=int(graph.degree(right_id)),
        left_chunk_ids=_chunk_ids(graph, left_id),
        right_chunk_ids=_chunk_ids(graph, right_id),
        left_documents=left_documents,
        right_documents=right_documents,
        cross_document=cross_document,
        normalized_label_equal=normalized_label_equal,
        review_priority=review_priority,
    )


def generate_resolution_candidates(
    graph: nx.Graph,
    *,
    fuzzy_minimum_score: float = 0.72,
    measurement_minimum_score: float = 0.90,
    domain_profile: ScientificDomainProfile,
) -> tuple[list[ResolutionCandidate], CandidateSummary]:
    profile = domain_profile
    token = _ACTIVE_DOMAIN_PROFILE.set(profile)
    try:
        return _generate_resolution_candidates_impl(
            graph,
            fuzzy_minimum_score=fuzzy_minimum_score,
            measurement_minimum_score=measurement_minimum_score,
        )
    finally:
        _ACTIVE_DOMAIN_PROFILE.reset(token)


def _generate_resolution_candidates_impl(
    graph: nx.Graph,
    *,
    fuzzy_minimum_score: float = 0.72,
    measurement_minimum_score: float = 0.90,
) -> tuple[list[ResolutionCandidate], CandidateSummary]:
    profile = _active_domain_profile()
    components = _component_index(graph)
    nodes_by_type: dict[str, list[str]] = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        nodes_by_type[str(data.get("type", ""))].append(str(node_id))

    candidates: dict[str, ResolutionCandidate] = {}
    exact_count = fuzzy_cross = fuzzy_intra = measurement_count = 0

    for node_type, node_ids in nodes_by_type.items():
        if node_type in CLAIM_NODE_TYPES or node_type == "MeasurementGroup":
            continue
        if (
            node_type not in profile.resolution.resolvable_node_types
            and node_type != "Measurement"
        ):
            continue
        for index, left_id in enumerate(sorted(node_ids)):
            for right_id in sorted(node_ids)[index + 1:]:
                signature_equal = _type_signature(graph, left_id) == _type_signature(graph, right_id)
                kind = "measurement_duplicate" if node_type == "Measurement" else (
                    "exact_entity" if signature_equal else "fuzzy_entity"
                )
                candidate = _candidate(
                    graph,
                    left_id=left_id,
                    right_id=right_id,
                    kind=kind,
                    component_index=components,
                )
                if candidate is None:
                    continue
                threshold = measurement_minimum_score if node_type == "Measurement" else fuzzy_minimum_score
                if not signature_equal and candidate.total_score < threshold:
                    continue
                if node_type == "Measurement" and candidate.total_score < measurement_minimum_score:
                    continue
                candidates[candidate.candidate_id] = candidate
                if node_type == "Measurement":
                    measurement_count += 1
                elif signature_equal:
                    exact_count += 1
                elif candidate.left_component != candidate.right_component:
                    fuzzy_cross += 1
                else:
                    fuzzy_intra += 1

    priority_rank = {
        "automatic": 0,
        "high": 1,
        "normal": 2,
    }

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            priority_rank.get(
                item.review_priority,
                9,
            ),
            -item.total_score,
            item.node_type,
            item.left_id,
            item.right_id,
        ),
    )
    
    return ordered, CandidateSummary(
        total_candidates=len(ordered),
        exact_entity_candidates=exact_count,
        fuzzy_cross_component_candidates=fuzzy_cross,
        fuzzy_intra_component_candidates=fuzzy_intra,
        measurement_duplicate_candidates=measurement_count,
        auto_approved_candidates=sum(item.auto_approve for item in ordered),
        weak_components=len(set(components.values())),
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
    )


def write_candidates_csv(path: str | Path, candidates: Sequence[ResolutionCandidate]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [candidate.to_row() for candidate in candidates]
    fieldnames = list(rows[0]) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object: {path}")
            records.append(value)
    return records




def _is_automatic_decision_record(record: Mapping[str, Any]) -> bool:
    return (
        str(record.get("reviewer", "") or "").strip()
        == "automatic_registry_rule"
    )

def sync_decisions_jsonl(
    path: str | Path,
    candidates: Sequence[ResolutionCandidate],
) -> tuple[Path, dict[str, int]]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {str(item.get("candidate_id")): item for item in read_jsonl(path)}
    records: list[dict[str, Any]] = []
    preserved = created = auto_approved = refreshed_automatic = 0
    preserved_manual = 0
    current_candidate_ids = {candidate.candidate_id for candidate in candidates}
    for candidate in candidates:
        base = candidate.to_row()
        prior = existing.get(candidate.candidate_id)
        if prior is not None and not _is_automatic_decision_record(prior):
            # Preserve human/unreviewed decisions exactly.
            for key in (
                "decision", "approved", "canonical_id", "reviewer",
                "reviewed_at", "notes",
            ):
                if key in prior:
                    base[key] = prior[key]
            preserved += 1
        else:
            # Automatic pair decisions are regenerated from the current
            # candidate. canonical_id is intentionally null: pairwise choices
            # are not valid cluster-level representatives when candidates form
            # a transitive component.
            base.update({
                "decision": "same_entity" if candidate.auto_approve else "unreviewed",
                "approved": bool(candidate.auto_approve),
                "canonical_id": None,
                "reviewer": "automatic_registry_rule" if candidate.auto_approve else None,
                "reviewed_at": None,
                "notes": None,
            })
            if prior is None:
                created += 1
            else:
                refreshed_automatic += 1
            auto_approved += int(candidate.auto_approve)
        records.append(base)

    # Preserve explicit human decisions even when a future candidate generator
    # no longer emits the same pair. This allows reviewed transitive clusters
    # to remain stable across vocabulary and scoring changes. Missing graph
    # nodes are still treated as stale later by load_resolution_plan().
    for candidate_id, prior in sorted(existing.items()):
        if candidate_id in current_candidate_ids:
            continue
        if _is_automatic_decision_record(prior):
            continue
        decision = str(prior.get("decision", "unreviewed")).strip()
        reviewer = str(prior.get("reviewer", "") or "").strip()
        if not reviewer or decision == "unreviewed":
            continue
        records.append(prior)
        preserved_manual += 1

    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path, {
        "preserved": preserved,
        "created": created,
        "auto_approved": auto_approved,
        "refreshed_automatic": refreshed_automatic,
        "preserved_manual": preserved_manual,
        "removed_stale": max(
            0,
            len(existing)
            - preserved
            - refreshed_automatic
            - preserved_manual,
        ),
    }


def _duplicate_label_counts(graph: nx.Graph) -> dict[str, int]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for node_id, data in graph.nodes(data=True):
        groups[(str(data.get("type", "")), normalize_scientific_text(_node_label(graph, str(node_id))))].append(str(node_id))
    return {
        "groups": sum(len(items) > 1 for items in groups.values()),
        "measurement_groups": sum(key[0] == "Measurement" and len(items) > 1 for key, items in groups.items()),
    }


def _graph_summary(graph: nx.Graph) -> dict[str, Any]:
    components = list(nx.weakly_connected_components(graph)) if graph.is_directed() else list(nx.connected_components(graph))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len(components),
        "largest_component_sizes": sorted((len(item) for item in components), reverse=True)[:10],
        "duplicate_normalized_labels": _duplicate_label_counts(graph),
    }


def build_raw_canonical_report(
    *,
    raw_graph: nx.Graph,
    canonical_graph: nx.Graph,
    candidate_summary: Mapping[str, Any],
    resolution_summary: Mapping[str, Any],
    domain_profile: ScientificDomainProfile,
) -> dict[str, Any]:
    profile = domain_profile
    token = _ACTIVE_DOMAIN_PROFILE.set(profile)
    try:
        raw = _graph_summary(raw_graph)
        canonical = _graph_summary(canonical_graph)
    finally:
        _ACTIVE_DOMAIN_PROFILE.reset(token)
    return {
        "raw": raw,
        "canonical": canonical,
        "difference": {
            "nodes_merged_or_dropped": raw["nodes"] - canonical["nodes"],
            "edge_difference": raw["edges"] - canonical["edges"],
        },
        "candidate_summary": dict(candidate_summary),
        "resolution_summary": dict(resolution_summary),
    }


def format_raw_canonical_report(report: Mapping[str, Any]) -> str:
    raw, canonical = report["raw"], report["canonical"]
    return "\n".join([
        "Paper-level resolution report",
        f"Raw nodes/edges: {raw['nodes']}/{raw['edges']}",
        f"Canonical nodes/edges: {canonical['nodes']}/{canonical['edges']}",
        f"Components: {raw['components']} -> {canonical['components']}",
        f"Nodes merged/dropped: {report['difference']['nodes_merged_or_dropped']}",
        f"Resolution: {json.dumps(report['resolution_summary'], ensure_ascii=False)}",
    ]) + "\n"
