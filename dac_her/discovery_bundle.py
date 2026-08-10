from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx
import numpy as np

from dac_her.discovery_contracts import (
    DiscoveryBundle,
    DiscoveryInspiration,
    DiscoveryScoreBreakdown,
)


_ALIGNMENT_CLASSES = {"registry_alignment", "pattern_alignment"}
_ALIGNMENT_NODE_TYPES = {"CorpusAlignment", "CorpusPattern"}
_GENERIC_ENTITY_TYPES = {
    "CATALYST",
    "CATALYSTMODEL",
    "MATERIAL",
    "SUPPORT",
    "METAL",
    "REACTION",
}
_MECHANISM_NODE_MARKERS = ("MECHANISM", "MECHANISTIC")
_MECHANISM_RELATION_MARKERS = (
    "MECHANISM",
    "INTERPRETED_AS",
    "INFLUENC",
    "MODULAT",
    "FACILITAT",
    "PROMOT",
    "REGULAT",
    "CONTROL",
    "CORRELAT",
    "CAUSE",
    "ENABLE",
    "ENHANC",
    "LOWER",
    "STABIL",
    "TRANSFER",
    "SPILLOVER",
    "TUN",
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9Δδ*+./_-]+")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _sha256_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        return None
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return None
    return array / norm


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or left.shape != right.shape:
        return 0.0
    return _clip01(float(np.dot(left, right)))


def _edge_id(step: dict[str, Any], index: int) -> str:
    value = str(step.get("selected_original_edge_id") or step.get("navigation_edge_id") or "").strip()
    if value:
        return value
    return f"{step.get('source','')}|{step.get('relation','')}|{step.get('target','')}|{index}"


def _edge_set(row: dict[str, Any]) -> frozenset[str]:
    return frozenset(
        _edge_id(step, i)
        for i, step in enumerate(row.get("steps", []))
        if isinstance(step, dict)
    )


def _edge_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def _node_label(graph: nx.DiGraph, node_id: str) -> str:
    if node_id not in graph:
        return node_id
    attrs = graph.nodes[node_id]
    return str(
        attrs.get("label")
        or attrs.get("statement")
        or attrs.get("name")
        or node_id
    ).strip()


def _normalized_type(attrs: Mapping[str, Any]) -> str:
    return "".join(
        character
        for character in str(attrs.get("type", "")).upper()
        if character.isalnum()
    )


def _is_alignment_node(attrs: Mapping[str, Any]) -> bool:
    return (
        str(attrs.get("corpus_node_kind", "")) == "alignment_hub"
        or str(attrs.get("type", "")) in _ALIGNMENT_NODE_TYPES
        or str(attrs.get("graph_layer", "")) == "corpus_alignment"
    )


def _is_mechanism_node(node_id: str, attrs: Mapping[str, Any]) -> bool:
    node_type = _normalized_type(attrs)
    if any(marker in node_type for marker in _MECHANISM_NODE_MARKERS):
        return True
    local_id = str(node_id).split("::")[-1].strip().lower()
    return local_id.startswith("mech_")


def _is_mechanism_step(step: Mapping[str, Any]) -> bool:
    relation = str(step.get("relation", "")).strip().upper()
    return any(marker in relation for marker in _MECHANISM_RELATION_MARKERS)


def _is_alignment_step(step: Mapping[str, Any]) -> bool:
    return str(step.get("edge_class", "")) in _ALIGNMENT_CLASSES


def _is_generic_entity_node(node_id: str, attrs: Mapping[str, Any]) -> bool:
    if _is_alignment_node(attrs) or _is_mechanism_node(node_id, attrs):
        return False
    return _normalized_type(attrs) in _GENERIC_ENTITY_TYPES


def _scientific_subgraph(graph: nx.DiGraph) -> nx.Graph:
    keep_nodes = [
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if not _is_alignment_node(dict(attrs))
    ]
    sub = graph.subgraph(keep_nodes).copy()
    drop_edges: list[tuple[str, str]] = []
    for left, right, attrs in sub.edges(data=True):
        if str(attrs.get("edge_class", "")) in _ALIGNMENT_CLASSES:
            drop_edges.append((left, right))
    sub.remove_edges_from(drop_edges)
    return sub.to_undirected()


def _communities(graph: nx.DiGraph) -> dict[str, int]:
    scientific = _scientific_subgraph(graph)
    if scientific.number_of_nodes() == 0:
        return {}
    try:
        groups = nx.community.louvain_communities(scientific, seed=17)
    except (AttributeError, TypeError):
        groups = nx.community.greedy_modularity_communities(scientific)
    result: dict[str, int] = {}
    for index, group in enumerate(groups):
        for node_id in group:
            result[str(node_id)] = index
    next_id = len(groups)
    for node_id in scientific.nodes:
        key = str(node_id)
        if key not in result:
            result[key] = next_id
            next_id += 1
    return result


def _relation_counts(graph: nx.DiGraph) -> Counter[str]:
    counter: Counter[str] = Counter()
    for _, _, attrs in graph.edges(data=True):
        if str(attrs.get("edge_class", "")) in _ALIGNMENT_CLASSES:
            continue
        relation = str(attrs.get("relation", "RELATED_TO")).strip().upper() or "RELATED_TO"
        counter[relation] += 1
    return counter


def _path_relation_rarity(row: dict[str, Any], counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    values: list[float] = []
    denom = math.log(total + 1.0)
    for step in row.get("steps", []):
        if not isinstance(step, dict) or _is_alignment_step(step):
            continue
        relation = str(step.get("relation", "RELATED_TO")).strip().upper() or "RELATED_TO"
        freq = max(1, counts.get(relation, 1))
        values.append(math.log((total + 1.0) / freq) / denom)
    return _clip01(sum(values) / len(values)) if values else 0.0


def _community_span(row: dict[str, Any], community_by_node: dict[str, int]) -> float:
    communities = {
        community_by_node[str(node_id)]
        for node_id in row.get("nodes", [])
        if str(node_id) in community_by_node
    }
    if len(communities) <= 1:
        return 0.0
    return _clip01((len(communities) - 1) / 2.0)


def _mechanistic_score(quality: dict[str, Any]) -> float:
    band = str(quality.get("mechanistic_content", "low")).lower()
    band_score = {"high": 1.0, "medium": 0.65, "low": 0.0}.get(band, 0.0)
    density = max(
        float(quality.get("mechanistic_edge_density", 0.0) or 0.0),
        float(quality.get("mechanistic_node_density", 0.0) or 0.0),
    )
    return _clip01(max(band_score, density))


def _endpoint_score(quality: dict[str, Any], row: dict[str, Any]) -> float:
    raw = quality.get("endpoint_pair_score")
    if raw is None:
        pair = row.get("endpoint_pair")
        if isinstance(pair, dict):
            raw = pair.get("pair_score")
    if raw is None:
        source = row.get("source_match") if isinstance(row.get("source_match"), dict) else {}
        target = row.get("target_match") if isinstance(row.get("target_match"), dict) else {}
        sims = [
            float(x.get("semantic_similarity", 0.0) or 0.0)
            for x in (source, target)
            if x
        ]
        raw = sum(sims) / len(sims) if sims else 0.0
    return _clip01((float(raw) - 0.45) / 0.45)


def _paper_ids(row: dict[str, Any]) -> list[str]:
    values = row.get("visited_paper_ids", row.get("source_paper_ids", []))
    return sorted({str(x) for x in values if str(x).strip()})


def _candidate_requires_verification(row: dict[str, Any]) -> bool:
    if bool(row.get("requires_verification", False)):
        return True
    quality = row.get("path_quality") if isinstance(row.get("path_quality"), dict) else {}
    if float(quality.get("candidate_fraction", 0.0) or 0.0) > 0.0:
        return True
    return any(
        _as_bool(step.get("requires_verification", False))
        for step in row.get("steps", [])
        if isinstance(step, dict)
    )


def _render_path(graph: nx.DiGraph, row: dict[str, Any], max_chars: int = 1800) -> str:
    nodes = [str(x) for x in row.get("nodes", [])]
    if not nodes:
        return ""
    parts = [_node_label(graph, nodes[0])]
    for index, step in enumerate(row.get("steps", [])):
        if not isinstance(step, dict):
            continue
        relation = str(step.get("relation", "RELATED_TO"))
        target = str(step.get("target", nodes[min(index + 1, len(nodes) - 1)]))
        parts.append(f" --{relation}--> {_node_label(graph, target)}")
    text = "".join(parts)
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _semantic_text(graph: nx.DiGraph, row: dict[str, Any]) -> str:
    """Text used for redundancy/diversity rather than user-facing rendering.

    Candidate-unit routes can share a long confirmed prefix/suffix with the
    grounding bundle while introducing a genuinely different provisional
    bridge in the middle. Alpha3 therefore compares the candidate *core*
    (unit + distinct anchors + proposed S/R/O) rather than mean-pooling the
    entire route for those paths. Ordinary paths retain full-route semantics.
    """
    explicit = str(row.get("candidate_unit_semantic_text", "")).strip()
    if explicit:
        return explicit
    unit = row.get("candidate_unit")
    if isinstance(unit, dict) and unit:
        parts = [
            str(unit.get("label", "")),
            str(unit.get("proposed_subject", "")),
            str(unit.get("proposed_relation", "")),
            str(unit.get("proposed_object", "")),
            str(unit.get("entry_anchor_label", "")),
            str(unit.get("exit_anchor_label", "")),
        ]
        text = " | ".join(part for part in parts if part.strip())
        if text:
            return text
    return _render_path(graph, row)


def _lexical_tokens(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in _TOKEN_RE.findall(text) if len(token) >= 2)


def _lexical_similarity(left: str, right: str) -> float:
    a = _lexical_tokens(left)
    b = _lexical_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _path_embedding(row: dict[str, Any], semantic_index: Any | None) -> np.ndarray | None:
    """Mean-pool existing node-index embeddings; no extra model call is required."""
    if semantic_index is None:
        return None
    records = list(getattr(semantic_index, "records", []))
    embeddings = np.asarray(getattr(semantic_index, "embeddings", []), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        return None
    row_by_id = {
        str(record.get("node_id", "")): index
        for index, record in enumerate(records)
        if str(record.get("node_id", "")).strip()
    }
    vectors: list[np.ndarray] = []
    semantic_node_ids = row.get("candidate_unit_core_node_ids")
    if not isinstance(semantic_node_ids, list) or not semantic_node_ids:
        semantic_node_ids = row.get("nodes", [])
    for node_id in map(str, semantic_node_ids):
        index = row_by_id.get(node_id)
        if index is not None:
            vectors.append(embeddings[index])
    if not vectors:
        return None
    return _normalize_vector(np.mean(np.stack(vectors, axis=0), axis=0))


@dataclass(frozen=True)
class MechanisticContinuity:
    score: float
    band: str
    mechanism_before_alignment: bool
    mechanism_after_alignment: bool


def _segment_has_mechanism(
    graph: nx.DiGraph,
    nodes: list[str],
    steps: list[dict[str, Any]],
) -> bool:
    if any(_is_mechanism_step(step) for step in steps):
        return True
    return any(
        node_id in graph and _is_mechanism_node(node_id, dict(graph.nodes[node_id]))
        for node_id in nodes
    )


def _alignment_blocks(steps: list[dict[str, Any]]) -> list[tuple[int, int]]:
    indices = [index for index, step in enumerate(steps) if _is_alignment_step(step)]
    if not indices:
        return []
    blocks: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        blocks.append((start, previous))
        start = previous = index
    blocks.append((start, previous))
    return blocks


def _mechanistic_continuity(graph: nx.DiGraph, row: dict[str, Any]) -> MechanisticContinuity:
    nodes = [str(x) for x in row.get("nodes", [])]
    steps = [dict(step) for step in row.get("steps", []) if isinstance(step, dict)]
    blocks = _alignment_blocks(steps)
    if not blocks:
        return MechanisticContinuity(
            score=0.50,
            band="not_applicable",
            mechanism_before_alignment=False,
            mechanism_after_alignment=False,
        )

    best: tuple[float, bool, bool] = (0.0, False, False)
    for block_index, (start, end) in enumerate(blocks):
        previous_end = blocks[block_index - 1][1] if block_index > 0 else -1
        next_start = blocks[block_index + 1][0] if block_index + 1 < len(blocks) else len(steps)

        left_steps = steps[previous_end + 1 : start]
        # A segment covering edge indices [a,b) uses node indices [a,b].
        left_nodes = nodes[previous_end + 1 : start + 1]
        right_steps = steps[end + 1 : next_start]
        right_nodes = nodes[end + 1 : next_start + 1]

        before = _segment_has_mechanism(graph, left_nodes, left_steps)
        after = _segment_has_mechanism(graph, right_nodes, right_steps)
        score = 1.0 if before and after else (0.25 if before or after else 0.0)
        candidate = (score, before, after)
        if candidate > best:
            best = candidate

    score, before, after = best
    band = "high" if score >= 0.75 else ("medium" if score > 0.0 else "low")
    return MechanisticContinuity(
        score=score,
        band=band,
        mechanism_before_alignment=before,
        mechanism_after_alignment=after,
    )


@dataclass(frozen=True)
class GenericHopDiagnostics:
    generic_entity_fraction: float
    max_generic_run_length: int
    generic_burden: float
    registry_hop_fraction: float


def _generic_hop_diagnostics(graph: nx.DiGraph, row: dict[str, Any]) -> GenericHopDiagnostics:
    nodes = [str(x) for x in row.get("nodes", [])]
    internal = nodes[1:-1] if len(nodes) >= 3 else []
    flags: list[bool] = []
    for node_id in internal:
        if node_id not in graph:
            continue
        attrs = dict(graph.nodes[node_id])
        if _is_alignment_node(attrs):
            # Alignment hubs are separately penalized and do not reset an entity run.
            continue
        flags.append(_is_generic_entity_node(node_id, attrs))

    generic_count = sum(flags)
    generic_fraction = _clip01(generic_count / len(flags)) if flags else 0.0
    max_run = current = 0
    for flag in flags:
        if flag:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    run_burden = _clip01(max(0, max_run - 1) / 4.0)
    generic_burden = max(generic_fraction, run_burden)

    steps = [dict(step) for step in row.get("steps", []) if isinstance(step, dict)]
    registry_count = sum(_is_alignment_step(step) for step in steps)
    registry_fraction = _clip01(registry_count / len(steps)) if steps else 0.0
    return GenericHopDiagnostics(
        generic_entity_fraction=generic_fraction,
        max_generic_run_length=max_run,
        generic_burden=generic_burden,
        registry_hop_fraction=registry_fraction,
    )


@dataclass(frozen=True)
class DiscoveryPolicy:
    top_k: int = 8
    max_per_paper_signature: int = 2
    max_edge_jaccard: float = 0.70
    cross_paper_mechanistic_reserve: int = 2
    min_reserved_continuity: float = 0.75
    candidate_exploration_reserve: int = 4
    min_reserved_candidate_unit_score: float = 0.25

    semantic_diversity_enabled: bool = True
    semantic_similarity_threshold: float = 0.88
    semantic_relaxed_threshold: float = 0.94

    # Alpha2.1: discovery is allowed to under-fill. A path that is almost
    # identical to the grounding bundle is not a useful discovery inspiration
    # merely because top_k has not been filled.
    max_grounding_semantic_similarity: float = 0.95
    min_exploration_score: float = 0.05
    force_fill: bool = False

    endpoint_weight: float = 0.12
    mechanistic_weight: float = 0.20
    continuity_weight: float = 0.20
    cross_paper_weight: float = 0.12
    community_weight: float = 0.10
    rarity_weight: float = 0.08
    exploratory_mode_weight: float = 0.08

    redundancy_penalty_weight: float = 0.08
    semantic_grounding_penalty_weight: float = 0.16
    navigation_penalty_weight: float = 0.10
    reverse_penalty_weight: float = 0.04
    generic_entity_penalty_weight: float = 0.18
    registry_hop_penalty_weight: float = 0.08

    # Alpha3: reward candidate-unit routes that survived the dedicated
    # (source, unit, target) selector and suppress unrelated reaction detours.
    candidate_unit_quality_weight: float = 0.16
    reaction_switch_penalty_weight: float = 0.10


class DiscoveryBundleBuilder:
    """Build an exploration-oriented, non-evidentiary path bundle.

    Alpha2 adds three safeguards discovered in the 31-paper benchmark:
    1. mechanistic continuity must survive the actual cross-paper alignment,
    2. generic entity/registry hopping is explicitly penalized,
    3. semantic diversity is enforced using already-built node embeddings when
       available (with a deterministic lexical fallback otherwise).

    These are exploration heuristics, not scientific novelty claims.
    """

    def __init__(self, policy: DiscoveryPolicy | None = None) -> None:
        self.policy = policy or DiscoveryPolicy()

    def _score(
        self,
        *,
        row: dict[str, Any],
        mode: str,
        graph: nx.DiGraph,
        community_by_node: dict[str, int],
        relation_counts: Counter[str],
        grounding_edge_sets: list[frozenset[str]],
        semantic_grounding_redundancy: float,
    ) -> tuple[
        DiscoveryScoreBreakdown,
        list[str],
        MechanisticContinuity,
        GenericHopDiagnostics,
    ]:
        quality = row.get("path_quality") if isinstance(row.get("path_quality"), dict) else {}
        endpoint = _endpoint_score(quality, row)
        mechanism = _mechanistic_score(quality)
        papers = _paper_ids(row)
        cross_paper = _clip01((len(papers) - 1) / 2.0) if papers else 0.0
        community = _community_span(row, community_by_node)
        rarity = _path_relation_rarity(row, relation_counts)
        exploratory_bonus = 1.0 if mode == "exploratory" else 0.0
        continuity = _mechanistic_continuity(graph, row)
        generic = _generic_hop_diagnostics(graph, row)

        own_edges = _edge_set(row)
        redundancy = max(
            (_edge_jaccard(own_edges, other) for other in grounding_edge_sets),
            default=0.0,
        )
        navigation = _clip01(float(quality.get("navigation_edge_fraction", 0.0) or 0.0))
        reverse = _clip01(float(quality.get("reverse_fraction", 0.0) or 0.0))
        candidate_selection = (
            row.get("candidate_unit_selection")
            if isinstance(row.get("candidate_unit_selection"), dict)
            else {}
        )
        candidate_unit_quality = _clip01(float(candidate_selection.get("total", 0.0) or 0.0))
        reaction_switch_penalty = _clip01(
            float(candidate_selection.get("reaction_switch_penalty", 0.0) or 0.0)
        )

        p = self.policy
        total = (
            p.endpoint_weight * endpoint
            + p.mechanistic_weight * mechanism
            + p.continuity_weight * continuity.score
            + p.cross_paper_weight * cross_paper
            + p.community_weight * community
            + p.rarity_weight * rarity
            + p.exploratory_mode_weight * exploratory_bonus
            - p.redundancy_penalty_weight * redundancy
            - p.semantic_grounding_penalty_weight * semantic_grounding_redundancy
            - p.navigation_penalty_weight * navigation
            - p.reverse_penalty_weight * reverse
            - p.generic_entity_penalty_weight * generic.generic_burden
            - p.registry_hop_penalty_weight * generic.registry_hop_fraction
            + p.candidate_unit_quality_weight * candidate_unit_quality
            - p.reaction_switch_penalty_weight * reaction_switch_penalty
        )
        total = _clip01(total)

        reasons: list[str] = []
        path_type = str(quality.get("path_type", "UNKNOWN"))
        if path_type == "CROSS_PAPER_MECHANISTIC":
            reasons.append("cross_paper_mechanistic")
        if mechanism >= 0.65:
            reasons.append("mechanism_bearing")
        if len(papers) >= 2:
            reasons.append("cross_paper")
        if continuity.band == "high":
            reasons.append("mechanistic_continuity_high")
        elif continuity.band == "medium":
            reasons.append("mechanistic_continuity_one_sided")
        elif continuity.band == "low" and len(papers) >= 2:
            reasons.append("mechanistic_continuity_low")
        if community >= 0.5:
            reasons.append("spans_graph_communities")
        if rarity >= 0.6:
            reasons.append("rare_relation_pattern")
        if exploratory_bonus:
            reasons.append("exploratory_projection")
        if redundancy >= 0.7:
            reasons.append("high_grounding_edge_overlap")
        if semantic_grounding_redundancy >= 0.82:
            reasons.append("high_grounding_semantic_overlap")
        if navigation >= 0.6:
            reasons.append("navigation_heavy")
        if generic.generic_burden >= 0.60:
            reasons.append("generic_entity_hopping")
        if generic.registry_hop_fraction >= 0.30:
            reasons.append("registry_hop_heavy")
        if _candidate_requires_verification(row):
            reasons.append("candidate_requires_verification")
        if candidate_unit_quality > 0.0:
            reasons.append("candidate_unit_traversal")
        if candidate_unit_quality >= self.policy.min_reserved_candidate_unit_score:
            reasons.append("candidate_unit_quality_supported")
        if reaction_switch_penalty > 0.0:
            reasons.append("reaction_domain_switch")

        return (
            DiscoveryScoreBreakdown(
                endpoint_relevance=endpoint,
                mechanistic_content=mechanism,
                cross_paper_span=cross_paper,
                community_span=community,
                relation_rarity=rarity,
                exploratory_mode_bonus=exploratory_bonus,
                grounding_redundancy_penalty=redundancy,
                navigation_burden_penalty=navigation,
                reverse_burden_penalty=reverse,
                mechanistic_continuity=continuity.score,
                semantic_grounding_redundancy_penalty=semantic_grounding_redundancy,
                generic_entity_burden_penalty=generic.generic_burden,
                registry_hop_penalty=generic.registry_hop_fraction,
                candidate_unit_quality=candidate_unit_quality,
                reaction_domain_switch_penalty=reaction_switch_penalty,
                total=total,
            ),
            reasons,
            continuity,
            generic,
        )

    def build(
        self,
        traversal_payloads: Iterable[tuple[str, dict[str, Any], nx.DiGraph]],
        *,
        semantic_indexes: Mapping[str, Any] | None = None,
    ) -> DiscoveryBundle:
        payloads = list(traversal_payloads)
        if not payloads:
            raise ValueError("at least one traversal payload is required")

        corpus_ids = {str(payload.get("corpus_id", "")) for _, payload, _ in payloads}
        if len(corpus_ids) != 1:
            raise ValueError(f"all traversals must use one corpus_id, got: {sorted(corpus_ids)}")
        corpus_id = next(iter(corpus_ids))

        semantic_indexes = dict(semantic_indexes or {})
        warnings: list[str] = []
        semantic_enabled = self.policy.semantic_diversity_enabled
        available_indexes = {
            source_name: semantic_indexes.get(source_name)
            for source_name, _, _ in payloads
            if semantic_indexes.get(source_name) is not None
        }
        if semantic_enabled and len(available_indexes) == len(payloads):
            model_names = {
                str(getattr(index, "model_name", getattr(index, "manifest", {}).get("model_name", "")))
                for index in available_indexes.values()
            }
            dimensions = {
                int(np.asarray(getattr(index, "embeddings", [])).shape[1])
                for index in available_indexes.values()
                if np.asarray(getattr(index, "embeddings", [])).ndim == 2
            }
            if len(model_names) == 1 and len(dimensions) == 1:
                semantic_mode = "node_embedding"
                semantic_model_name = next(iter(model_names)) or None
            else:
                semantic_mode = "lexical_fallback"
                semantic_model_name = None
                warnings.append(
                    "semantic node indexes used incompatible model/dimension metadata; "
                    "falling back to lexical path diversity"
                )
        elif semantic_enabled:
            semantic_mode = "lexical_fallback"
            semantic_model_name = None
            warnings.append(
                "one or more traversal node indexes were unavailable; "
                "falling back to lexical path diversity"
            )
        else:
            semantic_mode = "disabled"
            semantic_model_name = None

        query_parts: list[str] = []
        source_files: list[str] = []
        grounding_edge_sets: list[frozenset[str]] = []
        grounding_representations: list[tuple[str, np.ndarray | None]] = []
        for source_name, payload, graph in payloads:
            source_files.append(source_name)
            query_parts.append(
                f"{payload.get('source_query','')}|{payload.get('semantic_stop_query','')}|{payload.get('target_query','')}"
            )
            if str(payload.get("mode", "")) != "exploratory":
                semantic_index = available_indexes.get(source_name)
                for row in payload.get("paths", []):
                    if not isinstance(row, dict):
                        continue
                    grounding_edge_sets.append(_edge_set(row))
                    grounding_representations.append(
                        (
                            _semantic_text(graph, row),
                            _path_embedding(row, semantic_index) if semantic_mode == "node_embedding" else None,
                        )
                    )

        def semantic_similarity_to_grounding(
            *,
            text: str,
            vector: np.ndarray | None,
        ) -> float:
            if semantic_mode == "disabled":
                return 0.0
            if semantic_mode == "node_embedding" and vector is not None:
                return max((_cosine(vector, other_vector) for _, other_vector in grounding_representations), default=0.0)
            return max((_lexical_similarity(text, other_text) for other_text, _ in grounding_representations), default=0.0)

        candidate_rows: list[dict[str, Any]] = []
        used_candidate_pool = True
        for source_name, payload, graph in payloads:
            rows = payload.get("candidate_paths")
            if not isinstance(rows, list) or not rows:
                rows = payload.get("paths", [])
                used_candidate_pool = False
                warnings.append(
                    f"{source_name}: candidate_paths missing; discovery used returned paths only. "
                    "Rerun traversal with --include-candidate-paths for full discovery search."
                )
            communities = _communities(graph)
            relation_counts = _relation_counts(graph)
            mode = str(payload.get("mode", ""))
            semantic_index = available_indexes.get(source_name)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rendered = _render_path(graph, row)
                semantic_text = _semantic_text(graph, row)
                vector = _path_embedding(row, semantic_index) if semantic_mode == "node_embedding" else None
                semantic_grounding = semantic_similarity_to_grounding(text=semantic_text, vector=vector)
                score, reasons, continuity, generic = self._score(
                    row=row,
                    mode=mode,
                    graph=graph,
                    community_by_node=communities,
                    relation_counts=relation_counts,
                    grounding_edge_sets=grounding_edge_sets,
                    semantic_grounding_redundancy=semantic_grounding,
                )
                copied = dict(row)
                copied["_discovery_source_file"] = source_name
                copied["_discovery_mode"] = mode
                copied["_discovery_graph"] = graph
                copied["_discovery_score"] = score
                copied["_discovery_reasons"] = reasons
                copied["_discovery_rendered"] = rendered
                copied["_discovery_semantic_text"] = semantic_text
                copied["_discovery_semantic_vector"] = vector
                copied["_discovery_semantic_grounding"] = semantic_grounding
                copied["_discovery_continuity"] = continuity
                copied["_discovery_generic"] = generic
                candidate_rows.append(copied)

        # Deduplicate stable path IDs across repeated traversal files; keep highest exploration score.
        best_by_path: dict[tuple[str, str], dict[str, Any]] = {}
        for row in candidate_rows:
            key = (str(row.get("_discovery_mode", "")), str(row.get("path_id", "")))
            previous = best_by_path.get(key)
            if previous is None or row["_discovery_score"].total > previous["_discovery_score"].total:
                best_by_path[key] = row
        candidates = sorted(
            best_by_path.values(),
            key=lambda row: (-float(row["_discovery_score"].total), str(row.get("path_id", ""))),
        )

        # Alpha2.1 hard discovery-quality gate. Alpha2 treated grounding
        # similarity only as a soft score penalty and then force-filled the
        # bundle, which admitted paths with grounding_sem ~= 1.00. That makes a
        # DiscoveryBundle a second grounding bundle rather than an exploration
        # surface. Under-fill is preferable to false diversity.
        quality_candidates: list[dict[str, Any]] = []
        blocked_grounding = 0
        blocked_score = 0
        for row in candidates:
            score_total = float(row["_discovery_score"].total)
            grounding_semantic = float(
                row.get("_discovery_semantic_grounding", 0.0) or 0.0
            )
            if (
                not self.policy.force_fill
                and score_total < self.policy.min_exploration_score
            ):
                blocked_score += 1
                continue
            if (
                not self.policy.force_fill
                and semantic_mode != "disabled"
                and grounding_representations
                and grounding_semantic
                > self.policy.max_grounding_semantic_similarity
            ):
                blocked_grounding += 1
                continue
            quality_candidates.append(row)

        if blocked_grounding:
            warnings.append(
                f"{blocked_grounding} candidate paths were excluded because "
                "their semantic similarity to the grounding bundle exceeded "
                f"{self.policy.max_grounding_semantic_similarity:.2f}."
            )
        if blocked_score:
            warnings.append(
                f"{blocked_score} candidate paths were excluded because their "
                "exploration score was below "
                f"{self.policy.min_exploration_score:.2f}."
            )

        selected: list[dict[str, Any]] = []
        selected_keys: set[tuple[str, str]] = set()
        paper_counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
        selected_edges: list[frozenset[str]] = []

        def semantic_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
            if semantic_mode == "disabled":
                return 0.0
            if semantic_mode == "node_embedding":
                value = _cosine(
                    left.get("_discovery_semantic_vector"),
                    right.get("_discovery_semantic_vector"),
                )
                if value > 0.0:
                    return value
            return _lexical_similarity(
                str(left.get("_discovery_semantic_text", left.get("_discovery_rendered", ""))),
                str(right.get("_discovery_semantic_text", right.get("_discovery_rendered", ""))),
            )

        def max_selected_semantic(row: dict[str, Any]) -> float:
            return max((semantic_similarity(row, other) for other in selected), default=0.0)

        def eligible(
            row: dict[str, Any],
            *,
            enforce_structure: bool,
            semantic_threshold: float | None,
        ) -> tuple[bool, float]:
            key = (str(row.get("_discovery_mode", "")), str(row.get("path_id", "")))
            if key in selected_keys:
                return False, 0.0
            papers = tuple(_paper_ids(row))
            own_edges = _edge_set(row)
            if enforce_structure:
                if paper_counts[papers] >= self.policy.max_per_paper_signature:
                    return False, 0.0
                if selected_edges and max(_edge_jaccard(own_edges, other) for other in selected_edges) > self.policy.max_edge_jaccard:
                    return False, 0.0
            semantic_overlap = max_selected_semantic(row)
            if semantic_threshold is not None and semantic_overlap > semantic_threshold:
                return False, semantic_overlap
            return True, semantic_overlap

        def add(row: dict[str, Any], semantic_overlap: float) -> None:
            key = (str(row.get("_discovery_mode", "")), str(row.get("path_id", "")))
            copied = dict(row)
            copied["_discovery_max_selected_semantic"] = semantic_overlap
            selected.append(copied)
            selected_keys.add(key)
            paper_counts[tuple(_paper_ids(row))] += 1
            selected_edges.append(_edge_set(row))

        strict_threshold = (
            self.policy.semantic_similarity_threshold
            if semantic_mode != "disabled"
            else None
        )
        relaxed_threshold = (
            self.policy.semantic_relaxed_threshold
            if semantic_mode != "disabled"
            else None
        )

        # Alpha3: reserve a small number of dedicated candidate-unit routes.
        # These have already been ranked as (source, unit, target) triples with
        # exactly one provisional unit and distinct entry/exit grounding anchors.
        candidate_reserved = [
            row
            for row in quality_candidates
            if str((row.get("path_quality") or {}).get("path_type", "")) == "CANDIDATE_EXPLORATION"
            and int(row.get("candidate_unit_count", 0) or 0) == 1
            and float((row.get("candidate_unit_selection") or {}).get("total", 0.0) or 0.0)
            >= self.policy.min_reserved_candidate_unit_score
        ]
        for row in candidate_reserved:
            if len(selected) >= min(self.policy.candidate_exploration_reserve, self.policy.top_k):
                break
            ok, overlap = eligible(row, enforce_structure=True, semantic_threshold=strict_threshold)
            if ok:
                add(row, overlap)

        # Reserve only mechanistic cross-paper paths whose mechanism survives both sides
        # of the actual alignment crossing. Alpha1 reserved any path merely labelled
        # CROSS_PAPER_MECHANISTIC, which promoted entity-hop routes in the benchmark.
        reserved = [
            row
            for row in quality_candidates
            if str((row.get("path_quality") or {}).get("path_type", "")) == "CROSS_PAPER_MECHANISTIC"
            and row["_discovery_continuity"].score >= self.policy.min_reserved_continuity
        ]
        for row in reserved:
            if len(selected) >= min(self.policy.cross_paper_mechanistic_reserve, self.policy.top_k):
                break
            ok, overlap = eligible(row, enforce_structure=True, semantic_threshold=strict_threshold)
            if ok:
                add(row, overlap)

        # Semantic diversity remains a gate even when structural diversity
        # is relaxed. The alpha2 final (False, None) pass force-filled near-
        # duplicates (selected_sem up to 1.00). Alpha2.1 intentionally allows
        # fewer than top_k inspirations.
        selection_passes = [
            (True, strict_threshold),
            (True, relaxed_threshold),
            (False, relaxed_threshold),
        ]
        if self.policy.force_fill:
            # Diagnostic ablation only: restore alpha2-style final fill.
            selection_passes.append((False, None))

        for enforce_structure, semantic_threshold in selection_passes:
            if len(selected) >= self.policy.top_k:
                break
            for row in (candidates if self.policy.force_fill else quality_candidates):
                if len(selected) >= self.policy.top_k:
                    break
                ok, overlap = eligible(
                    row,
                    enforce_structure=enforce_structure,
                    semantic_threshold=semantic_threshold,
                )
                if ok:
                    add(row, overlap)

        if len(selected) < min(self.policy.top_k, len(candidates)):
            warnings.append(
                "Discovery bundle intentionally under-filled: "
                f"selected {len(selected)} of requested {self.policy.top_k}. "
                "Do not relax discovery-quality gates merely to fill quota."
            )
        if not selected:
            warnings.append(
                "No discovery-distinct path survived the quality gates. "
                "This mechanism traversal appears to replay the grounding "
                "neighborhood; add an exploratory-mode traversal or broaden "
                "the retrieval question instead of generating from canonical "
                "grounding paths."
            )

        inspirations: list[DiscoveryInspiration] = []
        for rank, row in enumerate(selected, start=1):
            quality = row.get("path_quality") if isinstance(row.get("path_quality"), dict) else {}
            steps = [step for step in row.get("steps", []) if isinstance(step, dict)]
            path_id = str(row.get("path_id", ""))
            mode = str(row.get("_discovery_mode", ""))
            inspiration_id = _stable_id("discovery_inspiration", corpus_id, mode, path_id)
            reasons = list(row["_discovery_reasons"])
            selected_semantic = float(row.get("_discovery_max_selected_semantic", 0.0) or 0.0)
            if selected_semantic >= self.policy.semantic_similarity_threshold:
                reasons.append("semantic_diversity_relaxed")
            reasons.append(f"bundle_rank:{rank}")
            continuity: MechanisticContinuity = row["_discovery_continuity"]
            generic: GenericHopDiagnostics = row["_discovery_generic"]
            candidate_unit = row.get("candidate_unit") if isinstance(row.get("candidate_unit"), dict) else {}
            candidate_selection = (
                row.get("candidate_unit_selection")
                if isinstance(row.get("candidate_unit_selection"), dict)
                else {}
            )
            inspirations.append(
                DiscoveryInspiration(
                    inspiration_id=inspiration_id,
                    source_path_id=path_id,
                    source_corpus_id=corpus_id,
                    source_mode=mode,
                    path_type=str(quality.get("path_type", "UNKNOWN")),
                    paper_ids=_paper_ids(row),
                    node_ids=[str(x) for x in row.get("nodes", [])],
                    edge_ids=[_edge_id(step, i) for i, step in enumerate(steps)],
                    relation_sequence=[str(step.get("relation", "RELATED_TO")) for step in steps],
                    rendered_path=str(row.get("_discovery_rendered", "")),
                    exploration_score=float(row["_discovery_score"].total),
                    score_breakdown=row["_discovery_score"],
                    reason_codes=sorted(set(reasons)),
                    requires_verification=_candidate_requires_verification(row),
                    mechanism_before_alignment=continuity.mechanism_before_alignment,
                    mechanism_after_alignment=continuity.mechanism_after_alignment,
                    mechanistic_continuity_band=continuity.band,  # type: ignore[arg-type]
                    generic_entity_fraction=generic.generic_entity_fraction,
                    max_generic_run_length=generic.max_generic_run_length,
                    registry_hop_fraction=generic.registry_hop_fraction,
                    semantic_similarity_to_grounding=float(row.get("_discovery_semantic_grounding", 0.0) or 0.0),
                    max_semantic_similarity_to_selected=selected_semantic,
                    semantic_diversity_mode=semantic_mode,  # type: ignore[arg-type]
                    candidate_unit_id=str(candidate_unit.get("unit_id", "")),
                    candidate_unit_label=str(candidate_unit.get("label", "")),
                    candidate_entry_anchor_id=str(candidate_unit.get("entry_anchor_id", "")),
                    candidate_entry_anchor_label=str(candidate_unit.get("entry_anchor_label", "")),
                    candidate_exit_anchor_id=str(candidate_unit.get("exit_anchor_id", "")),
                    candidate_exit_anchor_label=str(candidate_unit.get("exit_anchor_label", "")),
                    candidate_proposed_subject=str(candidate_unit.get("proposed_subject", "")),
                    candidate_proposed_relation=str(candidate_unit.get("proposed_relation", "")),
                    candidate_proposed_object=str(candidate_unit.get("proposed_object", "")),
                    candidate_unit_score=float(candidate_selection.get("total", 0.0) or 0.0),
                    reaction_domain_switch_penalty=float(
                        candidate_selection.get("reaction_switch_penalty", 0.0) or 0.0
                    ),
                )
            )

        query_signature = " || ".join(sorted(set(query_parts)))
        bundle_id = _stable_id(
            "discovery_bundle",
            corpus_id,
            query_signature,
            *[x.inspiration_id for x in inspirations],
        )
        payload = {
            "schema_version": "discovery-bundle-v1",
            "bundle_id": bundle_id,
            "corpus_id": corpus_id,
            "query_signature": query_signature,
            "inspirations": [x.model_dump(mode="json") for x in inspirations],
            "source_traversal_files": source_files,
            "candidate_count": len(candidates),
            "selected_count": len(inspirations),
            "used_candidate_pool": used_candidate_pool,
            "warnings": sorted(set(warnings)),
            "semantic_diversity_mode": semantic_mode,
            "semantic_model_name": semantic_model_name,
            "semantic_similarity_threshold": self.policy.semantic_similarity_threshold,
            "policy_version": "discovery-policy-v3",
        }
        return DiscoveryBundle(**payload, bundle_sha256=_sha256_json(payload))


def load_traversal_with_graph(
    path: str | Path,
    *,
    project_root: str | Path = ".",
) -> tuple[str, dict[str, Any], nx.DiGraph]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected traversal JSON object: {path}")
    corpus_id = str(payload.get("corpus_id", "")).strip()
    mode = str(payload.get("mode", "")).strip()
    if not corpus_id or not mode:
        raise ValueError(f"traversal is missing corpus_id/mode: {path}")
    graph_path = (
        Path(project_root)
        / "data_dac"
        / "corpus"
        / corpus_id
        / mode
        / "navigation"
        / "graph.graphml"
    )
    graph = nx.read_graphml(graph_path)
    return str(path), payload, graph


def load_semantic_index_for_traversal(
    payload: dict[str, Any],
    *,
    project_root: str | Path = ".",
) -> Any | None:
    """Load the existing node embedding index without instantiating a model."""
    corpus_id = str(payload.get("corpus_id", "")).strip()
    mode = str(payload.get("mode", "")).strip()
    if not corpus_id or not mode:
        return None
    index_dir = (
        Path(project_root)
        / "data_dac"
        / "corpus"
        / corpus_id
        / mode
        / "navigation"
        / "node_index"
    )
    if not (index_dir / "manifest.json").exists():
        return None
    try:
        from dac_her.node_mapping import load_node_embedding_index

        return load_node_embedding_index(index_dir)
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError):
        return None
