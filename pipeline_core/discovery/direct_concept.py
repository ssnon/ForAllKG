from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import networkx as nx

from pipeline_core.discovery_semantics import is_mechanism_node
from pipeline_core.domain_profile import DiscoverySemantics
from pipeline_core.discovery.waypoint_selection import match_tier


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _as_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _jsonish_papers(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(
            sorted({
                str(item)
                for item in value
                if str(item).strip()
            })
        )
    text = str(value or "").strip()
    if not text:
        return ()
    try:
        import json

        parsed = json.loads(text)
        if isinstance(parsed, list):
            return tuple(
                sorted({
                    str(item)
                    for item in parsed
                    if str(item).strip()
                })
            )
    except Exception:
        pass
    return ()




@dataclass(frozen=True)
class DirectConceptHit:
    node_id: str
    node_type: str
    label: str
    source_similarity: float | None
    target_similarity: float | None
    minimum_similarity: float | None
    joint_similarity: float | None
    source_match_tier: int
    target_match_tier: int
    hit_tier: int
    source_exact: bool
    target_exact: bool
    source_contains: bool
    target_contains: bool
    evidence_status: str
    graph_layer: str
    requires_verification: bool
    source_paper_id: str
    source_paper_ids: tuple[str, ...]
    mechanism_bearing: bool
    quality_basis: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_paper_ids"] = list(
            self.source_paper_ids
        )
        return payload


class DirectConceptHitSelector:
    """Preserve same-node semantic answers without pretending they are paths."""

    def __init__(
        self,
        graph: nx.DiGraph,
        *,
        min_similarity: float = 0.60,
        discovery_semantics: DiscoverySemantics,
    ) -> None:
        self.graph = graph
        self.min_similarity = float(
            min_similarity
        )
        self.discovery_semantics = discovery_semantics

    def _similarity(
        self,
        match: dict[str, Any],
    ) -> float | None:
        if _as_bool(
            match.get("direct_node_id", False)
        ):
            return 1.0
        return _as_float(
            match.get("semantic_similarity")
        )

    def select(
        self,
        source_matches: Iterable[dict[str, Any]],
        target_matches: Iterable[dict[str, Any]],
        *,
        top_k: int = 5,
    ) -> list[DirectConceptHit]:
        source_by_id = {
            str(row["node_id"]): dict(row)
            for row in source_matches
        }
        target_by_id = {
            str(row["node_id"]): dict(row)
            for row in target_matches
        }

        hits: list[DirectConceptHit] = []

        for node_id in sorted(
            set(source_by_id)
            & set(target_by_id)
        ):
            source = source_by_id[node_id]
            target = target_by_id[node_id]
            source_similarity = self._similarity(
                source
            )
            target_similarity = self._similarity(
                target
            )

            direct = (
                _as_bool(
                    source.get(
                        "direct_node_id",
                        False,
                    )
                )
                and _as_bool(
                    target.get(
                        "direct_node_id",
                        False,
                    )
                )
            )

            numeric = (
                source_similarity is not None
                and target_similarity is not None
            )
            minimum = (
                min(
                    source_similarity,
                    target_similarity,
                )
                if numeric
                else None
            )
            joint = (
                math.sqrt(
                    max(0.0, source_similarity)
                    * max(0.0, target_similarity)
                )
                if numeric
                else None
            )

            if (
                not direct
                and (
                    minimum is None
                    or minimum
                    < self.min_similarity
                )
            ):
                continue

            attrs = (
                dict(self.graph.nodes[node_id])
                if node_id in self.graph
                else {}
            )
            node_type = str(
                source.get("node_type")
                or source.get("type")
                or target.get("node_type")
                or target.get("type")
                or attrs.get("type")
                or "Unknown"
            )
            label = str(
                source.get("label")
                or target.get("label")
                or attrs.get("label")
                or attrs.get("statement")
                or node_id
            )

            source_exact = (
                _as_bool(
                    source.get(
                        "direct_node_id",
                        False,
                    )
                )
                or _as_bool(
                    source.get(
                        "exact_label_match",
                        False,
                    )
                )
            )
            target_exact = (
                _as_bool(
                    target.get(
                        "direct_node_id",
                        False,
                    )
                )
                or _as_bool(
                    target.get(
                        "exact_label_match",
                        False,
                    )
                )
            )
            source_contains = _as_bool(
                source.get(
                    "label_contains_query",
                    False,
                )
            )
            target_contains = _as_bool(
                target.get(
                    "label_contains_query",
                    False,
                )
            )

            source_tier = match_tier(source)
            target_tier = match_tier(target)
            hit_tier = max(
                source_tier,
                target_tier,
            )

            if direct:
                basis = "direct_node_id"
            elif source_exact and target_exact:
                basis = "exact_on_both_queries"
            elif (
                source_contains
                and target_contains
            ):
                basis = "contains_both_queries"
            else:
                basis = "semantic_intersection"

            source_paper_id = str(
                source.get("source_paper_id")
                or target.get("source_paper_id")
                or attrs.get("source_paper_id")
                or ""
            )
            source_paper_ids = _jsonish_papers(
                source.get(
                    "source_paper_ids_json"
                )
                or target.get(
                    "source_paper_ids_json"
                )
                or attrs.get(
                    "source_paper_ids_json"
                )
                or []
            )
            if (
                source_paper_id
                and source_paper_id
                not in source_paper_ids
            ):
                source_paper_ids = tuple(
                    sorted(
                        set(source_paper_ids)
                        | {source_paper_id}
                    )
                )

            evidence_status = str(
                source.get("evidence_status")
                or target.get("evidence_status")
                or attrs.get("evidence_status")
                or ""
            )
            graph_layer = str(
                source.get("graph_layer")
                or target.get("graph_layer")
                or attrs.get("graph_layer")
                or ""
            )
            requires_verification = (
                _as_bool(
                    source.get(
                        "requires_verification",
                        False,
                    )
                )
                or _as_bool(
                    target.get(
                        "requires_verification",
                        False,
                    )
                )
                or _as_bool(
                    attrs.get(
                        "requires_verification",
                        False,
                    )
                )
            )

            hits.append(
                DirectConceptHit(
                    node_id=node_id,
                    node_type=node_type,
                    label=label,
                    source_similarity=(
                        source_similarity
                    ),
                    target_similarity=(
                        target_similarity
                    ),
                    minimum_similarity=minimum,
                    joint_similarity=joint,
                    source_match_tier=(
                        source_tier
                    ),
                    target_match_tier=(
                        target_tier
                    ),
                    hit_tier=hit_tier,
                    source_exact=source_exact,
                    target_exact=target_exact,
                    source_contains=(
                        source_contains
                    ),
                    target_contains=(
                        target_contains
                    ),
                    evidence_status=(
                        evidence_status
                    ),
                    graph_layer=graph_layer,
                    requires_verification=(
                        requires_verification
                    ),
                    source_paper_id=(
                        source_paper_id
                    ),
                    source_paper_ids=(
                        source_paper_ids
                    ),
                    mechanism_bearing=(
                        is_mechanism_node(
                            node_id,
                            dict(attrs, type=node_type),
                            self.discovery_semantics,
                        )
                    ),
                    quality_basis=basis,
                )
            )

        hits.sort(
            key=lambda item: (
                item.hit_tier,
                -(
                    item.minimum_similarity
                    if item.minimum_similarity
                    is not None
                    else -1.0
                ),
                -(
                    item.joint_similarity
                    if item.joint_similarity
                    is not None
                    else -1.0
                ),
                item.node_id,
            )
        )
        return hits[: max(0, top_k)]
