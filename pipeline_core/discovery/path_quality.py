from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx

from pipeline_core.discovery.discovery_semantics import (
    is_alignment_edge,
    is_alignment_node,
    is_mechanism_edge,
    is_mechanism_node,
    is_scaffold_edge,
    is_shared_entity_node,
    normalized_node_type,
)
from pipeline_core.domain.domain_profile import DiscoverySemantics


PATH_TYPE_CANDIDATE = "CANDIDATE_EXPLORATION"
PATH_TYPE_SHARED_ENTITY = "SHARED_ENTITY_BRIDGE"
PATH_TYPE_CROSS_PAPER_MECHANISTIC = "CROSS_PAPER_MECHANISTIC"
PATH_TYPE_DIRECT_MECHANISTIC = "DIRECT_MECHANISTIC"
PATH_TYPE_CROSS_PAPER_BRIDGE = "CROSS_PAPER_BRIDGE"
PATH_TYPE_SCAFFOLD = "SCAFFOLD_NAVIGATION"


_ALIGNMENT_CLASSES = {
    "registry_alignment",
    "pattern_alignment",
}




_OBSERVATION_MARKERS = (
    "OBSERVATION",
    "MEASURE",
    "REPORT",
    "QUANTIF",
)




def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _safe_fraction(
    numerator: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _band(
    value: float,
    *,
    medium: float,
    high: float,
) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def relation_role(
    step: dict[str, Any],
    discovery_semantics: DiscoverySemantics,
) -> str:
    semantics = discovery_semantics
    if is_alignment_edge(step):
        return "alignment"
    if is_scaffold_edge(step, semantics):
        return "scaffold"
    if is_mechanism_edge(step, semantics):
        return "mechanism"

    relation = str(
        step.get("relation", "")
    ).strip().upper()
    if any(
        marker in relation
        for marker in _OBSERVATION_MARKERS
    ):
        return "observation"
    return "scientific_other"


def _normalized_node_type(
    attrs: dict[str, Any],
) -> str:
    return normalized_node_type(attrs)


def _is_alignment_node(
    attrs: dict[str, Any],
) -> bool:
    return is_alignment_node(attrs)


def _is_mechanism_node(
    node_id: str,
    attrs: dict[str, Any],
    *,
    semantics: DiscoverySemantics,
) -> bool:
    return is_mechanism_node(
        node_id,
        attrs,
        semantics,
    )


@dataclass(frozen=True)
class PathQuality:
    path_type: str
    path_structure_type: str
    path_tags: tuple[str, ...]
    endpoint_semantic_tier: int
    endpoint_pair_score: float | None
    mechanism_edge_count: int
    mechanism_node_count: int
    mechanism_node_ids: tuple[str, ...]
    content_node_count: int
    observation_edge_count: int
    scaffold_edge_count: int
    alignment_edge_count: int
    other_scientific_edge_count: int
    mechanistic_density: float
    mechanistic_edge_density: float
    mechanistic_node_density: float
    mechanistic_content_score: float
    mechanism_bearing: bool
    mechanism_content_sources: tuple[str, ...]
    mechanistic_content_basis: str
    scaffold_density: float
    navigation_edge_fraction: float
    reverse_fraction: float
    candidate_fraction: float
    endpoint_relevance: str
    mechanistic_content: str
    navigation_burden: str
    reverse_burden: str
    visited_paper_count: int
    visited_paper_ids: tuple[str, ...]
    supporting_paper_ids: tuple[str, ...]
    hub_scope_paper_ids: tuple[str, ...]
    shared_entity_bridge: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path_tags"] = list(
            self.path_tags
        )
        payload["mechanism_node_ids"] = list(
            self.mechanism_node_ids
        )
        payload["mechanism_content_sources"] = list(
            self.mechanism_content_sources
        )
        payload["visited_paper_ids"] = list(
            self.visited_paper_ids
        )
        payload["supporting_paper_ids"] = list(
            self.supporting_paper_ids
        )
        payload["hub_scope_paper_ids"] = list(
            self.hub_scope_paper_ids
        )
        return payload


class PathQualityScorer:
    """Produce auditable, multi-axis path quality features.

    This deliberately avoids pretending that scientific path quality can be
    reduced to one validated scalar. The output is a set of deterministic
    dimensions plus a coarse path type that a later Graph Explorer agent can
    use according to its task.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        *,
        discovery_semantics: DiscoverySemantics,
    ) -> None:
        self.graph = graph
        self.discovery_semantics = discovery_semantics

    def _shared_entity_bridge(
        self,
        path_row: dict[str, Any],
    ) -> bool:
        nodes = [
            str(node_id)
            for node_id
            in path_row.get("nodes", [])
        ]
        if len(nodes) < 3:
            return False

        internal = nodes[1:-1]
        return any(
            is_shared_entity_node(
                node_id,
                dict(self.graph.nodes[node_id]),
                self.discovery_semantics,
            )
            for node_id in internal
            if node_id in self.graph
        )

    def score(
        self,
        path_row: dict[str, Any],
    ) -> PathQuality:
        steps = [
            dict(step)
            for step
            in path_row.get("steps", [])
        ]

        role_counts = {
            "mechanism": 0,
            "observation": 0,
            "scaffold": 0,
            "alignment": 0,
            "scientific_other": 0,
        }

        for step in steps:
            role = relation_role(step, self.discovery_semantics)
            role_counts[role] += 1

        path_nodes = [
            str(node_id)
            for node_id
            in path_row.get("nodes", [])
        ]
        content_node_ids: list[str] = []
        mechanism_node_ids: list[str] = []

        for node_id in path_nodes:
            if node_id not in self.graph:
                continue
            attrs = dict(
                self.graph.nodes[node_id]
            )
            if _is_alignment_node(attrs):
                continue
            content_node_ids.append(node_id)
            if _is_mechanism_node(
                node_id,
                attrs,
                semantics=self.discovery_semantics,
            ):
                mechanism_node_ids.append(
                    node_id
                )

        content_node_count = len(
            content_node_ids
        )
        mechanism_node_ids = sorted(
            set(mechanism_node_ids)
        )
        mechanism_node_count = len(
            mechanism_node_ids
        )

        hop_count = int(
            path_row.get(
                "hop_count",
                len(steps),
            )
        )
        scientific_edge_count = int(
            path_row.get(
                "scientific_edge_count",
                max(
                    0,
                    hop_count
                    - role_counts["alignment"],
                ),
            )
        )
        alignment_edge_count = int(
            path_row.get(
                "alignment_edge_count",
                role_counts["alignment"],
            )
        )
        reverse_edge_count = int(
            path_row.get(
                "reverse_edge_count",
                0,
            )
        )
        candidate_edge_count = int(
            path_row.get(
                "candidate_edge_count",
                0,
            )
        )

        mechanism_edge_count = (
            role_counts["mechanism"]
        )
        observation_edge_count = (
            role_counts["observation"]
        )
        scaffold_edge_count = (
            role_counts["scaffold"]
        )
        other_edge_count = (
            role_counts["scientific_other"]
        )

        mechanistic_edge_density = _safe_fraction(
            mechanism_edge_count,
            scientific_edge_count,
        )
        # Backward-compatible alias. In v2.4.4 the explicit name above
        # should be preferred because this density is edge-only.
        mechanistic_density = (
            mechanistic_edge_density
        )
        mechanistic_node_density = _safe_fraction(
            mechanism_node_count,
            content_node_count,
        )
        mechanistic_content_score = max(
            mechanistic_edge_density,
            mechanistic_node_density,
        )
        mechanism_bearing = (
            mechanism_edge_count > 0
            or mechanism_node_count > 0
        )
        mechanism_content_sources: list[str] = []
        if mechanism_edge_count > 0:
            mechanism_content_sources.append(
                "edge_relation"
            )
        if mechanism_node_count > 0:
            mechanism_content_sources.append(
                "mechanism_node"
            )

        # Mechanistic-content *band* is intentionally categorical rather
        # than thresholding a single density. Edge relations and
        # mechanism-bearing nodes are distinct evidence channels:
        # - both channels present  -> high
        # - one channel present    -> medium
        # - neither present        -> low
        #
        # Densities remain available as descriptive continuous features.
        # This avoids anomalies where a path containing multiple explicit
        # mechanism nodes plus a mechanism relation could be rated below a
        # shorter path simply because its denominator was larger.
        if (
            mechanism_edge_count > 0
            and mechanism_node_count > 0
        ):
            mechanistic_content = "high"
            mechanistic_content_basis = (
                "edge_and_node"
            )
        elif mechanism_edge_count > 0:
            mechanistic_content = "medium"
            mechanistic_content_basis = (
                "edge_only"
            )
        elif mechanism_node_count > 0:
            mechanistic_content = "medium"
            mechanistic_content_basis = (
                "node_only"
            )
        else:
            mechanistic_content = "low"
            mechanistic_content_basis = "none"

        scaffold_density = _safe_fraction(
            scaffold_edge_count,
            scientific_edge_count,
        )
        navigation_fraction = _safe_fraction(
            scaffold_edge_count
            + alignment_edge_count,
            hop_count,
        )
        reverse_fraction = _safe_fraction(
            reverse_edge_count,
            hop_count,
        )
        candidate_fraction = _safe_fraction(
            candidate_edge_count,
            hop_count,
        )

        endpoint_pair = path_row.get(
            "endpoint_pair"
        )
        if isinstance(endpoint_pair, dict):
            endpoint_tier = int(
                endpoint_pair.get(
                    "semantic_tier",
                    99,
                )
            )
            raw_pair_score = (
                endpoint_pair.get(
                    "pair_score"
                )
            )
            pair_score = (
                float(raw_pair_score)
                if raw_pair_score is not None
                else None
            )
        else:
            endpoint_tier = 99
            pair_score = None

        if endpoint_tier == 0:
            endpoint_relevance = "exact"
        elif endpoint_tier == 1:
            endpoint_relevance = "mixed"
        elif endpoint_tier == 2:
            endpoint_relevance = "semantic"
        else:
            endpoint_relevance = "unknown"

        visited_papers = tuple(
            sorted({
                str(item)
                for item
                in path_row.get(
                    "visited_paper_ids",
                    path_row.get(
                        "source_paper_ids",
                        [],
                    ),
                )
                if str(item).strip()
            })
        )
        supporting_papers = tuple(
            sorted({
                str(item)
                for item
                in path_row.get(
                    "supporting_paper_ids",
                    [],
                )
                if str(item).strip()
            })
        )
        hub_scope_papers = tuple(
            sorted({
                str(item)
                for item
                in path_row.get(
                    "hub_scope_paper_ids",
                    [],
                )
                if str(item).strip()
            })
        )

        visited_count = int(
            path_row.get(
                "visited_paper_count",
                len(visited_papers),
            )
        )

        shared_entity = (
            self._shared_entity_bridge(
                path_row
            )
        )

        if candidate_edge_count > 0:
            path_type = (
                PATH_TYPE_CANDIDATE
            )
        elif (
            shared_entity
            and alignment_edge_count == 0
        ):
            path_type = (
                PATH_TYPE_SHARED_ENTITY
            )
        elif (
            visited_count >= 2
            and alignment_edge_count > 0
            and mechanism_edge_count > 0
        ):
            path_type = (
                PATH_TYPE_CROSS_PAPER_MECHANISTIC
            )
        elif (
            mechanism_edge_count > 0
            and alignment_edge_count == 0
        ):
            path_type = (
                PATH_TYPE_DIRECT_MECHANISTIC
            )
        elif (
            visited_count >= 2
            and alignment_edge_count > 0
        ):
            path_type = (
                PATH_TYPE_CROSS_PAPER_BRIDGE
            )
        else:
            path_type = (
                PATH_TYPE_SCAFFOLD
            )

        tags: set[str] = set()
        if endpoint_tier == 0:
            tags.add("exact_endpoints")
        elif endpoint_tier == 1:
            tags.add("one_exact_endpoint")
        elif endpoint_tier == 2:
            tags.add("semantic_endpoints")

        if visited_count >= 2:
            tags.add("cross_paper")
        if alignment_edge_count > 0:
            tags.add("uses_alignment")
        if candidate_edge_count > 0:
            tags.add("uses_candidate")
        if shared_entity:
            tags.add("shared_entity_bridge")
        if reverse_fraction >= 0.25:
            tags.add("reverse_heavy")
        if navigation_fraction >= 0.60:
            tags.add("navigation_heavy")
        if mechanism_edge_count > 0:
            tags.add("mechanism_edge_bearing")
        if mechanism_node_count > 0:
            tags.add("mechanism_node_bearing")
        if mechanism_bearing:
            tags.add("mechanism_bearing")
        if mechanistic_content == "high":
            tags.add("mechanism_rich")

        return PathQuality(
            path_type=path_type,
            path_structure_type=path_type,
            path_tags=tuple(sorted(tags)),
            endpoint_semantic_tier=(
                endpoint_tier
            ),
            endpoint_pair_score=pair_score,
            mechanism_edge_count=(
                mechanism_edge_count
            ),
            mechanism_node_count=(
                mechanism_node_count
            ),
            mechanism_node_ids=tuple(
                mechanism_node_ids
            ),
            content_node_count=(
                content_node_count
            ),
            observation_edge_count=(
                observation_edge_count
            ),
            scaffold_edge_count=(
                scaffold_edge_count
            ),
            alignment_edge_count=(
                alignment_edge_count
            ),
            other_scientific_edge_count=(
                other_edge_count
            ),
            mechanistic_density=(
                mechanistic_density
            ),
            mechanistic_edge_density=(
                mechanistic_edge_density
            ),
            mechanistic_node_density=(
                mechanistic_node_density
            ),
            mechanistic_content_score=(
                mechanistic_content_score
            ),
            mechanism_bearing=(
                mechanism_bearing
            ),
            mechanism_content_sources=tuple(
                mechanism_content_sources
            ),
            mechanistic_content_basis=(
                mechanistic_content_basis
            ),
            scaffold_density=(
                scaffold_density
            ),
            navigation_edge_fraction=(
                navigation_fraction
            ),
            reverse_fraction=(
                reverse_fraction
            ),
            candidate_fraction=(
                candidate_fraction
            ),
            endpoint_relevance=(
                endpoint_relevance
            ),
            mechanistic_content=(
                mechanistic_content
            ),
            navigation_burden=_band(
                navigation_fraction,
                medium=0.40,
                high=0.70,
            ),
            reverse_burden=_band(
                reverse_fraction,
                medium=0.20,
                high=0.40,
            ),
            visited_paper_count=(
                visited_count
            ),
            visited_paper_ids=(
                visited_papers
            ),
            supporting_paper_ids=(
                supporting_papers
            ),
            hub_scope_paper_ids=(
                hub_scope_papers
            ),
            shared_entity_bridge=(
                shared_entity
            ),
        )
