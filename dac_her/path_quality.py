from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx


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

_SCAFFOLD_RELATIONS = {
    "APPLIES_TO",
    "SUPPORTED_ON",
    "CATALYZES",
    "ALIGNS_TO_REGISTRY_ENTITY",
    "HAS_PAPER_MENTION",
    "INVOLVES_INTERMEDIATE",
    "PART_OF",
    "HAS_COMPONENT",
    "HAS_SUPPORT",
}

_SHARED_ENTITY_TYPES = {
    "Catalyst",
    "CatalystModel",
    "Material",
    "Support",
    "Metal",
    "CoordinationMotif",
}

_MECHANISM_MARKERS = (
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
) -> str:
    edge_class = str(
        step.get("edge_class", "")
    ).strip().lower()
    relation = str(
        step.get("relation", "")
    ).strip().upper()

    if edge_class in _ALIGNMENT_CLASSES:
        return "alignment"

    if relation in _SCAFFOLD_RELATIONS:
        return "scaffold"

    if any(
        marker in relation
        for marker in _MECHANISM_MARKERS
    ):
        return "mechanism"

    if any(
        marker in relation
        for marker in _OBSERVATION_MARKERS
    ):
        return "observation"

    return "scientific_other"


@dataclass(frozen=True)
class PathQuality:
    path_type: str
    path_tags: tuple[str, ...]
    endpoint_semantic_tier: int
    endpoint_pair_score: float | None
    mechanism_edge_count: int
    observation_edge_count: int
    scaffold_edge_count: int
    alignment_edge_count: int
    other_scientific_edge_count: int
    mechanistic_density: float
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
    ) -> None:
        self.graph = graph

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
            str(
                self.graph.nodes[node_id].get(
                    "type",
                    "",
                )
            )
            in _SHARED_ENTITY_TYPES
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
            role = relation_role(step)
            role_counts[role] += 1

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

        mechanistic_density = _safe_fraction(
            mechanism_edge_count,
            scientific_edge_count,
        )
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
        if mechanistic_density >= 0.25:
            tags.add("mechanism_rich")

        return PathQuality(
            path_type=path_type,
            path_tags=tuple(sorted(tags)),
            endpoint_semantic_tier=(
                endpoint_tier
            ),
            endpoint_pair_score=pair_score,
            mechanism_edge_count=(
                mechanism_edge_count
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
            mechanistic_content=_band(
                mechanistic_density,
                medium=0.10,
                high=0.30,
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
