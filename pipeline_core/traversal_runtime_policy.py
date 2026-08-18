"""Shared deterministic graph-traversal runtime policy helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable


DEFAULT_MAX_DEPTH = 8
DEFAULT_SEMANTIC_STOP_MAX_DEPTH = 12
DEFAULT_SEMANTIC_STOP_ABLATION_MAX_TRIPLES = 512


@dataclass(frozen=True)
class SemanticStopAblationGuard:
    enabled: bool
    applied: bool
    max_triples: int
    waypoint_count: int
    source_count_before: int
    target_count_before: int
    source_count_after: int
    target_count_after: int
    endpoint_pair_upper_bound: int
    traversal_triple_upper_bound: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_semantic_stop_max_depth(
    *,
    base_max_depth: int,
    semantic_stop_max_depth: int | None,
    base_max_depth_explicit: bool,
) -> int:
    """Resolve semantic-stop depth while preserving explicit legacy overrides."""
    if semantic_stop_max_depth is not None:
        return max(1, int(semantic_stop_max_depth))
    if base_max_depth_explicit:
        return max(1, int(base_max_depth))
    return DEFAULT_SEMANTIC_STOP_MAX_DEPTH


def guard_semantic_stop_ablation(
    source_matches: Iterable[dict[str, Any]],
    target_matches: Iterable[dict[str, Any]],
    *,
    waypoint_count: int,
    max_triples: int = DEFAULT_SEMANTIC_STOP_ABLATION_MAX_TRIPLES,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    SemanticStopAblationGuard,
]:
    """Bound the debug all-pairs semantic-stop Cartesian product."""
    sources = [dict(row) for row in source_matches]
    targets = [dict(row) for row in target_matches]
    waypoints = max(1, int(waypoint_count))
    budget = int(max_triples)

    if budget <= 0:
        pair_bound = len(sources) * len(targets)
        diagnostic = SemanticStopAblationGuard(
            enabled=False,
            applied=False,
            max_triples=budget,
            waypoint_count=waypoints,
            source_count_before=len(sources),
            target_count_before=len(targets),
            source_count_after=len(sources),
            target_count_after=len(targets),
            endpoint_pair_upper_bound=pair_bound,
            traversal_triple_upper_bound=pair_bound * waypoints,
        )
        return sources, targets, diagnostic

    pair_budget = max(1, budget // waypoints)
    side_cap = max(1, int(math.floor(math.sqrt(pair_budget))))
    guarded_sources = sources[:side_cap]
    guarded_targets = targets[:side_cap]

    pair_bound = len(guarded_sources) * len(guarded_targets)
    triple_bound = pair_bound * waypoints
    applied = (
        len(guarded_sources) < len(sources)
        or len(guarded_targets) < len(targets)
    )

    diagnostic = SemanticStopAblationGuard(
        enabled=True,
        applied=applied,
        max_triples=budget,
        waypoint_count=waypoints,
        source_count_before=len(sources),
        target_count_before=len(targets),
        source_count_after=len(guarded_sources),
        target_count_after=len(guarded_targets),
        endpoint_pair_upper_bound=pair_bound,
        traversal_triple_upper_bound=triple_bound,
    )
    return guarded_sources, guarded_targets, diagnostic
