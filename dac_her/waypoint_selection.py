from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


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
    default: float = 0.0,
) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def match_tier(match: dict[str, Any]) -> int:
    """Semantic waypoint tier: exact/direct, contains, then embedding-only."""
    if (
        _as_bool(match.get("direct_node_id", False))
        or _as_bool(match.get("exact_label_match", False))
    ):
        return 0
    if _as_bool(
        match.get(
            "label_contains_query",
            False,
        )
    ):
        return 1
    return 2


@dataclass(frozen=True)
class WaypointDiagnostic:
    node_id: str
    label: str
    semantic_similarity: float
    ranking_score: float
    semantic_tier: int
    exact_label_match: bool
    label_contains_query: bool
    selected: bool
    waypoint_rank: int | None
    selection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WaypointSelector:
    """Select semantic-stop candidates without letting graph cost redefine the stop.

    The selector intentionally uses only query-to-node relevance. Graph reachability
    is evaluated later by TraversalEngine for each source/target endpoint pair.
    """

    def select(
        self,
        matches: Iterable[dict[str, Any]],
        *,
        top_k: int = 8,
    ) -> tuple[
        list[WaypointDiagnostic],
        list[WaypointDiagnostic],
    ]:
        rows = [dict(item) for item in matches]
        diagnostics: list[WaypointDiagnostic] = []

        for row in rows:
            direct = _as_bool(
                row.get("direct_node_id", False)
            )
            similarity = _as_float(
                row.get("semantic_similarity"),
                1.0 if direct else 0.0,
            )
            ranking_score = _as_float(
                row.get("ranking_score"),
                similarity,
            )
            diagnostics.append(
                WaypointDiagnostic(
                    node_id=str(row["node_id"]),
                    label=str(
                        row.get("label")
                        or row["node_id"]
                    ),
                    semantic_similarity=similarity,
                    ranking_score=ranking_score,
                    semantic_tier=match_tier(row),
                    exact_label_match=(
                        direct
                        or _as_bool(
                            row.get(
                                "exact_label_match",
                                False,
                            )
                        )
                    ),
                    label_contains_query=_as_bool(
                        row.get(
                            "label_contains_query",
                            False,
                        )
                    ),
                    selected=False,
                    waypoint_rank=None,
                    selection_reason="eligible",
                )
            )

        diagnostics.sort(
            key=lambda item: (
                item.semantic_tier,
                -item.semantic_similarity,
                -item.ranking_score,
                item.node_id,
            )
        )

        selected: list[WaypointDiagnostic] = []
        selected_ids: set[str] = set()
        for item in diagnostics:
            if len(selected) >= max(0, top_k):
                break
            if item.node_id in selected_ids:
                continue
            payload = item.to_dict()
            payload["selected"] = True
            payload["waypoint_rank"] = len(selected) + 1
            payload["selection_reason"] = "selected_waypoint"
            chosen = WaypointDiagnostic(**payload)
            selected.append(chosen)
            selected_ids.add(item.node_id)

        selected_by_id = {
            item.node_id: item
            for item in selected
        }
        updated: list[WaypointDiagnostic] = []
        for item in diagnostics:
            if item.node_id in selected_by_id:
                updated.append(
                    selected_by_id[item.node_id]
                )
            else:
                updated.append(item)

        return selected, updated


def waypoint_relevance_pool(
    paths: Iterable[dict[str, Any]],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    """Keep the best waypoint tiers needed to supply an agent-facing bundle.

    If the best available waypoint tier already has at least ``top_k`` paths,
    lower-relevance waypoint tiers cannot enter the diversity selector. If not,
    the next tier is admitted, and so on. This makes waypoint relevance a hard
    semantic priority while still allowing fallback when the best stops cannot
    provide enough valid paths.
    """
    rows = [dict(row) for row in paths]
    if top_k <= 0 or not rows:
        return [], ()

    tiers = sorted({
        int(
            (
                row.get("waypoint")
                if isinstance(
                    row.get("waypoint"),
                    dict,
                )
                else {}
            ).get("semantic_tier", 99)
        )
        for row in rows
    })

    allowed: list[int] = []
    count = 0
    for tier in tiers:
        allowed.append(tier)
        count += sum(
            1
            for row in rows
            if int(
                (
                    row.get("waypoint")
                    if isinstance(
                        row.get("waypoint"),
                        dict,
                    )
                    else {}
                ).get("semantic_tier", 99)
            )
            == tier
        )
        if count >= top_k:
            break

    allowed_set = set(allowed)
    pooled = [
        row
        for row in rows
        if int(
            (
                row.get("waypoint")
                if isinstance(
                    row.get("waypoint"),
                    dict,
                )
                else {}
            ).get("semantic_tier", 99)
        )
        in allowed_set
    ]
    return pooled, tuple(allowed)
