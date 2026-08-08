from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


def _endpoint_signature(
    row: dict[str, Any],
) -> tuple[str, str]:
    pair = row.get("endpoint_pair")
    if isinstance(pair, dict):
        source = str(
            pair.get("source_node_id")
            or row.get("source")
            or ""
        )
        target = str(
            pair.get("target_node_id")
            or row.get("target")
            or ""
        )
        return source, target

    source_match = row.get("source_match")
    target_match = row.get("target_match")
    source = (
        str(source_match.get("node_id", ""))
        if isinstance(source_match, dict)
        else str(row.get("source", ""))
    )
    target = (
        str(target_match.get("node_id", ""))
        if isinstance(target_match, dict)
        else str(row.get("target", ""))
    )
    return source, target


def _paper_signature(
    row: dict[str, Any],
) -> tuple[str, ...]:
    papers = row.get(
        "visited_paper_ids",
        row.get(
            "source_paper_ids",
            [],
        ),
    )
    return tuple(
        sorted({
            str(item)
            for item in papers
            if str(item).strip()
        })
    )


def _edge_signature(
    row: dict[str, Any],
) -> frozenset[str]:
    edge_ids: set[str] = set()

    for index, step in enumerate(
        row.get("steps", [])
    ):
        if not isinstance(step, dict):
            continue

        edge_id = str(
            step.get(
                "selected_original_edge_id",
                "",
            )
        ).strip()
        if not edge_id:
            edge_id = str(
                step.get(
                    "navigation_edge_id",
                    "",
                )
            ).strip()
        if not edge_id:
            edge_id = (
                f"{step.get('source', '')}"
                f"|{step.get('relation', '')}"
                f"|{step.get('target', '')}"
                f"|{index}"
            )

        edge_ids.add(edge_id)

    return frozenset(edge_ids)


def edge_jaccard(
    left: frozenset[str],
    right: frozenset[str],
) -> float:
    if not left and not right:
        return 0.0

    union = left | right
    if not union:
        return 0.0

    return (
        len(left & right)
        / len(union)
    )


def _scientific_endpoints(
    step: dict[str, Any],
) -> tuple[str, str]:
    direction = str(
        step.get(
            "scientific_direction",
            "",
        )
    ).strip()

    if " -> " in direction:
        left, right = direction.split(
            " -> ",
            1,
        )
        return (
            left.strip(),
            right.strip(),
        )

    source = str(
        step.get("source", "")
    )
    target = str(
        step.get("target", "")
    )

    if (
        str(
            step.get(
                "traversal_direction",
                "forward",
            )
        )
        == "reverse"
    ):
        return target, source

    return source, target


def render_step_safe(
    step: dict[str, Any],
) -> list[str]:
    """Render traversal without reversing scientific semantics in prose."""
    source = str(
        step.get("source", "")
    )
    target = str(
        step.get("target", "")
    )
    relation = str(
        step.get(
            "relation",
            "RELATED_TO",
        )
    )
    traversal_direction = str(
        step.get(
            "traversal_direction",
            "forward",
        )
    )

    scientific_source, scientific_target = (
        _scientific_endpoints(step)
    )

    if traversal_direction == "reverse":
        return [
            (
                "TRAVERSE "
                f"{source} -> {target} "
                "[reverse navigation only]"
            ),
            (
                "    SCIENTIFIC "
                f"{scientific_source} "
                f"-- {relation} --> "
                f"{scientific_target}"
            ),
        ]

    return [
        (
            "SCIENTIFIC "
            f"{scientific_source} "
            f"-- {relation} --> "
            f"{scientific_target}"
        )
    ]


@dataclass(frozen=True)
class PathBundlePolicy:
    max_per_endpoint_pair: int = 2
    max_per_paper_signature: int = 2
    max_edge_jaccard: float = 0.80

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PathBundleDiagnostic:
    path_id: str
    base_rank: int
    selected: bool
    bundle_rank: int | None
    selection_pass: str
    endpoint_signature: tuple[str, str]
    paper_signature: tuple[str, ...]
    max_edge_jaccard_with_selected: float
    most_overlapping_path_id: str | None
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["endpoint_signature"] = list(
            self.endpoint_signature
        )
        payload["paper_signature"] = list(
            self.paper_signature
        )
        payload["rejection_reasons"] = list(
            self.rejection_reasons
        )
        return payload


@dataclass
class PathBundleSelection:
    selected_paths: list[dict[str, Any]]
    diagnostics: list[
        PathBundleDiagnostic
    ]
    policy: PathBundlePolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "selected_path_ids": [
                str(row["path_id"])
                for row in self.selected_paths
            ],
            "diagnostics": [
                item.to_dict()
                for item in self.diagnostics
            ],
        }


class PathBundleSelector:
    """Choose a diverse, deterministic agent-facing subset of paths.

    Candidate ordering is preserved as the scientific preference signal.
    Diversity constraints are applied first, then relaxed only if needed
    to fill the requested bundle size.
    """

    def __init__(
        self,
        *,
        policy: PathBundlePolicy | None = None,
    ) -> None:
        self.policy = (
            policy
            or PathBundlePolicy()
        )

    def _max_overlap(
        self,
        edge_set: frozenset[str],
        selected: list[
            dict[str, Any]
        ],
        edge_sets: dict[
            str,
            frozenset[str],
        ],
    ) -> tuple[float, str | None]:
        best = 0.0
        best_path: str | None = None

        for row in selected:
            path_id = str(
                row["path_id"]
            )
            overlap = edge_jaccard(
                edge_set,
                edge_sets[path_id],
            )
            if (
                overlap > best
                or (
                    overlap == best
                    and overlap > 0
                    and (
                        best_path is None
                        or path_id < best_path
                    )
                )
            ):
                best = overlap
                best_path = path_id

        return best, best_path

    def select(
        self,
        candidate_paths: Iterable[
            dict[str, Any]
        ],
        *,
        top_k: int,
    ) -> PathBundleSelection:
        rows = [
            dict(row)
            for row in candidate_paths
        ]

        if top_k <= 0 or not rows:
            return PathBundleSelection(
                selected_paths=[],
                diagnostics=[],
                policy=self.policy,
            )

        base_rank = {
            str(row["path_id"]): index
            for index, row in enumerate(
                rows,
                start=1,
            )
        }
        endpoint_signatures = {
            str(row["path_id"]): (
                _endpoint_signature(row)
            )
            for row in rows
        }
        paper_signatures = {
            str(row["path_id"]): (
                _paper_signature(row)
            )
            for row in rows
        }
        edge_sets = {
            str(row["path_id"]): (
                _edge_signature(row)
            )
            for row in rows
        }

        selected: list[
            dict[str, Any]
        ] = []
        selected_ids: set[str] = set()
        endpoint_counts: dict[
            tuple[str, str],
            int,
        ] = {}
        paper_counts: dict[
            tuple[str, ...],
            int,
        ] = {}

        diagnostic_state: dict[
            str,
            dict[str, Any],
        ] = {}

        def attempt(
            row: dict[str, Any],
            *,
            pass_name: str,
            enforce_endpoint: bool,
            enforce_papers: bool,
            enforce_overlap: bool,
        ) -> bool:
            path_id = str(
                row["path_id"]
            )
            if path_id in selected_ids:
                return False

            endpoint = (
                endpoint_signatures[path_id]
            )
            papers = (
                paper_signatures[path_id]
            )
            max_overlap, overlap_path = (
                self._max_overlap(
                    edge_sets[path_id],
                    selected,
                    edge_sets,
                )
            )

            reasons: list[str] = []

            if (
                enforce_endpoint
                and endpoint_counts.get(
                    endpoint,
                    0,
                )
                >= self.policy.max_per_endpoint_pair
            ):
                reasons.append(
                    "endpoint_pair_cap"
                )

            if (
                enforce_papers
                and paper_counts.get(
                    papers,
                    0,
                )
                >= self.policy.max_per_paper_signature
            ):
                reasons.append(
                    "paper_signature_cap"
                )

            if (
                enforce_overlap
                and selected
                and max_overlap
                > self.policy.max_edge_jaccard
            ):
                reasons.append(
                    "edge_overlap"
                )

            diagnostic_state[path_id] = {
                "base_rank": (
                    base_rank[path_id]
                ),
                "selected": not reasons,
                "bundle_rank": (
                    len(selected) + 1
                    if not reasons
                    else None
                ),
                "selection_pass": pass_name,
                "endpoint_signature": (
                    endpoint
                ),
                "paper_signature": papers,
                "max_edge_jaccard_with_selected": (
                    max_overlap
                ),
                "most_overlapping_path_id": (
                    overlap_path
                ),
                "rejection_reasons": tuple(
                    reasons
                ),
            }

            if reasons:
                return False

            selected.append(row)
            selected_ids.add(path_id)
            endpoint_counts[endpoint] = (
                endpoint_counts.get(
                    endpoint,
                    0,
                )
                + 1
            )
            paper_counts[papers] = (
                paper_counts.get(
                    papers,
                    0,
                )
                + 1
            )
            return True

        passes = [
            (
                "strict_diversity",
                True,
                True,
                True,
            ),
            (
                "relaxed_edge_overlap",
                True,
                True,
                False,
            ),
            (
                "relaxed_all_diversity",
                False,
                False,
                False,
            ),
        ]

        for (
            pass_name,
            enforce_endpoint,
            enforce_papers,
            enforce_overlap,
        ) in passes:
            for row in rows:
                if len(selected) >= top_k:
                    break

                attempt(
                    row,
                    pass_name=pass_name,
                    enforce_endpoint=(
                        enforce_endpoint
                    ),
                    enforce_papers=(
                        enforce_papers
                    ),
                    enforce_overlap=(
                        enforce_overlap
                    ),
                )

            if len(selected) >= top_k:
                break

        diagnostics: list[
            PathBundleDiagnostic
        ] = []

        for row in rows:
            path_id = str(
                row["path_id"]
            )
            state = diagnostic_state.get(
                path_id
            )

            if state is None:
                state = {
                    "base_rank": (
                        base_rank[path_id]
                    ),
                    "selected": False,
                    "bundle_rank": None,
                    "selection_pass": (
                        "not_examined_after_bundle_filled"
                    ),
                    "endpoint_signature": (
                        endpoint_signatures[
                            path_id
                        ]
                    ),
                    "paper_signature": (
                        paper_signatures[
                            path_id
                        ]
                    ),
                    "max_edge_jaccard_with_selected": (
                        0.0
                    ),
                    "most_overlapping_path_id": (
                        None
                    ),
                    "rejection_reasons": (),
                }

            diagnostics.append(
                PathBundleDiagnostic(
                    path_id=path_id,
                    **state,
                )
            )

        diagnostic_by_id = {
            item.path_id: item
            for item in diagnostics
        }

        enriched: list[
            dict[str, Any]
        ] = []
        for row in selected:
            path_id = str(
                row["path_id"]
            )
            copied = dict(row)
            copied[
                "bundle_selection"
            ] = diagnostic_by_id[
                path_id
            ].to_dict()
            enriched.append(copied)

        return PathBundleSelection(
            selected_paths=enriched,
            diagnostics=diagnostics,
            policy=self.policy,
        )
