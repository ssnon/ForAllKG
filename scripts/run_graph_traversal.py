from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

from domains.registry import get_domain_profile
from domains.extraction_registry import (
    get_extraction_adapter,
)
from pipeline_core.discovery.direct_concept import (
    DirectConceptHitSelector,
)
from pipeline_core.discovery.endpoint_selection import (
    EndpointPairSelector,
)
from pipeline_core.discovery.path_bundle import (
    PathBundlePolicy,
    PathBundleSelector,
    render_step_safe,
)
from pipeline_core.discovery.path_quality import (
    PathQualityScorer,
)
from pipeline_core.discovery.node_mapping import (
    NodeMapper,
    QueryConcept,
)
from pipeline_core.discovery.traversal_engine import (
    TraversalConstraints,
    TraversalEngine,
)
from pipeline_core.traversal_runtime_policy import (
    DEFAULT_SEMANTIC_STOP_ABLATION_MAX_TRIPLES,
    guard_semantic_stop_ablation,
    resolve_semantic_stop_max_depth,
)
from pipeline_core.discovery.waypoint_selection import (
    WaypointSelector,
    waypoint_relevance_pool,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _types(
    raw: str | None,
) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in (raw or "").split(",")
        if item.strip()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map scientific concepts to nodes, "
            "select graph-reachable endpoint pairs, "
            "and run policy-constrained traversal."
        )
    )
    parser.add_argument(
        "--corpus-id",
        required=True,
    )
    parser.add_argument(
        "--domain-profile",
        required=True,
    )
    parser.add_argument(
        "--data-root",
        default=None,
    )
    parser.add_argument(
        "--mode",
        choices=(
            "evidence",
            "mechanism",
            "exploratory",
        ),
        default="mechanism",
    )
    parser.add_argument(
        "--algorithm",
        choices=(
            "shortest",
            "top_n",
            "bounded_dfs",
            "semantic_stop",
        ),
        default="top_n",
    )

    source = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )
    source.add_argument(
        "--source",
        default=None,
    )
    source.add_argument(
        "--source-node-id",
        default=None,
    )

    target = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )
    target.add_argument(
        "--target",
        default=None,
    )
    target.add_argument(
        "--target-node-id",
        default=None,
    )

    parser.add_argument(
        "--stop",
        default=None,
    )
    parser.add_argument(
        "--stop-node-id",
        default=None,
    )
    parser.add_argument(
        "--source-types",
        default=None,
    )
    parser.add_argument(
        "--target-types",
        default=None,
    )
    parser.add_argument(
        "--stop-types",
        default=None,
    )
    parser.add_argument(
        "--node-map-k",
        type=int,
        default=20,
        help=(
            "Semantic endpoint candidate pool size. "
            "v2.4.1 defaults to 20."
        ),
    )
    parser.add_argument(
        "--endpoint-pair-k",
        type=int,
        default=12,
        help=(
            "Maximum graph-reachable endpoint pairs "
            "sent to TraversalEngine."
        ),
    )
    parser.add_argument(
        "--endpoint-paper-novelty-bonus",
        type=float,
        default=0.01,
        help=(
            "Per-side bounded bonus for a previously unseen source/target paper. "
            "Semantic tier remains a hard priority."
        ),
    )
    parser.add_argument(
        "--disable-endpoint-paper-diversity",
        action="store_true",
        help="Disable RDP1 endpoint paper-diversity reranking for ablation.",
    )
    parser.add_argument(
        "--disable-endpoint-selector",
        action="store_true",
        help=(
            "Use v2.4.0 all-pairs behavior "
            "for ablation/debugging."
        ),
    )
    parser.add_argument(
        "--waypoint-k",
        type=int,
        default=8,
        help=(
            "Maximum semantic-stop waypoint candidates after "
            "exact/contains/semantic tiering."
        ),
    )
    parser.add_argument(
        "--disable-waypoint-selector",
        action="store_true",
        help=(
            "Use all mapped semantic-stop candidates without "
            "v2.4.6 waypoint tier selection."
        ),
    )
    parser.add_argument(
        "--direct-hit-k",
        type=int,
        default=5,
        help=(
            "Maximum same-node DirectConceptHit records returned "
            "alongside traversal paths."
        ),
    )
    parser.add_argument(
        "--direct-hit-min-similarity",
        type=float,
        default=0.60,
        help=(
            "Minimum similarity required on both query sides for "
            "a semantic DirectConceptHit."
        ),
    )
    parser.add_argument(
        "--disable-direct-concept-hits",
        action="store_true",
        help=(
            "Disable the v2.4.6 same-node direct-answer channel."
        ),
    )
    parser.add_argument(
        "--bundle-max-per-endpoint-pair",
        type=int,
        default=2,
        help=(
            "Preferred maximum returned paths sharing "
            "the same endpoint pair before diversity relaxation."
        ),
    )
    parser.add_argument(
        "--bundle-max-per-paper-signature",
        type=int,
        default=2,
        help=(
            "Preferred maximum returned paths sharing "
            "the same visited-paper set before diversity relaxation."
        ),
    )
    parser.add_argument(
        "--bundle-max-edge-jaccard",
        type=float,
        default=0.80,
        help=(
            "Preferred maximum edge-set Jaccard overlap "
            "between returned paths before relaxation."
        ),
    )
    parser.add_argument(
        "--disable-path-quality-ranking",
        action="store_true",
        help=(
            "Disable RDP3 mechanism-mode scientific path-quality ordering. "
            "Endpoint relevance remains primary."
        ),
    )
    parser.add_argument(
        "--disable-bundle-coverage-first",
        action="store_true",
        help=(
            "Disable RDP2 paper-coverage/signature-first strict passes while "
            "retaining the existing PathBundleSelector caps and overlap rules."
        ),
    )
    parser.add_argument(
        "--disable-bundle-selector",
        action="store_true",
        help=(
            "Disable diversity-aware PathBundleSelector "
            "and use the previous top-k slicing behavior."
        ),
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--allow-candidate-endpoints",
        action="store_true",
    )
    parser.add_argument(
        "--allow-alignment-hub-endpoints",
        action="store_true",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help=(
            "Maximum path depth for ordinary traversal. "
            "For backward compatibility, an explicitly supplied value "
            "also overrides semantic-stop depth unless "
            "--semantic-stop-max-depth is supplied."
        ),
    )
    parser.add_argument(
        "--semantic-stop-max-depth",
        type=int,
        default=None,
        help=(
            "Maximum total source→waypoint→target depth for semantic-stop. "
            "Defaults to 12 when --max-depth is not explicitly supplied."
        ),
    )
    parser.add_argument(
        "--semantic-stop-ablation-max-triples",
        type=int,
        default=(
            DEFAULT_SEMANTIC_STOP_ABLATION_MAX_TRIPLES
        ),
        help=(
            "Safety cap for source×target×waypoint combinations when "
            "semantic_stop is used with --disable-endpoint-selector. "
            "Set <=0 to disable the guard."
        ),
    )
    parser.add_argument(
        "--max-alignment-edges",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--min-scientific-edges",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--navigation-graphml",
        default=None,
    )
    parser.add_argument(
        "--index-dir",
        default=None,
    )
    parser.add_argument(
        "--device",
        default=None,
    )
    parser.add_argument(
        "--include-candidate-paths",
        action="store_true",
        help=(
            "Persist the full pre-bundle candidate path pool in traversal JSON. "
            "Useful for DiscoveryBundle construction; disabled by default to keep artifacts compact."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    return parser.parse_args()


def _map(
    mapper: NodeMapper,
    text: str,
    allowed_types: tuple[str, ...],
    args: argparse.Namespace,
) -> list[dict]:
    return [
        item.to_dict()
        for item in mapper.map(
            QueryConcept(
                text=text,
                allowed_node_types=(
                    allowed_types
                ),
                allow_candidates=(
                    args.allow_candidate_endpoints
                ),
                allow_alignment_hubs=(
                    args.allow_alignment_hub_endpoints
                ),
                top_k=args.node_map_k,
                min_similarity=(
                    args.min_similarity
                ),
            )
        )
    ]


def _path_scientific_quality_key(
    row: dict,
) -> tuple:
    quality = row.get(
        "path_quality",
        {},
    )
    if not isinstance(quality, dict):
        quality = {}

    mechanism_band = str(
        quality.get(
            "mechanistic_content",
            "unknown",
        )
    ).strip().lower()
    mechanism_rank = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }.get(
        mechanism_band,
        3,
    )

    try:
        mechanism_score = float(
            quality.get(
                "mechanistic_content_score",
                0.0,
            )
        )
    except (TypeError, ValueError):
        mechanism_score = 0.0

    try:
        navigation_fraction = float(
            quality.get(
                "navigation_edge_fraction",
                1.0,
            )
        )
    except (TypeError, ValueError):
        navigation_fraction = 1.0

    try:
        reverse_fraction = float(
            quality.get(
                "reverse_fraction",
                1.0,
            )
        )
    except (TypeError, ValueError):
        reverse_fraction = 1.0

    return (
        mechanism_rank,
        -mechanism_score,
        navigation_fraction,
        reverse_fraction,
    )


def _path_sort_key(
    row: dict,
    *,
    quality_aware: bool = False,
) -> tuple:
    waypoint = row.get("waypoint")
    if isinstance(waypoint, dict):
        waypoint_tier = int(
            waypoint.get(
                "semantic_tier",
                99,
            )
        )
        waypoint_similarity = float(
            waypoint.get(
                "semantic_similarity",
                0.0,
            )
        )
        waypoint_rank = int(
            waypoint.get(
                "waypoint_rank",
                10**9,
            )
            or 10**9
        )
    else:
        waypoint_tier = 99
        waypoint_similarity = 0.0
        waypoint_rank = 10**9

    pair = row.get("endpoint_pair")
    if isinstance(pair, dict):
        tier = int(
            pair.get(
                "semantic_tier",
                99,
            )
        )
        pair_score = float(
            pair.get(
                "pair_score",
                0.0,
            )
        )
    else:
        tier = 99
        pair_score = 0.0

    quality_key = (
        _path_scientific_quality_key(
            row
        )
        if quality_aware
        else (
            0,
            0.0,
            0.0,
            0.0,
        )
    )

    return (
        waypoint_tier,
        -waypoint_similarity,
        waypoint_rank,
        tier,
        -pair_score,
        *quality_key,
        float(row["total_cost"]),
        int(row["hop_count"]),
        str(row["path_id"]),
    )


def _flag_was_explicit(
    flag: str,
    argv: list[str] | None = None,
) -> bool:
    rows = list(sys.argv[1:] if argv is None else argv)
    return any(
        item == flag
        or item.startswith(flag + "=")
        for item in rows
    )


def main() -> None:
    args = parse_args()
    domain_profile = get_domain_profile(
        args.domain_profile
    )
    extraction_adapter = get_extraction_adapter(
        domain_profile.profile_id
    )
    data_root = Path(
        args.data_root
        or extraction_adapter.default_data_root
    )
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    semantic_stop_max_depth = resolve_semantic_stop_max_depth(
        base_max_depth=args.max_depth,
        semantic_stop_max_depth=args.semantic_stop_max_depth,
        base_max_depth_explicit=_flag_was_explicit(
            "--max-depth"
        ),
    )
    effective_max_depth = (
        semantic_stop_max_depth
        if args.algorithm == "semantic_stop"
        else args.max_depth
    )

    if (
        args.algorithm == "semantic_stop"
        and not (
            args.stop
            or args.stop_node_id
        )
    ):
        raise ValueError(
            "semantic_stop requires --stop "
            "or --stop-node-id."
        )

    mode_root = (
        data_root
        / "corpus"
        / args.corpus_id
        / args.mode
    )
    navigation_root = (
        mode_root / "navigation"
    )
    graph_path = (
        Path(args.navigation_graphml)
        if args.navigation_graphml
        else navigation_root
        / "graph.graphml"
    )
    index_dir = (
        Path(args.index_dir)
        if args.index_dir
        else navigation_root
        / "node_index"
    )

    graph = nx.read_graphml(
        graph_path
    )
    engine = TraversalEngine(
        graph
    )
    quality_scorer = PathQualityScorer(
        graph,
        discovery_semantics=domain_profile.discovery,
    )

    mapper: NodeMapper | None = None

    def get_mapper() -> NodeMapper:
        nonlocal mapper
        if mapper is None:
            mapper = (
                NodeMapper.from_directory(
                    index_dir,
                    device=args.device,
                )
            )
        return mapper

    source_matches = (
        [
            {
                "node_id": (
                    args.source_node_id
                ),
                "direct_node_id": True,
                "exact_label_match": True,
            }
        ]
        if args.source_node_id
        else _map(
            get_mapper(),
            args.source,
            _types(args.source_types),
            args,
        )
    )

    target_matches = (
        [
            {
                "node_id": (
                    args.target_node_id
                ),
                "direct_node_id": True,
                "exact_label_match": True,
            }
        ]
        if args.target_node_id
        else _map(
            get_mapper(),
            args.target,
            _types(args.target_types),
            args,
        )
    )

    if args.stop_node_id:
        stop_matches: list[
            dict | None
        ] = [
            {
                "node_id": (
                    args.stop_node_id
                ),
                "direct_node_id": True,
                "exact_label_match": True,
            }
        ]
    elif args.stop:
        stop_matches = _map(
            get_mapper(),
            args.stop,
            _types(args.stop_types),
            args,
        )
    else:
        stop_matches = [None]

    if not source_matches:
        raise RuntimeError(
            "No source nodes matched."
        )
    if not target_matches:
        raise RuntimeError(
            "No target nodes matched."
        )
    if (
        args.algorithm == "semantic_stop"
        and not stop_matches
    ):
        raise RuntimeError(
            "No semantic-stop nodes matched."
        )

    if args.disable_direct_concept_hits:
        direct_concept_hits: list[dict] = []
    else:
        direct_concept_hits = [
            item.to_dict()
            for item in DirectConceptHitSelector(
                graph,
                min_similarity=(
                    args.direct_hit_min_similarity
                ),
                discovery_semantics=domain_profile.discovery,
            ).select(
                source_matches,
                target_matches,
                top_k=args.direct_hit_k,
            )
        ]

    waypoint_diagnostics: list[dict] = []
    if args.algorithm == "semantic_stop":
        actual_stop_matches = [
            dict(row)
            for row in stop_matches
            if row is not None
        ]
        if args.disable_waypoint_selector:
            waypoint_candidates = [
                {
                    "stop_match": row,
                    "waypoint_diagnostic": None,
                }
                for row in actual_stop_matches
            ]
        else:
            waypoint_selector = WaypointSelector()
            (
                selected_waypoints,
                waypoint_rows,
            ) = waypoint_selector.select(
                actual_stop_matches,
                top_k=args.waypoint_k,
            )
            waypoint_diagnostics = [
                item.to_dict()
                for item in waypoint_rows
            ]
            stop_by_id = {
                str(row["node_id"]): row
                for row in actual_stop_matches
            }
            waypoint_candidates = [
                {
                    "stop_match": stop_by_id[
                        item.node_id
                    ],
                    "waypoint_diagnostic": (
                        item.to_dict()
                    ),
                }
                for item in selected_waypoints
            ]
        if not waypoint_candidates:
            raise RuntimeError(
                "No semantic-stop waypoint candidates remained."
            )
    else:
        waypoint_candidates = [
            {
                "stop_match": None,
                "waypoint_diagnostic": None,
            }
        ]

    constraints = TraversalConstraints(
        mode=args.mode,
        top_k=args.top_k,
        max_depth=args.max_depth,
        semantic_stop_max_depth=(
            semantic_stop_max_depth
        ),
        max_alignment_edges=(
            args.max_alignment_edges
        ),
        min_scientific_edges=(
            args.min_scientific_edges
        ),
        max_expansions=(
            args.max_expansions
        ),
    )

    endpoint_pair_diagnostics: list[
        dict
    ] = []

    semantic_stop_ablation_guard = None

    if args.disable_endpoint_selector:
        endpoint_source_matches = source_matches
        endpoint_target_matches = target_matches
        if args.algorithm == "semantic_stop":
            (
                endpoint_source_matches,
                endpoint_target_matches,
                guard_diagnostic,
            ) = guard_semantic_stop_ablation(
                source_matches,
                target_matches,
                waypoint_count=len(waypoint_candidates),
                max_triples=(
                    args.semantic_stop_ablation_max_triples
                ),
            )
            semantic_stop_ablation_guard = (
                guard_diagnostic.to_dict()
            )
            if guard_diagnostic.applied:
                print(
                    "WARNING: semantic-stop no-selector ablation "
                    "was safety-capped: "
                    f"source {guard_diagnostic.source_count_before}"
                    f"->{guard_diagnostic.source_count_after}, "
                    f"target {guard_diagnostic.target_count_before}"
                    f"->{guard_diagnostic.target_count_after}, "
                    f"triple upper bound="
                    f"{guard_diagnostic.traversal_triple_upper_bound}."
                )

        endpoint_pairs = [
            {
                "source_match": (
                    source_match
                ),
                "target_match": (
                    target_match
                ),
                "pair_diagnostic": None,
            }
            for source_match
            in endpoint_source_matches
            for target_match
            in endpoint_target_matches
            if str(source_match["node_id"])
            != str(target_match["node_id"])
        ]
    else:
        selector = EndpointPairSelector(
            graph,
            paper_novelty_bonus=(
                0.0
                if args.disable_endpoint_paper_diversity
                else args.endpoint_paper_novelty_bonus
            ),
        )
        (
            selected_pairs,
            diagnostics,
        ) = selector.select(
            source_matches,
            target_matches,
            top_k=args.endpoint_pair_k,
            max_depth=effective_max_depth,
        )

        endpoint_pair_diagnostics = [
            item.to_dict()
            for item in diagnostics
        ]

        source_by_id = {
            str(item["node_id"]): item
            for item in source_matches
        }
        target_by_id = {
            str(item["node_id"]): item
            for item in target_matches
        }

        endpoint_pairs = [
            {
                "source_match": (
                    source_by_id[
                        pair.source_node_id
                    ]
                ),
                "target_match": (
                    target_by_id[
                        pair.target_node_id
                    ]
                ),
                "pair_diagnostic": (
                    pair.to_dict()
                ),
            }
            for pair in selected_pairs
        ]

    collected: dict[
        str,
        dict,
    ] = {}

    for endpoint_pair in endpoint_pairs:
        source_match = endpoint_pair[
            "source_match"
        ]
        target_match = endpoint_pair[
            "target_match"
        ]

        for waypoint_candidate in waypoint_candidates:
            stop_match = waypoint_candidate[
                "stop_match"
            ]
            waypoint_diagnostic = (
                waypoint_candidate[
                    "waypoint_diagnostic"
                ]
            )
            stop_node = (
                str(
                    stop_match["node_id"]
                )
                if stop_match
                else None
            )

            for path in engine.traverse(
                str(
                    source_match["node_id"]
                ),
                str(
                    target_match["node_id"]
                ),
                algorithm=args.algorithm,
                constraints=constraints,
                semantic_stop=stop_node,
            ):
                row = path.to_dict()
                row["source_match"] = (
                    source_match
                )
                row["target_match"] = (
                    target_match
                )
                row["stop_match"] = (
                    stop_match
                )
                row["waypoint"] = (
                    waypoint_diagnostic
                )
                row["endpoint_pair"] = (
                    endpoint_pair[
                        "pair_diagnostic"
                    ]
                )
                row["path_quality"] = (
                    quality_scorer.score(
                        row
                    ).to_dict()
                )
                collected[
                    path.path_id
                ] = row

    path_quality_ranking_enabled = (
        args.mode == "mechanism"
        and not args.disable_path_quality_ranking
    )

    all_paths = sorted(
        collected.values(),
        key=lambda row: _path_sort_key(
            row,
            quality_aware=(
                path_quality_ranking_enabled
            ),
        ),
    )

    if (
        args.algorithm == "semantic_stop"
        and not args.disable_waypoint_selector
    ):
        (
            bundle_candidate_paths,
            admitted_waypoint_tiers,
        ) = waypoint_relevance_pool(
            all_paths,
            top_k=args.top_k,
        )
    else:
        bundle_candidate_paths = all_paths
        admitted_waypoint_tiers = ()

    bundle_policy = PathBundlePolicy(
        max_per_endpoint_pair=(
            args.bundle_max_per_endpoint_pair
        ),
        max_per_paper_signature=(
            args.bundle_max_per_paper_signature
        ),
        max_edge_jaccard=(
            args.bundle_max_edge_jaccard
        ),
    )

    if args.disable_bundle_selector:
        paths = bundle_candidate_paths[
            : args.top_k
        ]
        bundle_selection = {
            "enabled": False,
            "policy": bundle_policy.to_dict(),
            "selected_path_ids": [
                str(row["path_id"])
                for row in paths
            ],
            "diagnostics": [],
        }
    else:
        bundle_result = PathBundleSelector(
            policy=bundle_policy,
            coverage_first=(
                not args.disable_bundle_coverage_first
            ),
        ).select(
            bundle_candidate_paths,
            top_k=args.top_k,
        )
        paths = (
            bundle_result.selected_paths
        )
        bundle_selection = {
            "enabled": True,
            "coverage_first_enabled": (
                not args.disable_bundle_coverage_first
            ),
            **bundle_result.to_dict(),
        }

    def type_groups(
        rows: list[dict],
    ) -> dict[str, list[str]]:
        groups: dict[
            str,
            list[str],
        ] = {}
        for row in rows:
            quality = row.get(
                "path_quality",
                {},
            )
            path_type = str(
                quality.get(
                    "path_type",
                    "UNKNOWN",
                )
            )
            groups.setdefault(
                path_type,
                [],
            ).append(
                str(row["path_id"])
            )
        return {
            path_type: path_ids
            for path_type, path_ids
            in sorted(groups.items())
        }

    candidate_path_groups = (
        type_groups(all_paths)
    )
    returned_path_groups = (
        type_groups(paths)
    )

    candidate_path_type_counts = {
        path_type: len(path_ids)
        for path_type, path_ids
        in candidate_path_groups.items()
    }
    returned_path_type_counts = {
        path_type: len(path_ids)
        for path_type, path_ids
        in returned_path_groups.items()
    }

    def path_paper_ids(
        rows: list[dict],
    ) -> list[str]:
        paper_ids: set[str] = set()
        for row in rows:
            values = row.get(
                "visited_paper_ids",
                row.get(
                    "source_paper_ids",
                    [],
                ),
            )
            for value in values:
                paper_id = str(value).strip()
                if paper_id:
                    paper_ids.add(paper_id)
        return sorted(paper_ids)

    candidate_paper_ids = path_paper_ids(
        bundle_candidate_paths
    )
    returned_paper_ids = path_paper_ids(
        paths
    )

    payload = {
        "corpus_id": args.corpus_id,
        "domain_profile_id": domain_profile.profile_id,
        "data_root": str(data_root),
        "mode": args.mode,
        "algorithm": args.algorithm,
        "source_query": args.source,
        "target_query": args.target,
        "semantic_stop_query": args.stop,
        "constraints": (
            constraints.to_dict()
        ),
        "depth_policy": {
            "base_max_depth": args.max_depth,
            "semantic_stop_max_depth": (
                semantic_stop_max_depth
            ),
            "effective_max_depth": (
                effective_max_depth
            ),
        },
        "effective_max_depth": (
            effective_max_depth
        ),
        "semantic_stop_ablation_guard": (
            semantic_stop_ablation_guard
        ),
        "source_matches": source_matches,
        "target_matches": target_matches,
        "stop_matches": (
            []
            if stop_matches == [None]
            else stop_matches
        ),
        "waypoint_selector_enabled": (
            args.algorithm == "semantic_stop"
            and not args.disable_waypoint_selector
        ),
        "waypoint_diagnostics": (
            waypoint_diagnostics
        ),
        "admitted_waypoint_tiers": list(
            admitted_waypoint_tiers
        ),
        "direct_concept_hits_enabled": (
            not args.disable_direct_concept_hits
        ),
        "direct_concept_hit_count": len(
            direct_concept_hits
        ),
        "direct_concept_hits": (
            direct_concept_hits
        ),
        "endpoint_selector_enabled": (
            not args.disable_endpoint_selector
        ),
        "endpoint_pair_selection_policy": {
            "paper_diversity_enabled": (
                not args.disable_endpoint_paper_diversity
                and not args.disable_endpoint_selector
            ),
            "paper_novelty_bonus": (
                0.0 if args.disable_endpoint_paper_diversity
                else args.endpoint_paper_novelty_bonus
            ),
            "semantic_tier_is_hard_priority": True,
        },
        "endpoint_pair_count": len(
            endpoint_pairs
        ),
        "endpoint_pairs": [
            item["pair_diagnostic"]
            for item in endpoint_pairs
            if item["pair_diagnostic"]
            is not None
        ],
        "endpoint_pair_diagnostics": (
            endpoint_pair_diagnostics
        ),
        "candidate_path_count": (
            len(all_paths)
        ),
        "waypoint_relevance_pool_count": (
            len(bundle_candidate_paths)
        ),
        "returned_path_count": len(
            paths
        ),
        "candidate_path_type_counts": (
            candidate_path_type_counts
        ),
        "returned_path_type_counts": (
            returned_path_type_counts
        ),
        "candidate_path_groups": (
            candidate_path_groups
        ),
        "candidate_paths_included": bool(
            args.include_candidate_paths
        ),
        "candidate_paths": (
            all_paths if args.include_candidate_paths else []
        ),
        "returned_path_groups": (
            returned_path_groups
        ),
        "bundle_selection": (
            bundle_selection
        ),
        "path_ranking_policy": {
            "quality_ranking_enabled": (
                path_quality_ranking_enabled
            ),
            "mechanism_mode_only": True,
            "priority_order": [
                "semantic_waypoint_relevance",
                "endpoint_semantic_tier",
                "endpoint_pair_score",
                "mechanistic_content_band",
                "mechanistic_content_score",
                "navigation_edge_fraction",
                "reverse_fraction",
                "total_cost",
                "hop_count",
            ],
        },
        "path_paper_coverage": {
            "candidate_distinct_paper_count": len(
                candidate_paper_ids
            ),
            "candidate_paper_ids": (
                candidate_paper_ids
            ),
            "returned_distinct_paper_count": len(
                returned_paper_ids
            ),
            "returned_paper_ids": (
                returned_paper_ids
            ),
        },
        # Backward-compatible aliases now refer to returned paths.
        "path_count": len(paths),
        "candidate_path_count_before_top_k": (
            len(all_paths)
        ),
        "path_type_counts": (
            returned_path_type_counts
        ),
        "path_groups": (
            returned_path_groups
        ),
        "paths": paths,
    }

    if args.output:
        output_path = Path(
            args.output
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "source_matches": [
                    {
                        "node_id": row[
                            "node_id"
                        ],
                        "label": row.get(
                            "label"
                        ),
                        "similarity": (
                            row.get(
                                "semantic_similarity"
                            )
                        ),
                    }
                    for row in source_matches
                ],
                "target_matches": [
                    {
                        "node_id": row[
                            "node_id"
                        ],
                        "label": row.get(
                            "label"
                        ),
                        "similarity": (
                            row.get(
                                "semantic_similarity"
                            )
                        ),
                    }
                    for row in target_matches
                ],
                "stop_matches": [
                    {
                        "node_id": row[
                            "node_id"
                        ],
                        "label": row.get(
                            "label"
                        ),
                        "similarity": (
                            row.get(
                                "semantic_similarity"
                            )
                        ),
                    }
                    for row in stop_matches
                    if row is not None
                ],
                "waypoint_count": (
                    len([
                        row
                        for row in waypoint_diagnostics
                        if row.get("selected")
                    ])
                ),
                "admitted_waypoint_tiers": list(
                    admitted_waypoint_tiers
                ),
                "direct_concept_hit_count": len(
                    direct_concept_hits
                ),
                "effective_max_depth": (
                    effective_max_depth
                ),
                "semantic_stop_ablation_guard": (
                    semantic_stop_ablation_guard
                ),
                "endpoint_pair_count": len(
                    endpoint_pairs
                ),
                "candidate_path_count": len(
                    all_paths
                ),
                "waypoint_relevance_pool_count": len(
                    bundle_candidate_paths
                ),
                "returned_path_count": len(
                    paths
                ),
                "candidate_path_type_counts": (
                    candidate_path_type_counts
                ),
                "returned_path_type_counts": (
                    returned_path_type_counts
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if direct_concept_hits:
        print()
        print("Direct concept hits:")
        for rank, hit in enumerate(
            direct_concept_hits,
            start=1,
        ):
            minimum = hit.get(
                "minimum_similarity"
            )
            minimum_text = (
                f"{float(minimum):.4f}"
                if minimum is not None
                else "NA"
            )
            print(
                f"  [{rank}] "
                f"tier={hit['hit_tier']} "
                f"min_sim={minimum_text} "
                f"basis={hit['quality_basis']}"
            )
            print(
                "      ",
                hit["label"],
                f"({hit['node_id']})",
            )

    if waypoint_diagnostics:
        print()
        print("Selected semantic waypoints:")
        for row in waypoint_diagnostics:
            if not row.get("selected"):
                continue
            print(
                f"  [{row['waypoint_rank']}] "
                f"tier={row['semantic_tier']} "
                f"sim={row['semantic_similarity']:.4f} "
                f"{row['label']}"
            )
        if admitted_waypoint_tiers:
            print(
                "  admitted tiers for bundle:",
                list(admitted_waypoint_tiers),
            )

    if endpoint_pair_diagnostics:
        print()
        print(
            "Selected endpoint pairs:"
        )

        selected_rows = [
            row
            for row
            in endpoint_pair_diagnostics
            if row.get("selected")
        ]

        if not selected_rows:
            print(
                "  NONE within max-depth="
                f"{effective_max_depth}"
            )
            blocked = [
                row
                for row
                in endpoint_pair_diagnostics
                if row.get(
                    "semantic_tier",
                    99,
                )
                <= 1
            ][:10]
            for row in blocked:
                print(
                    "  blocked:",
                    row["source_label"],
                    "->",
                    row["target_label"],
                    (
                        "reason="
                        + str(
                            row[
                                "selection_reason"
                            ]
                        )
                    ),
                    (
                        "hops="
                        + str(
                            row[
                                "shortest_hops"
                            ]
                        )
                    ),
                    (
                        "components="
                        + str(
                            row[
                                "source_component_id"
                            ]
                        )
                        + "/"
                        + str(
                            row[
                                "target_component_id"
                            ]
                        )
                    ),
                )
        else:
            for rank, row in enumerate(
                selected_rows,
                start=1,
            ):
                cost = row[
                    "shortest_weighted_cost"
                ]
                cost_text = (
                    f"{cost:.3f}"
                    if cost is not None
                    else "NA"
                )
                print(
                    f"  [{rank}] "
                    f"tier={row['semantic_tier']} "
                    f"score={row['pair_score']:.4f} "
                    f"hops={row['shortest_hops']} "
                    f"cost={cost_text}"
                )
                print(
                    "      source:",
                    row["source_label"],
                    (
                        f"(sim="
                        f"{row['source_similarity']:.4f}, "
                        f"exact={row['source_exact']}, "
                        f"component="
                        f"{row['source_component_id']}/"
                        f"{row['source_component_size']})"
                    ),
                )
                print(
                    "      target:",
                    row["target_label"],
                    (
                        f"(sim="
                        f"{row['target_similarity']:.4f}, "
                        f"exact={row['target_exact']}, "
                        f"component="
                        f"{row['target_component_id']}/"
                        f"{row['target_component_size']})"
                    ),
                )

    if not args.disable_bundle_selector:
        print()
        print("Path bundle selection:")
        for row in paths:
            selection = row.get(
                "bundle_selection",
                {},
            )
            print(
                "  ",
                str(row["path_id"]),
                "base_rank="
                + str(
                    selection.get(
                        "base_rank"
                    )
                ),
                "bundle_rank="
                + str(
                    selection.get(
                        "bundle_rank"
                    )
                ),
                "pass="
                + str(
                    selection.get(
                        "selection_pass"
                    )
                ),
                "edge_jaccard="
                + (
                    f"{float(selection.get('max_edge_jaccard_with_selected', 0.0)):.2f}"
                ),
            )

    for index, path in enumerate(
        paths,
        start=1,
    ):
        pair = path.get(
            "endpoint_pair"
        )
        pair_text = ""
        if isinstance(pair, dict):
            pair_text = (
                f" tier="
                f"{pair['semantic_tier']}"
                f" pair_score="
                f"{pair['pair_score']:.4f}"
            )

        waypoint = path.get(
            "waypoint"
        )
        waypoint_text = ""
        if isinstance(waypoint, dict):
            waypoint_text = (
                f" waypoint_tier="
                f"{waypoint['semantic_tier']}"
                f" waypoint_rank="
                f"{waypoint['waypoint_rank']}"
            )

        quality = path.get(
            "path_quality",
            {},
        )
        path_type = str(
            quality.get(
                "path_type",
                "UNKNOWN",
            )
        )
        mechanism_edge_density = float(
            quality.get(
                "mechanistic_edge_density",
                quality.get(
                    "mechanistic_density",
                    0.0,
                ),
            )
        )
        mechanism_node_density = float(
            quality.get(
                "mechanistic_node_density",
                0.0,
            )
        )
        mechanistic_content = str(
            quality.get(
                "mechanistic_content",
                "low",
            )
        )
        navigation_fraction = float(
            quality.get(
                "navigation_edge_fraction",
                0.0,
            )
        )

        print(
            f"\n[{index}] "
            f"type={path_type} "
            f"cost={path['total_cost']:.3f} "
            f"hops={path['hop_count']} "
            f"papers="
            f"{path['visited_paper_count']} "
            f"candidate="
            f"{path['candidate_edge_count']} "
            f"reverse="
            f"{path['reverse_edge_count']} "
            f"mech_edge_density="
            f"{mechanism_edge_density:.2f} "
            f"mech_node_density="
            f"{mechanism_node_density:.2f} "
            f"mech_content="
            f"{mechanistic_content} "
            f"nav_fraction="
            f"{navigation_fraction:.2f}"
            f"{pair_text}"
            f"{waypoint_text}"
        )

        print(
            "    visited_papers:",
            path.get(
                "visited_paper_ids",
                [],
            ),
        )
        print(
            "    supporting_papers:",
            path.get(
                "supporting_paper_ids",
                [],
            ),
        )
        if path.get(
            "hub_scope_paper_ids"
        ):
            print(
                "    hub_scope_papers:",
                path[
                    "hub_scope_paper_ids"
                ],
            )

        mechanism_nodes = quality.get(
            "mechanism_node_ids",
            [],
        )
        if mechanism_nodes:
            print(
                "    mechanism_nodes:",
                mechanism_nodes,
            )

        if isinstance(waypoint, dict):
            print(
                "    semantic_waypoint:",
                waypoint.get("label"),
                "node_id=",
                waypoint.get("node_id"),
            )

        for step in path["steps"]:
            for line in render_step_safe(
                step
            ):
                print(
                    "   ",
                    line,
                )

    if args.output:
        print(
            "\nSaved:",
            args.output,
        )


if __name__ == "__main__":
    main()
