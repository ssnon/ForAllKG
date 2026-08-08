from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from dac_her.endpoint_selection import (
    EndpointPairSelector,
)
from dac_her.node_mapping import (
    NodeMapper,
    QueryConcept,
)
from dac_her.traversal_engine import (
    TraversalConstraints,
    TraversalEngine,
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
        "--disable-endpoint-selector",
        action="store_true",
        help=(
            "Use v2.4.0 all-pairs behavior "
            "for ablation/debugging."
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


def _path_sort_key(
    row: dict,
) -> tuple:
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

    return (
        tier,
        -pair_score,
        float(row["total_cost"]),
        int(row["hop_count"]),
        str(row["path_id"]),
    )


def main() -> None:
    args = parse_args()

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
        PROJECT_ROOT
        / "data_dac"
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

    constraints = TraversalConstraints(
        mode=args.mode,
        top_k=args.top_k,
        max_depth=args.max_depth,
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

    if args.disable_endpoint_selector:
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
            in source_matches
            for target_match
            in target_matches
        ]
    else:
        selector = EndpointPairSelector(
            graph
        )
        (
            selected_pairs,
            diagnostics,
        ) = selector.select(
            source_matches,
            target_matches,
            top_k=args.endpoint_pair_k,
            max_depth=args.max_depth,
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

        for stop_match in stop_matches:
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
                row["endpoint_pair"] = (
                    endpoint_pair[
                        "pair_diagnostic"
                    ]
                )
                collected[
                    path.path_id
                ] = row

    paths = sorted(
        collected.values(),
        key=_path_sort_key,
    )[: args.top_k]

    payload = {
        "corpus_id": args.corpus_id,
        "mode": args.mode,
        "algorithm": args.algorithm,
        "source_query": args.source,
        "target_query": args.target,
        "semantic_stop_query": args.stop,
        "constraints": (
            constraints.to_dict()
        ),
        "source_matches": source_matches,
        "target_matches": target_matches,
        "stop_matches": (
            []
            if stop_matches == [None]
            else stop_matches
        ),
        "endpoint_selector_enabled": (
            not args.disable_endpoint_selector
        ),
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
        "path_count": len(paths),
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
                "endpoint_pair_count": len(
                    endpoint_pairs
                ),
                "path_count": len(paths),
            },
            ensure_ascii=False,
            indent=2,
        )
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
                f"{args.max_depth}"
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

        print(
            f"\n[{index}] "
            f"cost={path['total_cost']:.3f} "
            f"hops={path['hop_count']} "
            f"papers={path['cross_paper_count']} "
            f"candidate="
            f"{path['candidate_edge_count']} "
            f"reverse="
            f"{path['reverse_edge_count']}"
            f"{pair_text}"
        )

        for step in path["steps"]:
            marker = (
                " <-reverse"
                if step[
                    "traversal_direction"
                ]
                == "reverse"
                else ""
            )
            print(
                "  ",
                step["source"],
                "--",
                step["relation"],
                "-->",
                step["target"],
                marker,
            )

    if args.output:
        print(
            "\nSaved:",
            args.output,
        )


if __name__ == "__main__":
    main()
