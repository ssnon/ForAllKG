from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.discovery_bundle import (
    DiscoveryBundleBuilder,
    DiscoveryPolicy,
)
from scripts.discovery_bundle_runtime import (
    load_semantic_index_for_traversal,
    load_traversal_with_graph,
)
from domains.registry import get_domain_profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-evidentiary DiscoveryBundle v2.8.0-alpha3 from one or more traversal runs. "
            "Alpha3 adds candidate-unit-aware ranking/core semantics on top of alpha2 safeguards. "
            "Use --include-candidate-paths when producing traversal JSON for full coverage."
        )
    )
    parser.add_argument("--traversal", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-per-paper-signature", type=int, default=2)
    parser.add_argument("--max-edge-jaccard", type=float, default=0.70)
    parser.add_argument("--cross-paper-mechanistic-reserve", type=int, default=2)
    parser.add_argument("--candidate-exploration-reserve", type=int, default=4)
    parser.add_argument(
        "--min-reserved-candidate-unit-score",
        type=float,
        default=0.25,
        help="Minimum candidate-unit selector score for a reserved candidate exploration slot.",
    )
    parser.add_argument(
        "--min-reserved-continuity",
        type=float,
        default=0.75,
        help=(
            "Minimum cross-paper mechanistic-continuity score for a path to receive a reserved slot. "
            "1.0 means mechanism-bearing content occurs on both sides of the alignment crossing."
        ),
    )
    parser.add_argument(
        "--semantic-similarity-threshold",
        type=float,
        default=0.88,
        help="Strict maximum similarity allowed between selected discovery inspirations.",
    )
    parser.add_argument(
        "--semantic-relaxed-threshold",
        type=float,
        default=0.94,
        help="Fallback maximum similarity if strict semantic diversity under-fills the bundle.",
    )
    parser.add_argument(
        "--max-grounding-semantic-similarity",
        type=float,
        default=0.95,
        help=(
            "Hard maximum semantic similarity to the grounding bundle. "
            "Candidates above this value are not discovery inspirations."
        ),
    )
    parser.add_argument(
        "--min-exploration-score",
        type=float,
        default=0.05,
        help="Minimum exploration score required for a discovery inspiration.",
    )
    parser.add_argument(
        "--force-fill",
        action="store_true",
        help=(
            "Diagnostic ablation only: bypass discovery-quality under-fill "
            "and allow alpha2-style quota filling."
        ),
    )
    parser.add_argument(
        "--disable-semantic-diversity",
        action="store_true",
        help="Disable node-embedding/lexical semantic deduplication (diagnostic ablation only).",
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--domain-profile",
        default=None,
        help=(
            "Scientific domain profile. If omitted, infer from traversal "
            "artifacts and fall back to dac_her for legacy artifacts."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Optional explicit domain data root. Overrides traversal data_root "
            "and the extraction adapter default."
        ),
    )
    args = parser.parse_args()

    if not 0.0 <= args.semantic_similarity_threshold <= 1.0:
        raise ValueError("--semantic-similarity-threshold must be between 0 and 1")
    if not 0.0 <= args.semantic_relaxed_threshold <= 1.0:
        raise ValueError("--semantic-relaxed-threshold must be between 0 and 1")
    if args.semantic_relaxed_threshold < args.semantic_similarity_threshold:
        raise ValueError("--semantic-relaxed-threshold must be >= --semantic-similarity-threshold")
    if not 0.0 <= args.max_grounding_semantic_similarity <= 1.0:
        raise ValueError("--max-grounding-semantic-similarity must be between 0 and 1")
    if not 0.0 <= args.min_exploration_score <= 1.0:
        raise ValueError("--min-exploration-score must be between 0 and 1")

    payloads = [
        load_traversal_with_graph(
            path,
            project_root=args.project_root,
            domain_profile_id=args.domain_profile,
            data_root=args.data_root,
        )
        for path in args.traversal
    ]

    if args.domain_profile:
        domain_profile = get_domain_profile(args.domain_profile)
    else:
        explicit_domains = {
            str(payload.get("domain_profile_id", "")).strip()
            for _, payload, _ in payloads
            if str(payload.get("domain_profile_id", "")).strip()
        }
        if len(explicit_domains) > 1:
            raise ValueError(
                "traversal artifacts contain multiple domain profiles: "
                f"{sorted(explicit_domains)}"
            )
        domain_profile = get_domain_profile(
            next(iter(explicit_domains))
            if explicit_domains
            else "dac_her"
        )

    semantic_indexes = {}
    if not args.disable_semantic_diversity:
        for source_name, payload, _ in payloads:
            index = load_semantic_index_for_traversal(
                payload,
                project_root=args.project_root,
                domain_profile_id=domain_profile.profile_id,
                data_root=args.data_root,
            )
            if index is not None:
                semantic_indexes[source_name] = index

    bundle = DiscoveryBundleBuilder(
        DiscoveryPolicy(
            top_k=args.top_k,
            max_per_paper_signature=args.max_per_paper_signature,
            max_edge_jaccard=args.max_edge_jaccard,
            cross_paper_mechanistic_reserve=args.cross_paper_mechanistic_reserve,
            min_reserved_continuity=args.min_reserved_continuity,
            candidate_exploration_reserve=args.candidate_exploration_reserve,
            min_reserved_candidate_unit_score=args.min_reserved_candidate_unit_score,
            semantic_diversity_enabled=not args.disable_semantic_diversity,
            semantic_similarity_threshold=args.semantic_similarity_threshold,
            semantic_relaxed_threshold=args.semantic_relaxed_threshold,
            max_grounding_semantic_similarity=args.max_grounding_semantic_similarity,
            min_exploration_score=args.min_exploration_score,
            force_fill=args.force_fill,
        ),
        domain_profile=domain_profile,
    ).build(
        payloads,
        semantic_indexes=semantic_indexes,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("DiscoveryBundle built")
    print("Domain profile:", domain_profile.profile_id)
    print("Bundle ID:", bundle.bundle_id)
    print("Bundle SHA256:", bundle.bundle_sha256)
    print("Corpus:", bundle.corpus_id)
    print("Candidates:", bundle.candidate_count)
    print("Selected inspirations:", bundle.selected_count)
    print("Full candidate pool used:", bundle.used_candidate_pool)
    print("Semantic diversity:", bundle.semantic_diversity_mode)
    if bundle.semantic_model_name:
        print("Semantic model:", bundle.semantic_model_name)
    print("Semantic threshold:", bundle.semantic_similarity_threshold)
    print("Grounding semantic hard cap:", args.max_grounding_semantic_similarity)
    print("Minimum exploration score:", args.min_exploration_score)
    print("Force fill:", args.force_fill)

    for index, item in enumerate(bundle.inspirations, start=1):
        print(
            f"[{index}] score={item.exploration_score:.3f} "
            f"type={item.path_type} mode={item.source_mode} papers={len(item.paper_ids)} "
            f"continuity={item.mechanistic_continuity_band} "
            f"generic={item.generic_entity_fraction:.2f} "
            f"registry={item.registry_hop_fraction:.2f} "
            f"grounding_sem={item.semantic_similarity_to_grounding:.2f} "
            f"selected_sem={item.max_semantic_similarity_to_selected:.2f} "
            f"unit_score={item.candidate_unit_score:.2f} "
            f"reaction_switch={item.reaction_domain_switch_penalty:.2f}"
        )
        if item.candidate_unit_id:
            print(
                "     candidate_unit:",
                item.candidate_unit_label,
                "|",
                item.candidate_entry_anchor_label,
                "->",
                item.candidate_exit_anchor_label,
            )
        print("    ", item.rendered_path[:300])
    for warning in bundle.warnings:
        print("WARNING:", warning)
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
