from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.corpus_publication_runtime import (
    CorpusPublicationOptions,
    StrictBridgeCorpusPublisher,
)
from scripts.knowledge_backfill_runtime import (
    KnowledgeAwareBackfillCoordinator,
    KnowledgeBackfillOptions,
    KnowledgeBackfillPaths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Knowledge-aware corpus backfill: use Strict/Bridge outcomes as a "
            "feedback signal, request additional quality-pass M3.2 reserve "
            "papers, materialize/gate them, and rerun the resumable Strict -> "
            "Bridge -> corpus pipeline until the requested final target is met."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--paper-id-prefix", required=True)
    parser.add_argument("--target-count", required=True, type=int)
    parser.add_argument(
        "--target-status",
        choices=("STRICT_USABLE", "BRIDGE_USEFUL", "CORPUS_ELIGIBLE"),
        default="BRIDGE_USEFUL",
    )
    parser.add_argument("--oversample-factor", type=float, default=1.0)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-extra-candidates", type=int, default=100)

    parser.add_argument("--acquisition-profile", required=True, type=Path)
    parser.add_argument("--backfill-policy", required=True, type=Path)
    parser.add_argument("--source-policy", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--m2-assessments", required=True, type=Path)
    parser.add_argument("--quality-assessments", required=True, type=Path)
    parser.add_argument("--quality-gate-report", required=True, type=Path)
    parser.add_argument("--starting-m3-dir", required=True, type=Path)
    parser.add_argument("--supplementary-policy", type=Path, default=None)
    parser.add_argument("--m3-1-dir", type=Path, default=None)
    parser.add_argument("--materialization-policy", required=True, type=Path)
    parser.add_argument("--m4-dir", required=True, type=Path)
    parser.add_argument("--m4-config", required=True, type=Path)
    parser.add_argument("--gate-policy", required=True, type=Path)
    parser.add_argument("--m4-5-dir", required=True, type=Path)
    parser.add_argument("--strict-config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)

    parser.add_argument("--extract-concurrency", type=int, default=4)
    parser.add_argument("--bridge-concurrency", type=int, default=4)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--retry-failed-acquisition", action="store_true")
    parser.add_argument(
        "--retry-access-misses",
        action="store_true",
        help=(
            "Re-resolve cached unresolved/landing-only M3.2 candidates. "
            "Useful after enabling Unpaywall/OpenAlex credentials."
        ),
    )
    parser.add_argument("--retry-failed-materialization", action="store_true")
    parser.add_argument("--retry-failed-supplementary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--publish-on-success",
        action="store_true",
        help=(
            "After the knowledge target is satisfied, build lifecycle/funnel "
            "artifacts and publish corpus -> navigation -> node index."
        ),
    )
    parser.add_argument("--publication-output-dir", type=Path, default=None)
    parser.add_argument("--skip-node-index", action="store_true")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--include-alignment-hubs-in-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    paths = KnowledgeBackfillPaths(
        acquisition_profile=args.acquisition_profile,
        backfill_policy=args.backfill_policy,
        source_policy=args.source_policy,
        catalog=args.catalog,
        m2_assessments=args.m2_assessments,
        quality_assessments=args.quality_assessments,
        quality_gate_report=args.quality_gate_report,
        starting_m3_dir=args.starting_m3_dir,
        supplementary_policy=args.supplementary_policy,
        m3_1_dir=args.m3_1_dir,
        materialization_policy=args.materialization_policy,
        m4_dir=args.m4_dir,
        m4_config=args.m4_config,
        gate_policy=args.gate_policy,
        m4_5_dir=args.m4_5_dir,
        strict_config=args.strict_config,
        data_root=args.data_root,
        run_root=args.run_root,
    )
    coordinator = KnowledgeAwareBackfillCoordinator(
        project_root=root,
        corpus_id=args.corpus_id,
        domain_profile=args.domain_profile,
        paper_id_prefix=args.paper_id_prefix,
        paths=paths,
        options=KnowledgeBackfillOptions(
            target_count=args.target_count,
            target_status=args.target_status,
            oversample_factor=args.oversample_factor,
            max_rounds=args.max_rounds,
            max_extra_candidates=args.max_extra_candidates,
            extract_concurrency=args.extract_concurrency,
            bridge_concurrency=args.bridge_concurrency,
            heartbeat_seconds=args.heartbeat_seconds,
            retry_failed_acquisition=args.retry_failed_acquisition,
            retry_access_misses=args.retry_access_misses,
            retry_failed_materialization=args.retry_failed_materialization,
            retry_failed_supplementary=args.retry_failed_supplementary,
            dry_run=args.dry_run,
        ),
    )
    result = coordinator.run()

    if (
        args.publish_on_success
        and not args.dry_run
        and result.get("status") in {"target_reached", "target_already_satisfied"}
    ):
        latest_m3_dir = Path(
            str(result.get("latest_m3_dir") or args.starting_m3_dir)
        )
        publisher = StrictBridgeCorpusPublisher(
            project_root=root,
            corpus_id=args.corpus_id,
            domain_profile=args.domain_profile,
            data_root=args.data_root,
            selected_works_path=latest_m3_dir / "selected_works.jsonl",
            m3_dir=latest_m3_dir,
            m4_dir=args.m4_dir,
            m4_5_dir=args.m4_5_dir,
            outcomes_path=result.get("outcomes_path"),
            output_dir=args.publication_output_dir,
            options=CorpusPublicationOptions(
                mode="mechanism",
                target_count=args.target_count,
                target_status=args.target_status,
                build_node_index=not args.skip_node_index,
                embedding_model=args.embedding_model,
                embedding_device=args.embedding_device,
                embedding_batch_size=args.embedding_batch_size,
                include_alignment_hubs_in_index=(
                    args.include_alignment_hubs_in_index
                ),
            ),
        )
        result["publication"] = publisher.run()
        resolved_run_root = (
            args.run_root
            if args.run_root.is_absolute()
            else root / args.run_root
        )
        resolved_run_root.mkdir(parents=True, exist_ok=True)
        (resolved_run_root / "run.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    allowed = {"target_reached", "target_already_satisfied", "dry_run"}
    if result["status"] not in allowed and not args.dry_run:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
