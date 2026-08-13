from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.broad_corpus_pipeline import (
    BroadCorpusPilotPipeline,
    BroadPilotOptions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run extraction -> paper graph -> broad mechanism projection -> "
            "corpus graph -> extraction diagnostics -> coverage audit for "
            "selected abstract papers."
        )
    )
    parser.add_argument("--config", required=True, help="PR3-generated papers.yaml")
    parser.add_argument("--corpus-id", default="broad_catalysis_pilot_v1")
    parser.add_argument("--data-root", default="data_broad")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--paper-id", action="append", default=[])
    parser.add_argument("--extract-concurrency", type=int, default=1)
    parser.add_argument(
        "--max-abstract-source-tokens",
        type=int,
        default=1200,
        help=(
            "Skip full-text-like abstract packages before any LLM call when "
            "their selected Markdown exceeds this token estimate. Use 0 to "
            "disable the guard."
        ),
    )
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--broad-compact-schema",
        action="store_true",
        help=(
            "Use the experimental compact response schema for Broad initial "
            "graph generation. Pair with --force-extract for controlled A/B runs."
        ),
    )
    parser.add_argument(
        "--broad-compact-domain-recovery",
        action="store_true",
        help=(
            "Use the adapter-owned compact schema for targeted Broad "
            "domain-gate recovery. Requires --broad-compact-schema."
        ),
    )
    parser.add_argument(
        "--broad-prune-metric-vocabulary",
        action="store_true",
        help=(
            "Omit measurement-metric registry serialization from Broad "
            "extraction prompts while keeping metric validation active. "
            "Intended for PR6.1 controlled A/B runs."
        ),
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    error_group = parser.add_mutually_exclusive_group()
    error_group.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        default=True,
        help=(
            "Skip papers that fail extraction/graph/projection and build the "
            "Broad corpus from the usable subset (default)."
        ),
    )
    error_group.add_argument(
        "--fail-fast",
        dest="continue_on_error",
        action="store_false",
        help="Stop the pilot immediately when one paper fails.",
    )
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help=(
            "Retry a cached REJECTED/PARTIAL_CRITICAL extraction. By default "
            "known unusable papers are skipped without another LLM call."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = BroadCorpusPilotPipeline(
        project_root=PROJECT_ROOT,
        papers_yaml=args.config,
        corpus_id=args.corpus_id,
        options=BroadPilotOptions(
            data_root=args.data_root,
            extract_concurrency=args.extract_concurrency,
            force_extract=args.force_extract,
            broad_compact_schema=args.broad_compact_schema,
            broad_compact_domain_recovery=(
                args.broad_compact_domain_recovery
            ),
            broad_prune_metric_vocabulary=(
                args.broad_prune_metric_vocabulary
            ),
            allow_partial=args.allow_partial,
            skip_extraction=args.skip_extraction,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
            retry_rejected=args.retry_rejected,
            resume=not args.no_resume,
            max_abstract_source_tokens=args.max_abstract_source_tokens,
        ),
        requested_paper_ids=args.paper_id or None,
        paper_limit=args.limit,
    )
    manifest = pipeline.run()
    print("Broad corpus pilot finished")
    print("Manifest:", manifest)


if __name__ == "__main__":
    main()
