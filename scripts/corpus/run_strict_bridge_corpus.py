from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.corpus.strict_bridge_corpus_runtime import (
    StrictBridgeCorpusPipeline,
    StrictBridgePipelineOptions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a strict-ready generated paper config through Strict extraction -> "
            "paper graph -> Bridge -> GraphAgents projection -> usable-only corpus."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--source-manifest",
        default=None,
        help=(
            "Optional acquisition/pre-extraction artifact to bind into the run "
            "manifest by SHA256 (for example M4.5 extraction_plan.jsonl)."
        ),
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--mode",
        choices=("mechanism", "exploratory"),
        default="mechanism",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Run only this paper ID; repeat for a selected smoke-test subset.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--extract-concurrency", type=int, default=4)
    parser.add_argument("--bridge-concurrency", type=int, default=4)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-bridge", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-corpus", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    pipeline = StrictBridgeCorpusPipeline(
        project_root=root,
        config=args.config,
        corpus_id=args.corpus_id,
        domain_profile=args.domain_profile,
        data_root=args.data_root,
        source_manifest=args.source_manifest,
        requested_paper_ids=args.paper_id or None,
        paper_limit=args.limit,
        options=StrictBridgePipelineOptions(
            mode=args.mode,
            extract_concurrency=args.extract_concurrency,
            bridge_concurrency=args.bridge_concurrency,
            heartbeat_seconds=args.heartbeat_seconds,
            allow_partial=args.allow_partial,
            force_extract=args.force_extract,
            force_bridge=args.force_bridge,
            continue_on_error=not args.fail_fast,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            skip_corpus=args.skip_corpus,
        ),
    )
    summary = pipeline.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["status"] not in {"passed", "passed_with_paper_skips"} and not args.dry_run:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
