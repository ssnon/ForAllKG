from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.corpus_pipeline import FrozenCorpusPipeline, PipelineOptions
from dac_her.kg_config_adapter import load_and_generate_paper_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing GraphAgentsDAC extraction -> paper graph -> "
            "Bridge/projection -> corpus -> navigation -> node-index pipeline "
            "over a frozen Drive-ingestion corpus."
        )
    )
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--corpus-id", default=None)
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="evidence",
    )
    parser.add_argument("--papers-yaml", default=None)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help=(
            "Run only one frozen paper ID. Repeat for a smoke-test subset. "
            "The global corpus graph is built from only the selected IDs."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N configured papers (use a separate smoke corpus ID).",
    )
    parser.add_argument("--extract-concurrency", type=int, default=4)
    parser.add_argument("--bridge-concurrency", type=int, default=4)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-node-index", action="store_true")
    parser.add_argument("--include-alignment-hubs-in-index", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-bridge", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--index-batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    frozen_path = Path(args.frozen_manifest)
    if not frozen_path.is_absolute():
        frozen_path = (root / frozen_path).resolve()
    frozen_payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    corpus_id = args.corpus_id or str(frozen_payload.get("corpus_id") or "dac_her_frozen")

    if args.papers_yaml:
        papers_yaml = Path(args.papers_yaml)
        if not papers_yaml.is_absolute():
            papers_yaml = (root / papers_yaml).resolve()
    else:
        papers_yaml = root / "data_dac" / "generated_configs" / str(
            frozen_payload.get("corpus_id") or corpus_id
        ) / "papers.yaml"
        generated = load_and_generate_paper_config(
            frozen_path,
            papers_yaml,
            project_root=root,
        )
        papers_yaml = generated.papers_yaml
        print(f"[corpus-pipeline] generated papers.yaml: {papers_yaml}", flush=True)

    runner = FrozenCorpusPipeline(
        project_root=root,
        papers_yaml=papers_yaml,
        frozen_manifest=frozen_path,
        corpus_id=corpus_id,
        selected_paper_ids=args.paper_id or None,
        paper_limit=args.limit,
        options=PipelineOptions(
            mode=args.mode,
            extract_concurrency=args.extract_concurrency,
            bridge_concurrency=args.bridge_concurrency,
            heartbeat_seconds=args.heartbeat_seconds,
            fail_fast=args.fail_fast,
            skip_node_index=args.skip_node_index,
            include_alignment_hubs_in_index=args.include_alignment_hubs_in_index,
            force_extract=args.force_extract,
            force_bridge=args.force_bridge,
            allow_partial=args.allow_partial,
            device=args.device,
            index_batch_size=args.index_batch_size,
            dry_run=args.dry_run,
            resume=not args.no_resume,
        ),
    )
    summary = runner.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["status"] != "passed" and not args.dry_run:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
