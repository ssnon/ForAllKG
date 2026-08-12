from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dac_her.corpus_freeze import load_and_freeze
from dac_her.incremental_reconcile import (
    IncrementalCorpusReconciler,
    ReconcileOptions,
)
from dac_her.kg_config_adapter import load_and_generate_paper_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Desired-state updater for Drive -> Marker -> strict KG -> Bridge KG -> "
            "GraphAgents projection -> corpus. Existing current artifacts are skipped."
        )
    )
    parser.add_argument("--corpus-id", default="dac_her_drive_v1")
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="exploratory",
    )
    parser.add_argument("--domain-profile", default="dac_her")
    parser.add_argument(
        "--freshness",
        choices=("source", "semantic", "full"),
        default="semantic",
        help=(
            "Cache validity policy. semantic (default) invalidates strict LLM "
            "results only for scientific-contract changes; source ignores "
            "prompt/schema/vocabulary changes; full also tracks operational/code changes."
        ),
    )
    parser.add_argument(
        "--extract-model",
        default=None,
        help="Override OPENROUTER_EXTRACT_MODEL for freshness checks and extraction.",
    )
    parser.add_argument(
        "--extract-provider",
        default=None,
        help="Override OPENROUTER_PROVIDER for freshness checks and extraction.",
    )
    parser.add_argument("--kg-data-root", default="data_dac")
    parser.add_argument("--ingestion-data-root", default="data_dac/ingestion")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--skip-drive-sync", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--exclude-ingestion-warnings", action="store_true")
    parser.add_argument("--extract-concurrency", type=int, default=4)
    parser.add_argument("--bridge-concurrency", type=int, default=4)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-node-index", action="store_true")
    parser.add_argument("--include-alignment-hubs-in-index", action="store_true")
    parser.add_argument("--index-batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        choices=(
            "strict",
            "strict_graph",
            "bridge",
            "projection",
            "corpus",
            "navigation",
            "index",
        ),
        help="Force one stage even if its current artifact is valid. Repeatable.",
    )
    parser.add_argument(
        "--marker-arg",
        action="append",
        default=[],
        help="Forward an extra argument to sync_drive_corpus/marker_single.",
    )
    return parser.parse_args()


def _run_drive_sync(args: argparse.Namespace, root: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.sync_drive_corpus",
        "--corpus-id",
        args.corpus_id,
        "--data-root",
        args.ingestion_data_root,
    ]
    for value in args.marker_arg:
        command += ["--marker-arg", value]
    print("[update] Drive scan + incremental Marker sync", flush=True)
    code = subprocess.run(command, cwd=root).returncode
    if code != 0:
        raise SystemExit(code)


def _paths(args: argparse.Namespace, root: Path) -> tuple[Path, Path, Path]:
    ingestion_root = Path(args.ingestion_data_root)
    if not ingestion_root.is_absolute():
        ingestion_root = root / ingestion_root
    input_manifest = ingestion_root / "corpora" / args.corpus_id / "manifest.json"
    frozen_manifest = root / "data_dac" / "frozen_corpora" / args.corpus_id / "manifest.json"
    papers_yaml = root / "data_dac" / "generated_configs" / args.corpus_id / "papers.yaml"
    return input_manifest, frozen_manifest, papers_yaml


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()

    if args.status_only:
        args.skip_drive_sync = True

    if not args.skip_drive_sync and not args.dry_run:
        _run_drive_sync(args, root)
    elif not args.skip_drive_sync and args.dry_run:
        print(
            "[update] dry-run: Drive mutation is skipped; planning uses the current local ingestion manifest.",
            flush=True,
        )

    input_manifest, frozen_manifest, papers_yaml = _paths(args, root)
    if not input_manifest.is_file():
        raise SystemExit(
            f"Ingestion manifest not found: {input_manifest}\n"
            "Run scripts.sync_drive_corpus first or remove --skip-drive-sync."
        )

    if not args.status_only:
        print("[update] freeze/dedupe ingestion manifest", flush=True)
        frozen = load_and_freeze(
            input_manifest,
            frozen_manifest,
            project_root=root,
            include_warnings=not args.exclude_ingestion_warnings,
            verify_paths=True,
        )
        print(
            f"[update] frozen papers: {frozen['document_count']} "
            f"(deduplicated={frozen['deduplicated_document_count']})",
            flush=True,
        )
        generated = load_and_generate_paper_config(
            frozen_manifest,
            papers_yaml,
            project_root=root,
        )
        print(f"[update] generated papers.yaml: {generated.papers_yaml}", flush=True)
    else:
        if not frozen_manifest.is_file() or not papers_yaml.is_file():
            raise SystemExit(
                "--status-only requires existing frozen manifest and generated papers.yaml."
            )

    reconciler = IncrementalCorpusReconciler(
        project_root=root,
        papers_yaml=papers_yaml,
        frozen_manifest=frozen_manifest,
        corpus_id=args.corpus_id,
        options=ReconcileOptions(
            mode=args.mode,
            freshness=args.freshness,
            domain_profile=args.domain_profile,
            kg_data_root=args.kg_data_root,
            extract_model=args.extract_model,
            extract_provider=args.extract_provider,
            extract_concurrency=args.extract_concurrency,
            bridge_concurrency=args.bridge_concurrency,
            heartbeat_seconds=args.heartbeat_seconds,
            fail_fast=args.fail_fast,
            allow_partial=args.allow_partial,
            skip_node_index=args.skip_node_index,
            include_alignment_hubs_in_index=args.include_alignment_hubs_in_index,
            index_batch_size=args.index_batch_size,
            device=args.device,
            force_stages=frozenset(args.force_stage),
            dry_run=args.dry_run,
        ),
    )

    if args.status_only:
        print(f"[status] freshness={args.freshness}", flush=True)
        rows = reconciler.status_table()
        pending = 0
        for row in rows:
            ready = all(
                row[key] in {"ready", "n/a"}
                for key in ("strict", "strict_graph", "bridge", "projection")
            )
            if not ready:
                pending += 1
            print(
                f"{row['paper_id']} | strict={row['strict']} | "
                f"strict_graph={row['strict_graph']} | bridge={row['bridge']} | "
                f"projection={row['projection']}",
                flush=True,
            )
        print(f"[status] papers={len(rows)} pending={pending}", flush=True)
        return

    report = reconciler.run()
    summary = {
        "status": report["status"],
        "paper_count": report["paper_count"],
        "failure_count": len(report["failures"]),
        "changed_any_projection": report["changed_any_projection"],
        "global": report["global"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "passed" and not args.dry_run:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
