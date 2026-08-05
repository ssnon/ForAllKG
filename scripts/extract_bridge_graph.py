from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import networkx as nx

import dac_her.bridge_extraction as bridge_extraction_module
import dac_her.bridge_graph as bridge_graph_module
import dac_her.bridge_policy as bridge_policy_module
import dac_her.bridge_prompts as bridge_prompts_module
import dac_her.bridge_schemas as bridge_schemas_module
import dac_her.bridge_validation as bridge_validation_module
import dac_her.scientific_signatures as scientific_signatures_module
from dac_her.bridge_extraction import extract_bridge_chunk
from dac_her.bridge_graph import (
    build_bridge_graph,
    save_bridge_graph,
    write_bridge_tables,
)
from dac_her.bridge_policy import BRIDGE_POLICY_VERSION
from dac_her.bridge_prompts import BRIDGE_PROMPT_VERSION
from dac_her.bridge_run_state import (
    bridge_run_directory,
    compute_bridge_run_metadata,
)
from dac_her.bridge_schemas import BridgeChunkGraph
from dac_her.run_state import (
    paper_output_root,
    read_json,
    resolve_run_directory,
    write_json,
)
from dac_her.schemas import KnowledgeGraph


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "papers.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract fingerprinted Bridge v2 relation patterns and frontier "
            "concepts from validated strict DAC-HER chunks."
        )
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("OPENROUTER_BRIDGE_MODEL")
            or os.getenv("OPENROUTER_EXTRACT_MODEL")
        ),
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("OPENROUTER_PROVIDER") or None,
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--canonical-graphml",
        default=None,
        help=(
            "Optional canonical graph used to remap raw anchor IDs. Default: "
            "data_dac/extracted/<paper>/<paper>.graphml when present."
        ),
    )
    return parser.parse_args()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _source_path_for_record(run_dir: Path, record: dict[str, Any]) -> Path | None:
    source_path_value = record.get("source_path")
    if source_path_value:
        path = Path(str(source_path_value))
        if path.exists():
            return path

    safe_chunk_id = str(record.get("chunk_id", "unknown")).replace(":", "__")
    deterministic = run_dir / "source_chunks" / f"{safe_chunk_id}.json"
    return deterministic if deterministic.exists() else None


def _load_rejections(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        value = record.get("rejections_path")
        if not value:
            continue
        path = Path(str(value))
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, list):
            rows.extend(item for item in payload if isinstance(item, dict))
    return rows


def _copy_latest_artifacts(source_dir: Path, legacy_dir: Path) -> None:
    """Maintain the historical <strict-run>/bridge path as a latest alias."""
    legacy_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "summary.json",
        "run.json",
        "bridge.raw.graphml",
        "bridge.graphml",
        "bridge_concepts.csv",
        "bridge_patterns.csv",
        "bridge_frontier.csv",
        "bridge_links.csv",
        "bridge_issues.csv",
        "bridge_rejected.csv",
    )
    for name in names:
        source = source_dir / name
        if source.exists():
            shutil.copyfile(source, legacy_dir / name)
    write_json(
        legacy_dir / "latest_run.json",
        {
            "bridge_run_id": source_dir.name,
            "bridge_run_directory": str(source_dir),
        },
    )


def main() -> None:
    args = parse_args()
    if not args.model:
        raise RuntimeError(
            "Set OPENROUTER_BRIDGE_MODEL or OPENROUTER_EXTRACT_MODEL, or pass --model."
        )
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")

    strict_run_dir = resolve_run_directory(
        project_root=PROJECT_ROOT,
        paper_id=args.paper_id,
        run_id=args.run_id,
    )
    active_payload = read_json(strict_run_dir / "active_chunks.json")
    strict_run_metadata = read_json(strict_run_dir / "run.json")
    active_payload = {
        **active_payload,
        "run_fingerprint": strict_run_metadata.get("run_fingerprint", ""),
    }
    if not active_payload.get("complete", False) and not args.allow_incomplete:
        raise RuntimeError(
            "Strict extraction is incomplete. Resolve failed chunks or pass "
            "--allow-incomplete explicitly."
        )

    chunk_records = active_payload.get("chunks")
    if not isinstance(chunk_records, list) or not chunk_records:
        raise RuntimeError("No active strict chunks are available.")

    paper_root = paper_output_root(PROJECT_ROOT, args.paper_id)
    canonical_path = (
        Path(args.canonical_graphml)
        if args.canonical_graphml
        else paper_root / f"{args.paper_id}.graphml"
    )
    canonical_graph = (
        nx.read_graphml(canonical_path, force_multigraph=True)
        if canonical_path.exists()
        else None
    )

    jobs: list[tuple[dict[str, Any], KnowledgeGraph, dict[str, Any], Path, Path]] = []
    strict_results: dict[str, KnowledgeGraph] = {}
    missing_sources: list[str] = []

    for record in chunk_records:
        strict_path = Path(str(record["output_path"]))
        strict_result = KnowledgeGraph.model_validate_json(
            strict_path.read_text(encoding="utf-8")
        )
        source_path = _source_path_for_record(strict_run_dir, record)
        if source_path is None:
            missing_sources.append(str(record.get("chunk_id", "unknown")))
            continue
        source_payload = read_json(source_path)
        strict_results[strict_result.chunk_id] = strict_result
        jobs.append((record, strict_result, source_payload, strict_path, source_path))

    if missing_sources:
        raise RuntimeError(
            "Active chunks do not contain source snapshots required for bridge "
            "extraction. Re-run `python -m scripts.extract_paper --paper-id "
            f"{args.paper_id}` without --force. Missing chunks: "
            f"{missing_sources[:8]!r}"
        )

    implementation_paths = (
        bridge_extraction_module.__file__,
        bridge_graph_module.__file__,
        bridge_policy_module.__file__,
        bridge_prompts_module.__file__,
        bridge_schemas_module.__file__,
        bridge_validation_module.__file__,
        scientific_signatures_module.__file__,
    )
    bridge_run_metadata = compute_bridge_run_metadata(
        strict_run_dir=strict_run_dir,
        active_payload=active_payload,
        model=args.model,
        provider=args.provider,
        strict_chunk_paths=[item[3] for item in jobs],
        source_chunk_paths=[item[4] for item in jobs],
        canonical_graph_path=canonical_path if canonical_graph is not None else None,
        implementation_paths=implementation_paths,
    )
    bridge_run_id = str(bridge_run_metadata["bridge_run_id"])
    bridge_dir = bridge_run_directory(strict_run_dir, bridge_run_id)
    chunk_dir = bridge_dir / "chunks"
    debug_dir = bridge_dir / "debug"
    manifest_path = bridge_dir / "manifest.jsonl"
    for directory in (bridge_dir, chunk_dir, debug_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("", encoding="utf-8")
    write_json(bridge_dir / "run.json", bridge_run_metadata)

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_map = {
            executor.submit(
                extract_bridge_chunk,
                strict_result=strict_result,
                source_payload=source_payload,
                model=args.model,
                provider=args.provider,
                output_dir=chunk_dir,
                debug_dir=debug_dir,
                force=args.force,
            ): strict_result.chunk_id
            for _, strict_result, source_payload, _, _ in jobs
        }
        for future in as_completed(future_map):
            chunk_id = future_map[future]
            try:
                record = future.result()
            except Exception as error:
                record = {
                    "status": "failed",
                    "paper_id": args.paper_id,
                    "chunk_id": chunk_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            record["bridge_run_id"] = bridge_run_id
            record["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
            records.append(record)
            _append_jsonl(manifest_path, record)
            print(
                f"[{record['status'].upper()}] {chunk_id} "
                f"patterns={record.get('pattern_count', 0)} "
                f"frontier={record.get('frontier_count', 0)} "
                f"rejected={record.get('rejection_count', 0)} "
                f"links={record.get('link_count', 0)}",
                flush=True,
            )

    failed = [record for record in records if record["status"] == "failed"]
    if failed and not args.allow_incomplete:
        write_json(
            bridge_dir / "summary.json",
            {
                "paper_id": args.paper_id,
                "strict_run_id": active_payload.get("run_id"),
                "bridge_run_id": bridge_run_id,
                "complete": False,
                "failed_chunks": failed,
                "prompt_version": BRIDGE_PROMPT_VERSION,
                "policy_version": BRIDGE_POLICY_VERSION,
            },
        )
        raise SystemExit(2)

    bridge_results = [
        BridgeChunkGraph.model_validate_json(
            Path(str(record["output_path"])).read_text(encoding="utf-8")
        )
        for record in records
        if record["status"] in {"success", "skipped"}
    ]
    rejections = _load_rejections(records)

    raw_graph, raw_issues = build_bridge_graph(
        bridge_results,
        strict_results=strict_results,
        canonical_graph=None,
    )
    canonical_bridge_graph, canonical_issues = build_bridge_graph(
        bridge_results,
        strict_results=strict_results,
        canonical_graph=canonical_graph,
    )

    for graph in (
        raw_graph,
        canonical_bridge_graph,
    ):
        graph.graph.update({
            "bridge_policy_version": (
                BRIDGE_POLICY_VERSION
            ),
            "bridge_prompt_version": (
                BRIDGE_PROMPT_VERSION
            ),
            "bridge_run_id": (
                bridge_run_id
            ),
            "bridge_run_fingerprint": (
                bridge_run_metadata[
                    "bridge_run_fingerprint"
                ]
            ),
        })

    raw_path = save_bridge_graph(raw_graph, bridge_dir / "bridge.raw.graphml")
    graph_path = save_bridge_graph(
        canonical_bridge_graph,
        bridge_dir / "bridge.graphml",
    )
    write_bridge_tables(
        canonical_bridge_graph,
        canonical_issues,
        rejections,
        bridge_dir,
    )

    latest_bridge_path = paper_root / f"{args.paper_id}.bridge.graphml"
    latest_bridge_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(graph_path, latest_bridge_path)

    pattern_count = sum(
        concept.retention_lane == "accepted_pattern"
        for result in bridge_results
        for concept in result.concepts
    )
    frontier_count = sum(
        concept.retention_lane == "paper_local_frontier"
        for result in bridge_results
        for concept in result.concepts
    )
    concept_count = pattern_count + frontier_count
    link_count = sum(len(result.links) for result in bridge_results)
    summary = {
        "paper_id": args.paper_id,
        "strict_run_id": active_payload.get("run_id"),
        "bridge_run_id": bridge_run_id,
        "bridge_run_fingerprint": bridge_run_metadata["bridge_run_fingerprint"],
        "complete": not failed,
        "prompt_version": BRIDGE_PROMPT_VERSION,
        "policy_version": BRIDGE_POLICY_VERSION,
        "model": args.model,
        "provider": args.provider or "",
        "chunks": len(bridge_results),
        "concepts": concept_count,
        "patterns": pattern_count,
        "frontier_concepts": frontier_count,
        "links": link_count,
        "rejected_candidates": len(rejections),
        "failed_chunks": failed,
        "raw_anchor_issues": len(raw_issues),
        "canonical_anchor_issues": len(canonical_issues),
        "canonical_graph_used": str(canonical_path) if canonical_graph is not None else "",
        "raw_graphml": str(raw_path),
        "bridge_graphml": str(graph_path),
        "latest_bridge_graphml": str(latest_bridge_path),
    }
    write_json(bridge_dir / "summary.json", summary)
    _copy_latest_artifacts(bridge_dir, strict_run_dir / "bridge")
    write_json(
        strict_run_dir / "latest_bridge_run.json",
        {
            "paper_id": args.paper_id,
            "strict_run_id": active_payload.get("run_id"),
            "bridge_run_id": bridge_run_id,
            "bridge_run_directory": str(bridge_dir),
            "bridge_run_fingerprint": bridge_run_metadata["bridge_run_fingerprint"],
        },
    )

    print("Bridge v2 extraction finished")
    print("Bridge run ID:", bridge_run_id)
    print("Patterns:", pattern_count)
    print("Frontier concepts:", frontier_count)
    print("Rejected candidates:", len(rejections))
    print("Links:", link_count)
    print("Canonical anchor issues:", len(canonical_issues))
    print("Saved:", latest_bridge_path)
    print("Run directory:", bridge_dir)


if __name__ == "__main__":
    main()
