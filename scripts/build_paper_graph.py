from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import networkx as nx

from dac_her.config import get_paper_config
from dac_her.graph_io import knowledge_graph_to_networkx, save_graphml
from dac_her.paper_graph_postprocess import (
    canonicalize_paper_graph,
    load_resolution_plan,
    merge_node_attributes,
)
from dac_her.resolution_candidates import (
    build_raw_canonical_report,
    format_raw_canonical_report,
    generate_resolution_candidates,
    sync_decisions_jsonl,
    write_candidates_csv,
)
from dac_her.claim_overlap import write_claim_overlap_audit
from dac_her.run_state import (
    paper_output_root,
    read_json,
    resolve_run_directory,
    write_json,
)
from dac_her.schemas import KnowledgeGraph
from dac_her.validation import validate_graph_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "papers.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw and canonical paper-level GraphML graphs."
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--run-id",
        default=None,
        help="Default: latest_run.json for the paper.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Build despite failed chunks. Not recommended.",
    )
    parser.add_argument(
        "--decisions",
        default=None,
        help=(
            "Override paper-level decisions file. Supported: reviewed .jsonl "
            "or legacy aliases .json."
        ),
    )
    parser.add_argument(
        "--no-resolution",
        action="store_true",
        help="Build canonical graph identical to raw graph.",
    )
    return parser.parse_args()


def merge_chunk_graph(
    merged: nx.MultiDiGraph,
    chunk_graph: nx.MultiDiGraph,
    *,
    chunk_id: str,
) -> None:
    for node_id, node_data in chunk_graph.nodes(data=True):
        node_id = str(node_id)
        if node_id in merged:
            merged.nodes[node_id].update(
                merge_node_attributes(
                    dict(merged.nodes[node_id]),
                    dict(node_data),
                )
            )
        else:
            merged.add_node(node_id, **dict(node_data))

    for source, target, local_key, edge_data in chunk_graph.edges(
        keys=True,
        data=True,
    ):
        global_key = f"{chunk_id}:{local_key}"
        merged.add_edge(
            str(source),
            str(target),
            key=global_key,
            **dict(edge_data),
        )


def resolve_decisions_path(
    *,
    paper_resolution_file: Path | None,
    paper_root: Path,
    override: str | None,
    disabled: bool,
) -> Path | None:
    if disabled:
        return None
    if override:
        return Path(override).resolve()
    if paper_resolution_file is not None:
        return paper_resolution_file.resolve()

    default_jsonl = paper_root / "resolution" / "decisions.jsonl"
    if default_jsonl.exists():
        return default_jsonl.resolve()
    return None


def _candidate_summary(run_dir: Path) -> dict[str, object]:
    path = run_dir / "resolution" / "candidate_summary.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    paper = get_paper_config(
        args.config,
        project_root=PROJECT_ROOT,
        paper_id=args.paper_id,
    )
    run_dir = resolve_run_directory(
        project_root=PROJECT_ROOT,
        paper_id=paper.paper_id,
        run_id=args.run_id,
    )

    run_metadata = read_json(run_dir / "run.json")
    active_payload = read_json(run_dir / "active_chunks.json")

    if active_payload.get("paper_id") != paper.paper_id:
        raise ValueError(
            "active_chunks.json paper_id does not match the requested paper."
        )
    if active_payload.get("run_id") != run_metadata.get("run_id"):
        raise ValueError(
            "active_chunks.json and run.json refer to different runs."
        )
    if not active_payload.get("complete", False) and not args.allow_incomplete:
        raise RuntimeError(
            "The extraction run is incomplete. Resolve failed chunks before "
            "building, or pass --allow-incomplete explicitly."
        )

    chunk_records = active_payload.get("chunks")
    if not isinstance(chunk_records, list) or not chunk_records:
        raise RuntimeError("No active chunks are available for graph building.")

    merged = nx.MultiDiGraph(
        paper_id=paper.paper_id,
        run_id=str(run_metadata["run_id"]),
        run_fingerprint=str(run_metadata["run_fingerprint"]),
        graph_stage="raw_merged",
    )

    loaded_chunks = 0
    loaded_chunk_ids: list[str] = []

    for record in chunk_records:
        if not isinstance(record, dict):
            raise ValueError("Invalid chunk record in active_chunks.json.")

        json_path = Path(str(record["output_path"]))
        if not json_path.exists():
            raise FileNotFoundError(f"Active chunk JSON not found: {json_path}")

        result = KnowledgeGraph.model_validate_json(
            json_path.read_text(encoding="utf-8")
        )
        validate_graph_provenance(
            result,
            paper_id=paper.paper_id,
            chunk_id=str(record["chunk_id"]),
            section=str(record["section"]),
            document_id=str(record.get("document_id", "main")),
            document_role=str(record.get("document_role", "main")),
            page_ids=list(record.get("page_ids", [])),
            asset_ids=list(record.get("asset_ids", [])),
        )

        chunk_graph = knowledge_graph_to_networkx(result)
        merge_chunk_graph(merged, chunk_graph, chunk_id=result.chunk_id)
        loaded_chunks += 1
        loaded_chunk_ids.append(result.chunk_id)

    raw_graphml_path = run_dir / "raw_merged.graphml"
    save_graphml(merged, raw_graphml_path)

    paper_root = paper_output_root(PROJECT_ROOT, paper.paper_id)
    resolution_dir = run_dir / "resolution"
    resolution_dir.mkdir(parents=True, exist_ok=True)

    # Candidate generation is non-destructive. Only exact registry-safe
    # Metal/Reaction candidates are auto-approved; all others remain review.
    candidates, generated_candidate_summary = generate_resolution_candidates(merged)
    write_candidates_csv(resolution_dir / "candidates.csv", candidates)
    candidate_summary_payload = generated_candidate_summary.to_dict()
    write_json(resolution_dir / "candidate_summary.json", candidate_summary_payload)

    stable_resolution_dir = paper_root / "resolution"
    stable_resolution_dir.mkdir(parents=True, exist_ok=True)
    default_decisions_jsonl = stable_resolution_dir / "decisions.jsonl"
    sync_decisions_jsonl(default_decisions_jsonl, candidates)

    decisions_path = resolve_decisions_path(
        paper_resolution_file=paper.resolution_file,
        paper_root=paper_root,
        override=args.decisions,
        disabled=args.no_resolution,
    )
    resolution_plan = load_resolution_plan(
        decisions_path,
        graph=merged,
    )

    canonical = canonicalize_paper_graph(
        merged,
        aliases=resolution_plan.aliases,
        drop_node_ids=resolution_plan.drop_node_ids,
    )
    canonical.graph["graph_stage"] = "canonical"
    canonical.graph["resolution_file"] = (
        str(decisions_path) if decisions_path is not None else ""
    )
    canonical.graph["resolution_source_format"] = (
        resolution_plan.source_format
    )
    canonical.graph["approved_same_entity_decisions"] = (
        resolution_plan.approved_same_entity
    )
    canonical.graph["applied_resolution_aliases"] = (
        resolution_plan.applied_aliases
    )

    canonical_graphml_path = run_dir / "canonical.graphml"
    save_graphml(canonical, canonical_graphml_path)

    reloaded = nx.read_graphml(canonical_graphml_path)
    if reloaded.number_of_nodes() != canonical.number_of_nodes():
        raise AssertionError("GraphML node count changed after serialization.")
    if reloaded.number_of_edges() != canonical.number_of_edges():
        raise AssertionError("GraphML edge count changed after serialization.")

    for _, _, edge_data in reloaded.edges(data=True):
        for required in (
            "relation",
            "title",
            "chunk_id",
            "paper_id",
            "evidence_text",
            "document_id",
            "document_role",
            "evidence_pointers_json",
        ):
            if not edge_data.get(required):
                raise AssertionError(
                    f"Serialized edge is missing {required!r}."
                )

    latest_graphml_path = paper_root / f"{paper.paper_id}.graphml"
    latest_raw_path = paper_root / f"{paper.paper_id}.raw.graphml"
    shutil.copyfile(canonical_graphml_path, latest_graphml_path)
    shutil.copyfile(raw_graphml_path, latest_raw_path)

    comparison_report = build_raw_canonical_report(
        raw_graph=merged,
        canonical_graph=canonical,
        candidate_summary=candidate_summary_payload,
        resolution_summary=resolution_plan.summary(),
    )
    write_json(
        resolution_dir / "resolution_report.json",
        comparison_report,
    )
    report_text = format_raw_canonical_report(comparison_report)
    (resolution_dir / "resolution_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    write_json(
        stable_resolution_dir / "latest_resolution_report.json",
        comparison_report,
    )
    (stable_resolution_dir / "latest_resolution_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    claim_audit_summary = write_claim_overlap_audit(
        canonical,
        run_dir / "claim_audit",
    )

    build_summary = {
        "paper_id": paper.paper_id,
        "run_id": run_metadata["run_id"],
        "loaded_chunks": loaded_chunks,
        "loaded_chunk_ids": loaded_chunk_ids,
        "loaded_documents": sorted({
            str(record.get("document_id", "main"))
            for record in chunk_records
        }),
        "linked_asset_ids": sorted({
            str(asset_id)
            for record in chunk_records
            for asset_id in record.get("asset_ids", [])
        }),
        "raw_nodes": merged.number_of_nodes(),
        "raw_edges": merged.number_of_edges(),
        "canonical_nodes": canonical.number_of_nodes(),
        "canonical_edges": canonical.number_of_edges(),
        "resolution": resolution_plan.summary(),
        "resolution_candidates": candidate_summary_payload,
        "claim_overlap_audit": claim_audit_summary,
        "raw_graphml": str(raw_graphml_path),
        "canonical_graphml": str(canonical_graphml_path),
        "latest_graphml": str(latest_graphml_path),
        "resolution_report": str(
            resolution_dir / "resolution_report.json"
        ),
    }
    write_json(run_dir / "build_summary.json", build_summary)

    print("GraphML conversion successful")
    print("Paper:", paper.paper_id)
    print("Run ID:", run_metadata["run_id"])
    print("Chunks:", loaded_chunks)
    print("Raw nodes/edges:", merged.number_of_nodes(), merged.number_of_edges())
    print(
        "Canonical nodes/edges:",
        canonical.number_of_nodes(),
        canonical.number_of_edges(),
    )
    print(
        "Components:",
        comparison_report["raw"]["components"],
        "->",
        comparison_report["canonical"]["components"],
    )
    print(
        "Approved same_entity decisions:",
        resolution_plan.approved_same_entity,
    )
    print("Applied aliases:", resolution_plan.applied_aliases)
    print("Auto-approved safe candidates:", generated_candidate_summary.auto_approved_candidates)
    print("Claim-overlap review candidates:", claim_audit_summary["review_required"])
    print("Resolution report:", resolution_dir / "resolution_report.txt")
    print("Saved:", latest_graphml_path)


if __name__ == "__main__":
    main()
