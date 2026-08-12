from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from dac_her.broad_projection import (
    build_broad_mechanism_projection,
    summarize_broad_projection,
)
from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.domains.graph_registry import get_graph_adapter
from dac_her.domains.registry import get_domain_profile
from dac_her.extraction_quality import projection_quality_summary
from dac_her.graph_io import save_graphml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the bridge-free GraphAgents mechanism projection used by "
            "the broad abstract catalysis corpus."
        )
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--domain-profile", default="catalysis_mechanism")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--canonical-graphml", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_domain_profile(args.domain_profile)
    if profile.profile_id != "catalysis_mechanism":
        raise ValueError(
            "build_broad_projection is reserved for the "
            "catalysis_mechanism domain profile."
        )
    extraction_adapter = get_extraction_adapter(profile.profile_id)
    graph_adapter = get_graph_adapter(profile.profile_id)
    if graph_adapter.adapter_id != "catalysis_mechanism":
        raise RuntimeError(
            "Broad projection requires the catalysis_mechanism graph adapter."
        )

    data_root = Path(args.data_root or extraction_adapter.default_data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    paper_root = data_root / "extracted" / args.paper_id
    canonical_path = (
        Path(args.canonical_graphml)
        if args.canonical_graphml
        else paper_root / f"{args.paper_id}.graphml"
    )
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical graph not found: {canonical_path}")

    canonical_graph = nx.read_graphml(canonical_path, force_multigraph=True)
    canonical_domain = str(canonical_graph.graph.get("domain_profile_id", ""))
    if canonical_domain and canonical_domain != profile.profile_id:
        raise ValueError(
            "Canonical graph/domain mismatch: "
            f"{canonical_domain!r} != {profile.profile_id!r}"
        )

    relation_issues = graph_adapter.diagnose_relation_contracts(canonical_graph)
    errors = [issue for issue in relation_issues if issue.severity == "error"]
    if errors:
        raise RuntimeError(
            "Broad canonical graph violates graph-domain relation contracts: "
            + "; ".join(issue.message for issue in errors[:10])
        )

    projection, node_rows, evidence_rows = build_broad_mechanism_projection(
        canonical_graph
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else paper_root / "graphagents" / "mechanism"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = save_graphml(projection, output_dir / "graph.graphml")
    node_path = _write_jsonl(output_dir / "node_text.jsonl", node_rows)
    evidence_path = _write_jsonl(
        output_dir / "edge_evidence.jsonl", evidence_rows
    )
    summary = {
        "paper_id": args.paper_id,
        "domain_profile_id": profile.profile_id,
        "graph_adapter_id": graph_adapter.adapter_id,
        "data_root": str(data_root),
        "mode": "mechanism",
        "projection_builder": "broad_direct_abstract_v2_run_bound",
        "canonical_graphml": str(canonical_path),
        "source_extraction_run_id": str(
            canonical_graph.graph.get("run_id") or ""
        ),
        "source_extraction_run_fingerprint": str(
            canonical_graph.graph.get("run_fingerprint") or ""
        ),
        **projection_quality_summary(projection),
        **summarize_broad_projection(projection),
        "node_text_rows": len(node_rows),
        "edge_evidence_rows": len(evidence_rows),
        "relation_contract_warning_count": len(relation_issues),
        "graphml": str(graph_path),
        "node_text": str(node_path),
        "edge_evidence": str(evidence_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Broad mechanism projection built")
    print("Paper ID:", args.paper_id)
    print("Nodes/edges:", projection.number_of_nodes(), projection.number_of_edges())
    print("Direct mechanism edges:", summary["direct_mechanism_edges"])
    print("Saved:", graph_path)


if __name__ == "__main__":
    main()
