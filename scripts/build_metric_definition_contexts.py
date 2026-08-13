from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.domains.metric_definition_registry import get_metric_definition_adapter
from dac_her.domains.registry import get_domain_profile
from dac_her.metric_definition_context import audit_metric_definition_contexts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build grounded metric-definition sidecars from strict canonical "
            "scientific graphs."
        )
    )
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="exploratory",
    )
    parser.add_argument("--metric-definition-id", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_domain_profile(args.domain_profile)
    adapter = get_metric_definition_adapter(profile)
    extraction_adapter = get_extraction_adapter(profile.profile_id)

    data_root = Path(args.data_root or extraction_adapter.default_data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    corpus_root = data_root / "corpus" / args.corpus_id / args.mode
    manifest_path = corpus_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Corpus manifest not found: {manifest_path}")
    corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(corpus_manifest, dict):
        raise ValueError("Corpus manifest must be a JSON object.")

    corpus_domain = str(corpus_manifest.get("domain_profile_id", ""))
    if corpus_domain != profile.profile_id:
        raise ValueError(
            "Metric-definition corpus/domain mismatch: "
            f"{corpus_domain!r} != {profile.profile_id!r}."
        )

    if profile.corpus is not None:
        corpus_semantics_id = str(corpus_manifest.get("corpus_semantics_id", ""))
        if corpus_semantics_id != profile.corpus.semantics_id:
            raise ValueError(
                "Metric-definition corpus semantics mismatch: "
                f"{corpus_semantics_id!r} != {profile.corpus.semantics_id!r}."
            )

    if int(corpus_manifest.get("destructive_cross_paper_merges", -1)) != 0:
        raise ValueError(
            "Metric-definition sidecar requires a non-destructive corpus."
        )

    paper_ids = [
        str(value)
        for value in corpus_manifest.get("paper_ids", [])
        if str(value).strip()
    ]
    if not paper_ids:
        raise ValueError("Corpus manifest contains no paper IDs.")
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Corpus manifest contains duplicate paper IDs.")

    source_graphs: dict[str, nx.Graph] = {}
    source_rows: list[dict[str, object]] = []
    contexts = []

    for paper_id in paper_ids:
        graph_path = data_root / "extracted" / paper_id / f"{paper_id}.graphml"
        if not graph_path.exists():
            raise FileNotFoundError(f"Canonical graph not found: {graph_path}")
        graph = nx.read_graphml(graph_path, force_multigraph=True)
        graph_domain = str(graph.graph.get("domain_profile_id", ""))
        if graph_domain and graph_domain != profile.profile_id:
            raise ValueError(
                "Canonical graph/domain mismatch for "
                f"{paper_id}: {graph_domain!r} != {profile.profile_id!r}."
            )
        source_graphs[paper_id] = graph
        paper_contexts = adapter.extract_contexts(graph, paper_id)
        contexts.extend(paper_contexts)
        source_rows.append({
            "paper_id": paper_id,
            "canonical_graphml": str(graph_path),
            "canonical_graph_sha256": _sha256_file(graph_path),
            "metric_definition_context_count": len(paper_contexts),
        })

    audit = audit_metric_definition_contexts(
        contexts=contexts,
        source_graphs=source_graphs,
        adapter=adapter,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else corpus_root / "metric_definition" / args.metric_definition_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts_path = _write_jsonl(
        output_dir / "contexts.jsonl",
        (item.to_row() for item in contexts),
    )
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status_counts = Counter(item.definition_status for item in contexts)
    family_counts = Counter(item.definition_family for item in contexts)
    aggregation_counts = Counter(item.aggregation_scope for item in contexts)
    observable_counts = Counter(item.observable_key for item in contexts)
    summary = {
        "metric_definition_id": args.metric_definition_id,
        "domain_profile_id": profile.profile_id,
        "metric_definition_adapter_id": adapter.adapter_id,
        "metric_definition_semantics_id": adapter.semantics_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "corpus_manifest": str(manifest_path),
        "corpus_semantics_id": str(corpus_manifest.get("corpus_semantics_id", "")),
        "paper_ids": paper_ids,
        "paper_count": len(paper_ids),
        "source_graphs": source_rows,
        "context_count": len(contexts),
        "observable_counts": dict(sorted(observable_counts.items())),
        "definition_status_counts": dict(sorted(status_counts.items())),
        "definition_family_counts": dict(sorted(family_counts.items())),
        "aggregation_scope_counts": dict(sorted(aggregation_counts.items())),
        "explicit_formula_count": audit.explicit_formula_count,
        "structural_gate": audit.structural_gate,
        "outputs": {
            "contexts": str(contexts_path),
            "audit": str(audit_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Metric-definition contexts built")
    print("Metric-definition ID:", args.metric_definition_id)
    print("Domain profile:", profile.profile_id)
    print("Metric-definition semantics:", adapter.semantics_id)
    print("Papers:", len(paper_ids))
    print("Contexts:", len(contexts))
    print("Observables:", json.dumps(summary["observable_counts"], sort_keys=True))
    print("Definition status:", json.dumps(summary["definition_status_counts"], sort_keys=True))
    print("Definition families:", json.dumps(summary["definition_family_counts"], sort_keys=True))
    print("Aggregation scopes:", json.dumps(summary["aggregation_scope_counts"], sort_keys=True))
    print("Explicit formulas:", audit.explicit_formula_count)
    print("Structural gate:", audit.structural_gate)
    print("Saved:", output_dir)
    return 0 if audit.structural_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
