from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.domains.registry import get_domain_profile
from dac_her.domains.reproducibility_registry import get_reproducibility_adapter
from dac_her.reproducibility_evidence import audit_reproducibility_evidence


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
            "Build grounded reproducibility-quality sidecars from strict "
            "canonical scientific graphs."
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
    parser.add_argument("--reproducibility-id", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_domain_profile(args.domain_profile)
    adapter = get_reproducibility_adapter(profile)
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
            "Reproducibility corpus/domain mismatch: "
            f"{corpus_domain!r} != {profile.profile_id!r}."
        )

    if profile.corpus is not None:
        corpus_semantics_id = str(corpus_manifest.get("corpus_semantics_id", ""))
        if corpus_semantics_id != profile.corpus.semantics_id:
            raise ValueError(
                "Reproducibility corpus semantics mismatch: "
                f"{corpus_semantics_id!r} != {profile.corpus.semantics_id!r}."
            )

    if int(corpus_manifest.get("destructive_cross_paper_merges", -1)) != 0:
        raise ValueError(
            "Reproducibility sidecar requires a non-destructive corpus."
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
    evidence = []

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
        paper_evidence = adapter.extract_evidence(graph, paper_id)
        evidence.extend(paper_evidence)
        source_rows.append({
            "paper_id": paper_id,
            "canonical_graphml": str(graph_path),
            "canonical_graph_sha256": _sha256_file(graph_path),
            "reproducibility_evidence_count": len(paper_evidence),
        })

    audit = audit_reproducibility_evidence(
        evidence=evidence,
        source_graphs=source_graphs,
        adapter=adapter,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else corpus_root / "reproducibility" / args.reproducibility_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = _write_jsonl(
        output_dir / "evidence.jsonl",
        (item.to_row() for item in evidence),
    )
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    kind_counts = Counter(item.evidence_kind for item in evidence)
    scope_counts = Counter(item.reproducibility_scope for item in evidence)
    summary = {
        "reproducibility_id": args.reproducibility_id,
        "domain_profile_id": profile.profile_id,
        "reproducibility_adapter_id": adapter.adapter_id,
        "reproducibility_semantics_id": adapter.semantics_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "corpus_manifest": str(manifest_path),
        "corpus_semantics_id": str(corpus_manifest.get("corpus_semantics_id", "")),
        "paper_ids": paper_ids,
        "paper_count": len(paper_ids),
        "source_graphs": source_rows,
        "evidence_count": len(evidence),
        "quantitative_evidence_count": sum(
            item.value_numeric is not None for item in evidence
        ),
        "evidence_kind_counts": dict(sorted(kind_counts.items())),
        "scope_counts": dict(sorted(scope_counts.items())),
        "source_mention_count": audit.source_mention_count,
        "consolidated_result_count": audit.consolidated_result_count,
        "possible_duplicate_result_pair_count": (
            audit.possible_duplicate_result_pair_count
        ),
        "possible_duplicate_result_cluster_count": (
            audit.possible_duplicate_result_cluster_count
        ),
        "structural_gate": audit.structural_gate,
        "outputs": {
            "evidence": str(evidence_path),
            "audit": str(audit_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Reproducibility evidence built")
    print("Reproducibility ID:", args.reproducibility_id)
    print("Domain profile:", profile.profile_id)
    print("Reproducibility semantics:", adapter.semantics_id)
    print("Papers:", len(paper_ids))
    print("Evidence:", len(evidence))
    print("Quantitative evidence:", summary["quantitative_evidence_count"])
    print("Evidence kinds:", json.dumps(summary["evidence_kind_counts"], sort_keys=True))
    print("Scopes:", json.dumps(summary["scope_counts"], sort_keys=True))
    print("Source mentions:", audit.source_mention_count)
    print("Consolidated exact results:", audit.consolidated_result_count)
    print(
        "Possible duplicate result pairs/clusters:",
        audit.possible_duplicate_result_pair_count,
        "/",
        audit.possible_duplicate_result_cluster_count,
    )
    print("Structural gate:", audit.structural_gate)
    print("Saved:", output_dir)
    return 0 if audit.structural_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
