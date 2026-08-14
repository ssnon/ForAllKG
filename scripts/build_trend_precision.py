from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from dac_her.domains.registry import get_domain_profile
from dac_her.domains.trend_precision_registry import get_trend_precision_adapter
from dac_her.domains.trend_registry import get_trend_adapter
from dac_her.trend_precision import audit_trend_precision


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL row must be an object at {path}:{line_number}."
                )
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
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
            "Annotate raw TrendEvidence with evidence-kind semantics and "
            "build paper-local scientific trend identities."
        )
    )
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--data-root", default="data_sers")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument(
        "--mode",
        choices=("evidence", "mechanism", "exploratory"),
        default="exploratory",
    )
    parser.add_argument("--trend-id", required=True)
    parser.add_argument("--precision-id", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_domain_profile(args.domain_profile)
    trend_adapter = get_trend_adapter(profile)
    precision_adapter = get_trend_precision_adapter(profile)
    if precision_adapter.trend_semantics_id != trend_adapter.semantics_id:
        raise ValueError(
            "Trend/precision semantics mismatch: "
            f"{trend_adapter.semantics_id!r} != "
            f"{precision_adapter.trend_semantics_id!r}."
        )

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    corpus_root = data_root / "corpus" / args.corpus_id / args.mode
    trend_root = corpus_root / "trend" / args.trend_id
    trend_summary_path = trend_root / "summary.json"
    evidence_path = trend_root / "evidence.jsonl"
    audit_path = trend_root / "audit.json"
    for path in (trend_summary_path, evidence_path, audit_path):
        if not path.exists():
            raise FileNotFoundError(path)

    trend_summary = json.loads(trend_summary_path.read_text(encoding="utf-8"))
    trend_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(trend_summary.get("structural_gate", False)):
        raise ValueError("Raw TrendEvidence structural gate is false.")
    if not bool(trend_audit.get("structural_gate", False)):
        raise ValueError("Raw TrendEvidence audit structural gate is false.")

    expected_bindings = {
        "domain_profile_id": profile.profile_id,
        "trend_id": args.trend_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "trend_semantics_id": trend_adapter.semantics_id,
    }
    for key, expected in expected_bindings.items():
        observed = str(trend_summary.get(key, ""))
        if observed != str(expected):
            raise ValueError(
                f"Trend sidecar binding mismatch for {key}: "
                f"{observed!r} != {expected!r}."
            )

    graphs: dict[str, nx.Graph] = {}
    source_hashes: dict[str, str] = {}
    for source_row in trend_summary.get("source_graphs", []):
        if not isinstance(source_row, dict):
            continue
        paper_id = str(source_row.get("paper_id", ""))
        graph_path = Path(str(source_row.get("canonical_graphml", "")))
        if not graph_path.is_absolute():
            graph_path = PROJECT_ROOT / graph_path
        if not graph_path.exists():
            raise FileNotFoundError(graph_path)
        observed_hash = _sha256_file(graph_path)
        expected_hash = str(source_row.get("canonical_graph_sha256", ""))
        if observed_hash != expected_hash:
            raise ValueError(
                f"Canonical graph hash drift for {paper_id}: "
                f"{observed_hash} != {expected_hash}."
            )
        graphs[paper_id] = nx.read_graphml(graph_path, force_multigraph=True)
        source_hashes[paper_id] = observed_hash

    evidence_rows = _read_jsonl(evidence_path)
    annotations = []
    for row in evidence_rows:
        paper_id = str(row.get("paper_id", ""))
        graph = graphs.get(paper_id)
        if graph is None:
            raise ValueError(f"Missing canonical graph for trend paper {paper_id!r}.")
        annotations.append(precision_adapter.annotate(row, graph))

    results = precision_adapter.consolidate(evidence_rows, annotations, graphs)
    audit = audit_trend_precision(
        evidence_rows=evidence_rows,
        annotations=annotations,
        results=results,
        adapter=precision_adapter,
    )

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else trend_root / "precision" / args.precision_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    annotations_path = _write_jsonl(
        output_root / "annotations.jsonl",
        (row.to_row() for row in annotations),
    )
    results_path = _write_jsonl(
        output_root / "local_results.jsonl",
        (row.to_row() for row in results),
    )
    precision_audit_path = output_root / "audit.json"
    precision_audit_path.write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    control_key_counts = Counter(
        str(row.get("independent_variable_key", "")) for row in evidence_rows
    )
    observable_key_counts = Counter(
        str(row.get("dependent_observable_key", "")) for row in evidence_rows
    )
    local_relation_counts = Counter(
        (
            row.independent_variable_key,
            row.dependent_observable_key,
            row.direction,
        )
        for row in results
    )
    summary = {
        "precision_id": args.precision_id,
        "domain_profile_id": profile.profile_id,
        "trend_id": args.trend_id,
        "trend_semantics_id": trend_adapter.semantics_id,
        "precision_semantics_id": precision_adapter.precision_semantics_id,
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "source_graph_sha256": source_hashes,
        "evidence_count": len(evidence_rows),
        "annotation_count": len(annotations),
        "local_result_count": len(results),
        "evidence_kind_counts": audit.evidence_kind_counts,
        "result_lane_counts": audit.result_lane_counts,
        "control_family_counts": audit.control_family_counts,
        "observable_semantics_counts": audit.observable_semantics_counts,
        "control_key_counts": dict(sorted(control_key_counts.items())),
        "observable_key_counts": dict(sorted(observable_key_counts.items())),
        "duplicate_claim_mentions_collapsed": audit.duplicate_claim_mentions_collapsed,
        "claim_result_count_with_multiple_mentions": audit.claim_result_count_with_multiple_mentions,
        "local_relation_counts": {
            "|".join(key): value for key, value in sorted(local_relation_counts.items())
        },
        "structural_gate": audit.structural_gate,
        "outputs": {
            "annotations": str(annotations_path),
            "local_results": str(results_path),
            "audit": str(precision_audit_path),
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Trend precision built")
    print("Precision ID:", args.precision_id)
    print("Trend semantics:", trend_adapter.semantics_id)
    print("Precision semantics:", precision_adapter.precision_semantics_id)
    print("Evidence / local results:", len(evidence_rows), "/", len(results))
    print("Evidence kinds:", json.dumps(audit.evidence_kind_counts, sort_keys=True))
    print("Control families:", json.dumps(audit.control_family_counts, sort_keys=True))
    print("Observable semantics:", json.dumps(audit.observable_semantics_counts, sort_keys=True))
    print("Duplicate claim mentions collapsed:", audit.duplicate_claim_mentions_collapsed)
    print("Structural gate:", audit.structural_gate)
    print("Saved:", output_root)
    return 0 if audit.structural_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
