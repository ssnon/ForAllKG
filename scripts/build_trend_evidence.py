from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx

from dac_her.domains.comparison_registry import get_comparison_adapter
from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.domains.registry import get_domain_profile
from dac_her.domains.trend_registry import get_trend_adapter
from dac_her.measurement_result_identity import (
    identity_source_hashes,
    load_measurement_result_identity_sidecar,
)
from dac_her.trend_domain import TrendEvidenceSource
from dac_her.trend_evidence import audit_trend_evidence


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(summary: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row.get("paper_id", "")): str(
            row.get("canonical_graph_sha256", "")
        )
        for row in summary.get("source_graphs", [])
        if isinstance(row, dict)
    }


def _load_comparison_sidecar(
    *,
    corpus_root: Path,
    comparison_id: str,
    profile,
    corpus_id: str,
    corpus_mode: str,
    measurement_result_identity_id: str,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
    dict[str, str],
]:
    comparison_adapter = get_comparison_adapter(profile)
    root = corpus_root / "comparison" / comparison_id
    summary_path = root / "summary.json"
    contexts_path = root / "contexts.jsonl"
    method_path = root / "method_contexts.jsonl"
    audit_path = root / "audit.json"
    for path in (summary_path, contexts_path, method_path, audit_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Comparison sidecar file not found: {path}"
            )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(audit, dict):
        raise ValueError("Comparison sidecar metadata must be JSON objects.")

    expected = {
        "comparison_id": comparison_id,
        "domain_profile_id": profile.profile_id,
        "corpus_id": corpus_id,
        "corpus_mode": corpus_mode,
        "comparison_semantics_id": comparison_adapter.semantics_id,
        "measurement_result_identity_id": measurement_result_identity_id,
    }
    for key, expected_value in expected.items():
        observed = str(summary.get(key, ""))
        if observed != str(expected_value):
            raise ValueError(
                "Comparison sidecar binding mismatch for "
                f"{key}: {observed!r} != {expected_value!r}."
            )
    if comparison_adapter.method_semantics is not None:
        observed_method = str(summary.get("method_semantics_id", ""))
        expected_method = comparison_adapter.method_semantics.semantics_id
        if observed_method != expected_method:
            raise ValueError(
                "Comparison sidecar MethodContext semantics mismatch: "
                f"{observed_method!r} != {expected_method!r}."
            )
    if not bool(summary.get("passes_structural_gate", False)):
        raise ValueError("Comparison sidecar structural gate is false.")
    if not bool(audit.get("passes_structural_gate", False)):
        raise ValueError("Comparison sidecar audit structural gate is false.")

    contexts_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(contexts_path):
        contexts_by_paper[str(row.get("paper_id", ""))].append(row)

    methods_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(method_path):
        methods_by_paper[str(row.get("paper_id", ""))].append(row)

    return (
        dict(contexts_by_paper),
        dict(methods_by_paper),
        summary,
        _source_hashes(summary),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build grounded, paper-local TrendEvidence sidecars. Domain "
            "adapters declare which frozen source sidecars are required."
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
    parser.add_argument("--trend-id", required=True)
    parser.add_argument(
        "--measurement-result-identity-id",
        default=None,
        help="MeasurementResultIdentity sidecar required by identity-aware trend adapters.",
    )
    parser.add_argument(
        "--comparison-id",
        default=None,
        help="Comparison sidecar supplying identity-aware ComparisonContext and MethodContext rows.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_domain_profile(args.domain_profile)
    adapter = get_trend_adapter(profile)
    extraction_adapter = get_extraction_adapter(profile.profile_id)

    if (
        "measurement_result_identity" in adapter.required_inputs
        and not args.measurement_result_identity_id
    ):
        raise ValueError(
            "--measurement-result-identity-id is required by trend adapter "
            f"{adapter.adapter_id!r}."
        )
    if (
        {"method_context", "comparison_context"} & adapter.required_inputs
        and not args.comparison_id
    ):
        raise ValueError(
            "--comparison-id is required by trend adapter "
            f"{adapter.adapter_id!r}."
        )

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
            "Trend corpus/domain mismatch: "
            f"{corpus_domain!r} != {profile.profile_id!r}."
        )
    if profile.corpus is not None:
        observed = str(corpus_manifest.get("corpus_semantics_id", ""))
        if observed != profile.corpus.semantics_id:
            raise ValueError(
                "Trend corpus semantics mismatch: "
                f"{observed!r} != {profile.corpus.semantics_id!r}."
            )
    if int(corpus_manifest.get("destructive_cross_paper_merges", -1)) != 0:
        raise ValueError("Trend sidecar requires a non-destructive corpus.")

    paper_ids = [
        str(value)
        for value in corpus_manifest.get("paper_ids", [])
        if str(value).strip()
    ]
    if not paper_ids:
        raise ValueError("Corpus manifest contains no paper IDs.")
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError("Corpus manifest contains duplicate paper IDs.")

    identity_by_paper: dict[str, list[Any]] = {}
    identity_summary: dict[str, Any] = {}
    identity_hashes: dict[str, str] = {}
    if args.measurement_result_identity_id:
        identity_by_paper, identity_summary, identity_audit = (
            load_measurement_result_identity_sidecar(
                corpus_root=corpus_root,
                identity_id=str(args.measurement_result_identity_id),
                profile_id=profile.profile_id,
                corpus_id=args.corpus_id,
                corpus_mode=args.mode,
            )
        )
        if not bool(identity_audit.get("structural_gate", False)):
            raise ValueError(
                "Measurement-result identity sidecar structural gate is false."
            )
        identity_hashes = identity_source_hashes(identity_summary)

    comparison_by_paper: dict[str, list[dict[str, Any]]] = {}
    method_by_paper: dict[str, list[dict[str, Any]]] = {}
    comparison_summary: dict[str, Any] = {}
    comparison_hashes: dict[str, str] = {}
    if args.comparison_id:
        (
            comparison_by_paper,
            method_by_paper,
            comparison_summary,
            comparison_hashes,
        ) = _load_comparison_sidecar(
            corpus_root=corpus_root,
            comparison_id=str(args.comparison_id),
            profile=profile,
            corpus_id=args.corpus_id,
            corpus_mode=args.mode,
            measurement_result_identity_id=str(
                args.measurement_result_identity_id or ""
            ),
        )

    sources: dict[str, TrendEvidenceSource] = {}
    source_rows: list[dict[str, object]] = []
    evidence = []

    for paper_id in paper_ids:
        graph_path = data_root / "extracted" / paper_id / f"{paper_id}.graphml"
        if not graph_path.exists():
            raise FileNotFoundError(f"Canonical graph not found: {graph_path}")
        graph_hash = _sha256_file(graph_path)
        graph = nx.read_graphml(graph_path, force_multigraph=True)
        graph_domain = str(graph.graph.get("domain_profile_id", ""))
        if graph_domain and graph_domain != profile.profile_id:
            raise ValueError(
                f"Canonical graph/domain mismatch for {paper_id}: "
                f"{graph_domain!r} != {profile.profile_id!r}."
            )

        identities = identity_by_paper.get(paper_id, [])
        if args.measurement_result_identity_id:
            if identity_hashes.get(paper_id, "") != graph_hash:
                raise ValueError(
                    "Measurement-result identity canonical graph hash mismatch "
                    f"for {paper_id}."
                )
            if not identities:
                raise ValueError(
                    f"Measurement-result identity rows missing for {paper_id}."
                )

        contexts = comparison_by_paper.get(paper_id, [])
        methods = method_by_paper.get(paper_id, [])
        if args.comparison_id:
            if comparison_hashes.get(paper_id, "") != graph_hash:
                raise ValueError(
                    f"Comparison canonical graph hash mismatch for {paper_id}."
                )
            if not contexts or not methods:
                raise ValueError(
                    f"Comparison/MethodContext rows missing for {paper_id}."
                )

        source = TrendEvidenceSource(
            graph=graph,
            paper_id=paper_id,
            measurement_result_rows=tuple(
                item.to_row() for item in identities
            ),
            method_context_rows=tuple(methods),
            comparison_context_rows=tuple(contexts),
        )
        sources[paper_id] = source
        paper_evidence = adapter.extract_evidence(source)
        evidence.extend(paper_evidence)
        source_rows.append({
            "paper_id": paper_id,
            "canonical_graphml": str(graph_path),
            "canonical_graph_sha256": graph_hash,
            "trend_evidence_count": len(paper_evidence),
            "measurement_result_count": len(identities),
            "method_context_count": len(methods),
            "comparison_context_count": len(contexts),
            "available_trend_inputs": sorted(source.available_inputs),
        })

    audit = audit_trend_evidence(
        evidence=evidence,
        sources=sources,
        adapter=adapter,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else corpus_root / "trend" / args.trend_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = _write_jsonl(
        output_dir / "evidence.jsonl",
        (item.to_row() for item in evidence),
    )
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    basis_counts = Counter(item.evidence_basis for item in evidence)
    direction_counts = Counter(item.direction for item in evidence)
    shape_counts = Counter(item.shape for item in evidence)
    control_counts = Counter(item.independent_variable_key for item in evidence)
    response_counts = Counter(item.dependent_observable_key for item in evidence)
    summary = {
        "trend_id": args.trend_id,
        "domain_profile_id": profile.profile_id,
        "trend_adapter_id": adapter.adapter_id,
        "trend_semantics_id": adapter.semantics_id,
        "contract_semantics_id": "trend_evidence_contract_v1_alpha4c1",
        "corpus_id": args.corpus_id,
        "corpus_mode": args.mode,
        "corpus_manifest": str(manifest_path),
        "corpus_semantics_id": str(corpus_manifest.get("corpus_semantics_id", "")),
        "measurement_result_identity_id": str(args.measurement_result_identity_id or ""),
        "measurement_result_identity_semantics_id": str(
            identity_summary.get("measurement_result_identity_semantics_id", "")
        ),
        "comparison_id": str(args.comparison_id or ""),
        "comparison_semantics_id": str(
            comparison_summary.get("comparison_semantics_id", "")
        ),
        "method_semantics_id": str(comparison_summary.get("method_semantics_id", "")),
        "paper_ids": paper_ids,
        "paper_count": len(paper_ids),
        "source_graphs": source_rows,
        "evidence_count": len(evidence),
        "quantitative_evidence_count": audit.quantitative_evidence_count,
        "claim_evidence_count": audit.claim_evidence_count,
        "source_asserted_causal_count": audit.source_asserted_causal_count,
        "evidence_basis_counts": dict(sorted(basis_counts.items())),
        "direction_counts": dict(sorted(direction_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "independent_variable_counts": dict(sorted(control_counts.items())),
        "dependent_observable_counts": dict(sorted(response_counts.items())),
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

    print("Trend evidence built")
    print("Trend ID:", args.trend_id)
    print("Domain profile:", profile.profile_id)
    print("Trend semantics:", adapter.semantics_id)
    print("Papers:", len(paper_ids))
    print("Evidence:", len(evidence))
    print("Quantitative evidence:", audit.quantitative_evidence_count)
    print("Claim evidence:", audit.claim_evidence_count)
    print("Evidence bases:", json.dumps(summary["evidence_basis_counts"], sort_keys=True))
    print("Controls:", json.dumps(summary["independent_variable_counts"], sort_keys=True))
    print("Responses:", json.dumps(summary["dependent_observable_counts"], sort_keys=True))
    print("Directions:", json.dumps(summary["direction_counts"], sort_keys=True))
    print("Shapes:", json.dumps(summary["shape_counts"], sort_keys=True))
    print("Structural gate:", audit.structural_gate)
    print("Saved:", output_dir)
    return 0 if audit.structural_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
