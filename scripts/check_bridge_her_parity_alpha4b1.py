from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx


_VOLATILE_GRAPH_KEYS = {
    "bridge_extraction_id",
    "bridge_extraction_fingerprint",
    "bridge_policy_run_id",
    "bridge_policy_run_fingerprint",
    "bridge_run_id",
    "bridge_run_fingerprint",
    "domain_profile_id",
    "bridge_adapter_id",
}

_VOLATILE_SUMMARY_KEYS = {
    "bridge_extraction_id",
    "bridge_extraction_fingerprint",
    "bridge_policy_run_id",
    "bridge_policy_run_fingerprint",
    "bridge_run_id",
    "bridge_run_fingerprint",
    "canonical_graph_used",
    "raw_graphml",
    "bridge_graphml",
    "latest_bridge_graphml",
    "candidate_graphml",
    "latest_candidate_bridge_graphml",
    "domain_profile_id",
    "bridge_adapter_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a frozen HER Bridge policy run with an alpha4b.1 "
            "candidate while ignoring run IDs/paths but requiring scientific "
            "chunk outputs and graph semantics to match."
        )
    )
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    return parser.parse_args()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _chunk_payloads(root: Path) -> dict[str, str]:
    chunk_dir = root / "chunks"
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Missing chunks directory: {chunk_dir}")
    rows: dict[str, str] = {}
    for path in sorted(chunk_dir.glob("*.json")):
        rows[path.name] = _canonical(_json(path))
    return rows


def _clean_summary(root: Path) -> dict[str, Any]:
    value = dict(_json(root / "summary.json"))
    for key in _VOLATILE_SUMMARY_KEYS:
        value.pop(key, None)
    return value


def _node_signature(node_id: str, attrs: dict[str, Any]) -> tuple:
    cleaned = {
        str(key): str(value)
        for key, value in attrs.items()
        if key not in _VOLATILE_GRAPH_KEYS
    }
    return node_id, _canonical(cleaned)


def _edge_signature(source: str, target: str, attrs: dict[str, Any]) -> tuple:
    cleaned = {
        str(key): str(value)
        for key, value in attrs.items()
        if key not in _VOLATILE_GRAPH_KEYS
    }
    return source, target, _canonical(cleaned)


def _graph_signature(path: Path) -> tuple[list[tuple], list[tuple]]:
    graph = nx.read_graphml(path, force_multigraph=True)
    nodes = sorted(
        _node_signature(str(node_id), dict(attrs))
        for node_id, attrs in graph.nodes(data=True)
    )
    edges = sorted(
        _edge_signature(str(source), str(target), dict(attrs))
        for source, target, attrs in graph.edges(data=True)
    )
    return nodes, edges


def main() -> int:
    args = parse_args()
    baseline = Path(args.baseline_dir).resolve()
    candidate = Path(args.candidate_dir).resolve()

    failures: list[str] = []

    if _chunk_payloads(baseline) != _chunk_payloads(candidate):
        failures.append("filtered/candidate/rejection chunk JSON differs")

    if _clean_summary(baseline) != _clean_summary(candidate):
        failures.append("non-volatile summary semantics differ")

    for name in ("bridge.graphml", "bridge.candidates.graphml"):
        left = baseline / name
        right = candidate / name
        if not left.exists() or not right.exists():
            failures.append(f"missing graph artifact: {name}")
            continue
        if _graph_signature(left) != _graph_signature(right):
            failures.append(f"scientific graph semantics differ: {name}")

    if failures:
        print("alpha4b.1 HER parity: FAIL")
        for failure in failures:
            print(" -", failure)
        return 1

    print("alpha4b.1 HER parity: PASS")
    print("Scientific chunk outputs, non-volatile summary fields, and Bridge graph semantics match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
