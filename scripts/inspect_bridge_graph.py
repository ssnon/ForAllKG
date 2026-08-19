from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import networkx as nx

from domains.dac_her.bridge_audit import write_bridge_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a Bridge v2 graph and write bridge-specific audits."
    )
    parser.add_argument("--paper-id", default=None)
    parser.add_argument("--graphml", default=None)
    parser.add_argument("--rejections", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _read_rejections(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    if not args.graphml and not args.paper_id:
        raise ValueError("Provide --paper-id or --graphml.")

    if args.graphml:
        graph_path = Path(args.graphml)
        paper_root = graph_path.parent
    else:
        paper_root = PROJECT_ROOT / "data_dac" / "extracted" / args.paper_id
        graph_path = paper_root / f"{args.paper_id}.bridge.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)

    rejection_path = Path(args.rejections) if args.rejections else None
    if rejection_path is None and args.paper_id:
        latest = paper_root / "latest_run.json"
        # Bridge v2 stores the convenient latest copy under the strict run's
        # bridge directory; users may also pass --rejections explicitly.
        candidates = sorted(
            paper_root.glob("runs/*/bridge/bridge_rejected.csv"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            rejection_path = candidates[0]

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else paper_root / "bridge_audit"
    )
    graph = nx.read_graphml(graph_path, force_multigraph=True)
    report = write_bridge_audit(
        graph,
        output_dir=output_dir,
        rejection_rows=_read_rejections(rejection_path),
    )

    print("Bridge audit finished")
    print("Graph:", graph_path)
    print("Patterns:", report["patterns"])
    print("Frontier concepts:", report["frontier_concepts"])
    print("Rejected candidates:", report["rejected_candidates"])
    print("Pattern issues:", report["pattern_issues"])
    print("Ready for projection:", report["ready_for_projection"])
    print("Saved:", output_dir / "bridge_audit.json")


if __name__ == "__main__":
    main()
