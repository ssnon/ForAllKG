from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from dac_her.broad_corpus_audit import write_broad_corpus_audit
from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.domains.registry import get_domain_profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit mechanism coverage in a Broad Catalysis corpus graph."
    )
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--domain-profile", default="catalysis_mechanism")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--paper-ids", nargs="*", default=[])
    parser.add_argument("--graphml", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_domain_profile(args.domain_profile)
    if profile.profile_id != "catalysis_mechanism":
        raise ValueError("Broad corpus audit requires catalysis_mechanism profile.")
    extraction_adapter = get_extraction_adapter(profile.profile_id)
    data_root = Path(args.data_root or extraction_adapter.default_data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    mode_root = data_root / "corpus" / args.corpus_id / "mechanism"
    graph_path = Path(args.graphml) if args.graphml else mode_root / "graph.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"Broad corpus graph not found: {graph_path}")

    graph = nx.read_graphml(graph_path, force_multigraph=True)
    output_dir = Path(args.output_dir) if args.output_dir else mode_root
    report_path, signatures_path = write_broad_corpus_audit(
        graph=graph,
        output_dir=output_dir,
        expected_paper_ids=args.paper_ids,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print("Broad corpus audit complete")
    print("Observed papers:", report["observed_paper_count"])
    print(
        "Mechanism-bearing papers:",
        report["mechanism_bearing_paper_count"],
        f"({report['mechanism_bearing_paper_fraction']:.1%})",
    )
    print("Direct mechanism edges:", report["direct_mechanism_edges"])
    print("Unique mechanism signatures:", report["unique_mechanism_signatures"])
    print("Recurring signatures:", report["recurring_mechanism_signatures"])
    print("Report:", report_path)
    print("Signatures:", signatures_path)


if __name__ == "__main__":
    main()
