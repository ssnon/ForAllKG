from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

from dac_her.domains.graph_registry import get_graph_adapter
from dac_her.graph_semantics import write_graph_semantics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit graph-stage semantic contracts for a GraphML paper graph."
    )
    parser.add_argument("--graphml", required=True)
    parser.add_argument("--domain-profile", default="sers_au_ag")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = nx.read_graphml(args.graphml)
    adapter = get_graph_adapter(args.domain_profile)
    output_dir = Path(args.output_dir or Path(args.graphml).with_suffix("")) / "graph_semantics_audit"
    summary = write_graph_semantics_report(
        output_dir,
        graph,
        graph_adapter=adapter,
    )
    print("Graph semantic audit complete")
    print("Graph:", args.graphml)
    print("Domain profile:", args.domain_profile)
    print("Relation issues:", summary["relation_contract_issue_count"])
    print("Duplicate label groups:", summary["duplicate_label_group_count"])
    print("Components:", summary["component_count"])
    print("Saved:", summary["report_dir"])


if __name__ == "__main__":
    main()
