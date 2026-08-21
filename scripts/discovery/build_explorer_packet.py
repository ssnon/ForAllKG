from __future__ import annotations

import argparse
from pathlib import Path

from domains.registry import get_domain_profile
from domains.extraction_registry import (
    get_extraction_adapter,
)
from pipeline_core.discovery.explorer_packet import GraphExplorerPacketBuilder, write_packet


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a provenance-complete, agent-facing GraphExplorerPacket from a frozen traversal result."
    )
    parser.add_argument("--traversal-result", required=True)
    parser.add_argument("--domain-profile", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument(
        "--objective",
        choices=(
            "map_evidence",
            "explain_connection",
            "compare_mechanisms",
            "identify_reported_design_levers",
        ),
        default="map_evidence",
    )
    parser.add_argument(
        "--substrate-version",
        default="traversal-substrate-v2.4.7",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traversal_path = Path(args.traversal_result)

    import json

    traversal = json.loads(traversal_path.read_text(encoding="utf-8"))
    corpus_id = str(traversal.get("corpus_id", ""))
    mode = str(traversal.get("mode", ""))
    if not corpus_id or not mode:
        raise ValueError("Traversal result must contain corpus_id and mode.")

    traversal_domain = str(
        traversal.get("domain_profile_id") or ""
    ).strip()
    requested_domain = str(
        args.domain_profile or traversal_domain
    ).strip()
    if not requested_domain:
        raise ValueError(
            "domain_profile_id is required via --domain-profile "
            "or the traversal artifact."
        )
    if traversal_domain and requested_domain != traversal_domain:
        raise ValueError(
            "Requested domain profile does not match traversal artifact: "
            f"requested={requested_domain!r}, "
            f"traversal={traversal_domain!r}"
        )
    domain_profile = get_domain_profile(
        requested_domain
    )
    extraction_adapter = get_extraction_adapter(
        domain_profile.profile_id
    )
    data_root = Path(
        args.data_root
        or traversal.get("data_root")
        or extraction_adapter.default_data_root
    )
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    corpus_dir = (
        Path(args.corpus_dir)
        if args.corpus_dir
        else data_root / "corpus" / corpus_id / mode
    )
    output = (
        Path(args.output)
        if args.output
        else traversal_path.with_name(traversal_path.stem + ".explorer_packet.json")
    )

    packet = GraphExplorerPacketBuilder(
        substrate_version=args.substrate_version,
        strict_provenance=True,
    ).build_from_files(
        traversal_result_path=traversal_path,
        corpus_dir=corpus_dir,
        question=args.question,
        objective=args.objective,
    )
    write_packet(packet, output)

    print("GraphExplorerPacket built")
    print("Packet ID:", packet.packet_id)
    print("Packet SHA256:", packet.packet_sha256)
    print("Task ID:", packet.task.task_id)
    print("Domain profile:", packet.domain_profile_id)
    print("Corpus:", packet.corpus.corpus_id, packet.corpus.projection_mode)
    print("Papers in scope:", len(packet.corpus.papers))
    print("Direct concept hits:", len(packet.direct_concept_hits))
    print("Selected paths:", len(packet.paths))
    print("Evidence nodes:", len(packet.evidence_catalog.nodes))
    print("Evidence edges:", len(packet.evidence_catalog.edges))
    print("Alignment contexts:", len(packet.alignment_contexts))
    print(
        "Provenance edges (grounded/recovered/alignment/missing):",
        packet.provenance_summary.pointer_grounded_edge_count,
        packet.provenance_summary.pointer_recovered_from_traversal_count,
        packet.provenance_summary.derived_alignment_edge_count,
        packet.provenance_summary.missing_pointer_edge_count,
    )
    print(
        "Suppressed alignment-member nodes:",
        packet.provenance_summary.suppressed_alignment_member_node_count,
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
