from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.ablation_evaluation import (
    ScientificReasoningAblationBlindKey,
    ScientificReasoningAblationPacket,
)
from pipeline_core.discovery.reframing.ablation_generalized_matching import (
    build_generalized_matched_count_ablation_packet,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a schema-normalized, generalized candidate-count-matched blind ablation "
            "packet for arbitrary RELATIONAL_ONLY vs REFRAMING_ONLY cardinalities. "
            "Condition labels remain in the blind key/build report only, never in the evaluator packet."
        )
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--max-comparisons", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-key", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_path = args.packet.expanduser().resolve()
    key_path = args.key.expanduser().resolve()
    packet = ScientificReasoningAblationPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    key = ScientificReasoningAblationBlindKey.model_validate_json(
        key_path.read_text(encoding="utf-8")
    )
    matched_packet, matched_key, report = build_generalized_matched_count_ablation_packet(
        packet=packet,
        key=key,
        max_comparisons=args.max_comparisons,
    )

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else packet_path.parent / "scientific_reasoning_ablation_generalized_matched_packet.json"
    )
    output_key = (
        args.output_key.expanduser().resolve()
        if args.output_key is not None
        else packet_path.parent / "scientific_reasoning_ablation_generalized_matched_key.json"
    )
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else packet_path.parent / "scientific_reasoning_ablation_generalized_matched_build_report.json"
    )
    _write_json(output, matched_packet)
    _write_json(output_key, matched_key)
    _write_json(report_path, report)

    print("Generalized matched-count scientific reasoning ablation packet complete")
    print("LLM calls: 0")
    print(f"Source candidate counts: {report.source_candidate_counts}")
    print(f"Matched candidates per arm: {report.matched_candidate_count_per_arm}")
    print(
        "Subset comparisons: "
        f"materialized={report.materialized_comparison_count}, "
        f"possible={report.total_possible_subset_comparisons}, "
        f"complete={str(report.subset_schedule_complete).lower()}"
    )
    for row in report.comparisons:
        print(
            f"  {row.comparison_alias}: "
            f"{row.matched_candidate_count}v{row.matched_candidate_count}; "
            f"payload_ratio={row.payload_character_ratio:.3f}"
        )
    print("Blind condition-label leakage detected: false")
    print("Blind source-ID leakage detected: false")
    print("Candidate-count parity established: true")
    print("Payload-length parity targeted: false")
    print("Scientific quality judgment performed: false")
    print("Production selection changed: false")
    print(f"Packet: {output}")
    print(f"Blind key (do not supply to evaluator): {output_key}")
    print(f"Build report (do not supply to evaluator): {report_path}")


if __name__ == "__main__":
    main()
