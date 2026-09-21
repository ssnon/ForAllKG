from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.ablation_evaluation import (
    ScientificReasoningAblationPacket,
)
from pipeline_core.discovery.reframing.ablation_schema_normalization import (
    build_schema_normalized_ablation_packet,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic common-denominator presentation of an existing blind "
            "scientific-reasoning ablation packet. This removes source-schema-rich fields "
            "symmetrically without judging scientific quality or changing blind identities."
        )
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_path = args.packet.expanduser().resolve()
    packet = ScientificReasoningAblationPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    normalized, report = build_schema_normalized_ablation_packet(packet)

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else packet_path.parent
        / "scientific_reasoning_ablation_schema_normalized_packet.json"
    )
    report_output = (
        args.report_output.expanduser().resolve()
        if args.report_output is not None
        else packet_path.parent
        / "scientific_reasoning_ablation_schema_normalization_report.json"
    )
    _write_json(output, normalized)
    _write_json(report_output, report)

    removed_baseline = sum(
        int(candidate.baseline_model_removed)
        for comparison in report.comparisons
        for arm in (comparison.arm_a, comparison.arm_b)
        for candidate in arm.candidates
    )
    removed_tests = sum(
        int(candidate.dedicated_discriminating_test_removed)
        for comparison in report.comparisons
        for arm in (comparison.arm_a, comparison.arm_b)
        for candidate in arm.candidates
    )
    downprojected_contrast_predictions = sum(
        candidate.contrast_prediction_count_downprojected
        for comparison in report.comparisons
        for arm in (comparison.arm_a, comparison.arm_b)
        for candidate in arm.candidates
    )
    print("Schema-normalized scientific reasoning ablation packet complete")
    print("LLM calls: 0")
    print(f"Packet ID preserved: {normalized.packet_id == packet.packet_id}")
    print(f"Comparisons preserved: {normalized.comparison_count}")
    print(f"Source candidates preserved: {normalized.source_candidate_count}")
    print(f"Explicit baseline fields removed: {removed_baseline}")
    print(f"Dedicated discriminating-test fields removed: {removed_tests}")
    print(
        "Contrast predictions downprojected to common prediction shape: "
        f"{downprojected_contrast_predictions}"
    )
    print("Blind key compatibility preserved: true")
    print("Schema field parity targeted: true")
    print("Candidate-count parity targeted: false")
    print("Payload-length parity targeted: false")
    print("Scientific quality judgment performed: false")
    print("Production selection changed: false")
    print(f"Packet: {output}")
    print(f"Normalization report: {report_output}")


if __name__ == "__main__":
    main()
