from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.final_output_blind_evaluation import (
    BlindFinalOutputEvaluationPacket,
    BlindFinalOutputEvaluationReport,
    FinalOutputBlindKey,
    unblind_final_output_evaluation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unblind a completed final-hypothesis output evaluation without any LLM calls."
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--blind-report", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    packet_path = args.packet.expanduser().resolve()
    blind_path = args.blind_report.expanduser().resolve()
    key_path = args.key.expanduser().resolve()
    packet = BlindFinalOutputEvaluationPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))
    blind = BlindFinalOutputEvaluationReport.model_validate_json(blind_path.read_text(encoding="utf-8"))
    key = FinalOutputBlindKey.model_validate_json(key_path.read_text(encoding="utf-8"))
    report = unblind_final_output_evaluation(packet=packet, blind_report=blind, key=key)
    output = args.output or blind_path.with_name("scientific_final_output_unblinded_evaluation.json")
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("Final hypothesis output evaluation unblinded")
    print("LLM calls: 0")
    print("Task:", report.source_task_id)
    print("Condition preferences:", report.condition_preference_counts)
    for row in report.judgments:
        print(f"  {row.dimension_id}: {row.preferred_condition}")
    print("Per-dimension results only: true")
    print("Overall score computed: false")
    print("Overall winner selected: false")
    print("Production selection changed: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
