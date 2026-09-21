from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.final_output_blind_evaluation import (
    build_blind_final_output_evaluation_packet,
)
from pipeline_core.discovery.reframing.integrated_hypothesis_shadow import (
    IntegratedScientificHypothesisShadowPortfolio,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an anonymous legacy-vs-integrated final-hypothesis portfolio comparison. "
            "The packet contains no condition labels or source hypothesis IDs."
        )
    )
    parser.add_argument("--shadow", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--key-output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    shadow_path = args.shadow.expanduser().resolve()
    shadow = IntegratedScientificHypothesisShadowPortfolio.model_validate_json(
        shadow_path.read_text(encoding="utf-8")
    )
    packet, key = build_blind_final_output_evaluation_packet(shadow)
    output = args.output or shadow_path.with_name("scientific_final_output_blind_packet.json")
    key_output = args.key_output or shadow_path.with_name("scientific_final_output_blind_key.json")
    output.write_text(packet.model_dump_json(indent=2) + "\n", encoding="utf-8")
    key_output.write_text(key.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("Final hypothesis output blind packet complete")
    print("LLM calls: 0")
    print("Task:", shadow.source_task_id)
    print("ARM_A hypotheses:", packet.arm_a.hypothesis_count)
    print("ARM_B hypotheses:", packet.arm_b.hypothesis_count)
    print("Condition labels hidden: true")
    print("Source IDs hidden: true")
    print("Candidate count treated as quality signal: false")
    print("Overall winner requested: false")
    print("Production selection changed: false")
    print("Packet:", output)
    print("Blind key (do not supply to evaluator):", key_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
