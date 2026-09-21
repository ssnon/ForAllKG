from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.reframing.ablation_blind_evaluator import (
    BlindScientificReasoningEvaluationReport,
    unblind_scientific_reasoning_evaluation,
)
from pipeline_core.discovery.reframing.ablation_evaluation import (
    ScientificReasoningAblationBlindKey,
    ScientificReasoningAblationPacket,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unblind a completed blind scientific-reasoning ablation evaluation. "
            "This stage performs no LLM call, scoring, ranking, or selection."
        )
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--blind-report", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_path = args.packet.expanduser().resolve()
    report_path = args.blind_report.expanduser().resolve()
    key_path = args.key.expanduser().resolve()

    packet = ScientificReasoningAblationPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    blind_report = BlindScientificReasoningEvaluationReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    key = ScientificReasoningAblationBlindKey.model_validate_json(
        key_path.read_text(encoding="utf-8")
    )
    unblinded = unblind_scientific_reasoning_evaluation(
        packet=packet,
        key=key,
        blind_report=blind_report,
        source_blind_report_sha256=_sha256_file(report_path),
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else report_path.parent / "scientific_reasoning_ablation_unblinded_evaluation.json"
    )
    _write_json(output, unblinded)

    print("Scientific reasoning ablation evaluation unblinded")
    print("LLM calls: 0")
    print(f"Task: {unblinded.source_task_id}")
    for comparison in unblinded.comparisons:
        print(
            f"  {comparison.comparison_alias} ({comparison.comparison_kind}): "
            + ", ".join(
                f"{key}={comparison.preference_counts[key]}"
                for key in sorted(comparison.preference_counts)
            )
        )
    print("Per-dimension results only: true")
    print("Overall score computed: false")
    print("Overall winner selected: false")
    print("Scientific quality ranking performed: false")
    print("Production selection changed: false")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
