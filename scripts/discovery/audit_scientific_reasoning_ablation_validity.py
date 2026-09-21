from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.ablation_blind_evaluator import (
    BlindScientificReasoningEvaluationReport,
    UnblindedScientificReasoningEvaluationReport,
)
from pipeline_core.discovery.reframing.ablation_evaluation import (
    ScientificReasoningAblationPacket,
)
from pipeline_core.discovery.reframing.ablation_validity_audit import (
    audit_scientific_reasoning_ablation_validity,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit blind scientific-reasoning ablation results for deterministic "
            "presentation/schema asymmetries. No LLM calls or judgment changes are performed."
        )
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--blind-report", required=True, type=Path)
    parser.add_argument("--unblinded-report", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_path = args.packet.expanduser().resolve()
    blind_path = args.blind_report.expanduser().resolve()
    unblinded_path = args.unblinded_report.expanduser().resolve()

    packet = ScientificReasoningAblationPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )
    blind = BlindScientificReasoningEvaluationReport.model_validate_json(
        blind_path.read_text(encoding="utf-8")
    )
    unblinded = UnblindedScientificReasoningEvaluationReport.model_validate_json(
        unblinded_path.read_text(encoding="utf-8")
    )
    audit = audit_scientific_reasoning_ablation_validity(
        packet=packet,
        blind_report=blind,
        unblinded_report=unblinded,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else packet_path.parent / "scientific_reasoning_ablation_validity_audit.json"
    )
    _write_json(output, audit)

    print("Scientific reasoning ablation validity audit complete")
    print("LLM calls: 0")
    print(f"Judge model: {audit.judge_model_name}")
    print(f"Comparisons: {audit.comparison_count}")
    print(
        "Presentation asymmetry dimensions: "
        f"high={audit.high_asymmetry_dimension_count}, "
        f"moderate={audit.moderate_asymmetry_dimension_count}, "
        f"low={audit.low_asymmetry_dimension_count}"
    )
    for comparison in audit.comparison_audits:
        print(
            f"  {comparison.comparison_alias} ({comparison.comparison_kind}): "
            f"high={comparison.high_asymmetry_dimension_count}, "
            f"moderate={comparison.moderate_asymmetry_dimension_count}, "
            f"low={comparison.low_asymmetry_dimension_count}; "
            f"candidate_diff={comparison.candidate_count_difference}; "
            f"payload_ratio={comparison.payload_character_ratio:.3f}"
        )
        for dimension in comparison.dimensions:
            if dimension.asymmetry_level != "low_observed_presentation_asymmetry":
                print(
                    f"    {dimension.dimension_id}: {dimension.asymmetry_level}; "
                    f"observed={dimension.observed_preference}; "
                    f"signals={','.join(dimension.asymmetry_signals) or '-'}"
                )
    print(f"Presentation parity established: {str(audit.presentation_parity_established).lower()}")
    print(f"Schema parity established: {str(audit.schema_parity_established).lower()}")
    print(
        "Scientific reasoning superiority established: "
        f"{str(audit.scientific_reasoning_superiority_established).lower()}"
    )
    print("Evaluator judgments modified: false")
    print("Production selection changed: false")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
