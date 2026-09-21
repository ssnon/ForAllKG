from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.evidence_tension import (
    extract_evidence_level_tensions,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.trigger_detection import (
    detect_scientific_reframe_triggers,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect zero-LLM scientific reframing triggers from the existing "
            "Explorer report plus grounded reframing evidence. When --packet "
            "is supplied, use packet-assisted evidence-level tension typing."
        )
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--packet", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tension-output", type=Path, default=None)
    args = parser.parse_args()

    evidence = ScientificReframeEvidencePacket.model_validate(_load(args.evidence))
    explorer_report = _load(args.report)
    explorer_packet = _load(args.packet) if args.packet is not None else None
    tension_report = extract_evidence_level_tensions(
        explorer_report=explorer_report,
        evidence=evidence,
        explorer_packet=explorer_packet,
    )
    result = detect_scientific_reframe_triggers(
        explorer_report=explorer_report,
        evidence=evidence,
        explorer_packet=explorer_packet,
        evidence_tension_report=tension_report,
    )
    output = args.output or (
        args.evidence.parent / "scientific_reframing_triggers.json"
    )
    tension_output = args.tension_output or (
        args.evidence.parent / "scientific_reframing_evidence_tensions.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tension_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tension_output.write_text(
        tension_report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Scientific reframe trigger detection complete")
    print("Task:", result.source_task_id)
    print("LLM calls: 0")
    print("Trigger resolution:", result.trigger_resolution_mode)
    print("Tension witnesses:", len(result.tension_witnesses))
    print("Evidence tension taxonomy:")
    if tension_report.tension_type_counts:
        for tension_type, count in sorted(tension_report.tension_type_counts.items()):
            print(f"  {tension_type}: {count}")
    else:
        print("  none")
    for witness in tension_report.witnesses:
        types = ",".join(witness.tension_types) or "none"
        print(
            f"  witness {witness.witness_id}: types={types}; "
            f"conditions={witness.relevant_condition_signature_count}; "
            f"independence={witness.independence_basis}"
        )
    print("Direct trigger signals:")
    if result.direct_trigger_signals:
        for signal in result.direct_trigger_signals:
            print(
                f"  {signal.kind}: operator={signal.target_operator_id}; "
                f"strength={signal.strength}; statements={len(signal.statement_ids)}; "
                f"gaps={len(signal.gap_statement_ids)}"
            )
    else:
        print("  none")
    print(
        "Condition diversity:",
        f"examples={result.condition_diversity.example_count};",
        f"signatures={result.condition_diversity.distinct_condition_signature_count};",
        f"papers={result.condition_diversity.paper_count};",
        f"present={str(result.condition_diversity.diversity_present).lower()}",
    )
    for row in result.assessments:
        print(
            f"  {row.operator_id}: decision={row.decision}; "
            f"witnesses={len(row.witness_ids)}; "
            f"direct_signals={len(row.direct_signal_ids)}; "
            f"trigger={str(row.scientific_trigger_signal).lower()}"
        )
        for reason in row.reasons:
            print("    reason:", reason)
    print("No scientific conflict, regime-boundary, novelty, ranking, or production authority was created.")
    print("Tensions:", tension_output)
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
