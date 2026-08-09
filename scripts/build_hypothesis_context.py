from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.explorer_contracts import ExplorationReport, GraphExplorerPacket
from dac_her.hypothesis_context import HypothesisContextBuilder


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic HypothesisContext v2.6.0")
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-invalid-source-report",
        action="store_true",
        help="Development-only escape hatch; production should require a validated ExplorationReport.",
    )
    args = parser.parse_args()

    packet = GraphExplorerPacket.model_validate(_load(args.packet))
    report = ExplorationReport.model_validate(_load(args.report))
    context = HypothesisContextBuilder().build(
        packet,
        report,
        require_valid_report=not args.allow_invalid_source_report,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(context.model_dump_json(indent=2), encoding="utf-8")

    eligible = sum(x.eligible_as_premise for x in context.evidence_statements)
    gaps = sum(x.eligible_as_gap for x in context.evidence_statements)
    candidate = sum(x.requires_verification for x in context.evidence_statements)
    alignment_blocked = sum(bool(x.alignment_path_ids) for x in context.evidence_statements)
    print("HypothesisContext built")
    print("Context ID:", context.context_id)
    print("Context SHA256:", context.context_sha256)
    print("Source report:", context.source_report_id)
    print("Evidence statements:", len(context.evidence_statements))
    print("Eligible positive premises:", eligible)
    print("Eligible research gaps:", gaps)
    print("Candidate-dependent statements:", candidate)
    print("Alignment-blocked statements:", alignment_blocked)
    print("Partial absence blocked papers:", len(context.partial_absence_blocked_paper_ids))
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
