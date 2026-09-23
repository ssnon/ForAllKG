from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.scientific_verifier_prospective_campaign import (
    build_scientific_verifier_prospective_campaign_report,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile terminal prospective verifier runs into one campaign integrity report.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--case", action="append", dest="cases", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    case_ids = list(args.cases)
    if not case_ids:
        case_ids = sorted(path.name for path in args.root.iterdir() if path.is_dir() and path.name.startswith("P"))
    report = build_scientific_verifier_prospective_campaign_report(root=args.root, case_ids=case_ids)
    output = args.output.expanduser().resolve() if args.output is not None else args.root.expanduser().resolve() / "scientific_verifier_prospective_campaign_report.json"
    _write(output, report)
    print("Scientific verifier prospective campaign report complete")
    print("Cases:", report.case_count)
    print("Disposition counts:", report.disposition_counts)
    print("Verifier reached/not reached:", report.verifier_reached_case_count, "/", report.verifier_not_reached_case_count)
    print("Frozen hypotheses/claims:", report.frozen_hypothesis_count, "/", report.frozen_claim_count)
    print("Old/new agreements/disagreements:", report.old_new_agreement_count, "/", report.old_new_disagreement_count)
    print("Comparison cells:", report.comparison_cells)
    for row in report.cases:
        print(" ", row.case_id, row.disposition, "status=", row.wrapper_status, "frozen_h=", row.frozen_hypothesis_count, "agree/disagree=", f"{row.agreement_count}/{row.disagreement_count}")
    print("Prospective integrity invariants: PASS")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
