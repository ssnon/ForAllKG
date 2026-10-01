
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.residual_novelty_aggregation_shadow import aggregate_residual_novelty


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology-report", required=True, type=Path)
    parser.add_argument("--claim-reviews", required=True, type=Path)
    parser.add_argument("--fulltext-escalation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = aggregate_residual_novelty(
        topology_report=load(args.topology_report),
        claim_reviews=load(args.claim_reviews),
        fulltext_escalation=load(args.fulltext_escalation),
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("===== RESIDUAL NOVELTY AGGREGATION SHADOW =====")
    print("Residual states:", report["residual_state_counts"])
    print("Dispositions:", report["disposition_counts"])
    for row in report["composites"]:
        print(
            row.get("hypothesis_id"),
            "| full=", row.get("full_relation_status"),
            "| topology=", row.get("topology_state"),
            "| components=",
            f'{len(row["aggregated_relation_backed_component_claim_ids"])}/'
            f'{len(row["aggregated_component_claim_ids"])}',
            "| evidence=", row["evidence_depth"],
            "| closure=", row["aggregated_component_closure"],
            "| residual=", row["aggregated_residual_state"],
            "| disposition=", row["aggregation_disposition"],
        )
    print("Report:", args.output)
    print("SHADOW_ONLY=True")
    print("NOVELTY_AUTHORITY_CREATED=False")
    print("N9_AUTHORITY_CREATED=False")
    print("N10_AUTHORITY_CREATED=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
