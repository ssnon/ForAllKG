
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.memory_tiered_fulltext_selection_shadow import (
    build_tiered_selection_plan,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--query-plan", required=True, type=Path)
    p.add_argument("--topology-report", required=True, type=Path)
    p.add_argument("--current-source-binding", required=True, type=Path)
    p.add_argument("--current-packet", required=True, type=Path)
    p.add_argument("--current-reviews", required=True, type=Path)
    p.add_argument("--memory-packet", required=True, type=Path)
    p.add_argument("--historical-source-binding", required=True, type=Path)
    p.add_argument("--historical-packet", required=True, type=Path)
    p.add_argument("--historical-reviews", required=True, type=Path)
    p.add_argument("--historical-fulltext", required=True, type=Path)
    p.add_argument("--current-slots", type=int, default=4)
    p.add_argument("--tier-a-slots", type=int, default=2)
    p.add_argument("--tier-b-slots", type=int, default=2)
    p.add_argument("--tier-c-slots", type=int, default=4)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    report = build_tiered_selection_plan(
        query_plan=load(args.query_plan),
        topology_report=load(args.topology_report),
        current_source_binding=load(args.current_source_binding),
        current_packet=load(args.current_packet),
        current_reviews=load(args.current_reviews),
        memory_packet=load(args.memory_packet),
        historical_source_binding=load(
            args.historical_source_binding
        ),
        historical_packet=load(args.historical_packet),
        historical_reviews=load(args.historical_reviews),
        historical_fulltext=load(args.historical_fulltext),
        current_slots=args.current_slots,
        tier_a_slots=args.tier_a_slots,
        tier_b_slots=args.tier_b_slots,
        tier_c_slots=args.tier_c_slots,
    )

    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print("===== MEMORY-TIERED FULL-TEXT SELECTION SHADOW =====")
    print("Targets:", report["target_count"])
    print("Policy:", report["selection_policy"])
    for row in report["targets"]:
        print(
            row["claim_id"],
            "|", row["status"],
            "| selected=", row.get("selected_count", 0),
            "| historical claims=", len(
                row.get("historical_claim_ids", [])
            ),
        )
        for item in row.get("selected", []):
            print(
                "   ",
                item["selection_lane"],
                "|", item.get("memory_tier"),
                "|", item["work_id"],
            )
    print("MEMORY_IS_EVIDENCE=False")
    print("SHADOW_ONLY=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
