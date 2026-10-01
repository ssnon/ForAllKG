
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.memory_tiered_fulltext_escalation_shadow import (
    run_tiered_escalation,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--query-plan", required=True, type=Path)
    p.add_argument("--prior-art-packet", required=True, type=Path)
    p.add_argument("--selection-plan", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--acquisition-dir", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=None)
    p.add_argument("--max-excerpt-chars", type=int, default=24000)
    p.add_argument("--parse-retries", type=int, default=3)
    args = p.parse_args()

    report = run_tiered_escalation(
        query_plan=load(args.query_plan),
        packet=load(args.prior_art_packet),
        selection_plan=load(args.selection_plan),
        output_root=args.acquisition_dir,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        max_excerpt_chars=args.max_excerpt_chars,
        parse_retries=args.parse_retries,
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

    print("===== MEMORY-TIERED FULL-TEXT ESCALATION SHADOW =====")
    print("Targets:", report["target_claim_count"])
    print(
        "Full-text relation-backed:",
        report["fulltext_relation_backed_target_count"],
    )
    for target in report["targets"]:
        print(
            target["claim_id"],
            "|", target["status"],
            "| candidates=", target["candidate_count"],
            "| backed=", len(
                target["fulltext_relation_backed_work_ids"]
            ),
        )
        for row in target["reviews"]:
            print(
                "   ",
                row.get("selection_lane"),
                "|", row.get("memory_tier"),
                "|", row.get("relationship"),
                "|", row.get("doi"),
                "|", row.get("title"),
            )
    print("MEMORY_IS_EVIDENCE=False")
    print("NEW_SEARCH_PERFORMED=False")
    print("SHADOW_ONLY=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
