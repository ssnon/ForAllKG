
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.fulltext_lower_order_escalation_shadow import run_escalation


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-plan", required=True, type=Path)
    parser.add_argument("--claim-reviews", required=True, type=Path)
    parser.add_argument("--prior-art-packet", required=True, type=Path)
    parser.add_argument("--topology-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--acquisition-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-candidates-per-claim", type=int, default=4)
    parser.add_argument("--max-excerpt-chars", type=int, default=24000)
    parser.add_argument("--parse-retries", type=int, default=3)
    args = parser.parse_args()

    report = run_escalation(
        query_plan=load(args.query_plan),
        claim_reviews=load(args.claim_reviews),
        packet=load(args.prior_art_packet),
        topology_report=load(args.topology_report),
        output_root=args.acquisition_dir,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        max_candidates_per_claim=args.max_candidates_per_claim,
        max_excerpt_chars=args.max_excerpt_chars,
        parse_retries=args.parse_retries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("===== FULL-TEXT LOWER-ORDER ESCALATION SHADOW =====")
    print("Targets:", report["target_claim_count"])
    print("Full-text relation-backed:", report["fulltext_relation_backed_target_count"])
    for row in report["targets"]:
        print(
            row["claim_id"],
            "|", row["status"],
            "| candidates=", row["candidate_count"],
            "| backed=", len(row["fulltext_relation_backed_work_ids"]),
        )
        for work in row["reviews"]:
            print(
                "   ",
                work["relationship"],
                "|", work.get("doi"),
                "|", work.get("title"),
            )
    print("Report:", args.output)
    print("SHADOW_ONLY=True")
    print("NEW_SEARCH_PERFORMED=False")
    print("NOVELTY_AUTHORITY_CREATED=False")
    print("N9_AUTHORITY_CREATED=False")
    print("N10_AUTHORITY_CREATED=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
