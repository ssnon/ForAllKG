
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.fulltext_diagnostic_sentinel_shadow import (
    run_diagnostic_sentinel,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--query-plan", required=True, type=Path)
    p.add_argument("--prior-art-packet", required=True, type=Path)
    p.add_argument("--claim-id", required=True)
    p.add_argument("--doi", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--acquisition-dir", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=None)
    p.add_argument("--max-excerpt-chars", type=int, default=24000)
    p.add_argument("--parse-retries", type=int, default=3)
    args = p.parse_args()

    report = run_diagnostic_sentinel(
        query_plan=load(args.query_plan),
        packet=load(args.prior_art_packet),
        claim_id=args.claim_id,
        doi=args.doi,
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

    print("===== FULL-TEXT DIAGNOSTIC SENTINEL SHADOW =====")
    print("claim:", report.get("claim_id"))
    print("doi:", report.get("doi"))
    print("status:", report.get("status"))
    print("relationship:", report.get("relationship"))
    print("confidence:", report.get("confidence"))
    print("evidence spans:", len(report.get("evidence_spans", [])))
    print("DIAGNOSTIC_ONLY=True")
    print("AGGREGATION_ELIGIBLE=False")
    print("NOVELTY_AUTHORITY_CREATED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
