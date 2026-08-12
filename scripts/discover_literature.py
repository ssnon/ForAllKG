from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dac_her.literature_discovery import (
    LiteratureRegistry,
    all_query_requests,
    load_query_plan,
    run_discovery,
    select_pilot_requests,
)
from dac_her.literature_discovery.providers import OpenAlexProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "literature" / "broad_catalysis_v1.yaml"
DEFAULT_REGISTRY = PROJECT_ROOT / "data_broad" / "registry" / "literature.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover scholarly literature using a versioned query plan. "
            "PR2 supports an OpenAlex pilot and writes candidates.jsonl + run.json."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--provider", choices=("openalex",), default="openalex")
    parser.add_argument("--pilot-query-count", type=int, default=5)
    parser.add_argument("--per-query-limit", type=int, default=100)
    parser.add_argument(
        "--all-queries",
        action="store_true",
        help="Run every query in the plan instead of the round-robin pilot subset.",
    )
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help="Persistent literature registry JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Run output directory. Defaults under data_broad/discovery/<plan>/<timestamp>.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--backoff-base-seconds", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected pilot queries without making network requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_query_plan(args.config)
    requests = (
        all_query_requests(plan, per_query_limit=args.per_query_limit)
        if args.all_queries
        else select_pilot_requests(
            plan,
            query_count=args.pilot_query_count,
            per_query_limit=args.per_query_limit,
        )
    )

    if args.dry_run:
        payload = {
            "plan_id": plan.plan_id,
            "provider": args.provider,
            "query_count": len(requests),
            "per_query_limit": args.per_query_limit,
            "maximum_raw_candidates": sum(item.limit for item in requests),
            "queries": [
                {
                    "mechanism_bucket": item.mechanism_bucket,
                    "query": item.query,
                    "limit": item.limit,
                }
                for item in requests
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if not requests:
        raise RuntimeError("No literature queries were selected")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "data_broad" / "discovery" / plan.plan_id / timestamp
    )
    registry = LiteratureRegistry(args.registry)

    provider = OpenAlexProvider(
        api_key=os.getenv("OPENALEX_API_KEY"),
        mailto=os.getenv("OPENALEX_MAILTO"),
        timeout=args.timeout,
        max_retries=args.max_retries,
        backoff_base_seconds=args.backoff_base_seconds,
    )
    artifacts = run_discovery(
        provider=provider,
        plan=plan,
        requests=requests,
        registry=registry,
        output_dir=output_dir,
    )

    print("Literature discovery complete")
    print("Provider:", provider.provider_name)
    print("Plan:", plan.plan_id)
    print("Queries:", len(requests))
    print("Raw candidates:", artifacts.raw_candidates)
    print("Unique candidates:", artifacts.unique_candidates)
    print("Candidates:", artifacts.candidates_path)
    print("Run metadata:", artifacts.run_path)
    print("Registry:", registry.path)


if __name__ == "__main__":
    main()
