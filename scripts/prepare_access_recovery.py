from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.corpus_acquisition.access_recovery import prepare_access_recovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare resumable M3.2 access recovery state. Propagate prior "
            "candidate states, invalidate requested/stale non-downloaded "
            "states, and learn hard 401/403/404/410/non-PDF endpoints without "
            "weakening scientific selection or bypassing access controls."
        )
    )
    parser.add_argument("--source-policy", required=True, type=Path)
    parser.add_argument("--source-m3-dir", required=True, type=Path)
    parser.add_argument("--output-m3-dir", required=True, type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-access-misses", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = prepare_access_recovery(
        source_policy_path=args.source_policy,
        source_m3_dir=args.source_m3_dir,
        output_m3_dir=args.output_m3_dir,
        retry_failed=args.retry_failed,
        retry_access_misses=args.retry_access_misses,
    )
    print("Access recovery prepared")
    print(
        "State:",
        f"copied={report['copied_state_count']}",
        f"refreshed={report['refreshed_state_count']}",
        f"suppressed_urls={report['suppressed_url_count']}",
    )
    print("Reasons:", report["refresh_reason_counts"])
    print("Hard failures:", report["hard_failure_code_counts"])
    print(
        "Resolver capabilities:",
        {
            "unpaywall": report["resolver_context"]["unpaywall_contact_available"],
            "openalex": report["resolver_context"]["openalex_lookup_available"],
            "catalog_fallback": report["resolver_context"][
                "catalog_oa_fallback_enabled"
            ],
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
