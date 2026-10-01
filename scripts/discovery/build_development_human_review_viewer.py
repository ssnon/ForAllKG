
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.development_human_review import (
    build_review_bundle,
    write_review_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a two-phase blind/reveal human-review viewer from an existing "
            "prospective development ablation. No LLM or literature-provider calls."
        )
    )
    parser.add_argument("--prospective-output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-arm-per-case", type=int, default=2)
    parser.add_argument("--seed", default="development-human-review-v1")
    parser.add_argument(
        "--include-unpaired-cases",
        action="store_true",
        help=(
            "Allow cases where one or more arms have zero hypotheses. "
            "Default is paired-case balancing."
        ),
    )
    args = parser.parse_args()

    bundle = build_review_bundle(
        args.prospective_output_root,
        per_arm_per_case=args.per_arm_per_case,
        seed=args.seed,
        paired_case_only=not args.include_unpaired_cases,
    )
    paths = write_review_bundle(bundle, args.output_dir)

    manifest = bundle["manifest"]
    print("Development human-review viewer built")
    print("bundle:", manifest["bundle_id"])
    print("sample count:", manifest["sample_count"])
    print("included cases:", len(manifest["included_cases"]))
    print("excluded cases:", len(manifest["excluded_cases"]))
    print("blind viewer:", paths["blind_html"])
    print("reveal viewer:", paths["reveal_html"])
    print("PRIVATE key:", paths["key"])
    print("DIAGNOSTIC_ONLY=True")
    print("SCIENTIFIC_SUPERIORITY_ESTABLISHED=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
