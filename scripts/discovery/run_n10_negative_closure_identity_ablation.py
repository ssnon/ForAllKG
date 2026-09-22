from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.n10_negative_closure_identity_ablation import (
    analyze_claim_detail,
    build_report,
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only N10 negative-closure ablation over already-reviewed "
            "closure artifacts. Compares production local-window identity matching "
            "against a same-token/same-75%-threshold dispersed constituent matcher "
            "and an identity-free search-coverage control. Positive evidence and "
            "production authority are never changed."
        )
    )
    parser.add_argument("--detail-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--minimum-negative-abstracts",
        type=int,
        default=3,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.minimum_negative_abstracts < 1:
        raise SystemExit("--minimum-negative-abstracts must be >= 1")
    if not args.detail_root.is_dir():
        raise SystemExit(f"detail root does not exist: {args.detail_root}")

    claims = []
    for claim_dir in sorted(
        path for path in args.detail_root.iterdir() if path.is_dir()
    ):
        plan_path = claim_dir / "closure_plan.json"
        reviews_path = claim_dir / "slot_reviews.json"
        prior_art_path = claim_dir / "prior_art.json"
        if not (
            plan_path.is_file()
            and reviews_path.is_file()
            and prior_art_path.is_file()
        ):
            continue

        closure_plan = _load(plan_path)
        slot_reviews = _load(reviews_path)
        prior_art = _load(prior_art_path)
        if not isinstance(closure_plan, dict):
            raise ValueError(f"closure_plan must be object: {plan_path}")
        if not isinstance(slot_reviews, list):
            raise ValueError(f"slot_reviews must be list: {reviews_path}")
        if not isinstance(prior_art, dict):
            raise ValueError(f"prior_art must be object: {prior_art_path}")

        claims.append(
            analyze_claim_detail(
                closure_plan=closure_plan,
                slot_reviews=slot_reviews,
                prior_art=prior_art,
                minimum_negative_abstracts=args.minimum_negative_abstracts,
            )
        )

    report = build_report(claims)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("N10 negative-closure identity ablation complete")
    print("Claims:", report.claim_count)
    print("Slots:", report.slot_count)
    print(
        "Strict UNASSESSED non-base slots:",
        report.strict_unassessed_nonbase_slot_count,
    )
    print(
        "Dispersed-constituent NOT_FOUND non-base slots:",
        report.dispersed_constituent_not_found_nonbase_slot_count,
    )
    print(
        "Search-coverage-only NOT_FOUND non-base slots:",
        report.search_coverage_only_not_found_nonbase_slot_count,
    )
    print(
        "FULL_RELATION NOT_FOUND strict/dispersed/control:",
        report.strict_full_relation_not_found_count,
        "/",
        report.dispersed_constituent_full_relation_not_found_count,
        "/",
        report.search_coverage_only_full_relation_not_found_count,
    )

    for claim in report.claims:
        print()
        print("Claim:", claim.claim_id)
        for slot in claim.slots:
            if slot.slot == "BASE_RELATION":
                continue
            print(
                " ",
                slot.slot,
                "| eligible strict/dispersed/control =",
                slot.strict_negative_eligible_count,
                "/",
                slot.dispersed_constituent_negative_eligible_count,
                "/",
                slot.search_coverage_only_negative_eligible_count,
                "| state =",
                slot.strict_shadow_state,
                "->",
                slot.dispersed_constituent_shadow_state,
                "/",
                slot.search_coverage_only_shadow_state,
            )

    print()
    print("Diagnostic only: true")
    print("Positive evidence semantics changed: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
