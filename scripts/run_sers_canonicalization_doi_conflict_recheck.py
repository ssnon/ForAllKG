from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.canonicalization_doi_conflict_recheck import (
    OUTPUT_ROOT,
    atomic_json,
    run_recheck,
    validate_source,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    output_root = ROOT / OUTPUT_ROOT

    try:
        raw_packet, source_report = validate_source(ROOT)
    except Exception as exc:
        print("canonicalization DOI-conflict recheck preflight: FAIL")
        print(" -", f"{type(exc).__name__}: {exc}")
        print("Network calls:", 0)
        return 2

    if output_root.exists():
        print("canonicalization DOI-conflict recheck preflight: FAIL")
        print(" - output root already exists:", output_root)
        print("Network calls:", 0)
        return 2

    print("Canonicalization DOI-conflict Hardening Offline Recheck")
    print("Source v1 run:", source_report["run_id"])
    print("Source raw packet:", raw_packet.packet_id)
    print("Raw works:", len(raw_packet.works))
    print("Expected collision groups:", 5)
    print("Network calls:", 0)
    print("Ranker used:", False)
    print("LLM calls:", 0)

    if args.preflight:
        print("Preflight: PASS")
        print("Write performed: False")
        return 0

    canonical, report = run_recheck(ROOT)
    if report["outcome"] != (
        "CANONICALIZATION_DOI_CONFLICT_HARDENING_PASS"
    ):
        print("Recheck result:", report["outcome"])
        print("Checks:", report["checks"])
        print("Write performed: False")
        return 2

    output_root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output_root / "canonical_prior_art_v2.json",
        canonical.model_dump(mode="json"),
    )
    atomic_json(
        output_root / "recheck_report.json",
        report,
    )
    atomic_json(
        output_root / "RECHECK_PASS.json",
        {
            "status": "recheck_pass",
            "run_id": report["run_id"],
            "run_sha256": report["run_sha256"],
            "canonical_packet_id": report["canonical_packet_id"],
            "canonical_packet_sha256":
                report["canonical_packet_sha256"],
            "canonical_packet_eligible_for_dev_ranker_validation":
                report[
                    "canonical_packet_eligible_for_dev_ranker_validation"
                ],
            "network_calls": 0,
            "llm_calls": 0,
        },
    )

    print()
    print("canonicalization DOI-conflict recheck: PASS")
    print("Run ID:", report["run_id"])
    print("Outcome:", report["outcome"])
    print("Checks:", report["checks"])
    print("Counts:", report["counts"])
    print(
        "Collision preservation:",
        report["collision_preservation"],
    )
    print(
        "Canonical packet eligible for DEV ranker validation:",
        report[
            "canonical_packet_eligible_for_dev_ranker_validation"
        ],
    )
    print("Network calls:", 0)
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
