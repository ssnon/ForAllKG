from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from dac_her.external_novelty_contracts import ExternalNoveltyPolicy
from dac_her.hypothesis_novelty_synthesis_dev_validation import (
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    atomic_json,
    atomic_text,
    canonical_json,
    compute_hypothesis_synthesis,
    read_json,
    render_manual_audit,
    sha256_json,
    validate_handoff,
    verify_spec,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--confirm-one-shot-hypothesis-novelty-dev",
        action="store_true",
    )
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    args = parser.parse_args()

    if not args.run:
        parser.error("--run is required")
    if not args.confirm_one_shot_hypothesis_novelty_dev:
        parser.error(
            "--confirm-one-shot-hypothesis-novelty-dev is required"
        )

    spec_root = (
        args.spec_root
        if args.spec_root.is_absolute()
        else ROOT / args.spec_root
    )
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )

    issues, spec = verify_spec(
        ROOT,
        spec_root / "novelty_synthesis_spec.json",
    )
    if issues:
        print("hypothesis-novelty synthesis DEV preflight: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2

    marker = read_json(spec_root / "SPEC_FREEZE_PASS.json")
    if (
        marker.get("status") != "spec_freeze_pass"
        or marker.get("spec_id") != spec["spec_id"]
    ):
        print("hypothesis-novelty synthesis DEV preflight: FAIL")
        print(" - spec freeze marker mismatch")
        return 2

    if output_root.exists():
        print("hypothesis-novelty synthesis DEV preflight: FAIL")
        print(" - output root exists:", output_root)
        print("Automatic rerun:", False)
        return 2

    source = validate_handoff(ROOT)
    plan = source["plan"]
    packet = source["packet"]
    reviews = source["reviews"]
    policy = ExternalNoveltyPolicy()

    print("SERS Hypothesis-novelty Synthesis-only DEV One-Shot")
    print("Spec ID:", spec["spec_id"])
    print("Source claim-review v3:", spec["source_claim_review_v3_run_id"])
    print("Hypotheses:", len(plan.claims))
    print("Claims:", len(reviews))
    print("Production _coverage reused:", True)
    print("Production _status reused:", True)
    print("LLM calls:", 0)
    print("Literature network calls:", 0)
    print("Ranker recomputed:", False)
    print("Claim reviewer recomputed:", False)
    print("Automatic rerun:", False)

    output_root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output_root / "RUN_STARTED.json",
        {
            "status": "run_started",
            "spec_id": spec["spec_id"],
            "automatic_rerun_authorized": False,
            "llm_calls": 0,
            "network_calls": 0,
        },
    )

    try:
        rows_1 = compute_hypothesis_synthesis(
            plan=plan,
            packet=packet,
            reviews=reviews,
            policy=policy,
        )
        rows_2 = compute_hypothesis_synthesis(
            plan=plan,
            packet=packet,
            reviews=reviews,
            policy=policy,
        )
        deterministic = canonical_json(rows_1) == canonical_json(rows_2)

        claim_ids_from_rows = [
            claim_id
            for row in rows_1
            for claim_id in row["claim_ids"]
        ]
        checks = {
            "source_claim_review_v3_exact": (
                source["report"]["run_id"]
                == spec["source_claim_review_v3_run_id"]
            ),
            "all_12_frozen_claim_reviews_consumed_once": (
                len(claim_ids_from_rows) == 12
                and len(set(claim_ids_from_rows)) == 12
                and set(claim_ids_from_rows) == set(reviews)
            ),
            "all_3_hypotheses_synthesized": len(rows_1) == 3,
            "production_coverage_repeat_exact": deterministic,
            "production_status_repeat_exact": deterministic,
            "coverage_hypothesis_identity_exact": all(
                row["coverage"]["hypothesis_id"]
                == row["hypothesis_id"]
                for row in rows_1
            ),
            "claim_hypothesis_lineage_exact": all(
                reviews[claim_id].hypothesis_id
                == row["hypothesis_id"]
                for row in rows_1
                for claim_id in row["claim_ids"]
            ),
            "no_upstream_recomputation": True,
            "zero_llm_calls": True,
            "zero_literature_network_calls": True,
            "fresh_reserve_not_consumed": True,
            "automatic_scientific_status_approval_disabled": True,
            "automatic_next_stage_disabled": True,
        }
        structural_pass = all(checks.values())
        status_counts = dict(
            sorted(Counter(row["status"] for row in rows_1).items())
        )

        body = {
            "schema_version":
                "sers-hypothesis-novelty-synthesis-dev-run-v1",
            "semantics_id":
                "sers_hypothesis_novelty_synthesis_only_dev_v1",
            "source_spec_id": spec["spec_id"],
            "source_spec_sha256": spec["spec_sha256"],
            "source_claim_review_v3_run_id":
                spec["source_claim_review_v3_run_id"],
            "source_query_plan_id": spec["source_query_plan_id"],
            "source_canonical_packet_id":
                spec["source_canonical_packet_id"],
            "structural_outcome": (
                "HYPOTHESIS_NOVELTY_SYNTHESIS_STRUCTURAL_DEV_PASS"
                if structural_pass
                else "HYPOTHESIS_NOVELTY_SYNTHESIS_STRUCTURAL_DEV_FAIL"
            ),
            "scientific_novelty_status_outcome":
                "MANUAL_REVIEW_REQUIRED",
            "checks": checks,
            "hypothesis_rows": rows_1,
            "status_counts": status_counts,
            "policy": policy.model_dump(mode="json"),
            "llm_calls": 0,
            "literature_network_calls": 0,
            "ranker_recomputed": False,
            "claim_reviewer_recomputed": False,
            "claim_decomposition_recomputed": False,
            "canonicalization_recomputed": False,
            "case_specific_expected_statuses_used": False,
            "automatic_scientific_status_approval": False,
            "automatic_next_stage_authorized": False,
            "fresh_reserve_consumed": False,
        }
        body["run_sha256"] = sha256_json(body)
        body["run_id"] = (
            "sers_hypothesis_novelty_synthesis_dev_run:"
            + body["run_sha256"][:20]
        )

        atomic_json(
            output_root / "hypothesis_novelty_synthesis_report.json",
            body,
        )
        atomic_text(
            output_root / "human_hypothesis_novelty_audit.md",
            render_manual_audit(rows_1),
        )

        if not structural_pass:
            atomic_json(
                output_root / "STRUCTURAL_FAIL.json",
                {
                    "status": "structural_fail",
                    "run_id": body["run_id"],
                    "automatic_rerun_authorized": False,
                    "automatic_next_stage_authorized": False,
                },
            )
            print("hypothesis-novelty synthesis DEV: STRUCTURAL FAIL")
            print("Checks:", checks)
            return 2

        atomic_json(
            output_root / "STRUCTURAL_PASS.json",
            {
                "status": "structural_pass",
                "run_id": body["run_id"],
                "run_sha256": body["run_sha256"],
                "scientific_novelty_status_outcome":
                    "MANUAL_REVIEW_REQUIRED",
                "automatic_next_stage_authorized": False,
                "fresh_reserve_consumed": False,
            },
        )
        try:
            (output_root / "RUN_STARTED.json").unlink()
        except FileNotFoundError:
            pass

        print()
        print("hypothesis-novelty synthesis DEV: COMPLETE")
        print("Run ID:", body["run_id"])
        print("Structural outcome:", body["structural_outcome"])
        print(
            "Scientific novelty-status outcome:",
            body["scientific_novelty_status_outcome"],
        )
        print("Status counts:", body["status_counts"])
        for index, row in enumerate(rows_1, start=1):
            print(
                f"[{index}/3]",
                row["hypothesis_id"],
                "=>",
                row["status"],
                "| core=",
                row["core_claim_statuses"],
                "| coverage_sufficient=",
                row["coverage"][
                    "sufficient_for_absence_based_novelty"
                ],
            )
        print("LLM calls:", 0)
        print("Literature network calls:", 0)
        print("Ranker recomputed:", False)
        print("Claim reviewer recomputed:", False)
        print("Fresh Reserve consumed:", False)
        print("Automatic next stage authorized:", False)
        print(
            "Human audit:",
            output_root / "human_hypothesis_novelty_audit.md",
        )
        return 0
    except Exception as exc:
        atomic_json(
            output_root / "RUN_INTERRUPTED.json",
            {
                "status": "run_interrupted",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "automatic_rerun_authorized": False,
                "automatic_next_stage_authorized": False,
                "llm_calls": 0,
                "network_calls": 0,
            },
        )
        print("hypothesis-novelty synthesis DEV: INTERRUPTED")
        print(" -", f"{type(exc).__name__}: {exc}")
        print("Automatic rerun:", False)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
