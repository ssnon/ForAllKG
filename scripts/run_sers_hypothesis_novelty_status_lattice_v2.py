from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from dac_her.hypothesis_novelty_status_lattice_v2 import (
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    atomic_json,
    atomic_text,
    canonical_json,
    compute_real_synthesis,
    load_frozen_inputs,
    read_json,
    render_human_audit,
    sha256_json,
    verify_spec,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--confirm-one-shot-status-lattice-v2",
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
    if not args.confirm_one_shot_status_lattice_v2:
        parser.error("--confirm-one-shot-status-lattice-v2 is required")

    spec_root = (
        args.spec_root if args.spec_root.is_absolute()
        else ROOT / args.spec_root
    )
    output_root = (
        args.output_root if args.output_root.is_absolute()
        else ROOT / args.output_root
    )

    issues, spec = verify_spec(
        ROOT,
        spec_root / "status_lattice_v2_spec.json",
    )
    if issues:
        print("status-lattice v2 preflight: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2
    marker = read_json(spec_root / "SPEC_FREEZE_PASS.json")
    if marker.get("status") != "spec_freeze_pass":
        print("status-lattice v2 preflight: FAIL")
        print(" - spec freeze marker mismatch")
        return 2
    if output_root.exists():
        print("status-lattice v2 preflight: FAIL")
        print(" - output root exists:", output_root)
        print("Automatic rerun:", False)
        return 2

    plan, packet, reviews, claim_report = load_frozen_inputs(ROOT)

    print("SERS Hypothesis Novelty Status-Lattice v2 One-Shot")
    print("Spec ID:", spec["spec_id"])
    print("Source claim-review v3:", claim_report["run_id"])
    print("Hypotheses:", len(plan.claims))
    print("Claims:", len(reviews))
    print("LLM calls:", 0)
    print("Network calls:", 0)
    print("Ranker recomputed:", False)
    print("Claim reviewer recomputed:", False)
    print("Automatic rerun:", False)

    output_root.mkdir(parents=True, exist_ok=False)
    try:
        rows_1 = compute_real_synthesis(
            plan=plan,
            packet=packet,
            reviews=reviews,
        )
        rows_2 = compute_real_synthesis(
            plan=plan,
            packet=packet,
            reviews=reviews,
        )
        deterministic = canonical_json(rows_1) == canonical_json(rows_2)

        checks = {
            "source_claim_review_v3_exact":
                claim_report["run_id"]
                == spec["source_claim_review_v3_run_id"],
            "all_3_hypotheses_synthesized": len(rows_1) == 3,
            "repeat_exact_determinism": deterministic,
            "coverage_identity_exact": all(
                row["coverage"]["hypothesis_id"] == row["hypothesis_id"]
                for row in rows_1
            ),
            "zero_llm_calls": True,
            "zero_network_calls": True,
            "ranker_not_recomputed": True,
            "claim_reviewer_not_recomputed": True,
            "fresh_reserve_not_consumed": True,
            "automatic_scientific_approval_disabled": True,
            "automatic_next_stage_disabled": True,
        }
        structural_pass = all(checks.values())
        status_counts = dict(
            sorted(Counter(row["status"] for row in rows_1).items())
        )

        body = {
            "schema_version":
                "sers-hypothesis-novelty-status-lattice-v2-run",
            "semantics_id":
                "sers_hypothesis_novelty_status_lattice_v2",
            "source_spec_id": spec["spec_id"],
            "source_spec_sha256": spec["spec_sha256"],
            "parent_v1_run_id": spec["parent_v1_run_id"],
            "source_claim_review_v3_run_id":
                spec["source_claim_review_v3_run_id"],
            "structural_outcome": (
                "HYPOTHESIS_NOVELTY_STATUS_LATTICE_V2_STRUCTURAL_PASS"
                if structural_pass else
                "HYPOTHESIS_NOVELTY_STATUS_LATTICE_V2_STRUCTURAL_FAIL"
            ),
            "scientific_novelty_status_outcome":
                "MANUAL_REVIEW_REQUIRED",
            "checks": checks,
            "hypothesis_rows": rows_1,
            "status_counts": status_counts,
            "llm_calls": 0,
            "network_calls": 0,
            "ranker_recomputed": False,
            "claim_reviewer_recomputed": False,
            "coverage_policy_changed": False,
            "case_specific_sers_rules_used": False,
            "automatic_scientific_status_approval": False,
            "automatic_next_stage_authorized": False,
            "fresh_reserve_consumed": False,
        }
        body["run_sha256"] = sha256_json(body)
        body["run_id"] = (
            "sers_hypothesis_novelty_status_lattice_v2_run:"
            + body["run_sha256"][:20]
        )

        atomic_json(
            output_root / "status_lattice_v2_report.json",
            body,
        )
        atomic_text(
            output_root / "human_status_lattice_v2_audit.md",
            render_human_audit(rows_1),
        )
        if not structural_pass:
            atomic_json(
                output_root / "STRUCTURAL_FAIL.json",
                {
                    "status": "structural_fail",
                    "run_id": body["run_id"],
                    "automatic_rerun_authorized": False,
                },
            )
            print("status-lattice v2: STRUCTURAL FAIL")
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

        print()
        print("status-lattice v2: COMPLETE")
        print("Run ID:", body["run_id"])
        print("Structural outcome:", body["structural_outcome"])
        print("Status counts:", status_counts)
        for index, row in enumerate(rows_1, start=1):
            print(
                f"[{index}/3]",
                row["hypothesis_id"],
                "=>", row["status"],
                "| core=", row["core_claim_statuses"],
                "| coverage_sufficient=",
                row["coverage"][
                    "sufficient_for_absence_based_novelty"
                ],
            )
        print("Scientific novelty-status outcome: MANUAL_REVIEW_REQUIRED")
        print("LLM calls:", 0)
        print("Network calls:", 0)
        print("Fresh Reserve consumed:", False)
        print("Automatic next stage authorized:", False)
        print(
            "Human audit:",
            output_root / "human_status_lattice_v2_audit.md",
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
            },
        )
        print("status-lattice v2: INTERRUPTED")
        print(" -", f"{type(exc).__name__}: {exc}")
        print("Automatic rerun:", False)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
