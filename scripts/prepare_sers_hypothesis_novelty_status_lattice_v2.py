from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dac_her.hypothesis_novelty_status_lattice_v2 import (
    DEFAULT_SPEC_ROOT,
    atomic_json,
    build_spec,
    characterize_status_lattice,
    verify_spec,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")

    output_root = (
        args.output_root if args.output_root.is_absolute()
        else ROOT / args.output_root
    )
    if output_root.exists():
        print("status-lattice v2 spec freeze: FAIL")
        print(" - output root exists:", output_root)
        return 2

    try:
        spec = build_spec(ROOT)
        lattice = characterize_status_lattice()
    except Exception as exc:
        print("status-lattice v2 spec freeze: FAIL")
        print(" -", f"{type(exc).__name__}: {exc}")
        print("Writes performed:", 0)
        return 2

    print("SERS Hypothesis Novelty Status-Lattice v2 Specification")
    print("Prospective spec ID:", spec["spec_id"])
    print("Parent v1 run:", spec["parent_v1_run_id"])
    print("Source claim-review v3:", spec["source_claim_review_v3_run_id"])
    print("Status-lattice cases:", spec["status_lattice_case_count"])
    print("New status:", "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP")
    print("Title-only fail-closed:", True)
    print("New-combination requires relation-backed core:", True)
    print("Coverage policy changed:", False)
    print("LLM calls:", 0)
    print("Network calls:", 0)
    print("Scientific status approval:", False)

    output_root.mkdir(parents=True, exist_ok=False)
    try:
        atomic_json(output_root / "status_lattice_v2_spec.json", spec)
        atomic_json(output_root / "status_lattice_v2_audit.json", lattice)
        issues, verified = verify_spec(
            ROOT,
            output_root / "status_lattice_v2_spec.json",
        )
        if issues:
            raise RuntimeError(
                "spec verification failed:\n- "
                + "\n- ".join(issues)
            )
        atomic_json(
            output_root / "SPEC_FREEZE_PASS.json",
            {
                "status": "spec_freeze_pass",
                "spec_id": verified["spec_id"],
                "spec_sha256": verified["spec_sha256"],
                "llm_calls": 0,
                "network_calls": 0,
                "scientific_status_approval": False,
                "fresh_reserve_consumed": False,
            },
        )
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise

    print("status-lattice v2 specification: FROZEN")
    print("Spec ID:", spec["spec_id"])
    print()
    print("Manual semantic probes:")
    for row in lattice["manual_semantic_review_probes"]:
        print(
            " -", row["name"],
            row["core_statuses"],
            "+ coverage=", row["coverage_sufficient"],
            "=>", row["observed_status"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
