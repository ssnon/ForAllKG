from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.relational_atomic_binding_plan import (
    build_relational_atomic_binding_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze candidate-to-final Alpha6 lineage and determine which "
            "existing N10 canonical claims are specification-complete enough "
            "for a later literal-only relation-endpoint binding step."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plan = build_relational_atomic_binding_plan(
        run_dir=args.run_dir,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (
            args.run_dir.expanduser().resolve()
            / "relational_atomic_binding_plan.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            plan.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Relational atomic binding plan complete")
    print("Hypotheses:", plan.hypothesis_count)
    print(
        "Ready/not ready hypotheses:",
        plan.ready_hypothesis_count,
        "/",
        plan.not_ready_hypothesis_count,
    )
    print(
        "Claims/binding-ready/novelty-bearing-ready:",
        plan.claim_count,
        "/",
        plan.binding_ready_claim_count,
        "/",
        plan.novelty_bearing_binding_ready_claim_count,
    )
    print("Hypothesis states:", plan.hypothesis_status_counts)
    print("Claim states:", plan.claim_status_counts)
    for hypothesis in plan.hypotheses:
        print(
            " ",
            hypothesis.final_hypothesis_id,
            "<-",
            hypothesis.candidate_hypothesis_id,
            hypothesis.binding_status,
            "claims=",
            hypothesis.claim_count,
            "ready=",
            hypothesis.binding_ready_claim_count,
            "novelty-ready=",
            hypothesis.novelty_bearing_binding_ready_claim_count,
        )
        for claim in hypothesis.claims:
            if claim.binding_status == "READY_FOR_LITERAL_ENDPOINT_BINDING":
                print(
                    "   READY",
                    claim.claim_id,
                    claim.novelty_selection_role,
                )
            else:
                print(
                    "   SKIP",
                    claim.claim_id,
                    claim.reason_codes,
                )
    print("Endpoint binding performed: false")
    print("Verifier result observed: false")
    print("Production authority: false")
    print("Plan:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
