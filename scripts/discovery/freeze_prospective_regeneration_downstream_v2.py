from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_regeneration_downstream_v2 import (
    build_regeneration_downstream_v2_freeze,
    sha256_file,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def _git_state() -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return head, dirty


def _require_ancestor(older: str, newer: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "regeneration-unit freeze HEAD is not an ancestor of current HEAD"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze downstream-only evaluation semantics and stage budgets "
            "for prospective regeneration-unit v2 before a new cohort opens."
        )
    )
    parser.add_argument(
        "--regeneration-unit-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_path = args.regeneration_unit_freeze.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not source_path.is_file():
        raise ValueError(
            "missing regeneration-unit freeze: " + str(source_path)
        )
    if output_path.exists():
        raise ValueError(
            "regeneration downstream-v2 freeze output is write-once"
        )

    source = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    head, dirty = _git_state()
    _require_ancestor(source.repository_head_sha, head)

    frozen = build_regeneration_downstream_v2_freeze(
        regeneration_unit_freeze=source,
        regeneration_unit_freeze_file_sha256=sha256_file(source_path),
        repository_head_sha=head,
        repository_tracked_worktree_dirty=dirty,
    )
    write_json_exclusive(output_path, frozen)

    policy = frozen.policy
    budget = policy.budget

    print("Prospective regeneration downstream-v2 frozen")
    print("Freeze:", frozen.freeze_id)
    print("Repository HEAD:", frozen.repository_head_sha)
    print(
        "Source regeneration unit:",
        frozen.source_regeneration_unit_freeze_id,
    )
    print("Input authority:", policy.input_authority)
    print("Full E2E rerun allowed: false")
    print("Regenerated portfolio mutation allowed: false")
    print()
    print("Budget unit:", budget.budget_unit)
    print(
        "semantic critic / external novelty:",
        budget.semantic_critic_stage_max,
        "/",
        budget.external_novelty_stage_max,
    )
    print(
        "N9 intake / N9 closure / N10:",
        budget.n9_shadow_intake_stage_max,
        "/",
        budget.n9_full_closure_stage_max,
        "/",
        budget.n10_certification_stage_max,
    )
    print(
        "binding plan / Gate v2:",
        budget.binding_plan_build_max,
        "/",
        budget.preverifier_gate_v2_max,
    )
    print(
        "continuation / refinement / second regeneration:",
        budget.targeted_novelty_continuation_max,
        "/",
        budget.novelty_refinement_max,
        "/",
        budget.same_lineage_hypothesis_regeneration_max,
    )
    print("Provider parse retries max:", budget.provider_parse_retries_max)
    print("Downstream failure retry allowed: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
