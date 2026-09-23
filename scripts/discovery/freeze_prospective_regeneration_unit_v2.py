from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze prospective regeneration-unit v2 semantics before "
            "opening a new routed prospective cohort."
        )
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(
            "prospective regeneration-unit v2 freeze is write-once"
        )

    head, dirty = _git_state()
    frozen = build_regeneration_unit_v2_freeze(
        repository_head_sha=head,
        repository_tracked_worktree_dirty=dirty,
    )
    write_json_exclusive(output, frozen)

    policy = frozen.policy
    print("Prospective regeneration-unit v2 frozen")
    print("Freeze:", frozen.freeze_id)
    print("Repository HEAD:", frozen.repository_head_sha)
    print("Semantic name:", policy.semantic_name)
    print(
        "Structured generation calls max:",
        policy.structured_generation_calls_per_hypothesis_max,
    )
    print(
        "Structured repair calls allowed:",
        policy.structured_repair_calls_allowed,
    )
    print(
        "Max hypotheses per generation:",
        policy.max_hypotheses_per_generation_call,
    )
    print("Deterministic compiler:", policy.deterministic_compiler)
    print("Full E2E rerun is regeneration: false")
    print("Downstream evaluation is separate stage: true")
    print("Downstream budget must be frozen separately: true")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
