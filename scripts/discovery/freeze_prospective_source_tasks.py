from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from pipeline_core.discovery.prospective_source_task_freeze import (
    ProspectiveSourceTaskCampaignSpec,
    build_prospective_source_task_campaign_freeze,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Untracked evaluation/patch files do not alter committed pipeline
    # semantics. Only tracked modifications block the prospective freeze.
    tracked_status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    return head, bool(tracked_status.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze exactly five fresh P06-P10 scientific source tasks "
            "before hypothesis generation, reframing, N10, prior-art review, "
            "endpoint binding, or scientific-verifier execution."
        )
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec_path = args.spec.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not spec_path.is_file():
        raise ValueError("missing prospective source-task spec: " + str(spec_path))
    if output_path.exists():
        raise ValueError(
            "prospective source-task freeze is write-once; "
            "use a fresh output path"
        )

    spec = ProspectiveSourceTaskCampaignSpec.model_validate_json(
        spec_path.read_text(encoding="utf-8")
    )
    head, dirty = _git_state()

    freeze = build_prospective_source_task_campaign_freeze(
        spec=spec,
        source_spec_sha256=_sha256_file(spec_path),
        repository_head_sha=head,
        repository_tracked_worktree_dirty=dirty,
    )
    write_json_exclusive(output_path, freeze)

    print("Prospective source-task campaign frozen")
    print("Freeze:", freeze.freeze_id)
    print("Repository HEAD:", freeze.repository_head_sha)
    print("Tracked worktree dirty: false")
    print("Cases:", freeze.case_ids)
    print()
    for row in freeze.tasks:
        print(row.case_id, "|", row.task_id)
        print("  family:", row.relation_family)
        print("  source:", row.source)
        print("  target:", row.target)
        print("  question:", row.question)
        print("  run_dir:", row.run_dir)
    print()
    print("Hypotheses generated before freeze: false")
    print("Scientific reframing generated before freeze: false")
    print("Endpoint binding observed before freeze: false")
    print("Prior art observed before freeze: false")
    print("Old N10 observed before freeze: false")
    print("New verifier observed before freeze: false")
    print("Post-freeze task replacement allowed: false")
    print("Failed/abstained case replacement allowed: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
