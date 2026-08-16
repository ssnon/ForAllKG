from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from dac_her.hypothesis_novelty_synthesis_dev_validation import (
    EXPECTED_FREEZE_COMMIT,
    FROZEN_INPUT_ROOT,
    HANDOFF_SOURCE_FILES,
    atomic_json,
    sha256_file,
    sha256_json,
    validate_frozen_source,
)

ROOT = Path.cwd()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(fd)
    try:
        shutil.copyfile(source, tmp_name)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.home() / "GraphAgentsDAC-clean",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FROZEN_INPUT_ROOT,
    )
    args = parser.parse_args()

    if not args.run:
        parser.error("--run is required")

    source_root = args.source_root.expanduser().resolve()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )

    try:
        target_branch = git(ROOT, "branch", "--show-current")
        target_head = git(ROOT, "rev-parse", "HEAD")
        source_head = git(source_root, "rev-parse", "HEAD")
    except Exception as exc:
        print("frozen v3 handoff: FAIL")
        print(" -", f"git preflight: {type(exc).__name__}: {exc}")
        print("Writes performed:", 0)
        return 2

    if target_branch != "feat/SERS-novelty-synthesis-next":
        print("frozen v3 handoff: FAIL")
        print(" - target branch mismatch:", target_branch)
        print("Writes performed:", 0)
        return 2
    if target_head != EXPECTED_FREEZE_COMMIT:
        print("frozen v3 handoff: FAIL")
        print(" - target HEAD mismatch:", target_head)
        print("Writes performed:", 0)
        return 2
    if source_head != EXPECTED_FREEZE_COMMIT:
        print("frozen v3 handoff: FAIL")
        print(" - source worktree HEAD mismatch:", source_head)
        print("Writes performed:", 0)
        return 2
    if output_root.exists():
        print("frozen v3 handoff: FAIL")
        print(" - output root exists:", output_root)
        print("Writes performed:", 0)
        return 2

    try:
        validated = validate_frozen_source(source_root)
    except Exception as exc:
        print("frozen v3 handoff: FAIL")
        print(" -", f"{type(exc).__name__}: {exc}")
        print("Writes performed:", 0)
        return 2

    source_files = validated["paths"]
    rows = {}
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        for name in HANDOFF_SOURCE_FILES:
            source = source_files[name]
            target = output_root / name
            atomic_copy(source, target)
            source_sha = sha256_file(source)
            target_sha = sha256_file(target)
            if source_sha != target_sha:
                raise RuntimeError(f"copy SHA mismatch: {name}")
            rows[name] = {
                "source_relative_path": str(HANDOFF_SOURCE_FILES[name]),
                "source_sha256": source_sha,
                "target_sha256": target_sha,
                "byte_count": target.stat().st_size,
            }

        manifest = {
            "schema_version":
                "sers-hypothesis-novelty-frozen-input-handoff-v1",
            "source_freeze_commit": EXPECTED_FREEZE_COMMIT,
            "source_claim_review_v3_spec_id":
                validated["spec"]["spec_id"],
            "source_claim_review_v3_run_id":
                validated["report"]["run_id"],
            "source_query_plan_id": validated["plan"].plan_id,
            "source_query_plan_sha256": validated["plan"].plan_sha256,
            "source_canonical_packet_id": validated["packet"].packet_id,
            "source_canonical_packet_sha256":
                validated["packet"].packet_sha256,
            "files": rows,
            "llm_calls": 0,
            "network_calls": 0,
            "fresh_reserve_consumed": False,
        }
        manifest["manifest_sha256"] = sha256_json(manifest)
        atomic_json(output_root / "handoff_manifest.json", manifest)
        atomic_json(
            output_root / "HANDOFF_PASS.json",
            {
                "status": "handoff_pass",
                "manifest_sha256": manifest["manifest_sha256"],
                "source_claim_review_v3_run_id":
                    validated["report"]["run_id"],
                "llm_calls": 0,
                "network_calls": 0,
                "fresh_reserve_consumed": False,
            },
        )

        # Validate the copied bundle itself, not only the source worktree.
        validate_frozen_source(output_root, flat=True)
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise

    print("frozen v3 handoff: PASS")
    print("Source:", source_root)
    print("Target:", output_root)
    print("Claim-review v3 run:", validated["report"]["run_id"])
    print("Hypotheses:", len(validated["plan"].claims))
    print("Claims:", len(validated["reviews"]))
    print("Canonical works:", len(validated["packet"].works))
    print("Manifest SHA256:", manifest["manifest_sha256"])
    print("LLM calls:", 0)
    print("Network calls:", 0)
    print("Fresh Reserve consumed:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
