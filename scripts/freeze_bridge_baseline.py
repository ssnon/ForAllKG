from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_POLICY_SET_ID = "5c3a56ec4102"

EXPECTED_RUNS = {
    "Kiwook_1": {
        "bridge_extraction_id": "b68eca2bd9b812d0",
        "bridge_policy_run_id": "218915dac6641fb9",
    },
    "Kiwook_2": {
        "bridge_extraction_id": "0d53689bf967d1c5",
        "bridge_policy_run_id": "4685fb0aae5d9d7a",
    },
    "Kiwook_3": {
        "bridge_extraction_id": "b55fa58e7fd1a782",
        "bridge_policy_run_id": "e69b507a5a6db84a",
    },
}

EXPECTED_METRICS = {
    "relation_candidates": 79,
    "confirmed_precision": 1.0,
    "confirmed_plus_candidate_recall": 1.0,
    "fatal_rejection_false_negative_count": 0,
    "accepted_without_evidence_pointers": 0,
    "semantic_candidates_without_evidence_pointers": 0,
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def csv_relation_count(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    relation_rows = [
        row
        for row in rows
        if str(row.get("in_relation_calibration", "")).strip().lower()
        in {"1", "true", "yes"}
    ]
    return len(rows), len(relation_rows)


def resolve_strict_run_dir(paper_root: Path) -> Path:
    pointer = read_json(paper_root / "latest_run.json")

    for key in (
        "run_directory",
        "strict_run_directory",
    ):
        value = pointer.get(key)
        if value:
            return Path(str(value)).expanduser().resolve()

    run_id = pointer.get("run_id")
    if run_id:
        candidate = paper_root / "runs" / str(run_id)
        if candidate.exists():
            return candidate.resolve()

    raise RuntimeError(
        f"Could not resolve strict run directory from {paper_root / 'latest_run.json'}"
    )


def verify_metric(report: dict[str, Any], key: str, expected: Any) -> None:
    actual = report.get(key)

    if isinstance(expected, float):
        if actual is None or abs(float(actual) - expected) > 1e-12:
            raise RuntimeError(
                f"Metric mismatch for {key}: expected {expected!r}, got {actual!r}"
            )
        return

    if actual != expected:
        raise RuntimeError(
            f"Metric mismatch for {key}: expected {expected!r}, got {actual!r}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the validated three-paper Bridge calibration baseline. "
            "The script verifies the expected immutable run IDs, snapshots "
            "ignored data_dac run directories into a tar.gz archive, and writes "
            "a hash-bearing freeze manifest."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--policy-set-id",
        default=DEFAULT_POLICY_SET_ID,
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow freezing with an uncommitted Git working tree.",
    )
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help=(
            "Write only the manifest and checksums. This is not recommended "
            "because data_dac/ is Git-ignored."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.project_root.expanduser().resolve()

    calibration_dir = root / "calibration" / "bridge_semantic"
    reports_dir = calibration_dir / "reports"
    report_path = reports_dir / f"report_{args.policy_set_id}.json"
    errors_path = reports_dir / f"errors_{args.policy_set_id}.csv"
    gold_path = calibration_dir / "gold.csv"
    predictions_path = calibration_dir / "predictions.csv"
    calibration_manifest_path = calibration_dir / "calibration_manifest.json"

    required = (
        report_path,
        errors_path,
        gold_path,
        predictions_path,
        calibration_manifest_path,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    report = read_json(report_path)

    if report.get("policy_set_id") != args.policy_set_id:
        raise RuntimeError(
            "Report policy_set_id mismatch: "
            f"{report.get('policy_set_id')!r} != {args.policy_set_id!r}"
        )

    for key, expected in EXPECTED_METRICS.items():
        verify_metric(report, key, expected)

    all_count, relation_count = csv_relation_count(gold_path)
    if (all_count, relation_count) != (93, 79):
        raise RuntimeError(
            f"Unexpected gold candidate counts: {(all_count, relation_count)!r}"
        )

    prediction_all, prediction_relation = csv_relation_count(predictions_path)
    if (prediction_all, prediction_relation) != (93, 79):
        raise RuntimeError(
            "Unexpected prediction candidate counts: "
            f"{(prediction_all, prediction_relation)!r}"
        )

    commit = git_output(root, "rev-parse", "HEAD")
    branch = git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = git_output(root, "status", "--porcelain")

    if status and not args.allow_dirty:
        raise RuntimeError(
            "Git working tree is not clean. Commit the policy, gold, evaluator, "
            "and projection changes before freezing, or pass --allow-dirty."
        )

    report_run_ids = sorted(
        str(value)
        for value in report.get("bridge_policy_run_ids", [])
    )
    expected_run_ids = sorted(
        item["bridge_policy_run_id"]
        for item in EXPECTED_RUNS.values()
    )
    if report_run_ids != expected_run_ids:
        raise RuntimeError(
            f"Report run IDs do not match expected current runs: "
            f"{report_run_ids!r} != {expected_run_ids!r}"
        )

    frozen_root = (
        calibration_dir
        / "frozen"
        / args.policy_set_id
    )
    frozen_root.mkdir(parents=True, exist_ok=True)

    run_records: dict[str, Any] = {}
    strict_run_dirs: dict[str, Path] = {}

    for paper_id, expected in EXPECTED_RUNS.items():
        paper_root = root / "data_dac" / "extracted" / paper_id
        strict_run_dir = resolve_strict_run_dir(paper_root)
        strict_run_dirs[paper_id] = strict_run_dir

        policy_pointer = read_json(
            strict_run_dir / "latest_bridge_policy_run.json"
        )
        extraction_pointer = read_json(
            strict_run_dir / "latest_bridge_extraction.json"
        )

        policy_run_id = str(
            policy_pointer["bridge_policy_run_id"]
        )
        extraction_id = str(
            extraction_pointer["bridge_extraction_id"]
        )

        if policy_run_id != expected["bridge_policy_run_id"]:
            raise RuntimeError(
                f"{paper_id}: policy run mismatch: "
                f"{policy_run_id!r} != {expected['bridge_policy_run_id']!r}"
            )
        if extraction_id != expected["bridge_extraction_id"]:
            raise RuntimeError(
                f"{paper_id}: extraction ID mismatch: "
                f"{extraction_id!r} != {expected['bridge_extraction_id']!r}"
            )

        policy_dir = Path(
            str(policy_pointer["bridge_policy_run_directory"])
        ).expanduser().resolve()
        extraction_dir = Path(
            str(extraction_pointer["bridge_extraction_directory"])
        ).expanduser().resolve()

        policy_run = read_json(policy_dir / "run.json")
        extraction_run = read_json(extraction_dir / "run.json")
        policy_summary = read_json(policy_dir / "summary.json")

        run_records[paper_id] = {
            "strict_run_directory": str(strict_run_dir),
            "bridge_extraction_directory": str(extraction_dir),
            "bridge_policy_run_directory": str(policy_dir),
            "bridge_extraction_id": extraction_id,
            "bridge_extraction_fingerprint": extraction_run.get(
                "bridge_extraction_fingerprint", ""
            ),
            "bridge_policy_run_id": policy_run_id,
            "bridge_policy_run_fingerprint": policy_run.get(
                "bridge_policy_run_fingerprint", ""
            ),
            "bridge_prompt_version": extraction_run.get(
                "bridge_prompt_version", ""
            ),
            "bridge_policy_version": policy_run.get(
                "bridge_policy_version", ""
            ),
            "model": extraction_run.get("model", ""),
            "provider": extraction_run.get("provider", ""),
            "policy_summary": policy_summary,
        }

    snapshot_files = {
        "gold.csv": gold_path,
        "predictions.csv": predictions_path,
        "calibration_manifest.json": calibration_manifest_path,
        f"report_{args.policy_set_id}.json": report_path,
        f"errors_{args.policy_set_id}.csv": errors_path,
    }

    snapshot_dir = frozen_root / "calibration_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for name, source in snapshot_files.items():
        shutil.copy2(source, snapshot_dir / name)

    archive_path: Path | None = None
    archive_sha256 = ""

    if not args.skip_archive:
        archive_dir = (
            root
            / "data_dac"
            / "frozen_baselines"
        )
        archive_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        archive_path = archive_dir / (
            f"bridge_baseline_{args.policy_set_id}.tar.gz"
        )

        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(
                snapshot_dir,
                arcname="calibration_snapshot",
            )

            for paper_id, strict_run_dir in strict_run_dirs.items():
                archive.add(
                    strict_run_dir,
                    arcname=f"strict_runs/{paper_id}",
                )

                paper_root = root / "data_dac" / "extracted" / paper_id
                for suffix in (
                    ".graphml",
                    ".bridge.graphml",
                    ".bridge.candidates.graphml",
                ):
                    path = paper_root / f"{paper_id}{suffix}"
                    if path.exists():
                        archive.add(
                            path,
                            arcname=f"latest_aliases/{paper_id}/{path.name}",
                        )

        archive_sha256 = sha256_file(archive_path)

    file_hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in required
    }

    manifest = {
        "baseline_name": (
            "dac-her-bridge-v2.3.3-frozen-3paper"
        ),
        "frozen_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "policy_set_id": args.policy_set_id,
        "git": {
            "source_code_commit": commit,
            "source_branch": branch,
            "working_tree_was_clean_before_freeze": not bool(status),
            "dirty_status_before_freeze": status.splitlines(),
            "note": (
                "The annotated freeze tag should point to the later commit "
                "that adds this manifest and calibration snapshot. "
                "source_code_commit identifies the clean code/gold state "
                "used to generate the frozen run."
            ),
        },
        "calibration_split": {
            "calibration_papers": list(EXPECTED_RUNS),
            "held_out_papers": [
                "Kiwook_4",
                "Kiwook_5",
                "Kiwook_6",
                "Kiwook_7",
                "Kiwook_8",
                "Kiwook_9",
                "Kiwook_10",
            ],
            "rule": (
                "No prompt, schema, model/provider, relation cue, repair, "
                "candidate partition, or projection-policy tuning is allowed "
                "during the seven-paper held-out evaluation."
            ),
        },
        "runs": run_records,
        "metrics": {
            key: report.get(key)
            for key in (
                "relation_candidates",
                "automatic_lane_counts",
                "confirmed_precision",
                "confirmed_recall",
                "confirmed_plus_candidate_recall",
                "fatal_rejection_precision",
                "fatal_rejection_false_negative_count",
                "accepted_relation_label_accuracy",
                "accepted_argument_tuple_accuracy",
                "accepted_without_evidence_pointers",
                "semantic_candidates_without_evidence_pointers",
            )
        },
        "calibration_file_sha256": file_hashes,
        "archive": {
            "path": (
                str(archive_path.relative_to(root))
                if archive_path is not None
                else ""
            ),
            "sha256": archive_sha256,
            "note": (
                "data_dac/ is Git-ignored. Preserve this archive in a durable "
                "release or research-artifact store; do not rely on the Git tag "
                "alone to retain generated run data."
            ),
        },
        "gold_semantics": {
            "candidate_key_is_primary_identity": True,
            "gold_policy_run_id_columns_are_adjudication_snapshot_metadata": True,
            "current_policy_run_ids_are_recorded_under_runs": True,
        },
    }

    manifest_path = frozen_root / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Bridge baseline freeze prepared")
    print("Policy set:", args.policy_set_id)
    print("Git commit:", commit)
    print("Manifest:", manifest_path)
    if archive_path is not None:
        print("Archive:", archive_path)
        print("Archive SHA256:", archive_sha256)
    print()
    print("Next commands:")
    print(
        "  git add scripts/freeze_bridge_baseline.py "
        f"calibration/bridge_semantic/frozen/{args.policy_set_id}"
    )
    print(
        '  git commit -m "record three-paper Bridge baseline freeze"'
    )
    print(
        "  git tag -a bridge-v2.3.3-frozen-3paper "
        '-m "Bridge calibration policy set '
        f'{args.policy_set_id}"'
    )


if __name__ == "__main__":
    main()
