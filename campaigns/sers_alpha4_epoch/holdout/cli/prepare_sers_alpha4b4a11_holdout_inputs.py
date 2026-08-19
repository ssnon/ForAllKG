from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import networkx as nx

from dac_her.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
    measurement_value_payload_issues,
)
from campaigns.sers_alpha4_epoch.holdout.cli.run_sers_alpha4b4_holdout import (
    assert_no_manual_resolution,
    atomic_write_json,
    read_json,
    resolve_latest_strict_run,
    sha256_file,
)
from campaigns.sers_alpha4_epoch.holdout.cli.run_sers_alpha4b4a11_holdout import (
    DEFAULT_PROTOCOL,
    validate_protocol,
    verify_calibration_freeze,
    verify_frozen_blobs,
    verify_holdout_input_refreeze,
    verify_runtime_semantics,
)


from campaigns.sers_alpha4_epoch.paths import PROJECT_ROOT


def _strict_snapshot_unchanged(
    root: Path,
    snapshot: dict[str, Any],
) -> None:
    for item in snapshot.values():
        if not item:
            continue
        path = root / str(item["path"])
        if not path.exists() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(
                f"Frozen strict-run input changed during canonical "
                f"refreeze: {path}"
            )


def _run_command(
    *,
    command: list[str],
    root: Path,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(command))
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"Canonical refreeze command failed ({code}): {command!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-materialize SERS_2/6/10 canonical GraphML from the same "
            "frozen strict runs under the alpha4b.4a Measurement merge "
            "invariant, before the refrozen holdout epoch starts."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT.resolve()
    protocol_path = (
        args.protocol
        if args.protocol.is_absolute()
        else (root / args.protocol)
    ).resolve()
    protocol = read_json(protocol_path)
    validate_protocol(protocol)

    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if branch and branch != "feat/SERS-specification-v2.9.1":
        print(
            f"[WARNING] branch={branch!r}; exact frozen hashes are "
            "authoritative."
        )

    verify_frozen_blobs(root, protocol)
    verify_runtime_semantics(protocol)
    verify_calibration_freeze(root, protocol)

    refreeze = protocol["holdout_input_refreeze"]
    report_path = (root / str(refreeze["report"])).resolve()
    if report_path.exists():
        verified = verify_holdout_input_refreeze(root, protocol)
        print("Holdout canonical inputs already refrozen and verified.")
        print("Report:", root / verified["path"])
        return 0

    evaluation_root = report_path.parent
    manifest_path = evaluation_root / "manifest.json"
    logs_root = evaluation_root / "logs"
    snapshots_root = evaluation_root / "pre_refreeze_canonical"
    evaluation_root.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("protocol_sha256") != sha256_file(protocol_path):
            raise RuntimeError(
                "Input-refreeze protocol changed after preparation started."
            )
    else:
        manifest = {
            "refreeze_id": refreeze["refreeze_id"],
            "status": "running",
            "protocol_path": str(protocol_path.relative_to(root)),
            "protocol_sha256": sha256_file(protocol_path),
            "paper_ids": list(protocol["holdout_papers"]),
            "paper_records": {},
        }
        atomic_write_json(manifest_path, manifest)

    data_root = root / str(protocol["data_root"])
    config_path = root / str(protocol["config_path"])

    for paper_id in protocol["holdout_papers"]:
        records = manifest.setdefault("paper_records", {})
        prior = records.get(paper_id)
        if isinstance(prior, dict) and prior.get("status") == "complete":
            canonical = (
                data_root
                / "extracted"
                / paper_id
                / f"{paper_id}.graphml"
            )
            if (
                canonical.exists()
                and sha256_file(canonical)
                == prior.get("post_canonical_sha256")
            ):
                print(f"[SKIP VERIFIED] {paper_id}")
                continue
            raise RuntimeError(
                f"Completed input-refreeze artifact drifted: {paper_id}"
            )

        paper_root = data_root / "extracted" / paper_id
        assert_no_manual_resolution(paper_root)
        run_id, run_dir, strict_snapshot = resolve_latest_strict_run(
            root,
            paper_root,
        )
        canonical = paper_root / f"{paper_id}.graphml"

        record: dict[str, Any] = {
            "status": "running",
            "strict_run_id": run_id,
            "strict_run_directory": str(run_dir.relative_to(root)),
            "strict_run_files": strict_snapshot,
        }

        if canonical.exists():
            backup = snapshots_root / paper_id / canonical.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copy2(canonical, backup)
            record["pre_canonical"] = {
                "path": str(canonical.relative_to(root)),
                "sha256": sha256_file(canonical),
                "snapshot": str(backup.relative_to(root)),
            }
        else:
            record["pre_canonical"] = {
                "path": str(canonical.relative_to(root)),
                "exists": False,
            }

        records[paper_id] = record
        atomic_write_json(manifest_path, manifest)

        build_command = [
            sys.executable,
            "-m",
            "scripts.build_paper_graph",
            "--paper-id",
            paper_id,
            "--config",
            str(config_path),
            "--domain-profile",
            str(protocol["domain_profile"]),
            "--data-root",
            str(protocol["data_root"]),
            "--run-id",
            run_id,
        ]
        _run_command(
            command=build_command,
            root=root,
            log_path=logs_root / f"{paper_id}_build_paper_graph.log",
        )

        if not canonical.exists():
            raise FileNotFoundError(canonical)
        graph = nx.read_graphml(canonical, force_multigraph=True)
        observed_invariant = str(
            graph.graph.get("measurement_merge_invariant_id", "")
        )
        if observed_invariant != MEASUREMENT_MERGE_INVARIANT_ID:
            raise RuntimeError(
                f"{paper_id} canonical graph lacks refrozen Measurement "
                f"merge invariant: {observed_invariant!r}."
            )
        xor_issues = measurement_value_payload_issues(graph)
        if xor_issues:
            raise RuntimeError(
                f"{paper_id} still violates Measurement XOR after "
                f"canonical refreeze: {xor_issues[:5]!r}"
            )

        assert_no_manual_resolution(paper_root)
        _strict_snapshot_unchanged(root, strict_snapshot)

        collision_report = run_dir / "id_collisions.csv"
        record.update(
            {
                "status": "complete",
                "post_canonical_sha256": sha256_file(canonical),
                "post_canonical_nodes": graph.number_of_nodes(),
                "post_canonical_edges": graph.number_of_edges(),
                "measurement_merge_invariant_id": observed_invariant,
                "measurement_xor_issue_count": 0,
                "id_collision_report": (
                    {
                        "path": str(collision_report.relative_to(root)),
                        "sha256": sha256_file(collision_report),
                    }
                    if collision_report.exists()
                    else None
                ),
            }
        )
        atomic_write_json(manifest_path, manifest)

    paper_records = manifest["paper_records"]
    report = {
        "refreeze_id": refreeze["refreeze_id"],
        "holdout_epoch": protocol["holdout_epoch"],
        "paper_ids": list(protocol["holdout_papers"]),
        "measurement_merge_invariant_id": (
            MEASUREMENT_MERGE_INVARIANT_ID
        ),
        "llm_calls_performed": False,
        "strict_extraction_policy": "reuse_same_frozen_strict_runs",
        "canonical_policy": (
            "deterministic_rematerialization_after_domain_independent_"
            "measurement_merge_invariant_fix"
        ),
        "paper_records": paper_records,
        "passes_input_refreeze": all(
            record.get("status") == "complete"
            and record.get("measurement_xor_issue_count") == 0
            and record.get("measurement_merge_invariant_id")
            == MEASUREMENT_MERGE_INVARIANT_ID
            for record in paper_records.values()
        ),
        "count_thresholds_used_for_acceptance": False,
        "note": (
            "Canonical node/edge/collision counts are observations only. "
            "This preparation applies the already-calibrated generic merge "
            "invariant to held-out frozen strict runs and performs no LLM "
            "generation or holdout-specific tuning."
        ),
    }
    atomic_write_json(report_path, report)
    manifest["status"] = "complete"
    manifest["report"] = str(report_path.relative_to(root))
    atomic_write_json(manifest_path, manifest)

    verify_holdout_input_refreeze(root, protocol)
    print()
    print("Holdout canonical input refreeze: PASS")
    print("Invariant:", MEASUREMENT_MERGE_INVARIANT_ID)
    print("Papers:", ", ".join(protocol["holdout_papers"]))
    print("Report:", report_path)
    print(
        "Next: run the new alpha4b.4a11 holdout runner under a fresh "
        "campaign ID."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
