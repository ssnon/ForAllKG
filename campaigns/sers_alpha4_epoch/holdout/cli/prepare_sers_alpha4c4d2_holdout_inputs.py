from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import networkx as nx

from campaigns.sers_alpha4_epoch.holdout.alpha4c4d2_holdout_support import (
    Alpha4c4d2Error,
    DATA_ROOT,
    ROOT,
    atomic_json,
    canonical_snapshot,
    manual_decisions,
    read_json,
    read_jsonl,
    resolve_strict_source,
    verify_locked_input_record,
    verify_strict_source_unchanged,
)
from dac_her.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
    measurement_value_payload_issues,
)


PROTOCOL = (
    ROOT / "configs/heldout/sers_alpha4c4d2_trend_holdout_v2_run.json"
)


def verify_protocol() -> dict:
    protocol = read_json(PROTOCOL)
    if protocol.get("phase") != "alpha4c.4d.2":
        raise Alpha4c4d2Error("Unexpected protocol phase.")
    if protocol.get("source_split_sha256") != '6eebae74732070408e920154ba898841c879a8e97cb866a8971cb9feef526966':
        raise Alpha4c4d2Error("v2 split SHA drifted.")
    if protocol.get("holdout_papers") != ['Kiwook_SERS_21', 'Kiwook_SERS_38', 'Kiwook_SERS_12', 'Kiwook_SERS_28', 'Kiwook_SERS_17', 'Kiwook_SERS_22', 'Kiwook_SERS_23', 'Kiwook_SERS_11']:
        raise Alpha4c4d2Error("v2 holdout paper set drifted.")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "campaigns.sers_alpha4_epoch.holdout.cli.verify_sers_alpha4c4d1_holdout_v2_protocol",
        ],
        cwd=ROOT,
        check=True,
    )
    return protocol


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    # Resume-safe audit rule: the first snapshot at a campaign path is
    # immutable. A failed preparation may already have captured the genuine
    # pre-refreeze state; rerunning must never overwrite it with the migrated
    # canonical graph.
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Inspect frozen Strict quality and canonical epoch readiness "
            "without modifying canonical graphs or creating the input lock."
        ),
    )
    args = parser.parse_args()

    protocol = verify_protocol()
    eval_root = ROOT / protocol["evaluation_root"]
    prep_root = eval_root / "input_preparation"
    lock_path = eval_root / "canonical_input_lock.json"

    if lock_path.exists():
        lock = read_json(lock_path)
        for paper_id in protocol["holdout_papers"]:
            verify_locked_input_record(
                paper_id,
                lock["papers"][paper_id],
            )
        print("alpha4c.4d.2 canonical input lock already exists: PASS")
        print("Locked papers:", len(protocol["holdout_papers"]))
        return 0

    sources = {
        paper_id: resolve_strict_source(paper_id)
        for paper_id in protocol["holdout_papers"]
    }

    readiness = []
    needs_refreeze = []
    for paper_id in protocol["holdout_papers"]:
        source = sources[paper_id]
        snap = canonical_snapshot(paper_id)
        invariant_ok = (
            snap["canonical_present"]
            and snap["measurement_merge_invariant_id"]
            == MEASUREMENT_MERGE_INVARIANT_ID
            and snap["measurement_xor_issue_count"] == 0
        )
        if not invariant_ok:
            needs_refreeze.append(paper_id)

        readiness.append({
            "paper_id": paper_id,
            "strict_quality":
                source["extraction_quality"]["graph_materialization_status"],
            "active_complete_flag":
                source["active_payload_complete_flag"],
            "requires_allow_incomplete":
                source["requires_allow_incomplete"],
            "canonical_present": snap["canonical_present"],
            "canonical_invariant":
                snap["measurement_merge_invariant_id"],
            "canonical_xor_issue_count":
                snap["measurement_xor_issue_count"],
            "needs_refreeze": not invariant_ok,
        })

    print("alpha4c.4d.2 input readiness")
    for row in readiness:
        print(
            " -", row["paper_id"],
            "quality=" + row["strict_quality"],
            "active_complete=" + str(row["active_complete_flag"]),
            "canonical_invariant=" + repr(row["canonical_invariant"]),
            "refreeze=" + str(row["needs_refreeze"]),
            (
                "[--allow-incomplete]"
                if row["requires_allow_incomplete"]
                else ""
            ),
        )
    print("Refreeze required:", len(needs_refreeze))

    if args.preflight_only:
        print(
            "No canonical graph/resolution file was modified and no "
            "input lock/Trend output was created."
        )
        return 0

    prep_root.mkdir(parents=True, exist_ok=True)
    records = {}

    for paper_id in protocol["holdout_papers"]:
        source = sources[paper_id]
        verify_strict_source_unchanged(source)

        before = canonical_snapshot(paper_id)
        paper_root = DATA_ROOT / "extracted" / paper_id
        canonical_path = paper_root / f"{paper_id}.graphml"
        decisions_path = paper_root / "resolution" / "decisions.jsonl"

        before_decisions = read_jsonl(decisions_path)
        before_manual = manual_decisions(before_decisions)

        copy_if_exists(
            canonical_path,
            prep_root / "pre" / "canonical" / f"{paper_id}.graphml",
        )
        copy_if_exists(
            decisions_path,
            prep_root / "pre" / "resolution" /
            f"{paper_id}.decisions.jsonl",
        )

        invariant_ok = (
            before["canonical_present"]
            and before["measurement_merge_invariant_id"]
            == MEASUREMENT_MERGE_INVARIANT_ID
            and before["measurement_xor_issue_count"] == 0
        )

        build_command = None
        if not invariant_ok:
            build_command = [
                sys.executable,
                "-m",
                "scripts.build_paper_graph",
                "--paper-id",
                paper_id,
                "--config",
                "configs/papers_sers_au_ag.yaml",
                "--domain-profile",
                "sers_au_ag",
                "--data-root",
                "data_sers",
                "--run-id",
                source["run_id"],
            ]
            if source["attempt_id"]:
                build_command.extend([
                    "--attempt-id",
                    source["attempt_id"],
                ])
            if source["requires_allow_incomplete"]:
                build_command.append("--allow-incomplete")

            print("$", " ".join(build_command), flush=True)
            result = subprocess.run(build_command, cwd=ROOT)
            if result.returncode != 0:
                raise Alpha4c4d2Error(
                    f"{paper_id} canonical refreeze failed: "
                    f"exit={result.returncode}"
                )

        verify_strict_source_unchanged(source)

        after = canonical_snapshot(paper_id)
        if not after["canonical_present"]:
            raise Alpha4c4d2Error(
                f"{paper_id} canonical graph missing after preparation."
            )
        if (
            after["measurement_merge_invariant_id"]
            != MEASUREMENT_MERGE_INVARIANT_ID
        ):
            raise Alpha4c4d2Error(
                f"{paper_id} wrong Measurement merge invariant: "
                f"{after['measurement_merge_invariant_id']!r}"
            )
        if after["measurement_xor_issue_count"] != 0:
            raise Alpha4c4d2Error(
                f"{paper_id} canonical Measurement XOR failed."
            )

        after_decisions = read_jsonl(decisions_path)
        after_manual = manual_decisions(after_decisions)
        if before_manual != after_manual:
            raise Alpha4c4d2Error(
                f"{paper_id} manual resolution decisions changed."
            )

        copy_if_exists(
            canonical_path,
            prep_root / "post" / "canonical" / f"{paper_id}.graphml",
        )
        copy_if_exists(
            decisions_path,
            prep_root / "post" / "resolution" /
            f"{paper_id}.decisions.jsonl",
        )

        records[paper_id] = {
            **after,
            "strict_source": source,
            "canonical_refrozen": bool(build_command),
            "build_command": build_command,
            "manual_resolution_decisions_preserved": True,
        }

    lock = {
        "phase": "alpha4c.4d.2",
        "state": "frozen_v2_canonical_input_lock",
        "source_split_sha256": protocol["source_split_sha256"],
        "holdout_papers": protocol["holdout_papers"],
        "metric_definition_semantics_id":
            protocol["frozen_semantics"]["metric_definition"],
        "llm_calls_performed": False,
        "trend_outputs_generated": False,
        "papers": records,
    }
    atomic_json(lock_path, lock)

    for paper_id in protocol["holdout_papers"]:
        verify_locked_input_record(
            paper_id,
            lock["papers"][paper_id],
        )

    report = {
        "phase": "alpha4c.4d.2 input preparation",
        "state": "passed",
        "paper_count": len(records),
        "refrozen_count": sum(
            1 for row in records.values()
            if row["canonical_refrozen"]
        ),
        "llm_calls_performed": False,
        "trend_outputs_generated": False,
        "all_manual_resolution_decisions_preserved": True,
        "all_measurement_xor_clean": True,
        "input_lock":
            str(lock_path.relative_to(ROOT)),
    }
    atomic_json(
        prep_root / "input_preparation_report.json",
        report,
    )

    print("alpha4c.4d.2 input preparation: PASS")
    print("Papers:", len(records))
    print("Refrozen:", report["refrozen_count"])
    print("LLM calls performed: False")
    print("Trend outputs generated: False")
    print("Canonical input lock:", lock_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
