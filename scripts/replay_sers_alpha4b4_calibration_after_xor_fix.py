from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from dac_her.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
    measurement_value_payload_issues,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PAPERS = (
    "Kiwook_SERS_1",
    "Kiwook_SERS_5",
    "Kiwook_SERS_8",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(
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
            f"Calibration replay command failed ({code}): {command!r}"
        )


def _latest_run_id(root: Path, paper_root: Path) -> str:
    pointer = _read_json(paper_root / "latest_run.json")
    run_dir = Path(str(pointer.get("run_directory", ""))).expanduser()
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    run_json = _read_json(run_dir / "run.json")
    run_id = str(
        run_json.get("run_id")
        or pointer.get("run_id")
        or ""
    ).strip()
    if not run_id:
        raise RuntimeError(
            f"Cannot resolve frozen strict run for {paper_root.name}."
        )
    return run_id


def _snapshot(
    *,
    source: Path,
    destination: Path,
) -> dict[str, object]:
    if not source.exists():
        return {
            "path": str(source),
            "exists": False,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": str(source),
        "snapshot": str(destination),
        "exists": True,
        "sha256": _sha256(source),
    }


def _summary(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the frozen SERS 1/5/8 calibration after the "
            "alpha4b.4 measurement-merge invariant fix. No LLM calls are made; "
            "existing frozen Bridge graphs are reused."
        )
    )
    parser.add_argument(
        "--replay-id",
        default="sers_alpha4b4a_xorfix_calibration_replay_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT.resolve()
    data_root = root / "data_sers"
    config = root / "configs" / "papers_sers_au_ag.yaml"
    evaluation_root = (
        root
        / "evaluation"
        / "sers_alpha4b4"
        / "invariant_fix_xor_v1"
        / args.replay_id
    )
    logs_root = evaluation_root / "logs"
    snapshots_root = evaluation_root / "pre_fix_snapshots"

    old_calibration_root = (
        data_root
        / "corpus"
        / "sers_alpha4b3a_calibration"
        / "exploratory"
    )
    old = {
        "corpus": _summary(old_calibration_root / "manifest.json"),
        "reproducibility": _summary(
            old_calibration_root
            / "reproducibility"
            / "sers_alpha4b3b4a1_calibration"
            / "summary.json"
        ),
        "metric_definition": _summary(
            old_calibration_root
            / "metric_definition"
            / "sers_alpha4b3b4b1_calibration"
            / "summary.json"
        ),
        "comparison": _summary(
            old_calibration_root
            / "comparison"
            / "sers_alpha4b3b4c1_calibration"
            / "summary.json"
        ),
    }

    paper_records: dict[str, Any] = {}
    for paper_id in CALIBRATION_PAPERS:
        paper_root = data_root / "extracted" / paper_id
        canonical = paper_root / f"{paper_id}.graphml"
        bridge = paper_root / f"{paper_id}.bridge.graphml"
        candidate_bridge = (
            paper_root / f"{paper_id}.bridge.candidates.graphml"
        )
        if not bridge.exists() or not candidate_bridge.exists():
            raise FileNotFoundError(
                "Calibration replay intentionally reuses frozen Bridge "
                f"materialization, but it is missing for {paper_id}."
            )

        record: dict[str, Any] = {
            "strict_run_id": _latest_run_id(root, paper_root),
            "pre_fix_canonical": _snapshot(
                source=canonical,
                destination=(
                    snapshots_root / paper_id / canonical.name
                ),
            ),
            "bridge_sha256": _sha256(bridge),
            "candidate_bridge_sha256": _sha256(candidate_bridge),
        }

        build_command = [
            sys.executable,
            "-m",
            "scripts.build_paper_graph",
            "--paper-id",
            paper_id,
            "--config",
            str(config),
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            "data_sers",
            "--run-id",
            record["strict_run_id"],
        ]
        _run(
            command=build_command,
            root=root,
            log_path=logs_root / f"{paper_id}_build.log",
        )

        graph = nx.read_graphml(canonical, force_multigraph=True)
        issues = measurement_value_payload_issues(graph)
        if issues:
            raise RuntimeError(
                f"{paper_id} still violates Measurement XOR after rebuild: "
                f"{issues[:5]!r}"
            )
        invariant = str(
            graph.graph.get("measurement_merge_invariant_id", "")
        )
        if invariant != MEASUREMENT_MERGE_INVARIANT_ID:
            raise RuntimeError(
                f"{paper_id} canonical graph lacks invariant marker: "
                f"{invariant!r}."
            )

        audit_dir = evaluation_root / "canonical_audit" / paper_id
        _run(
            command=[
                sys.executable,
                "-m",
                "scripts.inspect_graphml",
                "--graphml",
                str(canonical),
                "--output-dir",
                str(audit_dir),
            ],
            root=root,
            log_path=logs_root / f"{paper_id}_audit.log",
        )

        for mode in ("evidence", "mechanism", "exploratory"):
            command = [
                sys.executable,
                "-m",
                "scripts.build_graphagents_projection",
                "--paper-id",
                paper_id,
                "--domain-profile",
                "sers_au_ag",
                "--data-root",
                "data_sers",
                "--mode",
                mode,
                "--canonical-graphml",
                str(canonical),
            ]
            if mode in {"mechanism", "exploratory"}:
                command.extend(
                    ["--bridge-graphml", str(bridge)]
                )
            if mode == "exploratory":
                command.extend(
                    [
                        "--candidate-bridge-graphml",
                        str(candidate_bridge),
                    ]
                )
            _run(
                command=command,
                root=root,
                log_path=(
                    logs_root / f"{paper_id}_projection_{mode}.log"
                ),
            )

        record.update(
            {
                "post_fix_canonical_sha256": _sha256(canonical),
                "post_fix_nodes": graph.number_of_nodes(),
                "post_fix_edges": graph.number_of_edges(),
                "measurement_xor_issue_count": 0,
                "measurement_merge_invariant_id": invariant,
            }
        )
        paper_records[paper_id] = record

    corpus_id = f"{args.replay_id}_corpus"
    reproducibility_id = f"{args.replay_id}_reproducibility"
    metric_definition_id = f"{args.replay_id}_metric_definition"
    comparison_id = f"{args.replay_id}_comparison"

    _run(
        command=[
            sys.executable,
            "-m",
            "scripts.build_corpus_graph",
            "--corpus-id",
            corpus_id,
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            "data_sers",
            "--paper-ids",
            *CALIBRATION_PAPERS,
            "--mode",
            "exploratory",
        ],
        root=root,
        log_path=logs_root / "corpus.log",
    )
    _run(
        command=[
            sys.executable,
            "-m",
            "scripts.build_reproducibility_evidence",
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            "data_sers",
            "--corpus-id",
            corpus_id,
            "--mode",
            "exploratory",
            "--reproducibility-id",
            reproducibility_id,
        ],
        root=root,
        log_path=logs_root / "reproducibility.log",
    )
    _run(
        command=[
            sys.executable,
            "-m",
            "scripts.build_metric_definition_contexts",
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            "data_sers",
            "--corpus-id",
            corpus_id,
            "--mode",
            "exploratory",
            "--metric-definition-id",
            metric_definition_id,
        ],
        root=root,
        log_path=logs_root / "metric_definition.log",
    )
    _run(
        command=[
            sys.executable,
            "-m",
            "scripts.build_comparison_contexts",
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            "data_sers",
            "--corpus-id",
            corpus_id,
            "--mode",
            "exploratory",
            "--comparison-id",
            comparison_id,
            "--metric-definition-id",
            metric_definition_id,
        ],
        root=root,
        log_path=logs_root / "comparison.log",
    )

    replay_root = data_root / "corpus" / corpus_id / "exploratory"
    new = {
        "corpus": _read_json(replay_root / "manifest.json"),
        "reproducibility": _read_json(
            replay_root
            / "reproducibility"
            / reproducibility_id
            / "summary.json"
        ),
        "metric_definition": _read_json(
            replay_root
            / "metric_definition"
            / metric_definition_id
            / "summary.json"
        ),
        "comparison": _read_json(
            replay_root
            / "comparison"
            / comparison_id
            / "summary.json"
        ),
    }

    report = {
        "replay_id": args.replay_id,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "reason": (
            "Holdout exposed a generic cross-chunk Measurement merge bug "
            "that could combine numeric and textual payloads."
        ),
        "measurement_merge_invariant_id": (
            MEASUREMENT_MERGE_INVARIANT_ID
        ),
        "llm_calls_performed": False,
        "bridge_policy": (
            "reuse_existing_frozen_bridge_materialization"
        ),
        "calibration_papers": list(CALIBRATION_PAPERS),
        "paper_records": paper_records,
        "old_frozen_calibration": old,
        "new_replay": new,
        "new_ids": {
            "corpus": corpus_id,
            "reproducibility": reproducibility_id,
            "metric_definition": metric_definition_id,
            "comparison": comparison_id,
        },
        "next_step": (
            "Review the replay. Do not resume holdout until the calibration "
            "freeze in sers_alpha4b4_protocol.json is explicitly updated."
        ),
    }
    evaluation_root.mkdir(parents=True, exist_ok=True)
    report_path = evaluation_root / "calibration_replay_report.json"
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    comparison = new["comparison"]
    metric = new["metric_definition"]
    repro = new["reproducibility"]
    print()
    print("Calibration replay complete")
    print("Invariant:", MEASUREMENT_MERGE_INVARIANT_ID)
    print("Corpus:", corpus_id)
    print(
        "Comparison contexts/assessments:",
        comparison.get("context_count"),
        "/",
        comparison.get("assessment_count"),
    )
    print(
        "Numeric ranking allowed:",
        comparison.get("numeric_ranking_allowed_count"),
    )
    print(
        "Metric definitions known/unknown:",
        metric.get("definition_status_counts", {}),
    )
    print(
        "Reproducibility evidence:",
        repro.get("evidence_count"),
    )
    print("Report:", report_path)
    print(
        "Holdout remains PAUSED until this replay is reviewed and "
        "the frozen protocol is reissued."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
