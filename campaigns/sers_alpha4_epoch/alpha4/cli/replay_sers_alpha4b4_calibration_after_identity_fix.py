from __future__ import annotations

import argparse
import hashlib
import json
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
from dac_her.measurement_result_identity import (
    MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
)


from campaigns.sers_alpha4_epoch.paths import PROJECT_ROOT
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
            f"Calibration identity replay failed ({code}): {command!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay SERS 1/5/8 calibration after alpha4b.4.1.1 "
            "Measurement Result Identity precision. Existing frozen Bridge "
            "graphs are reused; no LLM calls are made."
        )
    )
    parser.add_argument(
        "--replay-id",
        default="sers_alpha4b4a1_identity_calibration_replay_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT.resolve()
    data_root = root / "data_sers"
    evaluation_root = (
        root
        / "evaluation"
        / "sers_alpha4b4"
        / "measurement_result_identity_v1"
        / args.replay_id
    )
    logs_root = evaluation_root / "logs"

    canonical_records: dict[str, Any] = {}
    for paper_id in CALIBRATION_PAPERS:
        paper_root = data_root / "extracted" / paper_id
        canonical = paper_root / f"{paper_id}.graphml"
        bridge = paper_root / f"{paper_id}.bridge.graphml"
        candidate_bridge = (
            paper_root / f"{paper_id}.bridge.candidates.graphml"
        )
        for path in (canonical, bridge, candidate_bridge):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required frozen calibration artifact missing: {path}"
                )

        graph = nx.read_graphml(
            canonical,
            force_multigraph=True,
        )
        observed_invariant = str(
            graph.graph.get("measurement_merge_invariant_id", "")
        )
        if observed_invariant != MEASUREMENT_MERGE_INVARIANT_ID:
            raise RuntimeError(
                f"{paper_id} is not rebuilt under alpha4b.4.1: "
                f"{observed_invariant!r}."
            )
        xor_issues = measurement_value_payload_issues(graph)
        if xor_issues:
            raise RuntimeError(
                f"{paper_id} still violates Measurement XOR: "
                f"{xor_issues[:5]!r}"
            )

        canonical_records[paper_id] = {
            "canonical_graph_sha256": _sha256(canonical),
            "canonical_nodes": graph.number_of_nodes(),
            "canonical_edges": graph.number_of_edges(),
            "bridge_sha256": _sha256(bridge),
            "candidate_bridge_sha256": _sha256(candidate_bridge),
            "measurement_merge_invariant_id": observed_invariant,
            "measurement_xor_issue_count": 0,
        }

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
                    logs_root
                    / f"{paper_id}_projection_{mode}.log"
                ),
            )

    corpus_id = f"{args.replay_id}_corpus"
    identity_id = f"{args.replay_id}_measurement_identity"
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
            "scripts.build_measurement_result_identities",
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            "data_sers",
            "--corpus-id",
            corpus_id,
            "--mode",
            "exploratory",
            "--measurement-result-identity-id",
            identity_id,
        ],
        root=root,
        log_path=logs_root / "measurement_identity.log",
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
            "--measurement-result-identity-id",
            identity_id,
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
            "--measurement-result-identity-id",
            identity_id,
        ],
        root=root,
        log_path=logs_root / "comparison.log",
    )

    replay_root = data_root / "corpus" / corpus_id / "exploratory"
    identity_summary = _read_json(
        replay_root
        / "measurement_result_identity"
        / identity_id
        / "summary.json"
    )
    reproducibility_summary = _read_json(
        replay_root
        / "reproducibility"
        / reproducibility_id
        / "summary.json"
    )
    metric_summary = _read_json(
        replay_root
        / "metric_definition"
        / metric_definition_id
        / "summary.json"
    )
    comparison_summary = _read_json(
        replay_root
        / "comparison"
        / comparison_id
        / "summary.json"
    )

    old_frozen_root = (
        data_root
        / "corpus"
        / "sers_alpha4b3a_calibration"
        / "exploratory"
    )
    old = {
        "reproducibility": _read_json(
            old_frozen_root
            / "reproducibility"
            / "sers_alpha4b3b4a1_calibration"
            / "summary.json"
        ),
        "metric_definition": _read_json(
            old_frozen_root
            / "metric_definition"
            / "sers_alpha4b3b4b1_calibration"
            / "summary.json"
        ),
        "comparison": _read_json(
            old_frozen_root
            / "comparison"
            / "sers_alpha4b3b4c1_calibration"
            / "summary.json"
        ),
    }

    report = {
        "replay_id": args.replay_id,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "llm_calls_performed": False,
        "bridge_policy": "reuse_existing_frozen_bridge_materialization",
        "measurement_merge_invariant_id": (
            MEASUREMENT_MERGE_INVARIANT_ID
        ),
        "measurement_result_identity_semantics_id": (
            MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID
        ),
        "calibration_papers": list(CALIBRATION_PAPERS),
        "canonical_records": canonical_records,
        "new_ids": {
            "corpus": corpus_id,
            "measurement_result_identity": identity_id,
            "reproducibility": reproducibility_id,
            "metric_definition": metric_definition_id,
            "comparison": comparison_id,
        },
        "new_replay": {
            "measurement_result_identity": identity_summary,
            "reproducibility": reproducibility_summary,
            "metric_definition": metric_summary,
            "comparison": comparison_summary,
        },
        "old_frozen_calibration": old,
        "next_step": (
            "Review exact identity consolidation and downstream invariants. "
            "Holdout remains paused until a new frozen protocol is issued."
        ),
    }
    evaluation_root.mkdir(parents=True, exist_ok=True)
    report_path = evaluation_root / "calibration_identity_replay_report.json"
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

    print()
    print("Calibration measurement-result identity replay complete")
    print(
        "Identity semantics:",
        MEASUREMENT_RESULT_IDENTITY_SEMANTICS_ID,
    )
    print(
        "Source mentions / scientific results:",
        identity_summary.get("source_mention_count"),
        "/",
        identity_summary.get("scientific_result_count"),
    )
    print(
        "Consolidated exact results:",
        identity_summary.get("consolidated_exact_result_count"),
    )
    print(
        "Unresolved same-lineage groups:",
        identity_summary.get("unresolved_same_lineage_group_count"),
    )
    print(
        "Metric-definition contexts:",
        metric_summary.get("context_count"),
    )
    print(
        "Comparison contexts/assessments:",
        comparison_summary.get("context_count"),
        "/",
        comparison_summary.get("assessment_count"),
    )
    print(
        "Numeric ranking allowed:",
        comparison_summary.get("numeric_ranking_allowed_count"),
    )
    print(
        "Structural gates:",
        {
            "identity": identity_summary.get("structural_gate"),
            "reproducibility": reproducibility_summary.get(
                "structural_gate"
            ),
            "metric_definition": metric_summary.get("structural_gate"),
            "comparison": comparison_summary.get(
                "passes_structural_gate"
            ),
        },
    )
    print("Report:", report_path)
    print(
        "Holdout remains PAUSED until this replay is reviewed and "
        "the frozen protocol is reissued."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
