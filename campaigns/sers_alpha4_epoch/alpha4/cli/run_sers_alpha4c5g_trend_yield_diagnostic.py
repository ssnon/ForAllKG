from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f2_reserve import (
    load_pool_and_split,
)
from campaigns.sers_alpha4_epoch.readiness.canonical_readiness import (
    canonical_graph_snapshot,
)
from dac_her.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
)
from campaigns.sers_alpha4_epoch.support.trend_yield_diagnostic import (
    diagnose_development_trend_yield,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


ROOT = Path.cwd()
DEFAULT_POOL = Path(
    "evaluation/sers_alpha4c5f2/pool_v1/pool_manifest.json"
)
DEFAULT_SPLIT = Path(
    "evaluation/sers_alpha4c5f2/pool_v1/blind_split.json"
)
DEFAULT_OUTPUT = Path(
    "evaluation/sers_alpha4c5g/dev_v1"
)


class DiagnosticError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "alpha4c.5g development-only Trend yield/recall diagnostic. "
            "Runs deterministic evidence→Trend/Precision on the 53-paper "
            "development partition and creates read-only diagnostic "
            "sidecars. Reserve A/B are never consumed or used."
        )
    )
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        default=DEFAULT_POOL,
    )
    parser.add_argument(
        "--blind-split",
        type=Path,
        default=DEFAULT_SPLIT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--confirm-development-only",
        action="store_true",
    )
    return parser.parse_args()


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def command(
    *,
    log_path: Path,
    stage: str,
    args: list[str],
) -> None:
    started = time.time()
    print(f"\n[alpha4c.5g] {stage}")
    print("[alpha4c.5g] command:", " ".join(args))
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(
            result.stderr,
            end="",
            file=sys.stderr,
        )
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with log_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "stage": stage,
                    "command": args,
                    "returncode": result.returncode,
                    "elapsed_seconds": (
                        time.time() - started
                    ),
                    "stdout_tail": result.stdout[-8000:],
                    "stderr_tail": result.stderr[-8000:],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    if result.returncode != 0:
        raise DiagnosticError(
            f"{stage} failed with exit code "
            f"{result.returncode}"
        )


def py(
    *,
    log_path: Path,
    stage: str,
    module: str,
    args: list[str],
) -> None:
    command(
        log_path=log_path,
        stage=stage,
        args=[
            sys.executable,
            "-m",
            module,
            *args,
        ],
    )


def main() -> int:
    args = parse_args()
    if not args.confirm_development_only:
        raise SystemExit(
            "--confirm-development-only is required."
        )

    pool_path = rooted(args.pool_manifest)
    split_path = rooted(args.blind_split)
    output_dir = rooted(args.output_dir)
    if output_dir.exists():
        raise SystemExit(
            "Refusing to mix/reuse an existing alpha4c.5g diagnostic "
            f"directory: {output_dir}"
        )

    pool, split = load_pool_and_split(
        root=ROOT,
        pool_path=pool_path,
        split_path=split_path,
        verify_source_manifest=True,
    )
    development = list(split["development"])
    reserve_a = set(split["reserve_a"])
    reserve_b = set(split["reserve_b"])
    if len(development) != 53:
        raise DiagnosticError(
            f"Expected 53 development papers, got {len(development)}."
        )
    if set(development) & reserve_a:
        raise DiagnosticError(
            "Development overlaps Reserve A."
        )
    if set(development) & reserve_b:
        raise DiagnosticError(
            "Development overlaps Reserve B."
        )
    if (
        split.get(
            "reserve_b_sealed_for_future_confirmation"
        )
        is not True
    ):
        raise DiagnosticError(
            "Reserve B is not sealed in the blind split."
        )

    output_dir.mkdir(parents=True)
    work_data = output_dir / "work_data_sers"
    log_path = output_dir / "command_log.jsonl"
    canonical_lock_rows: list[
        dict[str, Any]
    ] = []

    # Readiness is asserted but never repaired: this is a read-only
    # scientific diagnostic with respect to source canonical graphs.
    for paper_id in development:
        source = (
            ROOT
            / "data_sers"
            / "extracted"
            / paper_id
            / f"{paper_id}.graphml"
        )
        if not source.exists():
            raise DiagnosticError(
                f"Canonical source missing: {source}"
            )
        snapshot = canonical_graph_snapshot(
            source,
            expected_domain_profile_id="sers_au_ag",
            expected_measurement_merge_invariant_id=(
                MEASUREMENT_MERGE_INVARIANT_ID
            ),
            include_issue_details=False,
        )
        if snapshot.get("ready") is not True:
            raise DiagnosticError(
                f"{paper_id}: development canonical is not ready: "
                f"{snapshot.get('readiness_issues')!r}. "
                "alpha4c.5g never refreezes source canonical graphs."
            )
        dest = (
            work_data
            / "extracted"
            / paper_id
            / f"{paper_id}.graphml"
        )
        dest.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(source, dest)
        if sha256_file(source) != sha256_file(dest):
            raise DiagnosticError(
                f"Canonical copy SHA mismatch: {paper_id}"
            )
        canonical_lock_rows.append(
            {
                "paper_id": paper_id,
                "source_path": str(source),
                "source_sha256": sha256_file(
                    source
                ),
                "diagnostic_copy": str(dest),
                "diagnostic_copy_sha256": (
                    sha256_file(dest)
                ),
                "readiness_issues": [],
            }
        )

    write_json(
        output_dir / "development_canonical_lock.json",
        {
            "diagnostic": "alpha4c.5g",
            "development_only": True,
            "paper_count": len(development),
            "paper_ids": development,
            "source_canonicals_modified": False,
            "canonical_rows": canonical_lock_rows,
        },
    )

    corpus_id = "sers_alpha4c5g_dev_v1_corpus"
    identity_id = "sers_alpha4c5g_dev_v1_measurement_identity"
    metric_id = "sers_alpha4c5g_dev_v1_metric_definition"
    comparison_id = "sers_alpha4c5g_dev_v1_comparison"
    trend_id = "sers_alpha4c5g_dev_v1_trend"
    precision_id = "sers_alpha4c5g_dev_v1_precision"

    for paper_id in development:
        py(
            log_path=log_path,
            stage=f"projection:{paper_id}",
            module="scripts.build_graphagents_projection",
            args=[
                "--paper-id",
                paper_id,
                "--domain-profile",
                "sers_au_ag",
                "--data-root",
                str(work_data),
                "--mode",
                "evidence",
            ],
        )

    py(
        log_path=log_path,
        stage="corpus",
        module="scripts.build_corpus_graph",
        args=[
            "--corpus-id",
            corpus_id,
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(work_data),
            "--paper-ids",
            *development,
            "--mode",
            "evidence",
            "--allow-critical-partial",
        ],
    )
    py(
        log_path=log_path,
        stage="measurement_result_identity",
        module="scripts.build_measurement_result_identities",
        args=[
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(work_data),
            "--corpus-id",
            corpus_id,
            "--mode",
            "evidence",
            "--measurement-result-identity-id",
            identity_id,
        ],
    )
    py(
        log_path=log_path,
        stage="metric_definition",
        module="scripts.build_metric_definition_contexts",
        args=[
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(work_data),
            "--corpus-id",
            corpus_id,
            "--mode",
            "evidence",
            "--metric-definition-id",
            metric_id,
            "--measurement-result-identity-id",
            identity_id,
        ],
    )
    py(
        log_path=log_path,
        stage="comparison",
        module="scripts.build_comparison_contexts",
        args=[
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(work_data),
            "--corpus-id",
            corpus_id,
            "--mode",
            "evidence",
            "--comparison-id",
            comparison_id,
            "--metric-definition-id",
            metric_id,
            "--measurement-result-identity-id",
            identity_id,
        ],
    )
    py(
        log_path=log_path,
        stage="trend",
        module="scripts.build_trend_evidence",
        args=[
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(work_data),
            "--corpus-id",
            corpus_id,
            "--mode",
            "evidence",
            "--trend-id",
            trend_id,
            "--measurement-result-identity-id",
            identity_id,
            "--comparison-id",
            comparison_id,
        ],
    )
    py(
        log_path=log_path,
        stage="trend_precision",
        module="scripts.build_trend_precision",
        args=[
            "--domain-profile",
            "sers_au_ag",
            "--data-root",
            str(work_data),
            "--corpus-id",
            corpus_id,
            "--mode",
            "evidence",
            "--trend-id",
            trend_id,
            "--precision-id",
            precision_id,
        ],
    )

    corpus_root = (
        work_data
        / "corpus"
        / corpus_id
        / "evidence"
    )
    identity_root = (
        corpus_root
        / "measurement_result_identity"
        / identity_id
    )
    comparison_root = (
        corpus_root
        / "comparison"
        / comparison_id
    )
    trend_root = (
        corpus_root
        / "trend"
        / trend_id
    )
    precision_root = (
        trend_root
        / "precision"
        / precision_id
    )

    identity_rows = read_jsonl(
        identity_root / "identities.jsonl"
    )
    if not identity_rows:
        # historical filename compatibility
        identity_rows = read_jsonl(
            identity_root / "results.jsonl"
        )
    method_rows = read_jsonl(
        comparison_root / "method_contexts.jsonl"
    )
    comparison_rows = read_jsonl(
        comparison_root / "contexts.jsonl"
    )
    trend_rows = read_jsonl(
        trend_root / "evidence.jsonl"
    )
    local_rows = read_jsonl(
        precision_root / "local_results.jsonl"
    )

    implementation_paths = [
        "dac_her/domains/sers_au_ag_trend.py",
        "scripts/build_trend_evidence.py",
        "scripts/build_trend_precision.py",
        "scripts/build_comparison_contexts.py",
        "scripts/build_metric_definition_contexts.py",
        "dac_her/measurement_result_identity.py",
    ]
    implementation_hashes = {
        rel: sha256_file(ROOT / rel)
        for rel in implementation_paths
        if (ROOT / rel).exists()
    }

    result = diagnose_development_trend_yield(
        paper_ids=development,
        canonical_paths={
            row["paper_id"]: Path(
                row["diagnostic_copy"]
            )
            for row in canonical_lock_rows
        },
        identity_rows=identity_rows,
        method_rows=method_rows,
        comparison_rows=comparison_rows,
        trend_rows=trend_rows,
        local_result_rows=local_rows,
        implementation_hashes=implementation_hashes,
    )

    diagnostic_root = (
        output_dir / "trend_yield_diagnostic"
    )
    write_json(
        diagnostic_root / "summary.json",
        result["summary"],
    )
    write_jsonl(
        diagnostic_root / "paper_classification.jsonl",
        result["papers"],
    )
    write_jsonl(
        diagnostic_root / "claim_candidates.jsonl",
        result["claim_candidates"],
    )
    write_jsonl(
        diagnostic_root / "numeric_candidates.jsonl",
        result["numeric_candidates"],
    )

    summary = result["summary"]
    print("\nalpha4c.5g Trend Yield / Recall Diagnostic: COMPLETE")
    print("Development papers:", summary["paper_count"])
    print(
        "Raw TrendEvidence:",
        summary["raw_trend_evidence_count"],
    )
    print(
        "Precision local results:",
        summary["precision_local_result_count"],
    )
    print(
        "Papers with Trend:",
        summary["papers_with_precision_trend"],
    )
    print(
        "Zero-yield papers:",
        summary["zero_yield_paper_count"],
    )
    print(
        "Primary classes:",
        summary["primary_class_counts"],
    )
    flags = summary["zero_yield_diagnostic_flags"]
    print(
        "A no-broad-candidate:",
        flags["A_no_broad_candidate_count"],
        f"({flags['A_no_broad_candidate_ratio']:.3f})",
    )
    print(
        "B claim-miss flag:",
        flags["B_claim_miss_flag_count"],
        f"({flags['B_claim_miss_flag_ratio']:.3f})",
    )
    print(
        "C numeric-block flag:",
        flags["C_numeric_block_flag_count"],
        f"({flags['C_numeric_block_flag_ratio']:.3f})",
    )
    print(
        "B/C overlap:",
        flags["BC_overlap_count"],
        f"({flags['BC_overlap_ratio']:.3f})",
    )
    print(
        "Claim candidates:",
        summary["claim_candidate_counts"],
    )
    print(
        "Numeric candidates:",
        summary["numeric_candidate_counts"],
    )
    print("Scientific semantics modified:", False)
    print("Acceptance semantics modified:", False)
    print("Count thresholds used:", False)
    print("LLM calls:", 0)
    print("Reserve A used:", False)
    print("Reserve B used:", False)
    print("Reserve B remains sealed:", True)
    print(
        "Summary:",
        diagnostic_root / "summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
