from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import networkx as nx

from dac_her.trend_rejection_attribution import (
    ALPHA4C5G1_ATTRIBUTION_ID,
    EXPECTED_5G_DIAGNOSTIC_ID,
    EXPECTED_5G_SUMMARY_SEMANTIC_SHA256,
    EXPECTED_TREND_IMPLEMENTATION_SHA256,
    attribute_claim_miss,
    attribute_numeric_miss,
    build_stratified_sample,
    read_json,
    read_jsonl,
    require_trend_helper_contract,
    semantic_sha256,
    sha256_file,
    summarize_attribution,
    write_json,
    write_jsonl,
)


ROOT = Path.cwd()
DEFAULT_5G_ROOT = Path(
    "evaluation/sers_alpha4c5g/dev_v1"
)
DEFAULT_OUTPUT = Path(
    "evaluation/sers_alpha4c5g1/dev_v1"
)


class AttributionError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "alpha4c.5g.1 development-only Trend rejection "
            "attribution. Reads the completed 5g diagnostic and "
            "the frozen development artifacts; calls zero LLMs and "
            "does not modify Trend semantics."
        )
    )
    parser.add_argument(
        "--source-5g-root",
        type=Path,
        default=DEFAULT_5G_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--sample-per-reason",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--confirm-development-only",
        action="store_true",
    )
    return parser.parse_args()


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _rows_by_paper(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        paper_id = str(
            row.get("paper_id", "")
        ).strip()
        if paper_id:
            result[paper_id].append(row)
    return dict(result)


def _load_identity_rows(
    corpus_root: Path,
    identity_id: str,
) -> list[dict[str, Any]]:
    root = (
        corpus_root
        / "measurement_result_identity"
        / identity_id
    )
    for name in (
        "identities.jsonl",
        "results.jsonl",
    ):
        path = root / name
        if path.exists():
            return read_jsonl(path)
    raise AttributionError(
        f"Measurement identity rows not found under {root}"
    )


def main() -> int:
    args = parse_args()
    if not args.confirm_development_only:
        raise SystemExit(
            "--confirm-development-only is required."
        )
    if args.sample_per_reason <= 0:
        raise SystemExit(
            "--sample-per-reason must be positive."
        )

    source_root = rooted(args.source_5g_root)
    output_dir = rooted(args.output_dir)
    if output_dir.exists():
        raise SystemExit(
            "Refusing to reuse/mix an existing 5g.1 attribution "
            f"directory: {output_dir}"
        )

    diagnostic_root = (
        source_root / "trend_yield_diagnostic"
    )
    summary_path = diagnostic_root / "summary.json"
    claim_path = diagnostic_root / "claim_candidates.jsonl"
    numeric_path = diagnostic_root / "numeric_candidates.jsonl"

    base_summary = read_json(summary_path)
    if (
        base_summary.get("diagnostic_id")
        != EXPECTED_5G_DIAGNOSTIC_ID
    ):
        raise AttributionError(
            "Unexpected 5g diagnostic ID."
        )
    observed_summary_sha = semantic_sha256(
        base_summary
    )
    if (
        observed_summary_sha
        != EXPECTED_5G_SUMMARY_SEMANTIC_SHA256
    ):
        raise AttributionError(
            "5g summary semantic SHA drifted: "
            f"{observed_summary_sha} != "
            f"{EXPECTED_5G_SUMMARY_SEMANTIC_SHA256}"
        )
    if base_summary.get("development_only") is not True:
        raise AttributionError(
            "Source 5g report is not development-only."
        )
    if base_summary.get("reserve_a_used") is not False:
        raise AttributionError(
            "Source 5g report used Reserve A."
        )
    if base_summary.get("reserve_b_used") is not False:
        raise AttributionError(
            "Source 5g report used Reserve B."
        )
    if (
        base_summary.get(
            "reserve_b_remains_sealed"
        )
        is not True
    ):
        raise AttributionError(
            "Reserve B is not sealed."
        )
    if int(base_summary.get("paper_count", -1)) != 53:
        raise AttributionError(
            "Source 5g report must contain 53 development papers."
        )

    # The 5g report itself freezes the exact local implementation
    # that generated the diagnostic. Verify those bytes before any
    # attribution so a GitHub/local drift cannot silently change
    # reason assignment.
    implementation_hashes = dict(
        base_summary.get(
            "implementation_sha256",
            {},
        )
    )
    for rel, expected in sorted(
        implementation_hashes.items()
    ):
        path = ROOT / rel
        if not path.exists():
            raise AttributionError(
                f"Frozen 5g implementation missing: {rel}"
            )
        observed = sha256_file(path)
        if observed != expected:
            raise AttributionError(
                f"Frozen 5g implementation drift: {rel}: "
                f"{observed} != {expected}"
            )

    trend_rel = "dac_her/domains/sers_au_ag_trend.py"
    trend_sha = implementation_hashes.get(
        trend_rel,
        "",
    )
    if trend_sha != EXPECTED_TREND_IMPLEMENTATION_SHA256:
        raise AttributionError(
            "Unexpected frozen local Trend implementation SHA."
        )

    trend_module = importlib.import_module(
        "dac_her.domains.sers_au_ag_trend"
    )
    require_trend_helper_contract(trend_module)
    trend_semantics_id = str(
        getattr(
            trend_module,
            "SERS_AU_AG_TREND_SEMANTICS_ID",
            "",
        )
    )

    claim_candidates = read_jsonl(claim_path)
    numeric_candidates = read_jsonl(numeric_path)
    claim_misses = [
        row
        for row in claim_candidates
        if not bool(
            row.get("admitted_by_current_trend")
        )
    ]
    numeric_misses = [
        row
        for row in numeric_candidates
        if not bool(
            row.get("admitted_by_current_trend")
        )
    ]

    expected_claim = dict(
        base_summary["claim_candidate_counts"]
    )
    expected_numeric = dict(
        base_summary["numeric_candidate_counts"]
    )
    if len(claim_candidates) != int(
        expected_claim["broad_candidates"]
    ):
        raise AttributionError(
            "Claim candidate count drifted."
        )
    if len(claim_misses) != int(
        expected_claim["candidate_misses"]
    ):
        raise AttributionError(
            "Claim miss count drifted."
        )
    if len(numeric_candidates) != int(
        expected_numeric["broad_series_candidates"]
    ):
        raise AttributionError(
            "Numeric candidate count drifted."
        )
    if len(numeric_misses) != int(
        expected_numeric["candidate_misses"]
    ):
        raise AttributionError(
            "Numeric miss count drifted."
        )

    work_data = source_root / "work_data_sers"
    corpus_id = "sers_alpha4c5g_dev_v1_corpus"
    identity_id = "sers_alpha4c5g_dev_v1_measurement_identity"
    comparison_id = "sers_alpha4c5g_dev_v1_comparison"

    corpus_root = (
        work_data
        / "corpus"
        / corpus_id
        / "evidence"
    )
    identity_rows = _load_identity_rows(
        corpus_root,
        identity_id,
    )
    comparison_root = (
        corpus_root
        / "comparison"
        / comparison_id
    )
    method_rows = read_jsonl(
        comparison_root / "method_contexts.jsonl"
    )
    comparison_rows = read_jsonl(
        comparison_root / "contexts.jsonl"
    )

    identity_by_paper = _rows_by_paper(
        identity_rows
    )
    methods_by_paper = _rows_by_paper(
        method_rows
    )
    comparison_by_paper = _rows_by_paper(
        comparison_rows
    )

    paper_ids = list(base_summary["paper_ids"])
    graph_by_paper: dict[str, nx.Graph] = {}
    for paper_id in paper_ids:
        graph_path = (
            work_data
            / "extracted"
            / paper_id
            / f"{paper_id}.graphml"
        )
        if not graph_path.exists():
            raise AttributionError(
                f"Frozen development graph missing: {graph_path}"
            )
        graph_by_paper[paper_id] = nx.read_graphml(
            graph_path,
            force_multigraph=True,
        )

    claim_attribution = [
        attribute_claim_miss(
            candidate=row,
            graph=graph_by_paper[
                str(row["paper_id"])
            ],
            trend_module=trend_module,
        )
        for row in claim_misses
    ]

    numeric_attribution = [
        attribute_numeric_miss(
            candidate=row,
            graph=graph_by_paper[
                str(row["paper_id"])
            ],
            trend_module=trend_module,
            identity_rows=identity_by_paper.get(
                str(row["paper_id"]),
                [],
            ),
            method_rows=methods_by_paper.get(
                str(row["paper_id"]),
                [],
            ),
            comparison_rows=comparison_by_paper.get(
                str(row["paper_id"]),
                [],
            ),
        )
        for row in numeric_misses
    ]

    sample_rows = build_stratified_sample(
        claim_rows=claim_attribution,
        numeric_rows=numeric_attribution,
        per_bucket=args.sample_per_reason,
    )
    summary = summarize_attribution(
        base_summary=base_summary,
        claim_rows=claim_attribution,
        numeric_rows=numeric_attribution,
        sample_rows=sample_rows,
        trend_semantics_id=trend_semantics_id,
        trend_implementation_sha256=trend_sha,
    )

    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "summary.json",
        summary,
    )
    write_jsonl(
        output_dir / "claim_attribution.jsonl",
        claim_attribution,
    )
    write_jsonl(
        output_dir / "numeric_attribution.jsonl",
        numeric_attribution,
    )
    write_jsonl(
        output_dir / "adjudication_sample.jsonl",
        sample_rows,
    )
    write_json(
        output_dir / "source_binding.json",
        {
            "attribution_id": ALPHA4C5G1_ATTRIBUTION_ID,
            "source_5g_summary": str(summary_path),
            "source_5g_summary_semantic_sha256": (
                observed_summary_sha
            ),
            "source_claim_candidates_sha256": (
                sha256_file(claim_path)
            ),
            "source_numeric_candidates_sha256": (
                sha256_file(numeric_path)
            ),
            "trend_implementation_sha256": trend_sha,
            "trend_semantics_id": trend_semantics_id,
            "development_paper_ids": paper_ids,
            "reserve_a_used": False,
            "reserve_b_used": False,
            "reserve_b_remains_sealed": True,
            "llm_calls": 0,
        },
    )

    print(
        "alpha4c.5g.1 Trend Candidate Rejection Attribution: COMPLETE"
    )
    print("Development papers:", summary["paper_count"])
    print("Claim misses:", summary["claim_miss_count"])
    print(
        "Claim reasons:",
        summary["claim_reason_counts"],
    )
    print("Numeric misses:", summary["numeric_miss_count"])
    print(
        "Numeric reasons:",
        summary["numeric_reason_counts"],
    )
    print("Adjudication sample:", summary["sample_count"])
    print("Trend semantics:", trend_semantics_id)
    print("Trend implementation SHA256:", trend_sha)
    print("Scientific semantics modified:", False)
    print("Acceptance semantics modified:", False)
    print("LLM calls:", 0)
    print("Reserve A used:", False)
    print("Reserve B used:", False)
    print("Reserve B remains sealed:", True)
    print("Summary:", output_dir / "summary.json")
    print(
        "Sample:",
        output_dir / "adjudication_sample.jsonl",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
