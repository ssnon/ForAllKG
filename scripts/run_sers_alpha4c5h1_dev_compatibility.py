from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
SOURCE_5G = Path("evaluation/sers_alpha4c5g/dev_v1")
CORPUS_ID = "sers_alpha4c5g_dev_v1_corpus"
IDENTITY_ID = "sers_alpha4c5g_dev_v1_measurement_identity"
COMPARISON_ID = "sers_alpha4c5g_dev_v1_comparison"

TREND_ID = "sers_alpha4c5h1_dev_compat_v1_trend"
PRECISION_ID = "sers_alpha4c5h1_dev_compat_v1_precision"
CONTEXT_ID = "sers_alpha4c5h1_dev_compat_v1_context"
ASSESSMENT_ID = "sers_alpha4c5h1_dev_compat_v1_assessment"

EXPECTED_TREND_SEMANTICS = (
    "sers_au_ag_trend_v6r2_alpha4c5g2r2"
)
EXPECTED_PRECISION_SEMANTICS = (
    "sers_au_ag_trend_precision_v5_alpha4c21211"
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def run(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("[5h.1 DEV compat]", " ".join(command))
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"{module} failed with exit code {result.returncode}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Development-only deterministic downstream compatibility "
            "smoke for frozen v6r2 Trend -> v5 Precision behavior -> "
            "CrossContext -> alpha4c.5a grounding. Zero LLM calls."
        )
    )
    parser.add_argument(
        "--source-5g-root",
        type=Path,
        default=SOURCE_5G,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "evaluation/sers_alpha4c5h1/dev_compat_v1"
        ),
    )
    parser.add_argument(
        "--confirm-development-only",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_development_only:
        raise SystemExit("--confirm-development-only is required.")

    source = (
        args.source_5g_root
        if args.source_5g_root.is_absolute()
        else ROOT / args.source_5g_root
    )
    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )
    if output.exists():
        raise SystemExit(
            f"Refusing existing DEV compatibility output: {output}"
        )

    data_root = source / "work_data_sers"
    corpus_root = (
        data_root / "corpus" / CORPUS_ID / "evidence"
    )
    if not corpus_root.exists():
        raise RuntimeError(
            f"5g Development corpus missing: {corpus_root}"
        )

    trend_root = corpus_root / "trend" / TREND_ID
    if trend_root.exists():
        raise RuntimeError(
            f"DEV compatibility Trend artifact already exists: {trend_root}"
        )

    run(
        "scripts.build_trend_evidence_alpha4c5h1",
        "--domain-profile", "sers_au_ag",
        "--data-root", str(data_root),
        "--corpus-id", CORPUS_ID,
        "--mode", "evidence",
        "--trend-id", TREND_ID,
        "--measurement-result-identity-id", IDENTITY_ID,
        "--comparison-id", COMPARISON_ID,
    )
    run(
        "scripts.build_trend_precision_alpha4c5h1",
        "--domain-profile", "sers_au_ag",
        "--data-root", str(data_root),
        "--corpus-id", CORPUS_ID,
        "--mode", "evidence",
        "--trend-id", TREND_ID,
        "--precision-id", PRECISION_ID,
    )

    precision_root = (
        trend_root / "precision" / PRECISION_ID
    )
    local_results = precision_root / "local_results.jsonl"
    local_count = sum(
        bool(line.strip())
        for line in local_results.read_text(
            encoding="utf-8"
        ).splitlines()
    )

    context_summary = None
    assessment_summary = None
    context_root = (
        precision_root / "cross_context" / CONTEXT_ID
    )
    assessment_root = (
        context_root / "assessment" / ASSESSMENT_ID
    )
    if local_count > 0:
        run(
            "scripts.build_cross_context_profiles_alpha4c5h1",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(data_root),
            "--corpus-id", CORPUS_ID,
            "--mode", "evidence",
            "--trend-id", TREND_ID,
            "--precision-id", PRECISION_ID,
            "--context-id", CONTEXT_ID,
        )
        run(
            "scripts.build_cross_context_assessments_alpha4c5h1",
            "--domain-profile", "sers_au_ag",
            "--data-root", str(data_root),
            "--corpus-id", CORPUS_ID,
            "--mode", "evidence",
            "--trend-id", TREND_ID,
            "--precision-id", PRECISION_ID,
            "--context-id", CONTEXT_ID,
            "--assessment-id", ASSESSMENT_ID,
        )
        context_summary = load_json(context_root / "summary.json")
        assessment_summary = load_json(
            assessment_root / "summary.json"
        )

    output.mkdir(parents=True)
    grounding = output / "trend_hypothesis_grounding.json"
    grounding_args = [
        "--trend-dir", str(trend_root),
        "--precision-dir", str(precision_root),
        "--domain-profile", "sers_au_ag",
        "--output", str(grounding),
    ]
    if local_count > 0:
        grounding_args.extend(
            [
                "--context-dir", str(context_root),
                "--assessment-dir", str(assessment_root),
            ]
        )
    run(
        "scripts.build_hypothesis_trend_grounding",
        *grounding_args,
    )

    trend_summary = load_json(trend_root / "summary.json")
    precision_summary = load_json(
        precision_root / "summary.json"
    )
    grounding_summary = load_json(grounding)

    conditions = {
        "trend_semantics_v6r2": (
            trend_summary.get("trend_semantics_id")
            == EXPECTED_TREND_SEMANTICS
        ),
        "frozen_dev_evidence_count_15": (
            trend_summary.get("evidence_count") == 15
        ),
        "trend_structural_gate": (
            trend_summary.get("structural_gate") is True
        ),
        "precision_source_trend_v6r2": (
            precision_summary.get("trend_semantics_id")
            == EXPECTED_TREND_SEMANTICS
        ),
        "precision_semantics_unchanged": (
            precision_summary.get("precision_semantics_id")
            == EXPECTED_PRECISION_SEMANTICS
        ),
        "precision_structural_gate": (
            precision_summary.get("structural_gate") is True
        ),
        "context_structural_gate": (
            True
            if local_count == 0
            else context_summary.get("structural_gate") is True
        ),
        "assessment_structural_gate": (
            True
            if local_count == 0
            else assessment_summary.get("structural_gate") is True
        ),
        "grounding_source_trend_v6r2": (
            grounding_summary.get("source_trend_semantics_id")
            == EXPECTED_TREND_SEMANTICS
        ),
        "grounding_source_precision_unchanged": (
            grounding_summary.get("source_precision_semantics_id")
            == EXPECTED_PRECISION_SEMANTICS
        ),
    }

    summary = {
        "evaluation_id":
            "sers_alpha4c5h1_dev_downstream_compatibility_v1",
        "development_only": True,
        "scientific_semantics_modified": False,
        "precision_algorithm_modified": False,
        "precision_trend_parent_metadata_rebound": True,
        "trend_semantics_id": EXPECTED_TREND_SEMANTICS,
        "precision_semantics_id": EXPECTED_PRECISION_SEMANTICS,
        "trend_evidence_count": trend_summary.get("evidence_count"),
        "precision_local_result_count":
            precision_summary.get("local_result_count"),
        "cross_context_assessment_count": (
            0
            if assessment_summary is None
            else assessment_summary.get("assessment_count", 0)
        ),
        "grounding_relation_count":
            grounding_summary.get("relation_count"),
        "conditions": conditions,
        "passes_downstream_compatibility": all(
            conditions.values()
        ),
        "count_thresholds_used_for_acceptance": False,
        "reserve_a_used": False,
        "reserve_b_used": False,
        "llm_calls": 0,
        "artifacts": {
            "trend": str(trend_root),
            "precision": str(precision_root),
            "context": (
                "" if local_count == 0 else str(context_root)
            ),
            "assessment": (
                "" if local_count == 0 else str(assessment_root)
            ),
            "grounding": str(grounding),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "alpha4c.5h.1 DEV downstream compatibility:",
        "PASS"
        if summary["passes_downstream_compatibility"]
        else "FAIL",
    )
    print("Trend evidence:", summary["trend_evidence_count"])
    print(
        "Precision local results:",
        summary["precision_local_result_count"],
    )
    print(
        "CrossContext assessments:",
        summary["cross_context_assessment_count"],
    )
    print(
        "Grounding relations:",
        summary["grounding_relation_count"],
    )
    print("Scientific semantics modified:", False)
    print("Precision algorithm modified:", False)
    print("Reserve B used:", False)
    print("LLM calls:", 0)
    print("Saved:", output / "summary.json")
    return (
        0
        if summary["passes_downstream_compatibility"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
