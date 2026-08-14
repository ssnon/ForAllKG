from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.sers_alpha4c211_regression_checks import (
    calibration_checks,
    checks_gate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ID = 'sers_alpha4b4a11_method_reconstruction_calibration_replay_v1_corpus'
DEFAULT_IDENTITY_ID = 'sers_alpha4b4a11_method_reconstruction_calibration_replay_v1_measurement_identity'
DEFAULT_COMPARISON_ID = 'sers_alpha4b4a11_method_reconstruction_calibration_replay_v1_comparison'
DEFAULT_TREND_ID = 'sers_alpha4c211_calibration_v1'
DEFAULT_PRECISION_ID = 'sers_alpha4c211_calibration_precision_v1'
EXPECTED_PAPERS = ['Kiwook_SERS_1', 'Kiwook_SERS_5', 'Kiwook_SERS_8']
PHASE_LABEL = 'alpha4c.2.1.1 calibration'
REPORT_NAME = 'calibration_report.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=PHASE_LABEL)
    parser.add_argument("--corpus-id", default=DEFAULT_CORPUS_ID)
    parser.add_argument(
        "--measurement-result-identity-id",
        default=DEFAULT_IDENTITY_ID,
    )
    parser.add_argument("--comparison-id", default=DEFAULT_COMPARISON_ID)
    parser.add_argument("--trend-id", default=DEFAULT_TREND_ID)
    parser.add_argument("--precision-id", default=DEFAULT_PRECISION_ID)
    parser.add_argument("--data-root", default="data_sers")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row must be object: {path}")
                rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    corpus_root = (
        PROJECT_ROOT
        / args.data_root
        / "corpus"
        / args.corpus_id
        / "exploratory"
    )
    manifest_path = corpus_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_papers = [
        str(value) for value in manifest.get("paper_ids", [])
    ]
    if observed_papers != EXPECTED_PAPERS:
        raise ValueError(
            f"Unexpected paper set/order: {observed_papers!r} "
            f"!= {EXPECTED_PAPERS!r}."
        )

    _run([
        sys.executable,
        "-m",
        "scripts.build_trend_evidence",
        "--domain-profile",
        "sers_au_ag",
        "--data-root",
        args.data_root,
        "--corpus-id",
        args.corpus_id,
        "--mode",
        "exploratory",
        "--trend-id",
        args.trend_id,
        "--measurement-result-identity-id",
        args.measurement_result_identity_id,
        "--comparison-id",
        args.comparison_id,
    ])
    _run([
        sys.executable,
        "-m",
        "scripts.build_trend_precision",
        "--domain-profile",
        "sers_au_ag",
        "--data-root",
        args.data_root,
        "--corpus-id",
        args.corpus_id,
        "--mode",
        "exploratory",
        "--trend-id",
        args.trend_id,
        "--precision-id",
        args.precision_id,
    ])

    trend_root = corpus_root / "trend" / args.trend_id
    precision_root = trend_root / "precision" / args.precision_id
    trend_summary = json.loads(
        (trend_root / "summary.json").read_text(encoding="utf-8")
    )
    precision_summary = json.loads(
        (precision_root / "summary.json").read_text(encoding="utf-8")
    )
    precision_audit = json.loads(
        (precision_root / "audit.json").read_text(encoding="utf-8")
    )
    evidence_rows = _read_jsonl(trend_root / "evidence.jsonl")
    annotation_rows = _read_jsonl(precision_root / "annotations.jsonl")
    local_results = _read_jsonl(precision_root / "local_results.jsonl")

    regression_checks = calibration_checks(
        evidence_rows,
        annotation_rows,
        local_results,
    )
    semantic_regression_gate = checks_gate(regression_checks)
    structural_gate = bool(
        trend_summary.get("structural_gate", False)
        and precision_summary.get("structural_gate", False)
        and precision_audit.get("structural_gate", False)
    )
    report = {
        "phase": PHASE_LABEL,
        "paper_ids": EXPECTED_PAPERS,
        "llm_calls_performed": False,
        "trend_id": args.trend_id,
        "trend_semantics_id": trend_summary.get("trend_semantics_id"),
        "precision_id": args.precision_id,
        "precision_semantics_id":
            precision_summary.get("precision_semantics_id"),
        "evidence_count": precision_summary.get("evidence_count"),
        "local_result_count":
            precision_summary.get("local_result_count"),
        "evidence_kind_counts":
            precision_summary.get("evidence_kind_counts", {}),
        "control_key_counts":
            precision_summary.get("control_key_counts", {}),
        "observable_key_counts":
            precision_summary.get("observable_key_counts", {}),
        "observable_semantics_counts":
            precision_summary.get("observable_semantics_counts", {}),
        "duplicate_claim_mentions_collapsed":
            precision_summary.get(
                "duplicate_claim_mentions_collapsed", 0
            ),
        "structural_gate": structural_gate,
        "regression_checks": regression_checks,
        "semantic_regression_gate": semantic_regression_gate,
        "issues": precision_audit.get("issues", []),
        "acceptance_note": (
            "No evidence-count target is encoded. The semantic regression "
            "gate checks only previously observed, explicitly named "
            "development/seen-regression claims and invariants."
        ),
    }
    report_path = precision_root / REPORT_NAME
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
    print(PHASE_LABEL)
    print(
        "Evidence / local results:",
        report["evidence_count"],
        "/",
        report["local_result_count"],
    )
    print(
        "Evidence kinds:",
        json.dumps(
            report["evidence_kind_counts"],
            sort_keys=True,
        ),
    )
    print(
        "Controls:",
        json.dumps(
            report["control_key_counts"],
            sort_keys=True,
        ),
    )
    print(
        "Observable semantics:",
        json.dumps(
            report["observable_semantics_counts"],
            sort_keys=True,
        ),
    )
    print(
        "Collapsed duplicate claim mentions:",
        report["duplicate_claim_mentions_collapsed"],
    )
    for name, row in sorted(regression_checks.items()):
        print(
            "Regression",
            name + ":",
            "PASS" if row.get("pass") else "FAIL",
            "-",
            row.get("detail"),
        )
    print("Structural gate:", structural_gate)
    print("Semantic regression gate:", semantic_regression_gate)
    print("Report:", report_path)
    print("Evidence:", trend_root / "evidence.jsonl")
    print("Annotations:", precision_root / "annotations.jsonl")
    print("Local results:", precision_root / "local_results.jsonl")
    return 0 if structural_gate and semantic_regression_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
