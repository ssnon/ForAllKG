from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ID = "sers_alpha4b4a11_method_reconstruction_calibration_replay_v1_corpus"
DEFAULT_IDENTITY_ID = "sers_alpha4b4a11_method_reconstruction_calibration_replay_v1_measurement_identity"
DEFAULT_COMPARISON_ID = "sers_alpha4b4a11_method_reconstruction_calibration_replay_v1_comparison"
DEFAULT_TREND_ID = "sers_alpha4c2_calibration_v1"
CALIBRATION_PAPERS = ["Kiwook_SERS_1", "Kiwook_SERS_5", "Kiwook_SERS_8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run alpha4c.2 SERS TrendEvidence calibration on the frozen "
            "SERS_1/5/8 evidence substrate. No LLM calls are performed."
        )
    )
    parser.add_argument("--corpus-id", default=DEFAULT_CORPUS_ID)
    parser.add_argument("--measurement-result-identity-id", default=DEFAULT_IDENTITY_ID)
    parser.add_argument("--comparison-id", default=DEFAULT_COMPARISON_ID)
    parser.add_argument("--trend-id", default=DEFAULT_TREND_ID)
    parser.add_argument("--data-root", default="data_sers")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_root = (
        PROJECT_ROOT / args.data_root / "corpus" / args.corpus_id / "exploratory"
    )
    manifest_path = corpus_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_papers = [str(value) for value in manifest.get("paper_ids", [])]
    if observed_papers != CALIBRATION_PAPERS:
        raise ValueError(
            "alpha4c.2 calibration must use exactly frozen SERS_1/5/8 in "
            f"order: {observed_papers!r}."
        )

    command = [
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
    ]
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        return completed.returncode

    output_root = corpus_root / "trend" / args.trend_id
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((output_root / "audit.json").read_text(encoding="utf-8"))
    report = {
        "phase": "alpha4c.2",
        "calibration_id": args.trend_id,
        "paper_ids": CALIBRATION_PAPERS,
        "llm_calls_performed": False,
        "trend_semantics_id": summary.get("trend_semantics_id"),
        "contract_semantics_id": summary.get("contract_semantics_id"),
        "evidence_count": summary.get("evidence_count"),
        "quantitative_evidence_count": summary.get("quantitative_evidence_count"),
        "claim_evidence_count": summary.get("claim_evidence_count"),
        "evidence_basis_counts": summary.get("evidence_basis_counts", {}),
        "independent_variable_counts": summary.get("independent_variable_counts", {}),
        "dependent_observable_counts": summary.get("dependent_observable_counts", {}),
        "direction_counts": summary.get("direction_counts", {}),
        "shape_counts": summary.get("shape_counts", {}),
        "structural_gate": bool(summary.get("structural_gate", False)),
        "audit_issues": audit.get("issues", []),
        "acceptance_note": (
            "No trend-count target is encoded. Review extracted provenance and "
            "scientific interpretation before freezing alpha4c.2 semantics."
        ),
    }
    report_path = output_root / "calibration_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print()
    print("alpha4c.2 calibration structural gate:", report["structural_gate"])
    print("Evidence:", report["evidence_count"])
    print("Quantitative / claim:", report["quantitative_evidence_count"], "/", report["claim_evidence_count"])
    print("Report:", report_path)
    print("Evidence JSONL:", output_root / "evidence.jsonl")
    return 0 if report["structural_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
