from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


from campaigns.sers_alpha4_epoch.paths import PROJECT_ROOT

SUITES = {
    "calibration": {
        "corpus_id":
            "sers_alpha4b4a11_method_reconstruction_calibration_replay_v1_corpus",
        "identity_id":
            "sers_alpha4b4a11_method_reconstruction_calibration_replay_v1_measurement_identity",
        "comparison_id":
            "sers_alpha4b4a11_method_reconstruction_calibration_replay_v1_comparison",
        "trend_id": "sers_alpha4c212_calibration_v1",
        "precision_id": "sers_alpha4c212_calibration_precision_v1",
    },
    "seen": {
        "corpus_id": "sers_alpha4b4a11_holdout_real_v1_corpus",
        "identity_id":
            "sers_alpha4b4a11_holdout_real_v1_measurement_identity",
        "comparison_id":
            "sers_alpha4b4a11_holdout_real_v1_comparison",
        "trend_id": "sers_alpha4c212_seen_regression_v1",
        "precision_id":
            "sers_alpha4c212_seen_regression_precision_v1",
    },
}


def _run(command: list[str]) -> None:
    print("$", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _record(
    checks: dict[str, dict[str, object]],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks[name] = {
        "pass": bool(passed),
        "detail": detail,
    }
    print(
        f"Regression {name}: "
        f"{'PASS' if passed else 'FAIL'} - {detail}"
    )


def _build(
    suite: str,
    data_root: str,
):
    cfg = SUITES[suite]
    _run([
        sys.executable,
        "-m",
        "scripts.build_trend_evidence",
        "--domain-profile", "sers_au_ag",
        "--data-root", data_root,
        "--corpus-id", cfg["corpus_id"],
        "--mode", "exploratory",
        "--trend-id", cfg["trend_id"],
        "--measurement-result-identity-id", cfg["identity_id"],
        "--comparison-id", cfg["comparison_id"],
    ])
    _run([
        sys.executable,
        "-m",
        "scripts.build_trend_precision",
        "--domain-profile", "sers_au_ag",
        "--data-root", data_root,
        "--corpus-id", cfg["corpus_id"],
        "--mode", "exploratory",
        "--trend-id", cfg["trend_id"],
        "--precision-id", cfg["precision_id"],
    ])

    root = (
        PROJECT_ROOT
        / data_root
        / "corpus"
        / cfg["corpus_id"]
        / "exploratory"
        / "trend"
        / cfg["trend_id"]
        / "precision"
        / cfg["precision_id"]
    )
    summary = json.loads(
        (root / "summary.json").read_text(encoding="utf-8")
    )
    return (
        root,
        summary,
        _jsonl(root / "local_results.jsonl"),
        _jsonl(root / "annotations.jsonl"),
    )


def _calibration_checks(results):
    checks = {}

    spectral = [
        row for row in results
        if row.get("paper_id") == "Kiwook_SERS_8"
        and row.get("independent_variable_key")
            == "spr_excitation_detuning"
        and row.get("dependent_observable_key")
            == "sers_enhancement_factor"
        and row.get("direction") == "negative"
        and row.get("control_family") == "optical_alignment"
        and row.get("observable_semantics")
            == "formal_sers_enhancement_factor"
    ]
    _record(
        checks,
        "sers8_spectral_axis_grounding",
        len(spectral) == 1,
        f"matching results: {len(spectral)}",
    )

    bad = [
        row for row in results
        if row.get("paper_id") == "Kiwook_SERS_8"
        and "mech_lspr_laser_matching_increases_ef"
            in (row.get("source_claim_ids") or [])
        and row.get("independent_variable_key")
            == "excitation_wavelength"
    ]
    _record(
        checks,
        "sers8_no_raw_excitation_axis_for_matching_claim",
        not bad,
        f"bad-axis results: {len(bad)}",
    )
    return checks


def _seen_checks(results, annotations):
    checks = {}

    shell = [
        row for row in results
        if row.get("paper_id") == "Kiwook_SERS_10"
        and row.get("result_lane") == "claim"
        and row.get("independent_variable_key") == "shell_thickness"
        and row.get("dependent_observable_key") == "raman_intensity"
        and row.get("direction") == "positive"
        and row.get("shape") == "saturating"
    ]
    expected = {
        "claim_shell_thickness_trend",
        "claim_sers_shell_thickness",
    }
    merged = (
        len(shell) == 1
        and expected.issubset(
            set(shell[0].get("source_claim_ids") or [])
        )
        and int(shell[0].get("support_mention_count", 0)) >= 2
    )
    _record(
        checks,
        "sers10_structural_family_consolidation",
        merged,
        (
            f"matching results: {len(shell)}; "
            f"support="
            f"{shell[0].get('support_mention_count') if shell else 0}"
        ),
    )

    dda = [
        row for row in results
        if row.get("paper_id") == "Kiwook_SERS_2"
        and row.get("independent_variable_key") == "nanogap_size"
        and "calculated_numeric" in (row.get("evidence_kinds") or [])
        and row.get("observable_semantics")
            == "model_derived_sers_enhancement_factor"
    ]
    _record(
        checks,
        "sers2_calculated_ef_preserved",
        len(dda) == 1,
        f"matching results: {len(dda)}",
    )

    ratio = [
        row for row in annotations
        if row.get("paper_id") == "Kiwook_SERS_6"
        and row.get("canonical_control_value_numeric") == 0.7
        and row.get("normalization_transform")
            == "au_ag_to_ag_over_au"
    ]
    _record(
        checks,
        "sers6_ratio_orientation_preserved",
        len(ratio) == 1,
        f"matching annotations: {len(ratio)}",
    )
    return checks


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "alpha4c.2.1.2 focused calibration + seen regression"
        )
    )
    parser.add_argument(
        "--suite",
        choices=("all", "calibration", "seen"),
        default="all",
    )
    parser.add_argument("--data-root", default="data_sers")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suites = (
        ("calibration", "seen")
        if args.suite == "all"
        else (args.suite,)
    )
    overall = True

    for suite in suites:
        print()
        print("=" * 72)
        print("alpha4c.2.1.2", suite)
        print("=" * 72)

        root, summary, results, annotations = _build(
            suite,
            args.data_root,
        )
        checks = (
            _calibration_checks(results)
            if suite == "calibration"
            else _seen_checks(results, annotations)
        )

        structural = bool(summary.get("structural_gate", False))
        semantic = all(
            bool(value["pass"])
            for value in checks.values()
        )
        overall = overall and structural and semantic

        report = {
            "phase": f"alpha4c.2.1.2 {suite}",
            "trend_semantics_id":
                "sers_au_ag_trend_v4_alpha4c212",
            "precision_semantics_id":
                "sers_au_ag_trend_precision_v3_alpha4c212",
            "evidence_count": summary.get("evidence_count"),
            "local_result_count": summary.get("local_result_count"),
            "duplicate_claim_mentions_collapsed":
                summary.get("duplicate_claim_mentions_collapsed"),
            "structural_gate": structural,
            "semantic_regression_gate": semantic,
            "regression_checks": checks,
            "acceptance_note": (
                "No evidence-count target is encoded. "
                "Only the alpha4c.2.1.2 repairs and the explicitly "
                "named preservation invariants are checked."
            ),
        }
        report_path = (
            root / "alpha4c212_regression_report.json"
        )
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

        print(
            "Evidence / local results:",
            report["evidence_count"],
            "/",
            report["local_result_count"],
        )
        print(
            "Collapsed duplicate claim mentions:",
            report["duplicate_claim_mentions_collapsed"],
        )
        print("Structural gate:", structural)
        print("Semantic regression gate:", semantic)
        print("Report:", report_path)
        print("Local results:", root / "local_results.jsonl")
        print("Annotations:", root / "annotations.jsonl")

    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
