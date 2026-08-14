from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUITES = {
    "calibration": {
        "corpus_id":
            "sers_alpha4b4a11_method_reconstruction_calibration_replay_v1_corpus",
        "trend_id":
            "sers_alpha4c21211_calibration_v1",
        "precision_id":
            "sers_alpha4c21211_calibration_precision_v1",
        "context_id":
            "sers_alpha4c3b_calibration_context_v1",
        "assessment_id":
            "sers_alpha4c3c_calibration_assessment_v1",
    },
    "seen": {
        "corpus_id":
            "sers_alpha4b4a11_holdout_real_v1_corpus",
        "trend_id":
            "sers_alpha4c21211_seen_regression_v1",
        "precision_id":
            "sers_alpha4c21211_seen_regression_precision_v1",
        "context_id":
            "sers_alpha4c3b_seen_context_v1",
        "assessment_id":
            "sers_alpha4c3c_seen_assessment_v1",
    },
}


def _run(command: list[str]) -> None:
    print("$", " ".join(command))
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        f"{'PASS' if passed else 'FAIL'} - "
        f"{detail}"
    )


def _build(suite: str):
    config = SUITES[suite]
    _run([
        sys.executable,
        "-m",
        "scripts.build_cross_context_assessments",
        "--domain-profile",
        "sers_au_ag",
        "--data-root",
        "data_sers",
        "--corpus-id",
        config["corpus_id"],
        "--mode",
        "exploratory",
        "--trend-id",
        config["trend_id"],
        "--precision-id",
        config["precision_id"],
        "--context-id",
        config["context_id"],
        "--assessment-id",
        config["assessment_id"],
    ])

    root = (
        PROJECT_ROOT
        / "data_sers"
        / "corpus"
        / config["corpus_id"]
        / "exploratory"
        / "trend"
        / config["trend_id"]
        / "precision"
        / config["precision_id"]
        / "cross_context"
        / config["context_id"]
        / "assessment"
        / config["assessment_id"]
    )
    summary = json.loads(
        (root / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (root / "audit.json").read_text(
            encoding="utf-8"
        )
    )
    contrasts = _read_jsonl(
        root / "pairwise_contrasts.jsonl"
    )
    assessments = _read_jsonl(
        root / "assessments.jsonl"
    )
    return (
        root,
        summary,
        audit,
        contrasts,
        assessments,
    )


def _common_checks(
    summary: dict[str, Any],
    audit: dict[str, Any],
    contrasts: list[dict[str, Any]],
    assessments: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    checks: dict[str, dict[str, object]] = {}

    _record(
        checks,
        "one_assessment_per_relation",
        (
            summary.get("relation_count")
            == summary.get("assessment_count")
            == len(assessments)
        ),
        (
            f"relations={summary.get('relation_count')}, "
            f"assessments={summary.get('assessment_count')}, "
            f"rows={len(assessments)}"
        ),
    )
    _record(
        checks,
        "complete_cross_paper_pair_generation",
        (
            summary.get(
                "expected_cross_paper_pair_count"
            )
            == summary.get(
                "pairwise_contrast_count"
            )
            == len(contrasts)
        ),
        (
            "expected="
            f"{summary.get('expected_cross_paper_pair_count')}, "
            "actual="
            f"{summary.get('pairwise_contrast_count')}"
        ),
    )

    same_paper_pairs = [
        row.get("contrast_id")
        for row in contrasts
        if row.get("left_paper_id")
        == row.get("right_paper_id")
    ]
    _record(
        checks,
        "same_paper_pairing_forbidden",
        not same_paper_pairs
        and summary.get(
            "same_paper_pairs_allowed"
        )
        is False,
        f"same_paper_pairs={same_paper_pairs}",
    )

    bad_reversal_status = [
        row.get("assessment_id")
        for row in assessments
        if row.get("reversal_pair_ids")
        and row.get("status") != "reversed"
    ]
    _record(
        checks,
        "strict_reversal_forces_reversed_status",
        not bad_reversal_status
        and summary.get("majority_vote_used")
        is False,
        f"bad={bad_reversal_status}",
    )

    single_paper_non_insufficient = [
        row.get("assessment_id")
        for row in assessments
        if len(row.get("paper_ids", [])) < 2
        and row.get("status") != "insufficient"
    ]
    _record(
        checks,
        "single_paper_relations_are_not_replications",
        not single_paper_non_insufficient,
        f"bad={single_paper_non_insufficient}",
    )

    _record(
        checks,
        "frozen_context_is_not_reprojected",
        (
            summary.get("context_reprojected")
            is False
            and summary.get(
                "numeric_ranking_reused_as_trend_policy"
            )
            is False
            and summary.get(
                "causal_status_promoted"
            )
            is False
        ),
        (
            "context_reprojected="
            f"{summary.get('context_reprojected')}"
        ),
    )

    _record(
        checks,
        "assessment_structural_gate",
        (
            summary.get("structural_gate") is True
            and audit.get("structural_gate") is True
            and audit.get("base_structural_gate")
            is True
        ),
        (
            f"summary={summary.get('structural_gate')}, "
            f"audit={audit.get('structural_gate')}, "
            f"base={audit.get('base_structural_gate')}"
        ),
    )
    return checks


def _calibration_checks(
    summary: dict[str, Any],
    audit: dict[str, Any],
    contrasts: list[dict[str, Any]],
    assessments: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    checks = _common_checks(
        summary,
        audit,
        contrasts,
        assessments,
    )

    unresolved_bad = [
        row.get("assessment_id")
        for row in assessments
        if not row.get("pairwise_contrast_ids")
        and row.get("status") != "insufficient"
    ]
    _record(
        checks,
        "no_pair_relations_remain_insufficient",
        not unresolved_bad,
        f"bad={unresolved_bad}",
    )
    return checks


def _seen_checks(
    summary: dict[str, Any],
    audit: dict[str, Any],
    contrasts: list[dict[str, Any]],
    assessments: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    checks = _common_checks(
        summary,
        audit,
        contrasts,
        assessments,
    )

    shell_raman = [
        row
        for row in assessments
        if (
            row.get("independent_variable_key")
            == "shell_thickness"
            and row.get("dependent_observable_key")
            == "raman_intensity"
            and row.get("control_family")
            == "structural"
            and row.get("observable_semantics")
            == "measured_signal_intensity"
        )
    ]
    shell_ok = (
        len(shell_raman) == 1
        and len(
            shell_raman[0].get(
                "member_result_ids",
                [],
            )
        )
        >= 2
        and len(
            shell_raman[0].get(
                "paper_ids",
                [],
            )
        )
        == 1
        and not shell_raman[0].get(
            "pairwise_contrast_ids"
        )
        and shell_raman[0].get("status")
        == "insufficient"
    )
    _record(
        checks,
        "sers10_same_paper_numeric_claim_not_counted_as_replication",
        shell_ok,
        (
            f"matching_assessments={len(shell_raman)}"
            + (
                ", members="
                f"{len(shell_raman[0].get('member_result_ids', []))}, "
                "papers="
                f"{len(shell_raman[0].get('paper_ids', []))}, "
                "pairs="
                f"{len(shell_raman[0].get('pairwise_contrast_ids', []))}, "
                "status="
                f"{shell_raman[0].get('status')}"
                if shell_raman
                else ""
            )
        ),
    )
    return checks


def main() -> int:
    overall = True

    for suite in ("calibration", "seen"):
        print()
        print("=" * 72)
        print("alpha4c.3c", suite)
        print("=" * 72)

        (
            root,
            summary,
            audit,
            contrasts,
            assessments,
        ) = _build(suite)

        checks = (
            _calibration_checks(
                summary,
                audit,
                contrasts,
                assessments,
            )
            if suite == "calibration"
            else _seen_checks(
                summary,
                audit,
                contrasts,
                assessments,
            )
        )
        semantic_gate = all(
            bool(row["pass"])
            for row in checks.values()
        )
        overall = overall and semantic_gate

        report = {
            "phase": f"alpha4c.3c {suite}",
            "assessment_semantics_id":
                "cross_context_trend_assessment_v1_alpha4c3c",
            "semantic_regression_gate":
                semantic_gate,
            "structural_gate":
                summary.get("structural_gate"),
            "relation_count":
                summary.get("relation_count"),
            "pairwise_contrast_count":
                summary.get(
                    "pairwise_contrast_count"
                ),
            "assessment_count":
                summary.get("assessment_count"),
            "status_counts":
                summary.get("status_counts"),
            "pair_role_counts":
                summary.get("pair_role_counts"),
            "checks": checks,
            "note": (
                "Current calibration/seen sets may contain no "
                "cross-paper same-relation overlap. That is not a "
                "failure; same-paper support must remain insufficient."
            ),
        }
        report_path = (
            root
            / "alpha4c3c_assessment_regression_report.json"
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
            "Relations / pairwise / assessments:",
            summary.get("relation_count"),
            "/",
            summary.get("pairwise_contrast_count"),
            "/",
            summary.get("assessment_count"),
        )
        print(
            "Statuses:",
            json.dumps(
                summary.get("status_counts", {}),
                sort_keys=True,
            ),
        )
        print(
            "Structural gate:",
            summary.get("structural_gate"),
        )
        print(
            "Semantic regression gate:",
            semantic_gate,
        )
        print("Report:", report_path)

    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
