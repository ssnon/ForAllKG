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


def _dimension(
    profile: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    matches = [
        item
        for item in profile.get("dimensions", [])
        if item.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {name!r} dimension."
        )
    return matches[0]


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


def _build(suite: str, data_root: str):
    config = SUITES[suite]
    _run([
        sys.executable,
        "-m",
        "scripts.build_cross_context_profiles",
        "--domain-profile",
        "sers_au_ag",
        "--data-root",
        data_root,
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
    ])

    root = (
        PROJECT_ROOT
        / data_root
        / "corpus"
        / config["corpus_id"]
        / "exploratory"
        / "trend"
        / config["trend_id"]
        / "precision"
        / config["precision_id"]
        / "cross_context"
        / config["context_id"]
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
    profiles = _read_jsonl(
        root / "context_profiles.jsonl"
    )
    return root, summary, audit, profiles


def _common_checks(
    summary: dict[str, Any],
    audit: dict[str, Any],
    profiles: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    checks: dict[str, dict[str, object]] = {}

    _record(
        checks,
        "one_profile_per_local_result",
        (
            summary.get("local_result_count")
            == summary.get("profile_count")
            == len(profiles)
        ),
        (
            f"local={summary.get('local_result_count')}, "
            f"profiles={summary.get('profile_count')}, "
            f"rows={len(profiles)}"
        ),
    )
    _record(
        checks,
        "no_paper_global_context_leakage",
        (
            summary.get("paper_global_leakage_count") == 0
            and summary.get(
                "paper_global_context_fallback_used"
            )
            is False
        ),
        (
            "leakage="
            f"{summary.get('paper_global_leakage_count')}"
        ),
    )
    _record(
        checks,
        "direct_measurement_provenance_resolved",
        summary.get(
            "unresolved_direct_measurement_count"
        )
        == 0,
        (
            "unresolved="
            f"{summary.get('unresolved_direct_measurement_count')}"
        ),
    )
    _record(
        checks,
        "projection_structural_gate",
        (
            summary.get("structural_gate") is True
            and audit.get("structural_gate") is True
        ),
        (
            f"summary={summary.get('structural_gate')}, "
            f"audit={audit.get('structural_gate')}"
        ),
    )

    direct_profiles = [
        profile
        for profile in profiles
        if profile.get("source_measurement_ids")
    ]
    direct_with_context = [
        profile
        for profile in direct_profiles
        if (
            profile.get(
                "source_comparison_context_ids"
            )
            and profile.get(
                "source_method_context_ids"
            )
        )
    ]
    _record(
        checks,
        "direct_measurement_profiles_bind_sidecars",
        len(direct_with_context) == len(direct_profiles),
        (
            f"direct={len(direct_profiles)}, "
            f"bound={len(direct_with_context)}"
        ),
    )

    no_direct_profiles = [
        profile
        for profile in profiles
        if not profile.get("source_measurement_ids")
    ]
    leaked_no_direct = []
    for profile in no_direct_profiles:
        if (
            profile.get("source_comparison_context_ids")
            or profile.get("source_method_context_ids")
        ):
            leaked_no_direct.append(
                profile.get("local_result_id")
            )
            continue
        for dimension in profile.get("dimensions", []):
            if dimension.get("status") not in {
                "unknown",
                "varied_control",
                "not_applicable",
            }:
                leaked_no_direct.append(
                    profile.get("local_result_id")
                )
                break
    _record(
        checks,
        "claim_or_unlinked_profiles_remain_context_unknown",
        not leaked_no_direct,
        f"leaked={leaked_no_direct}",
    )
    return checks


def _calibration_checks(
    summary: dict[str, Any],
    audit: dict[str, Any],
    profiles: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    checks = _common_checks(
        summary,
        audit,
        profiles,
    )

    concentration = [
        profile
        for profile in profiles
        if profile.get("independent_variable_key")
        == "analyte_concentration"
    ]
    concentration_ok = (
        len(concentration) == 1
        and _dimension(
            concentration[0],
            "analyte_concentration",
        ).get("status")
        == "varied_control"
    )
    _record(
        checks,
        "analyte_concentration_is_varied_control",
        concentration_ok,
        f"matching_profiles={len(concentration)}",
    )

    spectral = [
        profile
        for profile in profiles
        if profile.get("independent_variable_key")
        == "spr_excitation_detuning"
    ]
    spectral_ok = (
        len(spectral) == 1
        and _dimension(
            spectral[0],
            "excitation_wavelength",
        ).get("status")
        != "varied_control"
    )
    _record(
        checks,
        "spectral_detuning_does_not_mask_excitation_context",
        spectral_ok,
        f"matching_profiles={len(spectral)}",
    )
    return checks


def _seen_checks(
    summary: dict[str, Any],
    audit: dict[str, Any],
    profiles: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    checks = _common_checks(
        summary,
        audit,
        profiles,
    )

    shell_profiles = [
        profile
        for profile in profiles
        if profile.get("independent_variable_key")
        == "shell_thickness"
    ]
    _record(
        checks,
        "shell_thickness_not_misclassified_as_context_mask",
        bool(shell_profiles)
        and all(
            not any(
                dimension.get("status")
                == "varied_control"
                for dimension in profile.get(
                    "dimensions",
                    [],
                )
            )
            for profile in shell_profiles
        ),
        f"matching_profiles={len(shell_profiles)}",
    )
    return checks


def main() -> int:
    overall = True

    for suite in ("calibration", "seen"):
        print()
        print("=" * 72)
        print("alpha4c.3b", suite)
        print("=" * 72)

        root, summary, audit, profiles = _build(
            suite,
            "data_sers",
        )
        checks = (
            _calibration_checks(
                summary,
                audit,
                profiles,
            )
            if suite == "calibration"
            else _seen_checks(
                summary,
                audit,
                profiles,
            )
        )
        semantic_gate = all(
            bool(value["pass"])
            for value in checks.values()
        )
        overall = overall and semantic_gate

        report = {
            "phase": f"alpha4c.3b {suite}",
            "contract_semantics_id":
                "cross_context_trend_contract_v1_alpha4c3a",
            "context_semantics_id":
                "sers_au_ag_trend_context_v1_alpha4c3b",
            "semantic_regression_gate":
                semantic_gate,
            "checks": checks,
            "structural_gate":
                summary.get("structural_gate"),
            "profile_count":
                summary.get("profile_count"),
            "direct_measurement_profile_count":
                summary.get(
                    "direct_measurement_profile_count"
                ),
            "no_direct_measurement_profile_count":
                summary.get(
                    "no_direct_measurement_profile_count"
                ),
            "paper_global_leakage_count":
                summary.get(
                    "paper_global_leakage_count"
                ),
            "note": (
                "alpha4c.3b performs context projection only. "
                "No pairwise contrasts or final trend statuses "
                "are produced."
            ),
        }
        report_path = (
            root
            / "alpha4c3b_projection_regression_report.json"
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
            "Structural gate:",
            summary.get("structural_gate"),
        )
        print(
            "Semantic regression gate:",
            semantic_gate,
        )
        print("Report:", report_path)
        print(
            "Profiles:",
            root / "context_profiles.jsonl",
        )

    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
