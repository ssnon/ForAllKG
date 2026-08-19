from __future__ import annotations

from typing import Any, Iterable, Mapping


def _claim_rows(
    evidence_rows: Iterable[Mapping[str, Any]],
    claim_id: str,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in evidence_rows
        if claim_id in {
            str(value)
            for value in row.get("source_claim_ids", []) or []
        }
    ]


def _annotation_by_trend_id(
    annotation_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("trend_id", "")): row
        for row in annotation_rows
        if str(row.get("trend_id", "")).strip()
    }


def _match_row(
    rows: list[Mapping[str, Any]],
    **expected: object,
) -> tuple[bool, str]:
    if len(rows) != 1:
        return False, f"expected exactly one row, observed {len(rows)}"
    row = rows[0]
    mismatches = {
        key: (row.get(key), value)
        for key, value in expected.items()
        if row.get(key) != value
    }
    if mismatches:
        return False, f"field mismatches: {mismatches}"
    return True, "PASS"


def calibration_checks(
    evidence_rows: list[Mapping[str, Any]],
    annotation_rows: list[Mapping[str, Any]],
    local_results: list[Mapping[str, Any]],
) -> dict[str, dict[str, object]]:
    del annotation_rows, local_results
    checks: dict[str, dict[str, object]] = {}

    ok, detail = _match_row(
        _claim_rows(evidence_rows, "claim_optimal_agno3"),
        paper_id="Kiwook_SERS_1",
        independent_variable_key="silver_precursor_concentration",
        dependent_observable_key="raman_intensity",
        direction="non_monotonic",
        shape="single_optimum",
    )
    checks["sers1_rise_peak_fall"] = {"pass": ok, "detail": detail}

    ok, detail = _match_row(
        _claim_rows(evidence_rows, "claim_atp_detection"),
        paper_id="Kiwook_SERS_1",
        independent_variable_key="analyte_concentration",
        dependent_observable_key="raman_intensity",
        direction="positive",
        shape="monotonic",
    )
    checks["sers1_raman_peak_intensity"] = {"pass": ok, "detail": detail}

    ok, detail = _match_row(
        _claim_rows(
            evidence_rows,
            "claim_interior_nanogap_enhancement",
        ),
        paper_id="Kiwook_SERS_5",
        independent_variable_key="nanogap_presence",
        dependent_observable_key="sers_performance",
        direction="positive",
        shape="unspecified",
    )
    checks["sers5_nanogap_presence_not_size"] = {
        "pass": ok,
        "detail": detail,
    }

    ok, detail = _match_row(
        _claim_rows(evidence_rows, "claim_ef_increases_gold"),
        paper_id="Kiwook_SERS_8",
        independent_variable_key="gold_precursor_amount",
        dependent_observable_key="sers_enhancement_factor",
        direction="positive",
        shape="monotonic",
    )
    checks["sers8_formal_ef_precedence"] = {
        "pass": ok,
        "detail": detail,
    }
    return checks


def seen_regression_checks(
    evidence_rows: list[Mapping[str, Any]],
    annotation_rows: list[Mapping[str, Any]],
    local_results: list[Mapping[str, Any]],
) -> dict[str, dict[str, object]]:
    checks: dict[str, dict[str, object]] = {}
    annotation_by_id = _annotation_by_trend_id(annotation_rows)

    ratio_rows = _claim_rows(
        evidence_rows,
        "claim_ratio_10_7_highest_sers",
    )
    ok, detail = _match_row(
        ratio_rows,
        paper_id="Kiwook_SERS_6",
        independent_variable_key="ag_to_au_ratio",
        dependent_observable_key="raman_intensity",
        direction="non_monotonic",
        shape="single_optimum",
    )
    checks["sers6_plural_ratio_single_optimum"] = {
        "pass": ok,
        "detail": detail,
    }
    if ok:
        annotation = annotation_by_id.get(
            str(ratio_rows[0].get("trend_id", "")),
            {},
        )
        landmark_ok = (
            annotation.get("source_control_value_text") == "10:7"
            and annotation.get("canonical_control_value_numeric") == 0.7
            and annotation.get("normalization_transform")
            == "au_ag_to_ag_over_au"
        )
        checks["sers6_ratio_orientation_landmark"] = {
            "pass": landmark_ok,
            "detail": (
                "PASS"
                if landmark_ok
                else {
                    "source_control_value_text":
                        annotation.get("source_control_value_text"),
                    "canonical_control_value_numeric":
                        annotation.get("canonical_control_value_numeric"),
                    "normalization_transform":
                        annotation.get("normalization_transform"),
                }
            ),
        }
    else:
        checks["sers6_ratio_orientation_landmark"] = {
            "pass": False,
            "detail": "ratio trend row unavailable",
        }

    for claim_id, check_name in (
        ("claim_shell_thickness_trend", "sers10_shell_claim_primary"),
        ("claim_sers_shell_thickness", "sers10_shell_claim_restatement"),
    ):
        ok, detail = _match_row(
            _claim_rows(evidence_rows, claim_id),
            paper_id="Kiwook_SERS_10",
            independent_variable_key="shell_thickness",
            dependent_observable_key="raman_intensity",
            direction="positive",
            shape="saturating",
        )
        checks[check_name] = {"pass": ok, "detail": detail}

    merged = [
        row
        for row in local_results
        if row.get("paper_id") == "Kiwook_SERS_10"
        and row.get("result_lane") == "claim"
        and row.get("independent_variable_key") == "shell_thickness"
        and row.get("dependent_observable_key") == "raman_intensity"
        and row.get("direction") == "positive"
        and row.get("shape") == "saturating"
        and {
            "claim_shell_thickness_trend",
            "claim_sers_shell_thickness",
        }.issubset(set(row.get("source_claim_ids", []) or []))
    ]
    merged_ok = (
        len(merged) == 1
        and int(merged[0].get("support_mention_count", 0)) >= 2
    )
    checks["sers10_structural_family_consolidation"] = {
        "pass": merged_ok,
        "detail": (
            "PASS"
            if merged_ok
            else f"matching merged results: {len(merged)}"
        ),
    }

    calculated_ef = []
    for row in evidence_rows:
        if (
            row.get("paper_id") == "Kiwook_SERS_2"
            and row.get("evidence_basis")
            in {"controlled_numeric_pair", "controlled_numeric_series"}
            and row.get("dependent_observable_key")
            == "sers_enhancement_factor"
        ):
            annotation = annotation_by_id.get(
                str(row.get("trend_id", "")),
                {},
            )
            calculated_ef.append((row, annotation))
    model_ok = (
        len(calculated_ef) == 1
        and calculated_ef[0][1].get("evidence_kind")
        == "calculated_numeric"
        and calculated_ef[0][1].get("observable_semantics")
        == "model_derived_sers_enhancement_factor"
        and bool(calculated_ef[0][0].get("source_calculation_ids"))
    )
    checks["sers2_calculated_ef_model_semantics"] = {
        "pass": model_ok,
        "detail": (
            "PASS"
            if model_ok
            else [
                {
                    "trend_id": row.get("trend_id"),
                    "calculation_ids":
                        row.get("source_calculation_ids", []),
                    "evidence_kind":
                        annotation.get("evidence_kind"),
                    "observable_semantics":
                        annotation.get("observable_semantics"),
                }
                for row, annotation in calculated_ef
            ]
        ),
    }

    no_calculated_formal = all(
        not (
            row.get("evidence_kind") == "calculated_numeric"
            and row.get("observable_semantics")
            == "formal_sers_enhancement_factor"
        )
        for row in annotation_rows
    )
    checks["no_calculated_numeric_promoted_to_formal_empirical_ef"] = {
        "pass": no_calculated_formal,
        "detail": "PASS" if no_calculated_formal else "violation found",
    }
    return checks


def checks_gate(checks: Mapping[str, Mapping[str, object]]) -> bool:
    return bool(checks) and all(
        bool(row.get("pass", False))
        for row in checks.values()
    )
