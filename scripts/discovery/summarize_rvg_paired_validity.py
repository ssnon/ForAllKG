from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERDICT_RANK = {
    "FAIL": 0,
    "UNKNOWN": 1,
    "WARN": 2,
    "PASS": 3,
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _audit_by_alias(
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in payload.get("candidates", []):
        if not isinstance(row, dict):
            continue
        alias = str(row.get("candidate_alias") or "")
        if alias:
            result[alias] = row
    return result


def _dims(
    row: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(dim.get("dimension") or ""): dim
        for dim in row.get("dimensions", [])
        if isinstance(dim, dict)
        and str(dim.get("dimension") or "")
    }


def _transition(control: str, treatment: str) -> str:
    cr = VERDICT_RANK.get(control, -1)
    tr = VERDICT_RANK.get(treatment, -1)
    if cr < 0 or tr < 0:
        return "UNMAPPED"
    if tr > cr:
        return "IMPROVED"
    if tr < cr:
        return "WORSENED"
    return "SAME"


def build_paired_summary(
    *,
    control_audit: dict[str, Any],
    treatment_audit: dict[str, Any],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    control_by = _audit_by_alias(control_audit)
    treatment_by = _audit_by_alias(treatment_audit)

    paired_rows: list[dict[str, Any]] = []
    transition_counts = Counter()
    control_fail_counts = Counter()
    treatment_fail_counts = Counter()
    generated_pair_count = 0
    abstained_count = 0
    rejected_count = 0

    for map_row in mapping.get("records", []):
        control_alias = str(
            map_row.get("control_candidate_alias") or ""
        )
        status = str(
            map_row.get("treatment_status") or ""
        )

        control = control_by.get(control_alias)
        if control is None:
            raise RuntimeError(
                "missing control validity row: " + control_alias
            )

        control_dims = _dims(control)
        for name, dim in control_dims.items():
            if str(dim.get("verdict") or "") == "FAIL":
                control_fail_counts[name] += 1

        paired = {
            "control_candidate_alias": control_alias,
            "control_hypothesis_id": map_row.get(
                "control_hypothesis_id"
            ),
            "control_summary": control.get("summary"),
            "treatment_status": status,
            "identification_assessment": map_row.get(
                "identification_assessment"
            ),
            "treatment_hypothesis_id": map_row.get(
                "treatment_hypothesis_id"
            ),
            "exact_premise_identity": map_row.get(
                "exact_premise_identity"
            ),
            "exact_gap_identity": map_row.get(
                "exact_gap_identity"
            ),
            "exact_hypothesis_type_identity": map_row.get(
                "exact_hypothesis_type_identity"
            ),
            "dimension_transitions": [],
            "treatment_summary": None,
        }

        if status == "ABSTAINED_NOT_IDENTIFIABLE":
            abstained_count += 1
            paired["paired_outcome"] = (
                "SAFE_ABSTENTION_NOT_IDENTIFIABLE"
            )
            paired_rows.append(paired)
            continue

        if status != "GENERATED_TREATMENT":
            rejected_count += 1
            paired["paired_outcome"] = (
                "UNEVALUABLE_TREATMENT_GENERATION_FAILURE"
            )
            paired_rows.append(paired)
            continue

        treatment_alias = str(
            map_row.get("treatment_audit_alias") or ""
        )
        treatment = treatment_by.get(treatment_alias)
        if treatment is None:
            raise RuntimeError(
                "missing treatment validity row: "
                + treatment_alias
            )

        generated_pair_count += 1
        paired["treatment_audit_alias"] = treatment_alias
        paired["treatment_summary"] = treatment.get("summary")

        treatment_dims = _dims(treatment)

        dimensions = sorted(
            set(control_dims) | set(treatment_dims)
        )
        local_transitions = Counter()

        for name in dimensions:
            c = str(
                (control_dims.get(name) or {}).get(
                    "verdict", "UNKNOWN"
                )
            )
            t = str(
                (treatment_dims.get(name) or {}).get(
                    "verdict", "UNKNOWN"
                )
            )
            transition = _transition(c, t)
            transition_counts[transition] += 1
            local_transitions[transition] += 1

            if t == "FAIL":
                treatment_fail_counts[name] += 1

            paired["dimension_transitions"].append(
                {
                    "dimension": name,
                    "control_verdict": c,
                    "treatment_verdict": t,
                    "transition": transition,
                    "control_rationale": (
                        control_dims.get(name) or {}
                    ).get("rationale"),
                    "treatment_rationale": (
                        treatment_dims.get(name) or {}
                    ).get("rationale"),
                }
            )

        introduced_failures = [
            row
            for row in paired["dimension_transitions"]
            if (
                row["treatment_verdict"] == "FAIL"
                and row["control_verdict"] != "FAIL"
            )
        ]
        resolved_failures = [
            row
            for row in paired["dimension_transitions"]
            if (
                row["control_verdict"] == "FAIL"
                and row["treatment_verdict"] != "FAIL"
            )
        ]
        uncertainty_downgrades = [
            row
            for row in paired["dimension_transitions"]
            if (
                row["treatment_verdict"] == "UNKNOWN"
                and row["control_verdict"] in {"PASS", "WARN"}
            )
        ]

        paired["introduced_failures"] = [
            row["dimension"] for row in introduced_failures
        ]
        paired["resolved_failures"] = [
            row["dimension"] for row in resolved_failures
        ]
        paired["uncertainty_downgrades"] = [
            row["dimension"] for row in uncertainty_downgrades
        ]

        if introduced_failures:
            outcome = "GENERATED_WITH_HARD_VALIDITY_REGRESSION"
        elif resolved_failures and uncertainty_downgrades:
            outcome = (
                "GENERATED_WITH_HARD_VALIDITY_IMPROVEMENT"
                "_AND_UNCERTAINTY_DOWNGRADE"
            )
        elif resolved_failures:
            outcome = "GENERATED_WITH_HARD_VALIDITY_IMPROVEMENT"
        elif local_transitions["IMPROVED"] and uncertainty_downgrades:
            outcome = (
                "GENERATED_WITH_SOFT_VALIDITY_IMPROVEMENT"
                "_AND_UNCERTAINTY_DOWNGRADE"
            )
        elif local_transitions["IMPROVED"]:
            outcome = "GENERATED_WITH_SOFT_VALIDITY_IMPROVEMENT"
        elif uncertainty_downgrades:
            outcome = "GENERATED_WITH_UNCERTAINTY_DOWNGRADE"
        elif local_transitions["WORSENED"]:
            outcome = "GENERATED_WITH_SOFT_VALIDITY_REGRESSION"
        else:
            outcome = "GENERATED_WITH_NO_VALIDITY_CHANGE"

        paired["paired_outcome"] = outcome
        paired_rows.append(paired)

    body = {
        "schema_version": "rvg-paired-validity-summary-v1",
        "control_case_id": mapping.get("control_case_id"),
        "treatment_case_id": mapping.get("treatment_case_id"),
        "control_candidate_count": mapping.get(
            "control_candidate_count"
        ),
        "generated_pair_count": generated_pair_count,
        "safe_abstention_count": abstained_count,
        "generation_failure_count": rejected_count,
        "dimension_transition_counts": dict(
            sorted(transition_counts.items())
        ),
        "control_fail_dimension_counts": dict(
            sorted(control_fail_counts.items())
        ),
        "treatment_fail_dimension_counts": dict(
            sorted(treatment_fail_counts.items())
        ),
        "introduced_fail_dimension_counts": dict(
            sorted(
                Counter(
                    dim
                    for row in paired_rows
                    for dim in row.get("introduced_failures", [])
                ).items()
            )
        ),
        "resolved_fail_dimension_counts": dict(
            sorted(
                Counter(
                    dim
                    for row in paired_rows
                    for dim in row.get("resolved_failures", [])
                ).items()
            )
        ),
        "uncertainty_downgrade_dimension_counts": dict(
            sorted(
                Counter(
                    dim
                    for row in paired_rows
                    for dim in row.get("uncertainty_downgrades", [])
                ).items()
            )
        ),
        "paired_candidates": paired_rows,
        "same_grounded_premise_ablation": True,
        "relation_validity_has_rejection_authority": False,
        "production_selection_changed": False,
    }
    return body


def markdown(body: dict[str, Any]) -> str:
    lines = [
        "# RVG Paired Relation-Validity Evaluation v1",
        "",
        f"- control candidates: {body['control_candidate_count']}",
        f"- generated pairs: {body['generated_pair_count']}",
        f"- safe abstentions: {body['safe_abstention_count']}",
        f"- generation failures: {body['generation_failure_count']}",
        "",
        "## Dimension transitions",
        "",
    ]
    for key, value in body[
        "dimension_transition_counts"
    ].items():
        lines.append(f"- {key}: {value}")

    lines += [
        "",
        "## FAIL dimensions",
        "",
        "| Dimension | Control | Treatment |",
        "|---|---:|---:|",
    ]
    dims = sorted(
        set(body["control_fail_dimension_counts"])
        | set(body["treatment_fail_dimension_counts"])
    )
    for dim in dims:
        lines.append(
            "| "
            + dim
            + " | "
            + str(
                body["control_fail_dimension_counts"].get(
                    dim, 0
                )
            )
            + " | "
            + str(
                body["treatment_fail_dimension_counts"].get(
                    dim, 0
                )
            )
            + " |"
        )

    lines += [
        "",
        "## Candidates",
        "",
        "| Control | Treatment status | Outcome |",
        "|---|---|---|",
    ]
    for row in body["paired_candidates"]:
        lines.append(
            f"| {row['control_candidate_alias']} "
            f"| {row['treatment_status']} "
            f"| {row['paired_outcome']} |"
        )
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--control-audit", required=True, type=Path)
    p.add_argument("--treatment-audit", required=True, type=Path)
    p.add_argument("--mapping", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-md", required=True, type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    body = build_paired_summary(
        control_audit=load(
            args.control_audit.expanduser().resolve()
        ),
        treatment_audit=load(
            args.treatment_audit.expanduser().resolve()
        ),
        mapping=load(args.mapping.expanduser().resolve()),
    )
    write(args.output_json.expanduser().resolve(), body)
    args.output_md.expanduser().resolve().write_text(
        markdown(body),
        encoding="utf-8",
    )

    print("RVG paired validity summary complete")
    print(
        "generated pairs:",
        body["generated_pair_count"],
    )
    print(
        "safe abstentions:",
        body["safe_abstention_count"],
    )
    print(
        "transitions:",
        body["dimension_transition_counts"],
    )
    print(
        "control FAIL:",
        body["control_fail_dimension_counts"],
    )
    print(
        "treatment FAIL:",
        body["treatment_fail_dimension_counts"],
    )
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
