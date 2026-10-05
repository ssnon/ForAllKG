from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def _by_validity_alias(
    synthesis: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in synthesis.get("candidates", []):
        if not isinstance(row, dict):
            continue
        validity = row.get("validity") or {}
        alias = str(validity.get("candidate_alias") or "")
        if alias:
            result[alias] = row
    return result


def build_summary(
    *,
    control_synthesis: dict[str, Any],
    treatment_synthesis: dict[str, Any],
    mapping: dict[str, Any],
) -> dict[str, Any]:
    control_by = _by_validity_alias(control_synthesis)
    treatment_by = _by_validity_alias(treatment_synthesis)

    control_status = Counter()
    treatment_status = Counter()
    transitions = Counter()
    rows = []

    for record in mapping.get("records", []):
        if not isinstance(record, dict):
            continue

        control_alias = str(
            record.get("control_candidate_alias") or ""
        )
        status = str(record.get("treatment_status") or "")

        control = control_by.get(control_alias)
        if control is None:
            raise RuntimeError(
                "missing control synthesis row: " + control_alias
            )

        cnov = str(
            (control.get("fresh_search_novelty") or {}).get(
                "status"
            )
            or ""
        )
        if cnov:
            control_status[cnov] += 1

        row = {
            "control_candidate_alias": control_alias,
            "control_hypothesis_id": record.get(
                "control_hypothesis_id"
            ),
            "control_novelty_status": cnov,
            "treatment_status": status,
            "identification_assessment": record.get(
                "identification_assessment"
            ),
            "treatment_hypothesis_id": record.get(
                "treatment_hypothesis_id"
            ),
            "treatment_novelty_status": None,
            "novelty_transition": None,
            "same_grounded_premise": (
                record.get("exact_premise_identity") is True
            ),
        }

        if status == "ABSTAINED_NOT_IDENTIFIABLE":
            row["novelty_transition"] = (
                cnov + " -> SAFE_ABSTENTION_NOT_IDENTIFIABLE"
            )
            transitions[row["novelty_transition"]] += 1
            rows.append(row)
            continue

        if status != "GENERATED_TREATMENT":
            row["novelty_transition"] = (
                cnov + " -> UNEVALUABLE_GENERATION_FAILURE"
            )
            transitions[row["novelty_transition"]] += 1
            rows.append(row)
            continue

        treatment_alias = str(
            record.get("treatment_audit_alias") or ""
        )
        treatment = treatment_by.get(treatment_alias)
        if treatment is None:
            raise RuntimeError(
                "missing treatment synthesis row: "
                + treatment_alias
            )

        tnov = str(
            (treatment.get("fresh_search_novelty") or {}).get(
                "status"
            )
            or ""
        )
        if tnov:
            treatment_status[tnov] += 1

        transition = f"{cnov} -> {tnov}"
        transitions[transition] += 1
        row["treatment_novelty_status"] = tnov
        row["novelty_transition"] = transition
        rows.append(row)

    return {
        "schema_version": "rvg-paired-fresh-search-novelty-summary-v1",
        "control_case_id": mapping.get("control_case_id"),
        "treatment_case_id": mapping.get("treatment_case_id"),
        "control_status_counts": dict(sorted(control_status.items())),
        "treatment_status_counts_generated_only": dict(
            sorted(treatment_status.items())
        ),
        "transition_counts": dict(sorted(transitions.items())),
        "paired_candidates": rows,
        "novelty_status_is_ordinal": False,
        "same_grounded_premise_ablation": True,
        "external_prior_art_as_positive_premise": False,
        "production_selection_changed": False,
    }


def markdown(body: dict[str, Any]) -> str:
    lines = [
        "# RVG Paired Fresh-Search Novelty v1",
        "",
        "> Novelty status is treated as categorical, not ordinal.",
        "",
        "## Status counts",
        "",
        "### Control",
    ]
    for key, value in body["control_status_counts"].items():
        lines.append(f"- {key}: {value}")

    lines += ["", "### Treatment (generated only)"]
    for key, value in body[
        "treatment_status_counts_generated_only"
    ].items():
        lines.append(f"- {key}: {value}")

    lines += [
        "",
        "## Candidate transitions",
        "",
        "| Control | Novelty transition | Identification |",
        "|---|---|---|",
    ]
    for row in body["paired_candidates"]:
        lines.append(
            f"| {row['control_candidate_alias']} "
            f"| {row['novelty_transition']} "
            f"| {row['identification_assessment']} |"
        )
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--control-synthesis", required=True, type=Path)
    p.add_argument("--treatment-synthesis", required=True, type=Path)
    p.add_argument("--mapping", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-md", required=True, type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    body = build_summary(
        control_synthesis=load(
            args.control_synthesis.expanduser().resolve()
        ),
        treatment_synthesis=load(
            args.treatment_synthesis.expanduser().resolve()
        ),
        mapping=load(args.mapping.expanduser().resolve()),
    )
    write(args.output_json.expanduser().resolve(), body)
    args.output_md.expanduser().resolve().write_text(
        markdown(body),
        encoding="utf-8",
    )

    print("RVG paired fresh-search novelty summary complete")
    print("control:", body["control_status_counts"])
    print(
        "treatment:",
        body["treatment_status_counts_generated_only"],
    )
    print("transitions:", body["transition_counts"])
    print("NOVELTY_STATUS_IS_ORDINAL=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
