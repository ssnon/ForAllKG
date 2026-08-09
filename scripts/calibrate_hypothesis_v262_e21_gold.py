from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _expectation(case: dict, dimension: str) -> dict:
    rows = [
        row
        for row in case["expectations"]
        if row["dimension"] == dimension
    ]
    if len(rows) != 1:
        raise ValueError(
            f"{case['case_id']}: expected exactly one {dimension!r} expectation"
        )
    return rows[0]


def _worksheet_dimension(case: dict, dimension: str) -> dict:
    rows = [
        row
        for row in case["dimensions"]
        if row["dimension"] == dimension
    ]
    if len(rows) != 1:
        raise ValueError(
            f"{case['case_id']}: expected exactly one {dimension!r} worksheet row"
        )
    return rows[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the evidence-backed E2.1 semantic gold calibration. "
            "This encodes only observed, human-reviewed gray-zone bands and "
            "keeps structurally inactive candidate calibration canonical as "
            "not_applicable."
        )
    )
    parser.add_argument(
        "--k9-spec",
        default="benchmarks/hypothesis_v262/real_gold_spec.k9.example.json",
    )
    parser.add_argument(
        "--worksheet",
        default="benchmarks/hypothesis_v262/e2_review_worksheet.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k9_path = Path(args.k9_spec).resolve()
    worksheet_path = Path(args.worksheet).resolve()

    k9 = _load(k9_path)
    worksheet = _load(worksheet_path)

    if len(k9.get("cases", [])) != 1:
        raise SystemExit("K9 seed spec must contain exactly one case")
    k9_case = k9["cases"][0]
    if k9_case.get("case_id") != "real_k9_coordination_charge_transfer":
        raise SystemExit(
            "Unexpected K9 case_id: " + str(k9_case.get("case_id"))
        )

    # Structural applicability stays strict. Raw critic PASS is normalized
    # deterministically by E2.1 only when candidate dependency is inactive.
    row = _expectation(k9_case, "candidate_calibration")
    row["allowed_verdicts"] = ["not_applicable"]
    row["note"] = (
        "No candidate/provisional premise dependency is present. "
        "The canonical gold verdict is not_applicable. E2.1 may deterministically "
        "normalize a raw critic pass to not_applicable when the portfolio "
        "structurally contains no candidate-dependent hypothesis; warning/fail "
        "remain mismatches."
    )

    row = _expectation(k9_case, "inferential_proportionality")
    row["allowed_verdicts"] = ["pass", "warning"]
    row["note"] = (
        "Repeated human-reviewed critic runs place this bounded charge-transfer "
        "mediation hypothesis on the pass/warning boundary. Pass is reasonable "
        "because the mechanism is explicit, bounded, and testable; warning is "
        "reasonable because mediation specificity exceeds the supplied evidence. "
        "Fail is not acceptable."
    )

    cases = {row["case_id"]: row for row in worksheet.get("cases", [])}
    required = {
        "e2_candidate_live",
        "e2_alignment_live",
        "e2_partial_live",
        "e2_abstention_live",
    }
    missing = sorted(required - set(cases))
    if missing:
        raise SystemExit(f"Worksheet is missing E2 cases: {missing}")

    candidate = cases["e2_candidate_live"]
    row = _worksheet_dimension(candidate, "candidate_calibration")
    row["human_allowed_verdicts"] = ["pass", "warning"]
    row["human_note"] = (
        "Candidate dependence is explicitly preserved and never promoted to "
        "established evidence, so pass is reasonable. The proposed functional "
        "role is nevertheless stronger than the provisional association alone, "
        "so warning is also reasonable. Fail and not_applicable are not acceptable "
        "because candidate evidence is structurally active in this case."
    )

    alignment = cases["e2_alignment_live"]
    row = _worksheet_dimension(alignment, "inferential_proportionality")
    row["human_allowed_verdicts"] = ["pass", "warning"]
    row["human_note"] = (
        "The cross-paper coupling is explicitly hypothetical and bounded by "
        "assumptions, making pass reasonable. Functional coupling, directional "
        "dependence, and possible intermediate mechanistic steps are more specific "
        "than the separate reported premises, making warning reasonable. "
        "Fail is not acceptable."
    )

    partial = cases["e2_partial_live"]
    row = _worksheet_dimension(partial, "candidate_calibration")
    row["human_allowed_verdicts"] = ["not_applicable"]
    row["human_note"] = (
        "No candidate/provisional premise is present. The canonical human gold "
        "verdict remains not_applicable; a raw critic pass may only be treated "
        "as equivalent by the deterministic E2.1 applicability rule."
    )

    row = _worksheet_dimension(partial, "inferential_proportionality")
    row["human_allowed_verdicts"] = ["pass", "warning"]
    row["human_note"] = (
        "The association-to-partial-mediation step is stronger than the supplied "
        "premise and may reasonably merit a warning, but it is also a bounded, "
        "explicitly hypothetical inference with stated assumptions, so pass is "
        "acceptable. Fail is not acceptable."
    )

    if args.dry_run:
        print("E2.1 calibration dry-run passed.")
        print("Would update:", k9_path)
        print("Would update:", worksheet_path)
        return

    _write(k9_path, k9)
    _write(worksheet_path, worksheet)
    print("E2.1 semantic gold calibration applied")
    print("Updated:", k9_path)
    print("Updated:", worksheet_path)
    print(
        "Important: regenerate the frozen five-case gold after this calibration; "
        "do not edit the frozen gold JSON directly."
    )


if __name__ == "__main__":
    main()
