from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.materialization_survival_audit import (
    AUDITED_ARMS,
    build_cohort_survival_audit,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build a diagnostic-only materialization survival audit from an "
            "existing prospective-validation workspace. No LLM, retrieval, "
            "novelty, N9, feasibility, or production action is performed."
        )
    )
    p.add_argument("--prospective-output-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    return p


def main() -> int:
    args = parser().parse_args()
    root = args.prospective_output_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "materialization_survival.cohort.audit.json"
    )

    audit = build_cohort_survival_audit(prospective_output_root=root)
    _write(output, audit)

    print("===== MATERIALIZATION SURVIVAL AUDIT =====")
    print("artifact-observed/planned:", audit.artifact_observed_case_count, "/", audit.planned_case_count)
    print("execution-observed/planned:", audit.execution_observed_case_count, "/", audit.planned_case_count)
    print("artifact-only cases:", audit.artifact_only_case_count)
    print("partial materialization cases:", audit.partial_case_count)
    for arm in AUDITED_ARMS:
        print(
            arm,
            "| selected=", audit.arm_selected_candidate_counts.get(arm, 0),
            "| verification-ready=", audit.arm_verification_ready_counts.get(arm, 0),
            "| median yield=", audit.arm_verification_ready_fraction_medians.get(arm),
            "| downstream-complete cases=", audit.arm_downstream_complete_case_counts.get(arm, 0),
        )
        print(
            "   terminal statuses:",
            audit.arm_terminal_status_counts.get(arm, {}),
        )

    print("\n===== PER-CASE SURVIVAL MATRIX =====")
    for row in audit.per_case_survival_matrix:
        print(
            row.case_id,
            "| execution-observed=", row.execution_observed,
            "| execution-status=", row.execution_status,
        )
        for arm in row.arms:
            print(
                "  ", arm.arm,
                "| selected=", arm.selected_candidate_count,
                "| ready=", arm.verification_ready_count,
                "| yield=", round(arm.verification_ready_fraction, 3),
                "| lineage=", "COMPLETE" if arm.lineage_resolution_complete else "INCOMPLETE",
                "| statuses=", arm.terminal_status_counts,
            )

    print("\n===== COMMON-CASE PAIRED MATERIALIZATION =====")
    print(
        "common cases:",
        audit.common_materialization_case_count,
        audit.common_materialization_case_ids,
    )
    for row in audit.paired_comparisons:
        print(
            row.left_arm, "->", row.right_arm,
            "| cases=", row.case_count,
            "| median left=", row.median_left_yield,
            "| median right=", row.median_right_yield,
            "| median delta(right-left)=", row.median_delta_right_minus_left,
            "| right higher/equal/lower=",
            f"{row.right_higher_case_count}/{row.equal_case_count}/{row.right_lower_case_count}",
        )

    print("\n===== ORIGIN / OPERATOR / PROFILE TRANSITIONS =====")
    for row in audit.transition_cells:
        print(
            row.arm,
            "|", row.origin,
            "|", row.operator_or_form,
            "|", row.selection_profile,
            "| selected=", row.selected_count,
            "| ready=", row.verification_ready_count,
            "| yield=", round(row.verification_ready_fraction, 3),
            "| statuses=", row.status_counts,
        )

    print("\n===== REJECTION / ABSTENTION TAXONOMY =====")
    if not audit.issue_taxonomy:
        print("NONE")
    for row in audit.issue_taxonomy:
        print(
            row.issue_code,
            "| count=", row.count,
            "| status=", row.status_counts,
            "| arm=", row.arm_counts,
            "| operator=", row.operator_or_form_counts,
            "| profile=", row.profile_counts,
        )

    print("raw issue codes:", audit.raw_issue_code_counts)
    print(
        "taxonomy coverage:",
        audit.taxonomy_classified_failure_count,
        "/",
        audit.non_materialized_count,
        "| unclassified=", audit.taxonomy_unclassified_failure_count,
        "| exhaustive=", audit.taxonomy_exhaustive,
    )
    print("generation errors:", audit.generation_error_counts)
    incomplete_lineage = {
        case.case_id: {
            arm.arm: arm.unresolved_candidate_ids
            for arm in case.arms
            if not arm.lineage_resolution_complete
        }
        for case in audit.cases
        if not case.lineage_resolution_complete
    }
    print("lineage gaps:", incomplete_lineage)
    print("DIAGNOSTIC_ONLY=True")
    print("SCIENTIFIC_SUPERIORITY_ESTABLISHED=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
