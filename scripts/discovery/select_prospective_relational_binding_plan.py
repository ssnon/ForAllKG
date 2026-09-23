from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_relational_execution_plan import (
    select_prospective_relational_hypothesis,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select at most one binding-ready Alpha6 final hypothesis using "
            "pre-endpoint structural readiness only. Old-N10 status, endpoint "
            "binding, external novelty outcome, and verifier results are not "
            "selection features."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--selection-output", required=True, type=Path)
    parser.add_argument("--selected-plan-output", required=True, type=Path)
    args = parser.parse_args()

    source = RelationalAtomicBindingPlan.model_validate_json(
        args.plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    selection, selected_plan = select_prospective_relational_hypothesis(
        binding_plan=source,
    )

    write_json_exclusive(
        args.selection_output.expanduser().resolve(),
        selection,
    )
    if selected_plan is not None:
        write_json_exclusive(
            args.selected_plan_output.expanduser().resolve(),
            selected_plan,
        )

    print("Prospective relational hypothesis selection complete")
    print("Status:", selection.status)
    print("Eligible finals:", selection.eligible_final_hypothesis_ids)
    print("Selection rule:", selection.selection_rule)
    print("Endpoint outcome used for selection: false")
    print("Old N10 status used for selection: false")
    print("External novelty outcome used for selection: false")
    print("Verifier outcome used for selection: false")
    if selected_plan is not None:
        print("Selected final:", selection.selected_final_hypothesis_id)
        print("Selected plan:", args.selected_plan_output.expanduser().resolve())
        print("External report:", selection.selected_external_report)
    print("Selection report:", args.selection_output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
