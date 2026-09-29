from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_evidence_diversity import (
    HypothesisEvidenceDiversityReport,
)
from pipeline_core.discovery.portfolio_diversity_shadow import (
    build_portfolio_diversity_shadow_report,
)
from pipeline_core.discovery.prospective_novelty_validation_execution import (
    ProspectiveNoveltyExecutionPlan,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _external_statuses(path: Path) -> list[str]:
    if not path.is_file():
        return []

    payload = _load(path)
    statuses = []

    for row in payload.get("cards", []):
        if not isinstance(row, dict):
            continue
        value = row.get("status")
        if value is not None:
            statuses.append(str(value))

    return statuses


def _conceptual_status(run_dir: Path) -> dict[str, Any]:
    # S220 intentionally established the normalized contract before a generic
    # evaluator was wired into every domain. Do not turn "not measured" into
    # scientific absence.
    candidates = list(
        run_dir.glob(
            "**/*conceptual*knownness*.json"
        )
    )

    if not candidates:
        return {
            "measurement_status": "NOT_RUN",
            "first_gap_levels": [],
            "artifacts": [],
        }

    levels = []
    artifacts = []

    for path in sorted(candidates):
        try:
            payload = _load(path)
        except Exception:
            continue

        artifacts.append(str(path))

        for row in payload.get("records", []):
            if not isinstance(row, dict):
                continue
            value = row.get("first_gap_level")
            if value is not None:
                levels.append(str(value))

    return {
        "measurement_status":
            "AVAILABLE" if artifacts else "NOT_RUN",
        "first_gap_levels":
            levels,
        "artifacts":
            artifacts,
    }


def _portfolio_diversity(
    *,
    run_dir: Path,
    output: Path,
) -> dict[str, Any]:
    source = (
        run_dir
        / "hypothesis_axis_a4.evidence_diversity.json"
    )

    if not source.is_file():
        return {
            "measurement_status": "NOT_AVAILABLE",
        }

    evidence = (
        HypothesisEvidenceDiversityReport.model_validate_json(
            source.read_text(encoding="utf-8")
        )
    )

    report = build_portfolio_diversity_shadow_report(
        evidence=evidence,
    )

    output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "measurement_status": "AVAILABLE",
        "evidence_recommendation":
            report.evidence_recommendation,
        "operator_recommendation":
            report.operator_recommendation,
        "eligible_statement_count":
            report.eligible_statement_count,
        "used_statement_count":
            report.used_statement_count,
        "unused_eligible_statement_count":
            report.unused_eligible_statement_count,
        "exact_premise_set_duplicate_group_count":
            report.exact_premise_set_duplicate_group_count,
        "max_pairwise_statement_jaccard":
            report.max_pairwise_statement_jaccard,
        "operator_opportunity_count":
            report.operator_opportunity_count,
        "operator_measurement_scope":
            "NO_NOVELTY_GAP_PLAN_SUPPLIED",
        "artifact":
            str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect completed prospective novelty-validation runs into one "
            "missingness-aware report. Unmeasured shadow dimensions remain "
            "NOT_RUN/NOT_AVAILABLE rather than being imputed."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    plan = ProspectiveNoveltyExecutionPlan.model_validate_json(
        args.plan.read_text(encoding="utf-8")
    )

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(
            "prospective collector output is write-once; use a fresh path"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for case in plan.cases:
        run_dir = Path(case.run_dir)
        manifest_path = run_dir / "e2e_runner.manifest.json"

        if not manifest_path.is_file():
            rows.append(
                {
                    "prospective_case_id":
                        case.prospective_case_id,
                    "source_case_id":
                        case.source_case_id,
                    "domain_profile_id":
                        case.domain_profile_id,
                    "case_role":
                        case.case_role,
                    "run_status":
                        "NOT_RUN",
                    "evaluation_focus":
                        case.evaluation_focus,
                }
            )
            continue

        manifest = _load(manifest_path)

        direct = manifest.get(
            "direct_relationpattern_task_shadow",
            {},
        )
        direct_ho = manifest.get(
            "direct_higher_order_shadow",
            {},
        )
        direct_downstream = manifest.get(
            "direct_higher_order_downstream_shadow",
            {},
        )
        research_value = manifest.get(
            "research_value_shadow",
            {},
        )

        external = (
            run_dir
            / "external_novelty_a52.report.json"
        )

        diversity_out = (
            run_dir
            / "prospective.portfolio_diversity.shadow.json"
        )
        diversity = _portfolio_diversity(
            run_dir=run_dir,
            output=diversity_out,
        )

        conceptual = _conceptual_status(
            run_dir
        )

        row = {
            "prospective_case_id":
                case.prospective_case_id,
            "source_case_id":
                case.source_case_id,
            "domain_profile_id":
                case.domain_profile_id,
            "case_role":
                case.case_role,
            "evaluation_focus":
                case.evaluation_focus,
            "run_status":
                str(manifest.get("status", "UNKNOWN")),
            "grounding_algorithm_used":
                manifest.get("grounding_algorithm_used"),
            "initial_hypothesis_count":
                manifest.get("initial_hypothesis_count"),
            "final_hypothesis_count":
                manifest.get("final_hypothesis_count"),

            "direct_relation_recovery": {
                "measurement_status":
                    (
                        "AVAILABLE"
                        if direct.get("enabled")
                        else "NOT_RUN"
                    ),
                "accepted_relationpattern_count":
                    direct.get("accepted_relationpattern_count"),
                "stable_direct_count":
                    direct.get("stable_direct_count"),
                "stable_subordinate_count":
                    direct.get("stable_subordinate_count"),
                "stable_task_replacing_count":
                    direct.get("stable_task_replacing_count"),
                "unresolved_count":
                    direct.get("unresolved_count"),
            },

            "direct_higher_order": {
                "measurement_status":
                    (
                        "AVAILABLE"
                        if direct_ho.get("enabled")
                        else "NOT_RUN"
                    ),
                "eligible_modifier_count":
                    direct_ho.get("eligible_modifier_count"),
                "higher_order_topology_count":
                    direct_ho.get("higher_order_topology_count"),
                "selected_context_count":
                    direct_ho.get("selected_context_count"),
                "proposed_count":
                    direct_ho.get("proposed_count"),
            },

            "direct_higher_order_downstream": {
                "measurement_status":
                    (
                        "AVAILABLE"
                        if direct_downstream.get("enabled")
                        else "NOT_RUN"
                    ),
                "semantic_attempted_count":
                    direct_downstream.get("semantic_attempted_count"),
                "semantic_accepted_count":
                    direct_downstream.get("semantic_accepted_count"),
                "external_novelty_completed_count":
                    direct_downstream.get(
                        "external_novelty_completed_count"
                    ),
                "structural_view_count":
                    direct_downstream.get("structural_view_count"),
            },

            "main_external_novelty": {
                "measurement_status":
                    (
                        "AVAILABLE"
                        if external.is_file()
                        else "NOT_AVAILABLE"
                    ),
                "card_statuses":
                    _external_statuses(external),
                "artifact":
                    str(external)
                    if external.is_file()
                    else None,
            },

            "conceptual_knownness":
                conceptual,

            "portfolio_diversity":
                diversity,

            "research_value": {
                "measurement_status":
                    (
                        "AVAILABLE"
                        if research_value.get("status") == "complete"
                        else (
                            "UNSUPPORTED_DOMAIN"
                            if research_value.get("status")
                            == "not_supported_without_feasibility"
                            else "NOT_RUN"
                        )
                    ),
                "assessed_count":
                    research_value.get("assessed_count"),
                "hypothesis_count":
                    research_value.get("hypothesis_count"),
                "artifact":
                    research_value.get("artifact"),
                "capability_expected":
                    case.research_value_capability_expected,
            },
        }

        rows.append(row)

    payload = {
        "schema_version":
            "prospective-novelty-validation-collector-v1",
        "source_plan_id":
            plan.plan_id,
        "source_freeze_id":
            plan.source_freeze_id,
        "case_count":
            len(plan.cases),
        "collected_case_count":
            sum(
                row.get("run_status") != "NOT_RUN"
                for row in rows
            ),
        "cases":
            rows,
        "interpretation_policy": {
            "missing_measurement_is_scientific_negative":
                False,
            "not_run_is_zero":
                False,
            "production_selection_changed":
                False,
            "overall_winner_computed":
                False,
        },
    }

    output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Prospective novelty-validation collection complete")
    print("Plan:", plan.plan_id)
    print(
        "Collected:",
        payload["collected_case_count"],
        "/",
        payload["case_count"],
    )
    print()
    for row in rows:
        print(
            row["source_case_id"],
            "| run=",
            row["run_status"],
        )
        if row["run_status"] != "NOT_RUN":
            print(
                " direct stable=",
                row["direct_relation_recovery"].get(
                    "stable_direct_count"
                ),
                "| HO proposed=",
                row["direct_higher_order"].get(
                    "proposed_count"
                ),
                "| conceptual=",
                row["conceptual_knownness"].get(
                    "measurement_status"
                ),
                "| diversity=",
                row["portfolio_diversity"].get(
                    "evidence_recommendation"
                ),
                "| research-value=",
                row["research_value"].get(
                    "measurement_status"
                ),
            )
    print()
    print("Missing measurement interpreted as negative: false")
    print("Overall winner computed: false")
    print("Production selection changed: false")
    print("Artifact:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
