from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    PreVerifierContractGateV2Report,
    build_preverifier_contract_gate_v2,
)
from pipeline_core.discovery.prospective_regeneration_binding_gate_v2 import (
    build_compatibility_manifest_v2,
    build_compatibility_portfolio_v2,
    build_reachability_comparison_v2,
    build_regeneration_relational_atomic_binding_plan_v2,
)
from pipeline_core.discovery.prospective_regeneration_downstream_n10_v2 import (
    ProspectiveRegenerationExternalN10ReportV2,
)
from pipeline_core.discovery.prospective_regeneration_downstream_semantic_v2 import (
    ProspectiveRegenerationDownstreamSemanticReportV2,
)
from pipeline_core.discovery.prospective_routed_execution_plan_v2 import (
    ProspectiveRoutedExecutionPlanV2,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the regeneration-specific relational atomic binding plan, "
            "run deterministic Pre-Verifier Contract Gate v2, and compare "
            "initial/primary/regenerated strict reachability. No endpoint "
            "binding or verifier is executed."
        )
    )
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--semantic-report", required=True, type=Path)
    parser.add_argument("--external-n10-report", required=True, type=Path)
    args = parser.parse_args()

    execution = ProspectiveRoutedExecutionPlanV2.model_validate_json(
        args.execution_plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    cases = [row for row in execution.cases if row.case_id == args.case_id]
    if len(cases) != 1:
        raise ValueError("case is not present exactly once in execution plan")
    case = cases[0]

    semantic = ProspectiveRegenerationDownstreamSemanticReportV2.model_validate_json(
        args.semantic_report.expanduser().resolve().read_text(encoding="utf-8")
    )
    external_n10 = ProspectiveRegenerationExternalN10ReportV2.model_validate_json(
        args.external_n10_report.expanduser().resolve().read_text(encoding="utf-8")
    )
    if semantic.case_id != case.case_id or external_n10.case_id != case.case_id:
        raise ValueError("case lineage mismatch")
    if external_n10.source_semantic_report_id != semantic.report_id:
        raise ValueError("external-N10/semantic report ID mismatch")
    if external_n10.source_semantic_report_sha256 != semantic.report_sha256:
        raise ValueError("external-N10/semantic report SHA mismatch")

    semantic_by_source = {
        row.source_final_hypothesis_id: row
        for row in semantic.lineages
        if row.eligible_for_external_novelty
    }
    if len(semantic_by_source) != external_n10.lineage_count:
        raise ValueError("semantic/external-N10 lineage count mismatch")

    certification_paths = []
    source_portfolios = []
    for lineage in external_n10.lineages:
        semantic_row = semantic_by_source.get(lineage.source_final_hypothesis_id)
        if semantic_row is None:
            raise ValueError("external-N10 lineage absent from semantic report")
        downstream_dir = Path(semantic_row.downstream_dir).expanduser().resolve()
        certification = downstream_dir / "n10.certification.json"
        if not certification.is_file():
            raise ValueError(
                "missing regeneration N10 certification: " + str(certification)
            )
        certification_paths.append(certification)

        portfolio_path = Path(
            semantic_row.regenerated_portfolio_path
        ).expanduser().resolve()
        source_portfolios.append(
            HypothesisPortfolio.model_validate_json(
                portfolio_path.read_text(encoding="utf-8")
            )
        )

    routed = Path(case.routed_dispatch_path).expanduser().resolve().parent
    output_dir = routed / "regeneration_binding_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    compatibility_portfolio_path = (
        output_dir / "compatibility.candidate.portfolio.json"
    )
    compatibility_manifest_path = (
        output_dir / "compatibility.certification_manifest.json"
    )
    plan_path = output_dir / "relational_atomic_binding_plan.regeneration.json"
    gate_path = output_dir / "contract_gate_v2.regeneration.json"
    comparison_path = output_dir / "reachability_comparison.v2.json"

    for path in (
        compatibility_portfolio_path,
        compatibility_manifest_path,
        plan_path,
        gate_path,
        comparison_path,
    ):
        if path.exists():
            raise ValueError(
                "regeneration binding artifact is write-once: " + str(path)
            )

    compatibility_portfolio = build_compatibility_portfolio_v2(
        portfolios=source_portfolios,
    )
    write_json_exclusive(
        compatibility_portfolio_path,
        compatibility_portfolio,
    )

    manifest = build_compatibility_manifest_v2(
        external_n10=external_n10,
        compatibility_portfolio=compatibility_portfolio,
        certification_paths=certification_paths,
    )
    write_json_exclusive(compatibility_manifest_path, manifest)

    plan = build_regeneration_relational_atomic_binding_plan_v2(
        run_dir=Path(case.run_dir),
        external_n10=external_n10,
        certification_paths=certification_paths,
        compatibility_portfolio_path=compatibility_portfolio_path,
        compatibility_manifest_path=compatibility_manifest_path,
    )
    write_json_exclusive(plan_path, plan)

    gate = build_preverifier_contract_gate_v2(plan=plan)
    write_json_exclusive(gate_path, gate)

    initial_gate_path = routed / "pre_route_contract_gate_v2.json"
    primary_gate_path = routed / "primary_v2" / "contract_gate_v2.primary.json"
    if not initial_gate_path.is_file():
        raise ValueError("missing initial Gate v2 report")
    if not primary_gate_path.is_file():
        raise ValueError("missing primary Gate v2 report")

    initial_gate = PreVerifierContractGateV2Report.model_validate_json(
        initial_gate_path.read_text(encoding="utf-8")
    )
    primary_gate = PreVerifierContractGateV2Report.model_validate_json(
        primary_gate_path.read_text(encoding="utf-8")
    )

    comparison = build_reachability_comparison_v2(
        case_id=case.case_id,
        external_n10=external_n10,
        binding_plan=plan,
        initial_gate=initial_gate,
        primary_gate=primary_gate,
        regenerated_gate=gate,
    )
    write_json_exclusive(comparison_path, comparison)

    print("Regeneration binding-plan + Gate-v2 comparison complete")
    print("Case:", comparison.case_id)
    print("Binding plan:", plan.plan_id)
    print(
        "Hypotheses ready/not-ready:",
        plan.ready_hypothesis_count,
        "/",
        plan.not_ready_hypothesis_count,
    )
    print(
        "Claims/binding-ready/novelty-bearing-ready:",
        plan.claim_count,
        "/",
        plan.binding_ready_claim_count,
        "/",
        plan.novelty_bearing_binding_ready_claim_count,
    )
    print(
        "Gate claims ready/not-ready/novelty-ready:",
        gate.ready_claim_count,
        "/",
        gate.not_ready_claim_count,
        "/",
        gate.novelty_bearing_ready_claim_count,
    )
    print("Router hints:", gate.router_hint_counts)
    print()
    print(
        "Strict hypothesis reach initial -> primary -> regeneration:",
        comparison.initial_gate_ready_hypothesis_count,
        "->",
        comparison.primary_gate_ready_hypothesis_count,
        "->",
        comparison.regenerated_gate_ready_hypothesis_count,
        "/",
        comparison.lineage_count,
    )
    print(
        "N10 certified/unresolved/rejected:",
        comparison.n10_certified_count,
        "/",
        comparison.n10_unresolved_count,
        "/",
        comparison.n10_rejected_count,
    )
    for row in comparison.lineages:
        print(
            row.source_final_hypothesis_id,
            "->",
            row.regenerated_hypothesis_id,
            "| N10=",
            row.n10_certification_status,
            "| gate-ready=",
            row.gate_ready,
            "| novelty-ready claims=",
            row.novelty_bearing_gate_ready_claim_ids,
        )
    print()
    print("Endpoint binding performed: false")
    print("Verifier performed: false")
    print("Second regeneration performed: false")
    print("Scientific content mutated: false")
    print("Plan:", plan_path)
    print("Gate:", gate_path)
    print("Comparison:", comparison_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
