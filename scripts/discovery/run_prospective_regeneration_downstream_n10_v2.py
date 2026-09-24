from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.prospective_regeneration_downstream_n10_v2 import (
    RegenerationExternalN10LineageResultV2,
    RegenerationNoveltyArtifactBundleV2,
    build_external_n10_execution_report_v2,
    build_regeneration_novelty_certification_v2,
)
from pipeline_core.discovery.prospective_regeneration_downstream_semantic_v2 import (
    ProspectiveRegenerationDownstreamSemanticReportV2,
)
from pipeline_core.discovery.prospective_regeneration_downstream_v2 import (
    ProspectiveRegenerationDownstreamV2Freeze,
)
from pipeline_core.discovery.prospective_routed_execution_plan_v2 import (
    ProspectiveRoutedExecutionPlanV2,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def _run(module: str, argv: list[str]) -> None:
    subprocess.run([sys.executable, "-m", module, *argv], check=True)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--downstream-freeze", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--semantic-report", required=True, type=Path)
    args = parser.parse_args()

    execution = ProspectiveRoutedExecutionPlanV2.model_validate_json(
        args.execution_plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    downstream = ProspectiveRegenerationDownstreamV2Freeze.model_validate_json(
        args.downstream_freeze.expanduser().resolve().read_text(encoding="utf-8")
    )
    semantic = ProspectiveRegenerationDownstreamSemanticReportV2.model_validate_json(
        args.semantic_report.expanduser().resolve().read_text(encoding="utf-8")
    )

    if execution.source_regeneration_downstream_freeze_id != downstream.freeze_id:
        raise ValueError("execution-plan/downstream-freeze ID mismatch")
    if execution.source_regeneration_downstream_freeze_sha256 != downstream.freeze_sha256:
        raise ValueError("execution-plan/downstream-freeze SHA mismatch")
    if semantic.source_downstream_freeze_id != downstream.freeze_id:
        raise ValueError("semantic-report/downstream-freeze ID mismatch")
    if semantic.source_downstream_freeze_sha256 != downstream.freeze_sha256:
        raise ValueError("semantic-report/downstream-freeze SHA mismatch")

    cases = [row for row in execution.cases if row.case_id == args.case_id]
    if len(cases) != 1:
        raise ValueError("case is not present exactly once in execution plan")
    case = cases[0]
    if semantic.case_id != case.case_id:
        raise ValueError("semantic report/case mismatch")

    run_dir = Path(case.run_dir).expanduser().resolve()
    context_path = Path(case.hypothesis_context_path).expanduser().resolve()
    provider_plan = run_dir / "literature_provider_plan.json"
    if not provider_plan.is_file():
        raise ValueError("frozen initial-run literature provider plan missing: " + str(provider_plan))

    lineage_results = []

    for semantic_row in semantic.lineages:
        if not semantic_row.eligible_for_external_novelty:
            continue

        portfolio_path = Path(semantic_row.regenerated_portfolio_path).expanduser().resolve()
        portfolio = HypothesisPortfolio.model_validate_json(
            portfolio_path.read_text(encoding="utf-8")
        )
        if len(portfolio.hypotheses) != 1:
            raise ValueError("regenerated external-N10 lineage must contain one hypothesis")
        regenerated_id = portfolio.hypotheses[0].hypothesis_id

        downstream_dir = Path(semantic_row.downstream_dir).expanduser().resolve()
        external_prefix = downstream_dir / "external_novelty"
        external_plan = Path(str(external_prefix) + ".claims_queries.json")
        external_prior = Path(str(external_prefix) + ".prior_art.json")
        external_report = Path(str(external_prefix) + ".report.json")
        intake = downstream_dir / "n9.intake.json"
        full = downstream_dir / "n9.full.json"
        n10_candidate = downstream_dir / "n10.production.candidate.json"
        n10_production = downstream_dir / "n10.production.json"
        candidate_portfolio = downstream_dir / "n10.candidate.portfolio.json"
        certification_report = downstream_dir / "n10.certification.json"
        certified_portfolio = downstream_dir / "n10.certified.portfolio.json"

        outputs = [
            external_plan, external_prior, external_report,
            intake, full, n10_candidate, n10_production,
            candidate_portfolio, certification_report, certified_portfolio,
        ]
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise ValueError("external-N10 lineage outputs are write-once; existing=" + repr(existing))

        before_sha = _sha256_file(portfolio_path)

        _run(
            "scripts.discovery.run_external_novelty",
            [
                "--portfolio", str(portfolio_path),
                "--domain-profile", portfolio.domain_profile_id,
                "--model", case.critic_model,
                "--base-url", execution.settings.base_url,
                "--api-key-env", execution.settings.api_key_env,
                "--provider-plan", str(provider_plan),
                "--results-per-query", str(downstream.policy.external_novelty_results_per_query),
                "--max-ranked-works", str(min(8, downstream.policy.max_review_works_per_claim)),
                "--output-prefix", str(external_prefix),
                "--save-prompts",
            ],
        )

        _run(
            "scripts.discovery.build_nonobviousness_shadow",
            [
                "--query-plan", str(external_plan),
                "--external-report", str(external_report),
                "--portfolio", str(portfolio_path),
                "--output", str(intake),
            ],
        )

        intake_payload = _load_json(intake)
        ready_count = sum(
            len(row.get("ready_for_closure_claim_ids", []))
            for row in intake_payload.get("hypotheses", [])
            if isinstance(row, dict)
        )

        _run(
            "scripts.discovery.run_nonobviousness_full_shadow",
            [
                "--query-plan", str(external_plan),
                "--external-report", str(external_report),
                "--external-prior-art", str(external_prior),
                "--portfolio", str(portfolio_path),
                "--hypothesis-context", str(context_path),
                "--intake-shadow", str(intake),
                "--provider-plan", str(provider_plan),
                "--domain-profile", portfolio.domain_profile_id,
                "--model", case.critic_model,
                "--base-url", execution.settings.base_url,
                "--api-key-env", execution.settings.api_key_env,
                "--results-per-query", str(downstream.policy.external_novelty_results_per_query),
                "--max-ranked-works", str(min(8, downstream.policy.max_review_works_per_claim)),
                "--max-ready-claims", str(max(1, ready_count)),
                "--output", str(full),
            ],
        )

        _run(
            "scripts.discovery.build_nonobviousness_production_gate_v2_candidate",
            [
                "--query-plan", str(external_plan),
                "--intake-shadow", str(intake),
                "--full-shadow", str(full),
                "--output", str(n10_candidate),
            ],
        )

        _run(
            "scripts.discovery.build_nonobviousness_post_generation_production_gate_v2",
            ["--candidate-gate", str(n10_candidate), "--output", str(n10_production)],
        )

        artifact = RegenerationNoveltyArtifactBundleV2(
            source_final_hypothesis_id=semantic_row.source_final_hypothesis_id,
            regenerated_hypothesis_id=regenerated_id,
            source_portfolio=str(portfolio_path),
            query_plan=str(external_plan),
            prior_art=str(external_prior),
            external_report=str(external_report),
            n9_intake=str(intake),
            n9_full_closure=str(full),
            n10_candidate_gate=str(n10_candidate),
            n10_production_gate=str(n10_production),
            candidate_portfolio=str(candidate_portfolio),
            certification_report=str(certification_report),
            certified_portfolio=str(certified_portfolio),
        )
        candidate, certified, certification = build_regeneration_novelty_certification_v2(
            source_final_hypothesis_id=semantic_row.source_final_hypothesis_id,
            portfolio=portfolio,
            production_gate=_load_json(n10_production),
            artifact_bundle=artifact,
        )

        write_json_exclusive(candidate_portfolio, candidate)
        write_json_exclusive(certification_report, certification)
        write_json_exclusive(certified_portfolio, certified)

        if _sha256_file(portfolio_path) != before_sha:
            raise ValueError("regenerated portfolio mutated during external-N10 downstream")

        decision = certification.decisions[0]
        lineage_results.append(
            RegenerationExternalN10LineageResultV2(
                source_final_hypothesis_id=semantic_row.source_final_hypothesis_id,
                regenerated_hypothesis_id=regenerated_id,
                certification_status=decision.certification_status,
                n10_selection_class=decision.n10_selection_class,
            )
        )
        print(
            semantic_row.source_final_hypothesis_id,
            "| regenerated=", regenerated_id,
            "| N10=", decision.n10_selection_class,
            "->", decision.certification_status,
        )

    report = build_external_n10_execution_report_v2(
        semantic_report=semantic,
        downstream_freeze=downstream,
        lineage_results=lineage_results,
    )
    routed = Path(case.routed_dispatch_path).expanduser().resolve().parent
    output = routed / "regeneration_downstream_v2.external_n10.json"
    if output.exists():
        raise ValueError("external-N10 execution report is write-once")
    write_json_exclusive(output, report)

    print()
    print("Regeneration downstream-v2 external/N9/N10 complete")
    print("Case:", report.case_id)
    print("Report:", report.report_id)
    print("Lineages:", report.lineage_count)
    print("Certified/unresolved/rejected:", report.certified_count, "/", report.unresolved_count, "/", report.rejected_count)
    print("Stage invocations external/N9-intake/N9-full/N10:", report.external_novelty_stage_invocations, "/", report.n9_intake_stage_invocations, "/", report.n9_full_closure_stage_invocations, "/", report.n10_certification_stage_invocations)
    print("Novelty refinement performed: false")
    print("Bounded continuation performed: false")
    print("Binding plan/Gate v2 performed: false")
    print("Endpoint/verifier performed: false")
    print("Second regeneration performed: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
