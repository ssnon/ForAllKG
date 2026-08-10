from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.candidate_decision import CandidateDecisionEngine
from dac_her.experimental_runtime import ExperimentalRealizabilityRuntime
from dac_her.feasibility_intake import FeasibilityIntakeBuilder
from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from dac_her.hypothesis_semantic_contracts import HypothesisSemanticReview
from dac_her.physics_runtime import PhysicsFeasibilityRuntime
from dac_her.scope_compiler import HypothesisScopeCompiler
from dac_her.validation_specification import ValidationSpecificationCompiler


def _load(path: Path, model):
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.model_dump_json(indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run GraphAgentsDAC feasibility v0.2: hypothesis artifacts -> "
            "scientific scope -> validation specification -> physics -> generic "
            "experimental realizability -> candidate decision."
        )
    )
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--semantic-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    context = _load(args.context, HypothesisContext)
    portfolio = _load(args.portfolio, HypothesisPortfolio)
    semantic_review = _load(args.semantic_review, HypothesisSemanticReview)

    intake = FeasibilityIntakeBuilder().build(context, portfolio, semantic_review)
    scopes = HypothesisScopeCompiler().compile_intake(intake)
    specifications = ValidationSpecificationCompiler().compile_intake(intake, scopes)
    physics_reports = PhysicsFeasibilityRuntime().run_intake(
        intake,
        scopes,
        specifications,
    )
    experimental_reports = ExperimentalRealizabilityRuntime().run_intake(
        intake,
        physics_reports,
        scopes,
        specifications,
    )
    decisions = CandidateDecisionEngine().decide(
        intake,
        scopes,
        specifications,
        physics_reports,
        experimental_reports,
    )

    out = args.output_dir
    _write(out / "feasibility" / "intake.json", intake)
    for scope in scopes:
        _write(out / "scope" / f"{scope.hypothesis_id.replace(':', '_')}.json", scope)
    for spec in specifications:
        _write(
            out / "validation" / f"{spec.hypothesis_id.replace(':', '_')}.json",
            spec,
        )
    for report in physics_reports:
        _write(out / "physics" / f"{report.hypothesis_id.replace(':', '_')}.json", report)
    for report in experimental_reports:
        _write(
            out / "experimental" / f"{report.hypothesis_id.replace(':', '_')}.json",
            report,
        )
    _write(out / "decision" / "portfolio.json", decisions)

    manifest = {
        "schema_version": "feasibility-e2e-manifest-v02",
        "experimental_scope": "laboratory_agnostic",
        "scope_aware_planning": True,
        "source_context": str(args.context),
        "source_portfolio": str(args.portfolio),
        "source_semantic_review": str(args.semantic_review),
        "intake_id": intake.intake_id,
        "scientific_scope_ids": [row.scope_id for row in scopes],
        "validation_specification_ids": [row.specification_id for row in specifications],
        "physics_report_ids": [row.report_id for row in physics_reports],
        "experimental_report_ids": [row.report_id for row in experimental_reports],
        "decision_portfolio_id": decisions.decision_portfolio_id,
    }
    (out / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(decisions.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
