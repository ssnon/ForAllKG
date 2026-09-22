from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.reframing.atomic_specification_bridge_shadow import (
    run_atomic_specification_bridge_shadow,
)


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        value.model_dump(mode="json")
        if hasattr(value, "model_dump")
        else value
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only counterfactual N9 run that restores only uniquely "
            "authorized exact inferential-bridge source sentences to atomic claims "
            "through the existing typed required-bridge binding contract. No "
            "prediction/falsifier repair, novelty authority, or selection occurs."
        )
    )
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--query-plan", required=True, type=Path)
    parser.add_argument("--external-report", required=True, type=Path)
    parser.add_argument(
        "--specification-sanitization-audit",
        required=True,
        type=Path,
    )
    parser.add_argument("--bindings-output", required=True, type=Path)
    parser.add_argument("--baseline-intake-output", required=True, type=Path)
    parser.add_argument("--bridged-intake-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()

    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    plan = LiteratureQueryPlan.model_validate_json(
        args.query_plan.read_text(encoding="utf-8")
    )
    external = ExternalNoveltyReport.model_validate_json(
        args.external_report.read_text(encoding="utf-8")
    )
    audit = _load_object(args.specification_sanitization_audit)

    bindings, baseline, bridged, report = (
        run_atomic_specification_bridge_shadow(
            portfolio=portfolio,
            query_plan=plan,
            external_report=external,
            specification_sanitization_audit=audit,
        )
    )

    _write(args.bindings_output, bindings)
    _write(args.baseline_intake_output, baseline)
    _write(args.bridged_intake_output, bridged)
    _write(args.report_output, report)

    print("Atomic scientific specification bridge shadow complete")
    print("Claims:", report.claim_count)
    print("Exact-source bindings:", report.exact_source_binding_count)
    print(
        "READY_FOR_CLOSURE:",
        f"{report.baseline_ready_for_closure_count}",
        "->",
        f"{report.bridged_ready_for_closure_count}",
    )
    print(
        "Newly READY_FOR_CLOSURE:",
        report.newly_ready_for_closure_count,
    )
    print("State transitions:", report.state_transitions)
    print("Free-text bridge generated: false")
    print("Prediction/falsifier repaired: false")
    print("Production authority: false")
    print("Scientific selection changed: false")
    print("Bindings:", args.bindings_output)
    print("Bridged intake:", args.bridged_intake_output)
    print("Report:", args.report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
