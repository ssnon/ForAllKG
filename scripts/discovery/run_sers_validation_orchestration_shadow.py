#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio

from domains.sers.classical_em_readiness_contracts import (
    SERSClassicalEMReadinessBundle,
)
from domains.sers.validation_orchestration import SERSValidationOrchestrator
from domains.sers.validation_orchestration_contracts import SERSRouteEvidenceBundle
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlanBundle,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble shadow-only SERS multi-validator research state from a "
            "hypothesis portfolio, validation routes, optional classical-EM "
            "readiness, and optional structured evidence from non-EM routes."
        )
    )
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--validation-plans", required=True)
    parser.add_argument("--classical-em-readiness")
    parser.add_argument("--route-evidence")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    portfolio_path = Path(args.portfolio).expanduser().resolve()
    plans_path = Path(args.validation_plans).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    plans = SERSHypothesisValidationPlanBundle.model_validate_json(
        plans_path.read_text(encoding="utf-8")
    )

    readiness = None
    readiness_path = None
    if args.classical_em_readiness:
        readiness_path = Path(args.classical_em_readiness).expanduser().resolve()
        readiness = SERSClassicalEMReadinessBundle.model_validate_json(
            readiness_path.read_text(encoding="utf-8")
        )

    route_evidence = None
    route_evidence_path = None
    if args.route_evidence:
        route_evidence_path = Path(args.route_evidence).expanduser().resolve()
        route_evidence = SERSRouteEvidenceBundle.model_validate_json(
            route_evidence_path.read_text(encoding="utf-8")
        )

    orchestrator = SERSValidationOrchestrator()
    result = orchestrator.build(
        portfolio,
        plans,
        classical_em_readiness=readiness,
        route_evidence=route_evidence,
    )

    result_path = output_dir / "sers.validation_orchestration.shadow.json"
    manifest_path = output_dir / "sers.validation_orchestration_shadow.manifest.json"
    _write_json(result_path, result)

    route_state_counts = Counter(
        route.state for state in result.states for route in state.route_states
    )
    route_kind_counts = Counter(
        route.route_kind for state in result.states for route in state.route_states
    )
    overall_counts = Counter(row.overall_research_state for row in result.states)
    next_action_counts = Counter(row.recommended_next_action for row in result.states)

    sources: dict[str, str] = {
        "portfolio": str(portfolio_path),
        "portfolio_sha256": _sha256_file(portfolio_path),
        "validation_plans": str(plans_path),
        "validation_plans_sha256": _sha256_file(plans_path),
    }
    if readiness_path is not None:
        sources["classical_em_readiness"] = str(readiness_path)
        sources["classical_em_readiness_sha256"] = _sha256_file(readiness_path)
    if route_evidence_path is not None:
        sources["route_evidence"] = str(route_evidence_path)
        sources["route_evidence_sha256"] = _sha256_file(route_evidence_path)

    manifest = {
        "schema_version": "sers-validation-orchestration-shadow-manifest-v0",
        "orchestrator_version": orchestrator.orchestrator_version,
        "sources": sources,
        "state_count": result.state_count,
        "experiment_requirement_count": result.experiment_requirement_count,
        "route_state_counts": dict(sorted(route_state_counts.items())),
        "route_kind_counts": dict(sorted(route_kind_counts.items())),
        "overall_research_state_counts": dict(sorted(overall_counts.items())),
        "recommended_next_action_counts": dict(sorted(next_action_counts.items())),
        "interpretation": (
            "This artifact is a research-state and next-action planner. It may "
            "prioritize a missing validator route or produce non-procedural "
            "integrated SERS experiment requirements, but it creates no scientific "
            "verdict, hypothesis rejection, experiment promotion, self-feedback, "
            "or canonical graph authority."
        ),
        "shadow_only": True,
        "evidence_authority_created": False,
        "scientific_verdict_authority_created": False,
        "hypothesis_rejection_authority": False,
        "experiment_promotion_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"orchestration={result_path}")
    print(f"manifest={manifest_path}")
    print(f"state_count={result.state_count}")
    print(f"experiment_requirement_count={result.experiment_requirement_count}")
    print(
        "route_state_counts="
        + json.dumps(dict(sorted(route_state_counts.items())), sort_keys=True)
    )
    print(
        "overall_research_state_counts="
        + json.dumps(dict(sorted(overall_counts.items())), sort_keys=True)
    )
    print(
        "recommended_next_action_counts="
        + json.dumps(dict(sorted(next_action_counts.items())), sort_keys=True)
    )
    print("scientific_verdict_authority_created=false")
    print("experiment_promotion_authority=false")
    print("feedback_generation_authority=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
