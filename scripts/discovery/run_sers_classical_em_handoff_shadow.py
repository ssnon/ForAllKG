#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from domains.sers.classical_em_handoff import SERSClassicalEMHandoffBuilder
from domains.sers.fdtd_applicability_contracts import SERSFDTDApplicabilityBundle
from domains.sers.validation_routing_contracts import SERSHypothesisValidationPlanBundle


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
            "Connect SERS classical-EM validation routes to the existing FDTD "
            "simulation-spec stack without executing Meep or creating a whole-"
            "hypothesis verdict."
        )
    )
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--validation-plans", required=True)
    parser.add_argument(
        "--fdtd-applicability",
        help=(
            "Optional persisted applicability bundle from the same routing run. "
            "If omitted, deterministic applicability is recomputed and its bundle "
            "id must match the validation-plan lineage."
        ),
    )
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
    applicability = None
    if args.fdtd_applicability:
        applicability = SERSFDTDApplicabilityBundle.model_validate_json(
            Path(args.fdtd_applicability).expanduser().resolve().read_text(
                encoding="utf-8"
            )
        )

    handoffs, compilation, validation, applicability = (
        SERSClassicalEMHandoffBuilder().build(
            portfolio,
            plans,
            fdtd_applicability=applicability,
        )
    )

    applicability_path = output_dir / "sers_fdtd.routed_applicability.shadow.json"
    compilation_path = output_dir / "sers_fdtd.routed_simulation_specs.shadow.json"
    validation_path = output_dir / "sers_fdtd.routed_simulation_validation.shadow.json"
    handoff_path = output_dir / "sers.classical_em_handoffs.shadow.json"
    manifest_path = output_dir / "sers.classical_em_handoff_shadow.manifest.json"

    _write_json(applicability_path, applicability)
    _write_json(compilation_path, compilation)
    _write_json(validation_path, validation)
    _write_json(handoff_path, handoffs)

    disposition_counts = Counter(
        row.simulation_disposition for row in handoffs.handoffs
    )
    manifest = {
        "schema_version": "sers-classical-em-handoff-shadow-manifest-v0",
        "source_portfolio": str(portfolio_path),
        "source_portfolio_sha256": _sha256_file(portfolio_path),
        "source_validation_plans": str(plans_path),
        "source_validation_plans_sha256": _sha256_file(plans_path),
        "routed_hypothesis_count": handoffs.routed_hypothesis_count,
        "not_routed_hypothesis_ids": handoffs.not_routed_hypothesis_ids,
        "simulation_disposition_counts": dict(sorted(disposition_counts.items())),
        "artifacts": {
            "applicability": str(applicability_path),
            "compilation": str(compilation_path),
            "validation": str(validation_path),
            "handoffs": str(handoff_path),
        },
        "interpretation": (
            "The FDTD artifacts represent only routed classical-EM subclaims. "
            "They cannot establish the integrated SERS outcome or reject the "
            "whole scientific hypothesis."
        ),
        "shadow_only": True,
        "evidence_authority_created": False,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"handoffs={handoff_path}")
    print(f"compilation={compilation_path}")
    print(f"validation={validation_path}")
    print(f"applicability={applicability_path}")
    print(f"manifest={manifest_path}")
    print(f"routed_hypothesis_count={handoffs.routed_hypothesis_count}")
    print(f"not_routed_hypothesis_count={len(handoffs.not_routed_hypothesis_ids)}")
    print(
        "simulation_disposition_counts="
        + json.dumps(dict(sorted(disposition_counts.items())), sort_keys=True)
    )
    print("whole_hypothesis_verdict_permitted=false")
    print("physics_authority_created=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
