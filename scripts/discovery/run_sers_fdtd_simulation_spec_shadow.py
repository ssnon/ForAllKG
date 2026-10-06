#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio

from domains.sers.fdtd_applicability import (
    SERSFDTDApplicabilityAnalyzer,
)
from domains.sers.simulation_compiler import (
    SERSHypothesisSimulationCompiler,
)
from domains.sers.simulation_validator import (
    SERSDeterministicSimulationValidator,
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
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose SERS hypotheses by FDTD applicability, compile the "
            "FDTD-addressable portion into shadow simulation specifications, "
            "and validate whether those specs are concrete enough for a later "
            "Meep backend."
        )
    )
    parser.add_argument(
        "--portfolio",
        required=True,
        help="Path to a hypothesis-portfolio-v1 JSON artifact.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for shadow applicability/compilation artifacts.",
    )
    parser.add_argument(
        "--hypothesis-id",
        action="append",
        default=None,
        help=(
            "Optional hypothesis_id filter. Repeat to select multiple "
            "hypotheses."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    portfolio_path = Path(args.portfolio).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )

    requested_ids = (
        set(args.hypothesis_id)
        if args.hypothesis_id
        else None
    )

    analyzer = SERSFDTDApplicabilityAnalyzer()
    applicability = analyzer.analyze_portfolio(
        portfolio,
        hypothesis_ids=requested_ids,
    )

    compiler = SERSHypothesisSimulationCompiler()
    compilation = compiler.compile_portfolio(
        portfolio,
        hypothesis_ids=requested_ids,
        applicability_bundle=applicability,
    )

    validation = (
        SERSDeterministicSimulationValidator()
        .validate_bundle(compilation)
    )

    applicability_path = (
        output_dir / "sers_fdtd.applicability.shadow.json"
    )
    compilation_path = (
        output_dir / "sers_fdtd.simulation_specs.shadow.json"
    )
    validation_path = (
        output_dir / "sers_fdtd.simulation_validation.shadow.json"
    )
    manifest_path = (
        output_dir / "sers_fdtd.simulation_spec_shadow.manifest.json"
    )

    _write_json(applicability_path, applicability)
    _write_json(compilation_path, compilation)
    _write_json(validation_path, validation)

    applicability_counts: dict[str, int] = {}
    for report in applicability.reports:
        applicability_counts[report.applicability] = (
            applicability_counts.get(report.applicability, 0) + 1
        )

    disposition_counts: dict[str, int] = {}
    for report in validation.reports:
        disposition_counts[report.disposition] = (
            disposition_counts.get(report.disposition, 0) + 1
        )

    manifest = {
        "schema_version": "sers-fdtd-simulation-spec-shadow-manifest-v0.1",
        "source_portfolio_path": str(portfolio_path),
        "source_portfolio_sha256": _sha256_file(portfolio_path),
        "source_portfolio_id": portfolio.portfolio_id,
        "domain_profile_id": portfolio.domain_profile_id,
        "applicability_analyzer_version": analyzer.analyzer_version,
        "compiler_version": compiler.compiler_version,
        "spec_count": len(compilation.specs),
        "fdtd_applicability_counts": applicability_counts,
        "validation_disposition_counts": disposition_counts,
        "artifacts": {
            "applicability": str(applicability_path),
            "compilation": str(compilation_path),
            "validation": str(validation_path),
        },
        "authority": {
            "shadow_only": True,
            "physics_authority_created": False,
            "hypothesis_rejection_authority": False,
            "feedback_generation_authority": False,
            "canonical_graph_mutated": False,
        },
        "interpretation": (
            "not_applicable and partial are routing statements, not negative "
            "scientific verdicts. Non-FDTD subclaims remain unresolved by this "
            "shadow lane."
        ),
        "next_stage": (
            "A later patch may concretize/execute only the FDTD-computable "
            "subclaim while preserving non-FDTD subclaims separately."
        ),
    }
    _write_json(manifest_path, manifest)

    manifest["artifact_sha256"] = {
        "applicability": _sha256_file(applicability_path),
        "compilation": _sha256_file(compilation_path),
        "validation": _sha256_file(validation_path),
    }
    _write_json(manifest_path, manifest)

    print(f"applicability={applicability_path}")
    print(f"compilation={compilation_path}")
    print(f"validation={validation_path}")
    print(f"manifest={manifest_path}")
    print(
        "applicability_counts="
        + json.dumps(
            applicability_counts,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print(
        "dispositions="
        + json.dumps(
            disposition_counts,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
