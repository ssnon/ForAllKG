#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from domains.sers.simulation_concretization import (
    SERSConcretizationPlanner,
    SERSOperatorConcretizer,
)
from domains.sers.simulation_concretization_contracts import (
    SERSConcretizationOverrideBundle,
)
from domains.sers.simulation_contracts import (
    SERSSimulationCompilationBundle,
    SERSSimulationValidationBundle,
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
            "Plan and optionally apply explicit operator concretization to "
            "shadow-only SERS FDTD simulation specifications."
        )
    )
    parser.add_argument(
        "--specs",
        required=True,
        help="sers_fdtd.simulation_specs.shadow.json",
    )
    parser.add_argument(
        "--validation",
        required=True,
        help="sers_fdtd.simulation_validation.shadow.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for concretization artifacts.",
    )
    parser.add_argument(
        "--overrides",
        help=(
            "Optional operator override bundle JSON. If omitted, only the "
            "requirements and a fillable template are emitted."
        ),
    )
    return parser.parse_args()


def _template_for_requests(requests) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for request in requests.requests:
        if request.status != "ready_for_operator_input":
            continue
        fields = {row.field for row in request.requirements}
        row: dict[str, object] = {
            "hypothesis_id": request.hypothesis_id,
            "source_spec_id": request.source_spec_id,
            "source_label": "EDIT_ME_explicit_operator_concretization",
            "operator_note": (
                "Replace null/empty values only with scientifically justified "
                "simulation conditions. Do not treat these values as facts from "
                "the source hypothesis."
            ),
        }
        field_to_override = {
            "particle_radius_nm": "particle_radius_nm",
            "gap_nm": "gap_nm",
            "nanorod_length_nm": "nanorod_length_nm",
            "nanorod_diameter_nm": "nanorod_diameter_nm",
            "shell_thickness_nm": "shell_thickness_nm",
            "excitation_wavelength_nm": "excitation_wavelength_nm",
            "polarization": "polarization",
            "surrounding_medium": "surrounding_medium",
        }
        for requirement_field, override_field in field_to_override.items():
            if requirement_field in fields:
                row[override_field] = None
        if "sweep.values" in fields:
            row["sweep_values"] = []
        rows.append(row)
    return {
        "schema_version": (
            "sers-simulation-concretization-override-bundle-v0"
        ),
        "overrides": rows,
    }


def main() -> int:
    args = _parse_args()

    specs_path = Path(args.specs).expanduser().resolve()
    validation_path = Path(args.validation).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    compilation = SERSSimulationCompilationBundle.model_validate_json(
        specs_path.read_text(encoding="utf-8")
    )
    validation = SERSSimulationValidationBundle.model_validate_json(
        validation_path.read_text(encoding="utf-8")
    )

    planner = SERSConcretizationPlanner()
    requests = planner.plan(compilation, validation)

    request_path = output_dir / "sers_fdtd.concretization_requests.shadow.json"
    template_path = output_dir / "sers_fdtd.concretization_overrides.template.json"
    _write_json(request_path, requests)
    _write_json(template_path, _template_for_requests(requests))

    print(f"requests={request_path}")
    print(f"template={template_path}")

    status_counts: dict[str, int] = {}
    for row in requests.requests:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    print(
        "request_status_counts="
        + json.dumps(status_counts, ensure_ascii=False, sort_keys=True)
    )

    manifest = {
        "schema_version": "sers-fdtd-concretization-shadow-manifest-v0",
        "source_specs_path": str(specs_path),
        "source_specs_sha256": _sha256_file(specs_path),
        "source_validation_path": str(validation_path),
        "source_validation_sha256": _sha256_file(validation_path),
        "request_bundle_id": requests.bundle_id,
        "request_status_counts": status_counts,
        "authority": {
            "shadow_only": True,
            "physics_authority_created": False,
            "hypothesis_rejection_authority": False,
            "feedback_generation_authority": False,
            "canonical_graph_mutated": False,
        },
        "artifacts": {
            "requests": str(request_path),
            "override_template": str(template_path),
        },
    }

    if args.overrides:
        overrides_path = Path(args.overrides).expanduser().resolve()
        overrides = SERSConcretizationOverrideBundle.model_validate_json(
            overrides_path.read_text(encoding="utf-8")
        )
        concretized = SERSOperatorConcretizer().apply(
            compilation,
            requests,
            overrides,
        )

        # Re-wrap the concretized specs in the existing compilation contract so
        # the same deterministic validator is reused without creating a parallel
        # notion of numerical validity.
        concretized_compilation = compilation.model_copy(
            update={
                "bundle_id": "concretized:" + concretized.bundle_id,
                "specs": concretized.specs,
            }
        )
        concretized_validation = (
            SERSDeterministicSimulationValidator()
            .validate_bundle(concretized_compilation)
        )

        concretized_path = (
            output_dir / "sers_fdtd.concretized_specs.shadow.json"
        )
        concretized_validation_path = (
            output_dir / "sers_fdtd.concretized_validation.shadow.json"
        )
        records_path = (
            output_dir / "sers_fdtd.concretization_records.shadow.json"
        )

        _write_json(concretized_path, concretized_compilation)
        _write_json(concretized_validation_path, concretized_validation)
        _write_json(records_path, concretized)

        disposition_counts: dict[str, int] = {}
        for report in concretized_validation.reports:
            disposition_counts[report.disposition] = (
                disposition_counts.get(report.disposition, 0) + 1
            )

        manifest["source_overrides_path"] = str(overrides_path)
        manifest["source_overrides_sha256"] = _sha256_file(overrides_path)
        manifest["concretized_validation_disposition_counts"] = (
            disposition_counts
        )
        manifest["artifacts"].update({
            "concretization_records": str(records_path),
            "concretized_specs": str(concretized_path),
            "concretized_validation": str(concretized_validation_path),
        })

        print(f"records={records_path}")
        print(f"concretized_specs={concretized_path}")
        print(f"concretized_validation={concretized_validation_path}")
        print(
            "concretized_dispositions="
            + json.dumps(
                disposition_counts,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    manifest_path = output_dir / "sers_fdtd.concretization_shadow.manifest.json"
    _write_json(manifest_path, manifest)
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
