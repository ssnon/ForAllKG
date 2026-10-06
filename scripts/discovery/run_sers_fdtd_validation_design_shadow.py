#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from domains.sers.simulation_contracts import (
    SERSSimulationCompilationBundle,
    SERSSimulationValidationBundle,
)
from domains.sers.validation_design import (
    SERSValidationDesignBuilder,
    SERSValidationDesignPlanner,
)
from domains.sers.validation_design_contracts import (
    SERSValidationDesignOverrideBundle,
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


def _template(requests) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for request in requests.requests:
        if request.status != "ready_for_design_input":
            continue

        mandatory_wavelengths = sorted(set(
            request.source_constraints.reported_resonance_wavelengths_nm
            + (
                [request.source_constraints.baseline_wavelength_nm]
                if request.source_constraints.baseline_wavelength_nm is not None
                else []
            )
        ))
        rows.append({
            "hypothesis_id": request.hypothesis_id,
            "source_spec_id": request.source_spec_id,
            "source_label": "EDIT_ME_validation_design",
            "operator_note": (
                "Geometry/media/polarization values below are validation-design "
                "assumptions, not source facts. Preserve the mandatory source-"
                "derived wavelength anchors and add explicit screening points."
            ),
            "geometry_candidates": [
                {
                    "candidate_id": "EDIT_ME_geometry_01",
                    "label": "EDIT_ME_representative_geometry",
                    "nanorod_length_nm": None,
                    "nanorod_diameter_nm": None,
                    "shell_thickness_nm": None,
                    "rationale": "EDIT_ME_scientific_reason_for_this_candidate",
                }
            ],
            "wavelength_sweep_nm": mandatory_wavelengths,
            "polarizations": [],
            "surrounding_media": [],
        })
    return {
        "schema_version": "sers-validation-design-override-bundle-v0",
        "overrides": rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lift a SERS FDTD hypothesis into a shadow-only multi-case "
            "ValidationDesign and optionally expand it into solver-neutral "
            "simulation cases."
        )
    )
    parser.add_argument("--specs", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--design-overrides",
        help=(
            "Optional explicit validation-design override bundle. If omitted, "
            "emit only design requirements and a template."
        ),
    )
    return parser.parse_args()


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

    requests = SERSValidationDesignPlanner().plan(compilation, validation)
    request_path = output_dir / "sers_fdtd.validation_design_requests.shadow.json"
    template_path = output_dir / "sers_fdtd.validation_design.template.json"
    _write_json(request_path, requests)
    _write_json(template_path, _template(requests))

    status_counts: dict[str, int] = {}
    for row in requests.requests:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    print(f"requests={request_path}")
    print(f"template={template_path}")
    print(
        "request_status_counts="
        + json.dumps(status_counts, ensure_ascii=False, sort_keys=True)
    )

    manifest: dict[str, object] = {
        "schema_version": "sers-fdtd-validation-design-shadow-manifest-v0",
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
            "design_template": str(template_path),
        },
    }

    if args.design_overrides:
        overrides_path = Path(args.design_overrides).expanduser().resolve()
        overrides = SERSValidationDesignOverrideBundle.model_validate_json(
            overrides_path.read_text(encoding="utf-8")
        )
        designs, cases = SERSValidationDesignBuilder().build(
            compilation,
            requests,
            overrides,
        )

        designs_path = output_dir / "sers_fdtd.validation_designs.shadow.json"
        cases_path = output_dir / "sers_fdtd.simulation_cases.shadow.json"
        _write_json(designs_path, designs)
        _write_json(cases_path, cases)

        manifest["source_design_overrides_path"] = str(overrides_path)
        manifest["source_design_overrides_sha256"] = _sha256_file(overrides_path)
        manifest["design_count"] = len(designs.designs)
        manifest["simulation_case_count"] = len(cases.cases)
        manifest["artifacts"].update({
            "validation_designs": str(designs_path),
            "simulation_cases": str(cases_path),
        })

        print(f"designs={designs_path}")
        print(f"simulation_cases={cases_path}")
        print(f"simulation_case_count={len(cases.cases)}")

    manifest_path = output_dir / "sers_fdtd.validation_design_shadow.manifest.json"
    _write_json(manifest_path, manifest)
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
