#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from domains.sers.resonance_calibration import (
    SERSCompletedMeepRun,
    SERSResonanceCalibrationEvaluator,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


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
            "Extract deterministic resonance-calibration observations from completed "
            "SERS Meep shadow execution directories. No match threshold or physics "
            "authority is created."
        )
    )
    parser.add_argument(
        "--execution-dir",
        action="append",
        required=True,
        help="S2.2 output directory; repeat for paired polarization runs.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _collect_execution_dir(path: Path) -> tuple[list[SERSCompletedMeepRun], dict[str, Any]]:
    request_path = path / "sers_fdtd.meep_execution_requests.shadow.json"
    records_path = path / "sers_fdtd.meep_execution_records.shadow.json"
    if not request_path.exists() or not records_path.exists():
        raise FileNotFoundError(f"missing S2.2 request/record bundle under {path}")

    request_bundle = _load_json(request_path)
    record_bundle = _load_json(records_path)
    requests = {
        str(row["request_id"]): row
        for row in request_bundle.get("requests", [])
        if isinstance(row, dict) and row.get("request_id")
    }
    records = {
        str(row["request_id"]): row
        for row in record_bundle.get("records", [])
        if isinstance(row, dict) and row.get("request_id")
    }

    raw_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    for raw_path in sorted((path / "raw").glob("*.result.json")):
        raw = _load_json(raw_path)
        request_id = raw.get("request_id")
        if request_id:
            raw_by_id[str(request_id)] = (raw_path, raw)

    runs: list[SERSCompletedMeepRun] = []
    provenance = {
        "execution_dir": str(path),
        "request_bundle": str(request_path),
        "request_bundle_sha256": _sha256_file(request_path),
        "record_bundle": str(records_path),
        "record_bundle_sha256": _sha256_file(records_path),
        "accepted_results": [],
    }
    for request_id, record in sorted(records.items()):
        if record.get("execution_status") != "completed":
            continue
        if request_id not in requests:
            raise ValueError(f"completed record missing request {request_id}")
        raw_item = raw_by_id.get(request_id)
        if raw_item is None:
            raise ValueError(f"completed record missing raw worker result {request_id}")
        raw_path, raw = raw_item
        if raw.get("worker_status") != "completed":
            continue
        expected_sha = record.get("worker_result_sha256")
        actual_sha = _sha256_file(raw_path)
        if expected_sha and expected_sha != actual_sha:
            raise ValueError(f"worker result sha256 mismatch for {request_id}")
        runs.append(SERSCompletedMeepRun(request=requests[request_id], result=raw))
        provenance["accepted_results"].append({
            "request_id": request_id,
            "worker_result": str(raw_path),
            "worker_result_sha256": actual_sha,
        })
    return runs, provenance


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[SERSCompletedMeepRun] = []
    provenance: list[dict[str, Any]] = []
    for item in args.execution_dir:
        collected, source = _collect_execution_dir(Path(item).resolve())
        runs.extend(collected)
        provenance.append(source)
    if not runs:
        raise ValueError("no completed Meep worker results were found")

    bundle = SERSResonanceCalibrationEvaluator().evaluate(runs)
    bundle_path = output_dir / "sers_fdtd.resonance_calibration_observations.shadow.json"
    _write_json(bundle_path, bundle)

    quality_counts = Counter(row.quality_gate for row in bundle.observations)
    paired_count = sum(row.paired_polarization_complete for row in bundle.observations)
    manifest = {
        "schema_version": "sers-fdtd-resonance-calibration-shadow-manifest-v0",
        "source_execution_directories": provenance,
        "observation_bundle": str(bundle_path),
        "observation_bundle_sha256": _sha256_file(bundle_path),
        "execution_result_count": bundle.execution_result_count,
        "calibration_group_count": bundle.observation_count,
        "paired_polarization_group_count": paired_count,
        "quality_gate_counts": dict(sorted(quality_counts.items())),
        "calibration_scope": "distance_report_only_no_acceptance_threshold",
        "shadow_only": True,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    manifest_path = output_dir / "sers_fdtd.resonance_calibration_shadow.manifest.json"
    _write_json(manifest_path, manifest)

    print(f"observations={bundle_path}")
    print(f"manifest={manifest_path}")
    print(f"execution_result_count={bundle.execution_result_count}")
    print(f"calibration_group_count={bundle.observation_count}")
    print(f"paired_polarization_group_count={paired_count}")
    print("quality_gate_counts=" + json.dumps(dict(sorted(quality_counts.items())), sort_keys=True))
    print("physics_authority_created=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
