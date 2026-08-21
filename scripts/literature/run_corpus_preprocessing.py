from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.literature.knowledge_backfill_runtime import (
    write_dynamic_target_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPORT_SCHEMA = "corpus-preprocess-run-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _profile_identity(path: Path) -> tuple[str, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Acquisition profile must be a mapping: {path}")
    profile_id = str(payload.get("profile_id") or "").strip()
    domain_profile_id = str(payload.get("domain_profile_id") or "").strip()
    if not profile_id or not domain_profile_id:
        raise ValueError("Acquisition profile must define profile_id/domain_profile_id")
    return profile_id, domain_profile_id


def _infer_materialization_id(m4_dir: Path, override: str | None) -> str:
    if override and override.strip():
        return override.strip()
    for name in (
        "incremental_materialization_report.json",
        "materialization_report.json",
    ):
        path = m4_dir / name
        if path.is_file():
            payload = _read_json(path)
            value = str(payload.get("materialization_id") or "").strip()
            if value:
                return value
    raise ValueError(
        "Could not infer materialization_id from M4 reports; pass --materialization-id"
    )


def _latest_m3_from_manifest(run_root: Path) -> Path | None:
    path = run_root / "preprocess_run.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    value = str(payload.get("latest_m3_dir") or "").strip()
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not (candidate / "acquisition_report.json").is_file():
        return None
    return candidate


def _next_round_dir(run_root: Path) -> tuple[int, Path]:
    indices = []
    if run_root.is_dir():
        for path in run_root.glob("round_*"):
            if not path.is_dir():
                continue
            try:
                indices.append(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    index = max(indices, default=0) + 1
    return index, run_root / f"round_{index:03d}"


def _run(command: list[str], *, label: str, dry_run: bool) -> dict[str, Any]:
    print(f"[preprocess] {label} | start", flush=True)
    print("[preprocess]   $", " ".join(command), flush=True)
    started = time.monotonic()
    if dry_run:
        return {
            "label": label,
            "status": "dry_run",
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "command": command,
        }
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.monotonic() - started
    status = "passed" if completed.returncode == 0 else "failed"
    print(f"[preprocess] {label} | {status} | {elapsed:.1f}s", flush=True)
    return {
        "label": label,
        "status": status,
        "return_code": int(completed.returncode),
        "elapsed_seconds": elapsed,
        "command": command,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess-only corpus orchestrator: grow the public-OA main-paper "
            "selection to --target-count, discover supplementary artifacts, and "
            "run incremental M4. It intentionally stops before M4.5/Strict/Bridge."
        )
    )
    parser.add_argument("--target-count", required=True, type=int)
    parser.add_argument("--acquisition-profile", required=True, type=Path)
    parser.add_argument("--backfill-policy", required=True, type=Path)
    parser.add_argument("--source-policy", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--m2-assessments", required=True, type=Path)
    parser.add_argument("--quality-assessments", required=True, type=Path)
    parser.add_argument("--quality-gate-report", required=True, type=Path)
    parser.add_argument("--starting-m3-dir", required=True, type=Path)
    parser.add_argument("--supplementary-policy", required=True, type=Path)
    parser.add_argument("--materialization-policy", required=True, type=Path)
    parser.add_argument("--m4-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--generated-config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--paper-id-prefix", required=True)
    parser.add_argument("--materialization-id", default=None)
    parser.add_argument("--m4-workers", type=int, default=2)
    parser.add_argument("--retry-failed-acquisition", action="store_true")
    parser.add_argument("--retry-access-misses", action="store_true")
    parser.add_argument("--retry-failed-supplementary", action="store_true")
    parser.add_argument("--retry-failed-materialization", action="store_true")
    parser.add_argument("--skip-access-recovery", action="store_true")
    parser.add_argument("--skip-supplementary", action="store_true")
    parser.add_argument("--fresh-from-starting-m3", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target_count < 1:
        parser.error("--target-count must be >= 1")
    if args.m4_workers < 1:
        parser.error("--m4-workers must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    profile_id, domain_profile_id = _profile_identity(args.acquisition_profile)

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    round_index, round_root = _next_round_dir(run_root)
    round_root.mkdir(parents=True, exist_ok=True)
    output_m3 = round_root / "m3_2"
    dynamic_profile = round_root / "acquisition_profile.yaml"
    write_dynamic_target_profile(
        source_profile=args.acquisition_profile,
        output_path=dynamic_profile,
        target_total=args.target_count,
    )

    previous = None if args.fresh_from_starting_m3 else _latest_m3_from_manifest(run_root)
    source_m3 = (previous or args.starting_m3_dir).resolve()
    for required in (
        source_m3 / "acquisition_report.json",
        source_m3 / "selected_works.jsonl",
        source_m3 / "selection_report.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Completed starting M3 snapshot required: {required}")

    m4_dir = args.m4_dir.resolve()
    materialization_id = _infer_materialization_id(m4_dir, args.materialization_id)
    m31_dir = run_root / "m3_1"
    command_records: list[dict[str, Any]] = []
    started_at = _utc_now()

    if not args.skip_access_recovery:
        command = [
            sys.executable,
            "-m",
            "scripts.literature.prepare_access_recovery",
            "--source-policy",
            str(args.source_policy.resolve()),
            "--source-m3-dir",
            str(source_m3),
            "--output-m3-dir",
            str(output_m3),
        ]
        if args.retry_failed_acquisition:
            command.append("--retry-failed")
        if args.retry_access_misses:
            command.append("--retry-access-misses")
        record = _run(command, label="access_recovery", dry_run=args.dry_run)
        command_records.append(record)
        if record["return_code"] != 0:
            return _finish_failure(
                run_root=run_root,
                round_index=round_index,
                target_count=args.target_count,
                started_at=started_at,
                source_m3=source_m3,
                output_m3=output_m3,
                command_records=command_records,
                failed_stage="access_recovery",
            )

    command = [
        sys.executable,
        "-m",
        "scripts.literature.backfill_acquisition_ready_corpus",
        "--profile",
        str(dynamic_profile),
        "--backfill-policy",
        str(args.backfill_policy.resolve()),
        "--source-policy",
        str(args.source_policy.resolve()),
        "--catalog",
        str(args.catalog.resolve()),
        "--m2-assessments",
        str(args.m2_assessments.resolve()),
        "--quality-assessments",
        str(args.quality_assessments.resolve()),
        "--quality-gate-report",
        str(args.quality_gate_report.resolve()),
        "--m2-1-selected-works",
        str(source_m3 / "selected_works.jsonl"),
        "--m2-1-selection-report",
        str(source_m3 / "selection_report.json"),
        "--m3-dir",
        str(source_m3),
        "--output-dir",
        str(output_m3),
        "--backfill-id",
        f"preprocess_target_{args.target_count}_r{round_index:03d}",
    ]
    if args.retry_failed_acquisition:
        command.append("--retry-failed")
    record = _run(command, label="m3_2_backfill", dry_run=args.dry_run)
    command_records.append(record)
    if record["return_code"] != 0:
        return _finish_failure(
            run_root=run_root,
            round_index=round_index,
            target_count=args.target_count,
            started_at=started_at,
            source_m3=source_m3,
            output_m3=output_m3,
            command_records=command_records,
            failed_stage="m3_2_backfill",
        )

    selected_count = 0 if args.dry_run else _jsonl_count(output_m3 / "selected_works.jsonl")

    if not args.skip_supplementary:
        command = [
            sys.executable,
            "-m",
            "scripts.literature.discover_supplementary_artifacts",
            "--profile-id",
            profile_id,
            "--catalog",
            str(args.catalog.resolve()),
            "--selected-works",
            str(output_m3 / "selected_works.jsonl"),
            "--selection-report",
            str(output_m3 / "selection_report.json"),
            "--m3-dir",
            str(output_m3),
            "--supplementary-policy",
            str(args.supplementary_policy.resolve()),
            "--output-dir",
            str(m31_dir),
            "--acquisition-id",
            f"preprocess_target_{args.target_count}_supplementary",
        ]
        if args.retry_failed_supplementary:
            command.append("--retry-failed")
        record = _run(command, label="m3_1_supplementary", dry_run=args.dry_run)
        command_records.append(record)
        if record["return_code"] != 0:
            return _finish_failure(
                run_root=run_root,
                round_index=round_index,
                target_count=args.target_count,
                started_at=started_at,
                source_m3=source_m3,
                output_m3=output_m3,
                command_records=command_records,
                failed_stage="m3_1_supplementary",
            )

    command = [
        sys.executable,
        "-m",
        "scripts.literature.materialize_corpus_documents_incremental",
        "--profile-id",
        profile_id,
        "--domain-profile-id",
        domain_profile_id,
        "--data-root",
        str(args.data_root.resolve()),
        "--catalog",
        str(args.catalog.resolve()),
        "--selected-works",
        str(output_m3 / "selected_works.jsonl"),
        "--selection-report",
        str(output_m3 / "selection_report.json"),
        "--m3-dir",
        str(output_m3),
        "--materialization-policy",
        str(args.materialization_policy.resolve()),
        "--output-dir",
        str(m4_dir),
        "--generated-config",
        str(args.generated_config.resolve()),
        "--materialization-id",
        materialization_id,
        "--paper-id-prefix",
        args.paper_id_prefix,
        "--workers",
        str(args.m4_workers),
    ]
    if not args.skip_supplementary:
        command.extend(["--m3-1-dir", str(m31_dir)])
    if args.retry_failed_materialization:
        command.append("--retry-failed")
    record = _run(command, label="m4_incremental", dry_run=args.dry_run)
    command_records.append(record)
    if record["return_code"] != 0:
        return _finish_failure(
            run_root=run_root,
            round_index=round_index,
            target_count=args.target_count,
            started_at=started_at,
            source_m3=source_m3,
            output_m3=output_m3,
            command_records=command_records,
            failed_stage="m4_incremental",
        )

    materialization_report = {}
    incremental_report = {}
    if not args.dry_run:
        materialization_report = _read_json(m4_dir / "materialization_report.json")
        incremental_report = _read_json(m4_dir / "incremental_materialization_report.json")
    ready_count = int(materialization_report.get("extraction_ready_paper_count") or 0)
    main_materialized = int(materialization_report.get("main_materialized_count") or 0)
    main_failed = int(materialization_report.get("main_materialization_failed_count") or 0)
    acquisition_target_reached = selected_count >= args.target_count if not args.dry_run else False
    m4_target_reached = ready_count >= args.target_count if not args.dry_run else False
    status = (
        "dry_run"
        if args.dry_run
        else "target_reached"
        if acquisition_target_reached and m4_target_reached
        else "materialization_shortfall"
        if acquisition_target_reached
        else "acquisition_shortfall"
    )

    payload = {
        "schema_version": _REPORT_SCHEMA,
        "status": status,
        "round": round_index,
        "target_count": args.target_count,
        "source_m3_dir": str(source_m3),
        "latest_m3_dir": str(output_m3),
        "m3_1_dir": None if args.skip_supplementary else str(m31_dir),
        "m4_dir": str(m4_dir),
        "selected_count": selected_count,
        "acquisition_target_reached": acquisition_target_reached,
        "main_materialized_count": main_materialized,
        "main_materialization_failed_count": main_failed,
        "extraction_ready_count": ready_count,
        "m4_target_reached": m4_target_reached,
        "m4_workers": args.m4_workers,
        "cache_reused_document_count": int(
            incremental_report.get("cache_reused_document_count") or 0
        ),
        "executed_document_count": int(
            incremental_report.get("executed_document_count") or 0
        ),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "command_records": command_records,
    }
    _write_json_atomic(run_root / "preprocess_run.json", payload)
    _write_json_atomic(round_root / "run.json", payload)

    print()
    print("Corpus preprocessing complete")
    print("Status:", status)
    print("Selected main PDFs:", f"{selected_count}/{args.target_count}")
    print("M4 extraction-ready:", f"{ready_count}/{args.target_count}")
    print("M4 main failures:", main_failed)
    print("Latest M3:", output_m3)
    print("M4:", m4_dir)
    print("Run report:", run_root / "preprocess_run.json")
    return 0


def _finish_failure(
    *,
    run_root: Path,
    round_index: int,
    target_count: int,
    started_at: str,
    source_m3: Path,
    output_m3: Path,
    command_records: list[dict[str, Any]],
    failed_stage: str,
) -> int:
    payload = {
        "schema_version": _REPORT_SCHEMA,
        "status": "stage_failure",
        "round": round_index,
        "target_count": target_count,
        "failed_stage": failed_stage,
        "source_m3_dir": str(source_m3),
        "latest_m3_dir": str(output_m3),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "command_records": command_records,
    }
    _write_json_atomic(run_root / "preprocess_run.json", payload)
    _write_json_atomic(run_root / f"round_{round_index:03d}" / "run.json", payload)
    print(f"Corpus preprocessing failed at {failed_stage}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
