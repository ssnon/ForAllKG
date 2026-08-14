from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _run(label: str, command: list[str]) -> dict:
    print(f"[discovery-expansion] {label} | start", flush=True)
    print("[discovery-expansion]   $ " + " ".join(command), flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    elapsed = time.perf_counter() - started
    status = "passed" if completed.returncode == 0 else "failed"
    print(
        f"[discovery-expansion] {label} | {status} | {elapsed:.1f}s",
        flush=True,
    )
    record = {
        "label": label,
        "status": status,
        "return_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "command": command,
    }
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand the literature discovery pool append-only, rerun deterministic "
            "M2/M2.1 on the expanded catalog, and rebase an existing verified "
            "downloaded M3 snapshot onto the new catalog lineage."
        )
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--quality-policy", required=True, type=Path)
    parser.add_argument("--base-catalog", required=True, type=Path)
    parser.add_argument("--source-m3-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expansion-id", required=True)
    parser.add_argument(
        "--providers",
        default="semantic_scholar,crossref,openalex",
    )
    parser.add_argument("--results-per-query", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    m1 = root / "m1"
    m2 = root / "m2"
    m21 = root / "m2_1"
    m3 = root / "m3_rebased"
    command_records: list[dict] = []
    run_report = {
        "schema_version": "discovery-pool-expansion-run-v1",
        "status": "running",
        "expansion_id": args.expansion_id,
        "started_at": _utc_now(),
        "profile": str(args.profile.resolve()),
        "base_catalog": str(args.base_catalog.resolve()),
        "source_m3_dir": str(args.source_m3_dir.resolve()),
        "providers": [
            value.strip() for value in args.providers.split(",") if value.strip()
        ],
        "results_per_query": args.results_per_query,
        "outputs": {
            "m1": str(m1),
            "m2": str(m2),
            "m2_1": str(m21),
            "m3_rebased": str(m3),
        },
        "command_records": command_records,
    }
    _write_json(root / "run.json", run_report)

    try:
        command_records.append(
            _run(
                "m1_expand",
                [
                    sys.executable,
                    "-m",
                    "scripts.expand_literature_catalog",
                    "--profile",
                    str(args.profile.resolve()),
                    "--base-catalog",
                    str(args.base_catalog.resolve()),
                    "--output-dir",
                    str(m1),
                    "--expansion-id",
                    args.expansion_id,
                    "--providers",
                    args.providers,
                    "--results-per-query",
                    str(args.results_per_query),
                ],
            )
        )
        command_records.append(
            _run(
                "m2_select",
                [
                    sys.executable,
                    "-m",
                    "scripts.select_corpus_candidates",
                    "--profile",
                    str(args.profile.resolve()),
                    "--catalog",
                    str(m1 / "catalog.json"),
                    "--output-dir",
                    str(m2),
                ],
            )
        )
        command_records.append(
            _run(
                "m2_1_quality",
                [
                    sys.executable,
                    "-m",
                    "scripts.apply_corpus_quality_gate",
                    "--profile",
                    str(args.profile.resolve()),
                    "--quality-policy",
                    str(args.quality_policy.resolve()),
                    "--catalog",
                    str(m1 / "catalog.json"),
                    "--assessments",
                    str(m2 / "assessments.jsonl"),
                    "--selected-works",
                    str(m2 / "selected_works.jsonl"),
                    "--selection-report",
                    str(m2 / "selection_report.json"),
                    "--output-dir",
                    str(m21),
                    "--quality-gate-id",
                    f"{args.expansion_id}_quality",
                ],
            )
        )
        command_records.append(
            _run(
                "m3_rebase",
                [
                    sys.executable,
                    "-m",
                    "scripts.rebase_m3_to_expanded_catalog",
                    "--profile",
                    str(args.profile.resolve()),
                    "--catalog",
                    str(m1 / "catalog.json"),
                    "--assessments",
                    str(m2 / "assessments.jsonl"),
                    "--quality-assessments",
                    str(m21 / "quality_assessments.jsonl"),
                    "--source-m3-dir",
                    str(args.source_m3_dir.resolve()),
                    "--output-dir",
                    str(m3),
                    "--rebase-id",
                    f"{args.expansion_id}_m3_rebase",
                ],
            )
        )
    except subprocess.CalledProcessError as exc:
        run_report.update(
            {
                "status": "stage_failure",
                "failed_return_code": exc.returncode,
                "command_records": command_records,
                "updated_at": _utc_now(),
            }
        )
        _write_json(root / "run.json", run_report)
        return int(exc.returncode or 1)

    expansion_report = json.loads(
        (m1 / "expansion_report.json").read_text(encoding="utf-8")
    )
    quality_report = json.loads(
        (m21 / "quality_gate_report.json").read_text(encoding="utf-8")
    )
    rebase_report = json.loads(
        (m3 / "rebase_report.json").read_text(encoding="utf-8")
    )
    run_report.update(
        {
            "status": "ready_for_knowledge_backfill",
            "command_records": command_records,
            "expanded_catalog_id": expansion_report["expanded_catalog_id"],
            "base_work_count": expansion_report["base_work_count"],
            "new_work_count": expansion_report["new_work_count"],
            "expanded_work_count": expansion_report["expanded_work_count"],
            "quality_pass_count": quality_report.get("quality_pass_count"),
            "rebased_downloaded_count": rebase_report["retained_downloaded_count"],
            "updated_at": _utc_now(),
        }
    )
    _write_json(root / "run.json", run_report)

    print()
    print("Discovery pool expansion ready for knowledge-aware backfill")
    print("New canonical works:", run_report["new_work_count"])
    print("Expanded canonical works:", run_report["expanded_work_count"])
    print("Rebased verified PDFs:", run_report["rebased_downloaded_count"])
    print("Expanded catalog:", m1 / "catalog.json")
    print("Expanded M2 assessments:", m2 / "assessments.jsonl")
    print("Expanded quality assessments:", m21 / "quality_assessments.jsonl")
    print("Expanded quality report:", m21 / "quality_gate_report.json")
    print("Starting M3 snapshot:", m3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
