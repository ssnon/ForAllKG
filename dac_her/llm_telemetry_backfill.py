from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from dac_her.llm_telemetry import append_extraction_artifact_resolutions


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _telemetry_ids(path: Path) -> tuple[set[str], set[str]]:
    if not path.exists():
        return set(), set()
    call_ids: set[str] = set()
    resolved_call_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            call_id = row.get("call_id")
            if row.get("record_type") == "artifact_resolution":
                if call_id:
                    resolved_call_ids.add(str(call_id))
                continue
            if call_id:
                call_ids.add(str(call_id))
    return call_ids, resolved_call_ids


def backfill_extraction_resolutions(
    *,
    data_root: str | Path,
    paper_ids: Iterable[str],
    telemetry_path: str | Path,
) -> dict[str, Any]:
    data_root = Path(data_root)
    telemetry_path = Path(telemetry_path)
    call_ids, resolved_call_ids = _telemetry_ids(telemetry_path)
    allowed_call_ids = call_ids - resolved_call_ids
    papers_seen = 0
    papers_resolved = 0
    resolution_records = 0
    missing_runs: list[str] = []

    for paper_id_raw in paper_ids:
        paper_id = str(paper_id_raw)
        papers_seen += 1
        paper_root = data_root / "extracted" / paper_id
        pointer = _read_json(paper_root / "latest_run.json")
        if pointer is None or not pointer.get("run_directory"):
            missing_runs.append(paper_id)
            continue
        family_dir = Path(str(pointer["run_directory"]))
        attempt_raw = pointer.get("attempt_directory")
        attempt_dir = Path(str(attempt_raw)) if attempt_raw else None
        if attempt_dir is not None and attempt_dir.exists():
            run_dir = attempt_dir
        else:
            latest_attempt = _read_json(family_dir / "latest_attempt.json")
            latest_raw = (latest_attempt or {}).get("attempt_directory")
            latest_dir = Path(str(latest_raw)) if latest_raw else None
            run_dir = (
                latest_dir
                if latest_dir is not None and latest_dir.exists()
                else family_dir
            )
        active = _read_json(run_dir / "active_chunks.json")
        if active is None:
            missing_runs.append(paper_id)
            continue

        chunks = active.get("chunks")
        quarantined = active.get("quarantined_chunks")
        failed = active.get("failed_chunks")
        written = append_extraction_artifact_resolutions(
            telemetry_path,
            run_id=str(active.get("run_id") or pointer.get("run_id") or ""),
            paper_id=paper_id,
            materialization_status=str(
                active.get("graph_materialization_status") or "unknown"
            ),
            active_records=(chunks if isinstance(chunks, list) else []),
            quarantined_records=(
                quarantined if isinstance(quarantined, list) else []
            ),
            failed_records=(failed if isinstance(failed, list) else []),
            allowed_call_ids=allowed_call_ids,
        )
        if written:
            papers_resolved += 1
            resolution_records += written

    return {
        "papers_seen": papers_seen,
        "papers_resolved": papers_resolved,
        "resolution_records_appended": resolution_records,
        "telemetry_call_ids": len(call_ids),
        "already_resolved_call_ids": len(resolved_call_ids),
        "eligible_unresolved_call_ids": len(allowed_call_ids),
        "missing_runs": missing_runs,
    }
