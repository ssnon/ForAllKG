from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


BINDING_SCHEMA_VERSION = "graphagentsdac-broad-extraction-bindings-v1"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_attempt_directory(
    data_root: str | Path,
    paper_id: str,
    run_id: str,
    attempt_id: str,
) -> Path:
    data_root = Path(data_root)
    attempt_dir = (
        data_root
        / "extracted"
        / paper_id
        / "runs"
        / run_id
        / "attempts"
        / attempt_id
    )
    if not attempt_dir.is_dir():
        raise FileNotFoundError(f"Extraction attempt directory not found: {attempt_dir}")

    active = _read_json(attempt_dir / "active_chunks.json") or {}
    summary = _read_json(attempt_dir / "summary.json") or {}
    actual_run_id = str(active.get("run_id") or summary.get("run_id") or "").strip()
    actual_attempt_id = str(
        active.get("attempt_id") or summary.get("attempt_id") or ""
    ).strip()
    if actual_run_id and actual_run_id != run_id:
        raise ValueError(
            f"Bound run mismatch for {paper_id}: expected {run_id}, got {actual_run_id}"
        )
    if actual_attempt_id and actual_attempt_id != attempt_id:
        raise ValueError(
            f"Bound attempt mismatch for {paper_id}: expected {attempt_id}, "
            f"got {actual_attempt_id}"
        )
    return attempt_dir.resolve()


def binding_from_attempt(
    data_root: str | Path,
    paper_id: str,
    run_id: str,
    attempt_id: str,
    *,
    capture_matching_projection: bool = True,
) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    attempt_dir = resolve_attempt_directory(data_root, paper_id, run_id, attempt_id)
    active = _read_json(attempt_dir / "active_chunks.json") or {}
    summary = _read_json(attempt_dir / "summary.json") or {}
    fingerprint = str(
        active.get("run_fingerprint") or summary.get("run_fingerprint") or ""
    ).strip()
    status = str(
        active.get("graph_materialization_status")
        or summary.get("graph_materialization_status")
        or "unknown"
    ).strip()
    family_dir = attempt_dir.parent.parent
    binding: dict[str, Any] = {
        "paper_id": paper_id,
        "run_id": run_id,
        "run_fingerprint": fingerprint,
        "attempt_id": attempt_id,
        "graph_materialization_status": status,
        "run_family_directory": str(family_dir),
        "attempt_directory": str(attempt_dir),
    }

    if capture_matching_projection:
        projection_path = (
            data_root
            / "extracted"
            / paper_id
            / "graphagents"
            / "mechanism"
            / "summary.json"
        )
        projection = _read_json(projection_path)
        if projection is not None:
            source_run = str(projection.get("source_extraction_run_id") or "").strip()
            source_fp = str(
                projection.get("source_extraction_run_fingerprint") or ""
            ).strip()
            source_attempt = str(
                projection.get("source_extraction_attempt_id") or ""
            ).strip()
            if (
                source_run == run_id
                and (not fingerprint or source_fp == fingerprint)
                and source_attempt == attempt_id
            ):
                binding["projection_summary_snapshot"] = projection
    return binding


def binding_from_latest(
    data_root: str | Path,
    paper_id: str,
    *,
    capture_matching_projection: bool = True,
) -> dict[str, Any]:
    data_root = Path(data_root).resolve()
    pointer_path = data_root / "extracted" / paper_id / "latest_run.json"
    pointer = _read_json(pointer_path)
    if pointer is None:
        raise FileNotFoundError(f"latest_run.json not found/readable: {pointer_path}")
    run_id = str(pointer.get("run_id") or "").strip()
    attempt_id = str(pointer.get("attempt_id") or "").strip()
    if not run_id or not attempt_id:
        raise ValueError(
            f"Latest pointer for {paper_id} is not attempt-aware; "
            "explicit RUN_ID:ATTEMPT_ID is required for legacy data."
        )
    return binding_from_attempt(
        data_root,
        paper_id,
        run_id,
        attempt_id,
        capture_matching_projection=capture_matching_projection,
    )


def write_bindings(
    path: str | Path,
    *,
    corpus_id: str,
    data_root: str | Path,
    bindings: Mapping[str, Mapping[str, Any]],
) -> Path:
    path = Path(path)
    payload = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "source_data_root": str(Path(data_root).resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "papers": {
            str(paper_id): dict(binding)
            for paper_id, binding in sorted(bindings.items())
        },
    }
    _write_json(path, payload)
    return path


def load_bindings(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(path)
    payload = _read_json(path)
    if payload is None:
        raise FileNotFoundError(f"Bindings file not found/readable: {path}")
    papers = payload.get("papers")
    if not isinstance(papers, dict):
        raise ValueError(f"Bindings file has no papers mapping: {path}")
    bindings = {
        str(paper_id): dict(binding)
        for paper_id, binding in papers.items()
        if isinstance(binding, dict)
    }
    return payload, bindings


def build_bound_diagnostics_view(
    *,
    view_root: str | Path,
    data_root: str | Path,
    paper_ids: Iterable[str],
    bindings: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Build a read-only diagnostic view without mutating source extraction data."""

    view_root = Path(view_root)
    data_root = Path(data_root).resolve()
    missing: list[str] = []
    for raw_paper_id in paper_ids:
        paper_id = str(raw_paper_id)
        binding = bindings.get(paper_id)
        paper_root = view_root / "extracted" / paper_id
        paper_root.mkdir(parents=True, exist_ok=True)
        if not isinstance(binding, Mapping):
            missing.append(paper_id)
            # Deliberately leave the view without latest_run.json. The existing
            # diagnostics code will report this paper as run-missing instead of
            # silently falling back to the real data_root latest pointer.
            continue
        run_id = str(binding.get("run_id") or "").strip()
        attempt_id = str(binding.get("attempt_id") or "").strip()
        if not run_id or not attempt_id:
            missing.append(paper_id)
            continue
        attempt_dir = resolve_attempt_directory(data_root, paper_id, run_id, attempt_id)
        family_dir = attempt_dir.parent.parent
        active = _read_json(attempt_dir / "active_chunks.json") or {}
        summary = _read_json(attempt_dir / "summary.json") or {}
        fingerprint = str(
            binding.get("run_fingerprint")
            or active.get("run_fingerprint")
            or summary.get("run_fingerprint")
            or ""
        ).strip()
        pointer = {
            "paper_id": paper_id,
            "run_id": run_id,
            "run_fingerprint": fingerprint,
            "run_directory": str(family_dir),
            "attempt_id": attempt_id,
            "attempt_directory": str(attempt_dir),
            "attempt_layout_version": "run-attempt-provenance-v1",
        }
        _write_json(paper_root / "latest_run.json", pointer)

        projection = binding.get("projection_summary_snapshot")
        if isinstance(projection, Mapping):
            _write_json(
                paper_root / "graphagents" / "mechanism" / "summary.json",
                dict(projection),
            )
    return missing


def run_bound_diagnostics(
    *,
    data_root: str | Path,
    paper_ids: Iterable[str],
    bindings: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    preflight_outlier_ids: Iterable[str] = (),
) -> tuple[Path, Path, Path, list[str]]:
    """Run existing Broad diagnostics against exact historical attempts."""

    from dac_her.broad_extraction_diagnostics import write_broad_extraction_diagnostics

    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir)
    paper_ids = [str(value) for value in paper_ids]
    with tempfile.TemporaryDirectory(prefix="graphagents_bound_audit_") as temp_dir:
        view_root = Path(temp_dir)
        missing = build_bound_diagnostics_view(
            view_root=view_root,
            data_root=data_root,
            paper_ids=paper_ids,
            bindings=bindings,
        )
        report_path, rows_path, issues_path = write_broad_extraction_diagnostics(
            data_root=view_root,
            paper_ids=paper_ids,
            output_dir=output_dir,
            preflight_outlier_ids=preflight_outlier_ids,
        )

    report = _read_json(report_path) or {}
    report["diagnostic_scope"] = "explicit_extraction_bindings"
    report["source_data_root"] = str(data_root)
    report["bound_paper_count"] = len(paper_ids) - len(missing)
    report["missing_binding_paper_ids"] = missing
    _write_json(report_path, report)
    return report_path, rows_path, issues_path, missing
