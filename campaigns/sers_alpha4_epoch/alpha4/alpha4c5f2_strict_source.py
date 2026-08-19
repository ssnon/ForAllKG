from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from campaigns.sers_alpha4_epoch.holdout.alpha4c4d2_holdout_support import (
    Alpha4c4d2Error,
    DATA_ROOT,
    ROOT,
    quality_snapshot,
    read_json,
    require_equal,
    sha256,
    snapshot_optional,
)
from dac_her.extraction_quality import (
    QUALITY_PARTIAL_CRITICAL,
    QUALITY_REJECTED,
)


STRICT_SOURCE_LAYOUT_SEMANTICS_ID = (
    "strict_source_attempt_layout_v1_alpha4c5f21"
)


def _resolve_candidate_path(raw: object) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path.resolve()


def _resolve_concrete_run_directory(
    *,
    paper_id: str,
    pointer: Mapping[str, Any],
    family_dir: Path,
) -> tuple[Path, dict[str, Any] | None]:
    """Resolve legacy flat-run and current run-attempt layouts fail-closed.

    Current Strict extraction writes latest_run.json at the paper root.  Its
    run_directory is the run-family directory.  Concrete attempt artifacts
    live under attempts/<attempt_id> and are addressed either directly by the
    latest_run pointer or by run-family/latest_attempt.json.

    No scientific content is inspected here; only provenance/layout metadata.
    """
    pointer_attempt = _resolve_candidate_path(
        pointer.get("attempt_directory")
    )
    if pointer_attempt is not None:
        if not pointer_attempt.is_dir():
            raise Alpha4c4d2Error(
                f"{paper_id}: latest_run.json declares missing "
                f"attempt_directory: {pointer_attempt}"
            )
        return pointer_attempt, None

    latest_attempt_path = family_dir / "latest_attempt.json"
    if latest_attempt_path.exists():
        latest_attempt = read_json(latest_attempt_path)
        require_equal(
            f"{paper_id} latest-attempt paper_id",
            latest_attempt.get("paper_id"),
            paper_id,
        )
        if pointer.get("run_id"):
            require_equal(
                f"{paper_id} latest-attempt run_id",
                latest_attempt.get("run_id"),
                pointer.get("run_id"),
            )
        attempt_dir = _resolve_candidate_path(
            latest_attempt.get("attempt_directory")
        )
        if attempt_dir is None:
            raise Alpha4c4d2Error(
                f"{paper_id}: latest_attempt.json lacks "
                "attempt_directory."
            )
        if not attempt_dir.is_dir():
            raise Alpha4c4d2Error(
                f"{paper_id}: latest_attempt.json points to missing "
                f"directory: {attempt_dir}"
            )
        return attempt_dir, latest_attempt

    # Legacy flat layout: run.json and active_chunks.json live directly
    # in the run-family directory.
    return family_dir, None


def resolve_strict_source_attempt_aware(
    paper_id: str,
) -> dict[str, Any]:
    paper_root = DATA_ROOT / "extracted" / paper_id
    pointer_path = paper_root / "latest_run.json"
    if not pointer_path.exists():
        raise Alpha4c4d2Error(
            f"{paper_id}: latest_run.json missing."
        )

    pointer = read_json(pointer_path)
    pointer_paper = str(pointer.get("paper_id") or "").strip()
    if pointer_paper:
        require_equal(
            f"{paper_id} latest-run paper_id",
            pointer_paper,
            paper_id,
        )

    family_dir = _resolve_candidate_path(
        pointer.get("run_directory")
    )
    if family_dir is None:
        raise Alpha4c4d2Error(
            f"{paper_id}: latest_run.json lacks run_directory."
        )
    if not family_dir.is_dir():
        raise Alpha4c4d2Error(
            f"{paper_id}: run-family directory missing: {family_dir}"
        )

    concrete_dir, latest_attempt = _resolve_concrete_run_directory(
        paper_id=paper_id,
        pointer=pointer,
        family_dir=family_dir,
    )

    run_json_path = concrete_dir / "run.json"
    active_path = concrete_dir / "active_chunks.json"
    run_json = read_json(run_json_path)
    active = read_json(active_path)

    require_equal(
        f"{paper_id} active paper_id",
        active.get("paper_id"),
        paper_id,
    )

    pointer_run_id = str(pointer.get("run_id") or "").strip()
    run_id = str(
        active.get("run_id")
        or run_json.get("run_id")
        or pointer_run_id
        or ""
    ).strip()
    if not run_id:
        raise Alpha4c4d2Error(
            f"{paper_id}: run_id unresolved."
        )
    if pointer_run_id:
        require_equal(
            f"{paper_id} pointer/run run_id",
            run_id,
            pointer_run_id,
        )
    require_equal(
        f"{paper_id} active/run run_id",
        active.get("run_id"),
        run_json.get("run_id"),
    )

    attempt_id = str(
        active.get("attempt_id")
        or run_json.get("attempt_id")
        or pointer.get("attempt_id")
        or (
            latest_attempt.get("attempt_id")
            if latest_attempt
            else ""
        )
        or ""
    ).strip()

    pointer_attempt_id = str(
        pointer.get("attempt_id") or ""
    ).strip()
    if pointer_attempt_id and attempt_id:
        require_equal(
            f"{paper_id} pointer/active attempt_id",
            attempt_id,
            pointer_attempt_id,
        )

    quality = quality_snapshot(active)
    status = quality["graph_materialization_status"]
    if status == QUALITY_REJECTED:
        raise Alpha4c4d2Error(
            f"{paper_id}: frozen Strict source is REJECTED. "
            f"Reason={quality['classification_reason']}; "
            f"active={quality['active_chunk_count']}; "
            f"quarantined={quality['quarantined_chunk_count']}; "
            f"failed={quality['failed_chunk_count']}; "
            f"coverage={quality['source_token_coverage']!r}"
        )
    if not quality["positive_evidence_queries_allowed"]:
        raise Alpha4c4d2Error(
            f"{paper_id}: positive-evidence queries are not allowed."
        )

    chunks = active.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise Alpha4c4d2Error(
            f"{paper_id}: no active strict-valid chunks."
        )

    chunk_inputs: list[dict[str, Any]] = []
    for row in chunks:
        output = _resolve_candidate_path(
            row.get("output_path")
        )
        if output is None or not output.exists():
            raise Alpha4c4d2Error(
                f"{paper_id}: active chunk missing: {output}"
            )
        chunk_inputs.append(
            {
                "chunk_id": str(row.get("chunk_id") or ""),
                "path": str(output),
                "sha256": sha256(output),
            }
        )

    latest_attempt_path = family_dir / "latest_attempt.json"
    locator = concrete_dir / "locator_index.json"

    return {
        "paper_id": paper_id,
        "strict_source_layout_semantics_id": (
            STRICT_SOURCE_LAYOUT_SEMANTICS_ID
        ),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "run_directory": str(concrete_dir),
        "run_family_directory": str(family_dir),
        "attempt_layout": concrete_dir != family_dir,
        "active_payload_complete_flag": bool(
            active.get("complete", False)
        ),
        "extraction_quality": quality,
        "requires_allow_incomplete": (
            status == QUALITY_PARTIAL_CRITICAL
        ),
        "latest_run": {
            "path": str(pointer_path.relative_to(ROOT)),
            "sha256": sha256(pointer_path),
        },
        "latest_attempt": snapshot_optional(
            latest_attempt_path
        ),
        "run_json": {
            "path": str(run_json_path.relative_to(ROOT)),
            "sha256": sha256(run_json_path),
        },
        "active_chunks": {
            "path": str(active_path.relative_to(ROOT)),
            "sha256": sha256(active_path),
        },
        "locator_index": snapshot_optional(locator),
        "chunk_inputs": chunk_inputs,
    }


def verify_strict_source_attempt_aware_unchanged(
    source: Mapping[str, Any],
) -> None:
    for key in ("latest_run", "run_json", "active_chunks"):
        row = source[key]
        path = ROOT / str(row["path"])
        if not path.exists():
            raise Alpha4c4d2Error(
                f"{source['paper_id']}: {key} disappeared: {path}"
            )
        require_equal(
            f"{source['paper_id']} {key} SHA256",
            sha256(path),
            row["sha256"],
        )

    latest_attempt = source.get("latest_attempt")
    if isinstance(latest_attempt, Mapping):
        raw = str(latest_attempt.get("path") or "")
        if raw:
            path = ROOT / raw
            present = path.exists()
            require_equal(
                f"{source['paper_id']} latest_attempt presence",
                present,
                bool(latest_attempt.get("present")),
            )
            if present:
                require_equal(
                    f"{source['paper_id']} latest_attempt SHA256",
                    sha256(path),
                    latest_attempt.get("sha256"),
                )

    for row in source["chunk_inputs"]:
        path = Path(str(row["path"]))
        if not path.exists():
            raise Alpha4c4d2Error(
                f"{source['paper_id']}: chunk disappeared: {path}"
            )
        require_equal(
            f"{source['paper_id']} chunk "
            f"{row['chunk_id']} SHA256",
            sha256(path),
            row["sha256"],
        )

    active = read_json(
        ROOT / str(source["active_chunks"]["path"])
    )
    require_equal(
        f"{source['paper_id']} extraction quality",
        quality_snapshot(active),
        source["extraction_quality"],
    )
