from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_core.document_provenance import (
    document_source_fingerprints,
    sha256_file,
)

from dac_her.config import PaperConfig, paper_config_fingerprint_payload
from dac_her.extraction_policy import ExtractionPolicy
from dac_her.prompts import PROMPT_VERSION, SYSTEM_PROMPT


RUN_STATE_VERSION = "semantic-si-assets-run-v5-strict-recovery"
ATTEMPT_LAYOUT_VERSION = "run-attempt-provenance-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))




def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )




def compute_run_metadata(
    *,
    project_root: str | Path,
    paper: PaperConfig,
    policy: ExtractionPolicy,
    model: str,
    provider: str | None,
    schemas_path: str | Path,
    chunking_path: str | Path,
    runtime_options: dict[str, Any] | None = None,
    implementation_paths: tuple[str | Path, ...,] = (),
    prompt_version: str = PROMPT_VERSION,
    system_prompt: str = SYSTEM_PROMPT,
    domain_profile_id: str = "dac_her",
    data_root: str | Path = "data_dac",
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    schemas_path = Path(schemas_path).resolve()
    chunking_path = Path(chunking_path).resolve()

    metadata: dict[str, Any] = {
        "run_state_version": RUN_STATE_VERSION,
        "paper": paper_config_fingerprint_payload(paper),
        "document_sources": document_source_fingerprints(paper),
        "domain_profile_id": domain_profile_id,
        "data_root": str(data_root),
        "prompt_version": prompt_version,
        "prompt_sha256": sha256_text(system_prompt),
        "schema_sha256": sha256_file(schemas_path),
        "chunking_sha256": sha256_file(chunking_path),
        "policy": asdict(policy),
        "model": model,
        "provider": provider,
        "runtime_options": runtime_options or {},
        "vocabularies": [
            {
                "relative_path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
            }
            for path in sorted(
                (project_root / "configs" / "vocabularies").glob("*.yaml")
            )
        ],
        "project_root": str(project_root),
        "implementation_files": [
            {
                "relative_path": str(
                    Path(path)
                    .resolve()
                    .relative_to(
                        project_root
                    )
                ),
                "sha256": sha256_file(
                    path
                ),
            }
            for path in sorted(
                (
                    Path(item).resolve()
                    for item
                    in implementation_paths
                ),
                key=str,
            )
        ],
    }

    fingerprint = sha256_text(canonical_json(metadata))
    metadata["run_fingerprint"] = fingerprint
    metadata["run_id"] = fingerprint[:16]
    metadata["created_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    return metadata


def paper_output_root(
    project_root: str | Path,
    paper_id: str,
    data_root: str | Path = "data_dac",
) -> Path:
    data_root_path = Path(data_root)
    if not data_root_path.is_absolute():
        data_root_path = Path(project_root).resolve() / data_root_path
    return data_root_path.resolve() / "extracted" / paper_id


def run_directory(
    project_root: str | Path,
    paper_id: str,
    run_id: str,
    data_root: str | Path = "data_dac",
) -> Path:
    return paper_output_root(
        project_root, paper_id, data_root=data_root
    ) / "runs" / run_id


def attempt_directory(
    project_root: str | Path,
    paper_id: str,
    run_id: str,
    attempt_id: str,
    data_root: str | Path = "data_dac",
) -> Path:
    return (
        run_directory(
            project_root,
            paper_id,
            run_id,
            data_root=data_root,
        )
        / "attempts"
        / attempt_id
    )


def _latest_attempt_from_family(run_dir: Path) -> Path:
    pointer_path = run_dir / "latest_attempt.json"
    if not pointer_path.exists():
        return run_dir
    pointer = read_json(pointer_path)
    raw = pointer.get("attempt_directory")
    if not raw:
        return run_dir
    path = Path(str(raw))
    return path if path.exists() else run_dir


def write_json(
    path: str | Path,
    payload: Any,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_latest_attempt_pointer(
    *,
    project_root: str | Path,
    paper_id: str,
    run_metadata: dict[str, Any],
    attempt_id: str,
    data_root: str | Path = "data_dac",
) -> Path:
    run_id = str(run_metadata["run_id"])
    family_dir = run_directory(
        project_root,
        paper_id,
        run_id,
        data_root=data_root,
    )
    concrete_dir = attempt_directory(
        project_root,
        paper_id,
        run_id,
        attempt_id,
        data_root=data_root,
    )
    return write_json(
        family_dir / "latest_attempt.json",
        {
            "paper_id": paper_id,
            "run_id": run_id,
            "run_fingerprint": run_metadata["run_fingerprint"],
            "attempt_layout_version": ATTEMPT_LAYOUT_VERSION,
            "attempt_id": attempt_id,
            "attempt_directory": str(concrete_dir),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def write_latest_run_pointer(
    *,
    project_root: str | Path,
    paper_id: str,
    run_metadata: dict[str, Any],
    data_root: str | Path = "data_dac",
    attempt_id: str | None = None,
) -> Path:
    root = paper_output_root(project_root, paper_id, data_root=data_root)
    run_id = str(run_metadata["run_id"])
    family_dir = run_directory(
        project_root,
        paper_id,
        run_id,
        data_root=data_root,
    )
    payload: dict[str, Any] = {
        "paper_id": paper_id,
        "run_id": run_id,
        "run_fingerprint": run_metadata["run_fingerprint"],
        "run_directory": str(family_dir),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if attempt_id:
        payload["attempt_layout_version"] = ATTEMPT_LAYOUT_VERSION
        payload["attempt_id"] = attempt_id
        payload["attempt_directory"] = str(
            attempt_directory(
                project_root,
                paper_id,
                run_id,
                attempt_id,
                data_root=data_root,
            )
        )
    return write_json(root / "latest_run.json", payload)


def resolve_run_directory(
    *,
    project_root: str | Path,
    paper_id: str,
    run_id: str | None,
    data_root: str | Path = "data_dac",
    attempt_id: str | None = None,
) -> Path:
    pointer: dict[str, Any] | None = None
    if run_id:
        family_dir = run_directory(
            project_root, paper_id, run_id, data_root=data_root
        )
    else:
        pointer_path = (
            paper_output_root(project_root, paper_id, data_root=data_root)
            / "latest_run.json"
        )
        if not pointer_path.exists():
            raise FileNotFoundError(
                "No latest run pointer found for "
                f"{paper_id!r}: {pointer_path}"
            )
        pointer = read_json(pointer_path)
        family_dir = Path(pointer["run_directory"])

    if attempt_id:
        path = family_dir / "attempts" / attempt_id
    elif pointer and pointer.get("attempt_directory"):
        candidate = Path(str(pointer["attempt_directory"]))
        path = (
            candidate
            if candidate.exists()
            else _latest_attempt_from_family(family_dir)
        )
    else:
        path = _latest_attempt_from_family(family_dir)

    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")

    return path.resolve()
