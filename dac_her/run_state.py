from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pipeline_core.run_lifecycle as _run_lifecycle

from pipeline_core.document_provenance import (
    document_source_fingerprints,
    sha256_file,
)
from pipeline_core.serialization_primitives import (
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_text,
    write_json,
)

from pipeline_core.document_config import PaperConfig, paper_config_fingerprint_payload
from pipeline_core.extraction_policy import ExtractionPolicy
from domains.dac_her.prompts import PROMPT_VERSION, SYSTEM_PROMPT


RUN_STATE_VERSION = "semantic-si-assets-run-v5-strict-recovery"
ATTEMPT_LAYOUT_VERSION = "run-attempt-provenance-v1"












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
    return _run_lifecycle.paper_output_root(
        project_root,
        paper_id,
        data_root=data_root,
    )


def run_directory(
    project_root: str | Path,
    paper_id: str,
    run_id: str,
    data_root: str | Path = "data_dac",
) -> Path:
    return _run_lifecycle.run_directory(
        project_root,
        paper_id,
        run_id,
        data_root=data_root,
    )


def attempt_directory(
    project_root: str | Path,
    paper_id: str,
    run_id: str,
    attempt_id: str,
    data_root: str | Path = "data_dac",
) -> Path:
    return _run_lifecycle.attempt_directory(
        project_root,
        paper_id,
        run_id,
        attempt_id,
        data_root=data_root,
    )


def _latest_attempt_from_family(
    run_dir: Path,
) -> Path:
    return _run_lifecycle._latest_attempt_from_family(
        run_dir
    )






def write_latest_attempt_pointer(
    *,
    project_root: str | Path,
    paper_id: str,
    run_metadata: dict[str, Any],
    attempt_id: str,
    data_root: str | Path = "data_dac",
) -> Path:
    return _run_lifecycle.write_latest_attempt_pointer(
        project_root=project_root,
        paper_id=paper_id,
        run_metadata=run_metadata,
        attempt_id=attempt_id,
        data_root=data_root,
        attempt_layout_version=ATTEMPT_LAYOUT_VERSION,
        updated_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
    )


def write_latest_run_pointer(
    *,
    project_root: str | Path,
    paper_id: str,
    run_metadata: dict[str, Any],
    data_root: str | Path = "data_dac",
    attempt_id: str | None = None,
) -> Path:
    return _run_lifecycle.write_latest_run_pointer(
        project_root=project_root,
        paper_id=paper_id,
        run_metadata=run_metadata,
        data_root=data_root,
        attempt_layout_version=ATTEMPT_LAYOUT_VERSION,
        updated_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
        attempt_id=attempt_id,
    )


def resolve_run_directory(
    *,
    project_root: str | Path,
    paper_id: str,
    run_id: str | None,
    data_root: str | Path = "data_dac",
    attempt_id: str | None = None,
) -> Path:
    return _run_lifecycle.resolve_run_directory(
        project_root=project_root,
        paper_id=paper_id,
        run_id=run_id,
        data_root=data_root,
        attempt_id=attempt_id,
    )
