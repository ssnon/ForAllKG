"""Deterministic extraction-run metadata and fingerprint construction."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_core.document_config import (
    PaperConfig,
    paper_config_fingerprint_payload,
)
from pipeline_core.document_provenance import (
    document_source_fingerprints,
    sha256_file,
)
from pipeline_core.extraction_policy import ExtractionPolicy
from pipeline_core.serialization_primitives import (
    canonical_json,
    sha256_text,
)


RUN_STATE_VERSION = (
    "semantic-si-assets-run-v5-strict-recovery"
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
    prompt_version: str,
    system_prompt: str,
    domain_profile_id: str,
    data_root: str | Path,
    runtime_options: dict[str, Any] | None = None,
    implementation_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Build the deterministic extraction-run fingerprint payload.

    Scientific/domain policy is supplied explicitly by the caller.
    """

    project_root = Path(project_root).resolve()
    schemas_path = Path(schemas_path).resolve()
    chunking_path = Path(chunking_path).resolve()

    metadata: dict[str, Any] = {
        "run_state_version": RUN_STATE_VERSION,
        "paper": paper_config_fingerprint_payload(
            paper
        ),
        "document_sources": (
            document_source_fingerprints(
                paper
            )
        ),
        "domain_profile_id": domain_profile_id,
        "data_root": str(data_root),
        "prompt_version": prompt_version,
        "prompt_sha256": sha256_text(
            system_prompt
        ),
        "schema_sha256": sha256_file(
            schemas_path
        ),
        "chunking_sha256": sha256_file(
            chunking_path
        ),
        "policy": asdict(policy),
        "model": model,
        "provider": provider,
        "runtime_options": runtime_options or {},
        "vocabularies": [
            {
                "relative_path": str(
                    path.relative_to(
                        project_root
                    )
                ),
                "sha256": sha256_file(
                    path
                ),
            }
            for path in sorted(
                (
                    project_root
                    / "configs"
                    / "vocabularies"
                ).glob("*.yaml")
            )
        ],
        "project_root": str(
            project_root
        ),
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

    fingerprint = sha256_text(
        canonical_json(
            metadata
        )
    )

    metadata[
        "run_fingerprint"
    ] = fingerprint

    metadata[
        "run_id"
    ] = fingerprint[:16]

    metadata[
        "created_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    return metadata
