from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from domains.dac_her.bridge_policy import (
    BRIDGE_POLICY_VERSION,
)
from domains.dac_her.bridge_prompts import (
    BRIDGE_PROMPT_VERSION,
)
from pipeline_core.corpus.bridge.bridge_run_state import (
    bridge_extraction_directory,
    bridge_policy_run_directory,
    bridge_run_directory,
    compute_bridge_extraction_metadata as _compute_bridge_extraction_metadata,
    compute_bridge_policy_run_metadata as _compute_bridge_policy_run_metadata,
    compute_bridge_run_metadata as _compute_bridge_run_metadata,
    file_sha256,
    stable_json_hash,
)


def _include_domain_identity(
    *,
    domain_profile_id: str,
    bridge_adapter_id: str,
) -> bool:
    return not (
        domain_profile_id == "dac_her"
        and bridge_adapter_id == "dac_her"
    )


def compute_bridge_extraction_metadata(
    *,
    strict_run_dir: str | Path,
    active_payload: dict[str, Any],
    model: str,
    provider: str | None,
    strict_chunk_paths: Iterable[str | Path],
    source_chunk_paths: Iterable[str | Path],
    implementation_paths: Iterable[str | Path],
    runtime_options: dict[str, Any] | None = None,
    bridge_prompt_version: str = BRIDGE_PROMPT_VERSION,
    domain_profile_id: str = "dac_her",
    bridge_adapter_id: str = "dac_her",
) -> dict[str, Any]:
    return _compute_bridge_extraction_metadata(
        strict_run_dir=strict_run_dir,
        active_payload=active_payload,
        model=model,
        provider=provider,
        strict_chunk_paths=strict_chunk_paths,
        source_chunk_paths=source_chunk_paths,
        implementation_paths=implementation_paths,
        runtime_options=runtime_options,
        bridge_prompt_version=bridge_prompt_version,
        domain_profile_id=domain_profile_id,
        bridge_adapter_id=bridge_adapter_id,
        include_domain_identity_in_fingerprint=(
            _include_domain_identity(
                domain_profile_id=domain_profile_id,
                bridge_adapter_id=bridge_adapter_id,
            )
        ),
    )


def compute_bridge_policy_run_metadata(
    *,
    strict_run_dir: str | Path,
    extraction_metadata: dict[str, Any],
    raw_chunk_paths: Iterable[str | Path],
    canonical_graph_path: str | Path | None,
    implementation_paths: Iterable[str | Path],
    bridge_policy_version: str = BRIDGE_POLICY_VERSION,
    domain_profile_id: str = "dac_her",
    bridge_adapter_id: str = "dac_her",
) -> dict[str, Any]:
    return _compute_bridge_policy_run_metadata(
        strict_run_dir=strict_run_dir,
        extraction_metadata=extraction_metadata,
        raw_chunk_paths=raw_chunk_paths,
        canonical_graph_path=canonical_graph_path,
        implementation_paths=implementation_paths,
        bridge_policy_version=bridge_policy_version,
        domain_profile_id=domain_profile_id,
        bridge_adapter_id=bridge_adapter_id,
        include_domain_identity_in_fingerprint=(
            _include_domain_identity(
                domain_profile_id=domain_profile_id,
                bridge_adapter_id=bridge_adapter_id,
            )
        ),
    )


def compute_bridge_run_metadata(
    *,
    strict_run_dir: str | Path,
    active_payload: dict[str, Any],
    model: str,
    provider: str | None,
    strict_chunk_paths: Iterable[str | Path],
    source_chunk_paths: Iterable[str | Path],
    canonical_graph_path: str | Path | None,
    implementation_paths: Iterable[str | Path],
    bridge_prompt_version: str = BRIDGE_PROMPT_VERSION,
    bridge_policy_version: str = BRIDGE_POLICY_VERSION,
    domain_profile_id: str = "dac_her",
    bridge_adapter_id: str = "dac_her",
) -> dict[str, Any]:
    return _compute_bridge_run_metadata(
        strict_run_dir=strict_run_dir,
        active_payload=active_payload,
        model=model,
        provider=provider,
        strict_chunk_paths=strict_chunk_paths,
        source_chunk_paths=source_chunk_paths,
        canonical_graph_path=canonical_graph_path,
        implementation_paths=implementation_paths,
        bridge_prompt_version=bridge_prompt_version,
        bridge_policy_version=bridge_policy_version,
        domain_profile_id=domain_profile_id,
        bridge_adapter_id=bridge_adapter_id,
        include_domain_identity_in_fingerprint=(
            _include_domain_identity(
                domain_profile_id=domain_profile_id,
                bridge_adapter_id=bridge_adapter_id,
            )
        ),
    )
