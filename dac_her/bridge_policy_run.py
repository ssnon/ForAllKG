from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from domains.dac_her.bridge_policy import (
    BRIDGE_POLICY_VERSION,
)
from domains.dac_her.bridge_prompts import (
    BRIDGE_PROMPT_VERSION,
)
from pipeline_core.bridge_domain import (
    BridgeDomainAdapter,
)
from pipeline_core.bridge_policy_run import (
    materialize_bridge_policy_run as _materialize_bridge_policy_run,
)
from pipeline_core.bridge_schemas import (
    BridgeChunkGraph,
)

from dac_her.bridge_extraction import (
    bridge_raw_output_path,
)
from dac_her.bridge_filtering import (
    filter_bridge_raw_chunk,
)
from dac_her.bridge_graph import (
    build_bridge_graph,
    save_bridge_graph,
    write_bridge_tables,
)
from pipeline_core.corpus.schemas import KnowledgeGraph


def _resolve_bridge_adapter(
    bridge_adapter: BridgeDomainAdapter | None,
) -> BridgeDomainAdapter:
    if bridge_adapter is not None:
        return bridge_adapter

    from dac_her.domains.bridge_registry import (
        get_bridge_adapter,
    )

    return get_bridge_adapter("dac_her")


def _include_domain_identity(
    bridge_adapter: BridgeDomainAdapter,
) -> bool:
    return not (
        bridge_adapter.domain_profile_id == "dac_her"
        and bridge_adapter.adapter_id == "dac_her"
    )


def materialize_bridge_policy_run(
    *,
    project_root: str | Path,
    paper_id: str,
    strict_run_dir: str | Path,
    active_payload: dict[str, Any],
    extraction_dir: str | Path,
    canonical_path: str | Path,
    strict_results: dict[
        str,
        KnowledgeGraph,
    ],
    source_payloads: dict[
        str,
        dict[str, Any],
    ],
    policy_implementation_paths: Iterable[
        str | Path
    ],
    bridge_adapter: BridgeDomainAdapter | None = None,
    data_root: str | Path = "data_dac",
) -> dict[str, Any]:
    bridge_adapter = _resolve_bridge_adapter(
        bridge_adapter
    )

    return _materialize_bridge_policy_run(
        project_root=project_root,
        paper_id=paper_id,
        strict_run_dir=strict_run_dir,
        active_payload=active_payload,
        extraction_dir=extraction_dir,
        canonical_path=canonical_path,
        strict_results=strict_results,
        source_payloads=source_payloads,
        policy_implementation_paths=(
            policy_implementation_paths
        ),
        bridge_adapter=bridge_adapter,
        data_root=data_root,
        bridge_raw_output_path_fn=(
            bridge_raw_output_path
        ),
        filter_bridge_raw_chunk_fn=(
            filter_bridge_raw_chunk
        ),
        build_bridge_graph_fn=(
            build_bridge_graph
        ),
        save_bridge_graph_fn=(
            save_bridge_graph
        ),
        write_bridge_tables_fn=(
            write_bridge_tables
        ),
        include_domain_identity_in_fingerprint=(
            _include_domain_identity(
                bridge_adapter
            )
        ),
        publish_legacy_aliases=True,
    )
