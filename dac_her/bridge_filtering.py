from __future__ import annotations

from pathlib import Path
from typing import Any

from dac_her.bridge_relation_repairs import (
    apply_deterministic_relation_repairs,
)
from pipeline_core.graph_io import (
    knowledge_graph_to_networkx,
)
from dac_her.schemas import KnowledgeGraph
from pipeline_core.bridge_domain import (
    BridgeDomainAdapter,
)
from pipeline_core.bridge_filtering import (
    bridge_candidate_issues_path,
    bridge_candidates_path,
    bridge_output_path,
    bridge_rejections_path,
    bridge_relation_repairs_path,
    filter_bridge_raw_chunk as _filter_bridge_raw_chunk,
)
from pipeline_core.bridge_schemas import (
    BridgeChunkGraph,
)


def _resolve_bridge_adapter(
    bridge_adapter: BridgeDomainAdapter | None,
) -> BridgeDomainAdapter:
    if bridge_adapter is not None:
        return bridge_adapter

    from dac_her.domains.bridge_registry import (
        get_bridge_adapter,
    )

    return get_bridge_adapter("dac_her")


def _catalog(
    result: KnowledgeGraph,
    bridge_adapter: BridgeDomainAdapter,
) -> list[dict[str, Any]]:
    return bridge_adapter.strict_node_catalog_builder(
        knowledge_graph_to_networkx(
            result
        )
    )


def filter_bridge_raw_chunk(
    *,
    raw_result: BridgeChunkGraph,
    strict_result: KnowledgeGraph,
    source_payload: dict[str, Any],
    output_dir: str | Path,
    bridge_adapter: BridgeDomainAdapter | None = None,
) -> dict[str, Any]:
    bridge_adapter = _resolve_bridge_adapter(
        bridge_adapter
    )

    strict_nodes = _catalog(
        strict_result,
        bridge_adapter,
    )

    return _filter_bridge_raw_chunk(
        raw_result=raw_result,
        strict_nodes=strict_nodes,
        chunk_id=strict_result.chunk_id,
        source_payload=source_payload,
        output_dir=output_dir,
        bridge_adapter=bridge_adapter,
        relation_repairer=(
            apply_deterministic_relation_repairs
        ),
    )
