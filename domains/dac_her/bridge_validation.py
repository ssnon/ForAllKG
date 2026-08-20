"""DAC-HER compatibility binding for shared Bridge validation."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping

from pipeline_core.bridge_schemas import BridgeChunkGraph
from domains.dac_her.scientific_signatures import (
    strong_anchor_context_issues,
)
from pipeline_core.bridge_validation import (
    _CAUSAL_RELATIONS,
    _CORRELATIONAL_RELATIONS,
    _is_subspan,
    _normalized_source,
    _strict_catalog,
    bridge_validation_issues as _core_bridge_validation_issues,
)


def bridge_validation_issues(
    result: BridgeChunkGraph,
    *,
    paper_id: str,
    chunk_id: str,
    document_id: str,
    document_role: str,
    page_ids: Iterable[int],
    asset_ids: Iterable[str],
    core_text: str,
    strict_nodes: Iterable[dict[str, Any] | str] | None = None,
    strict_node_ids: Iterable[str] | None = None,
    anchor_context_issues_fn: Callable[..., list[str]] = (
        strong_anchor_context_issues
    ),
) -> list[str]:
    return _core_bridge_validation_issues(
        result,
        paper_id=paper_id,
        chunk_id=chunk_id,
        document_id=document_id,
        document_role=document_role,
        page_ids=page_ids,
        asset_ids=asset_ids,
        core_text=core_text,
        strict_nodes=strict_nodes,
        strict_node_ids=strict_node_ids,
        anchor_context_issues_fn=anchor_context_issues_fn,
    )


def validate_bridge_chunk(
    result: BridgeChunkGraph,
    **kwargs: Any,
) -> None:
    issues = bridge_validation_issues(
        result,
        **kwargs,
    )

    if issues:
        raise ValueError(
            "Bridge graph validation failed:\n- "
            + "\n- ".join(issues)
        )


def bind_bridge_validation(
    anchor_context_issues_fn: Callable[..., list[str]],
) -> tuple[
    Callable[..., list[str]],
    Callable[..., None],
]:
    def issues(
        result: BridgeChunkGraph,
        **kwargs: Any,
    ) -> list[str]:
        return _core_bridge_validation_issues(
            result,
            anchor_context_issues_fn=(
                anchor_context_issues_fn
            ),
            **kwargs,
        )

    def validate(
        result: BridgeChunkGraph,
        **kwargs: Any,
    ) -> None:
        found = issues(
            result,
            **kwargs,
        )

        if found:
            raise ValueError(
                "Bridge graph validation failed:\n- "
                + "\n- ".join(found)
            )

    return issues, validate
