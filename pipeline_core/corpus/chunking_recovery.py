from __future__ import annotations

from dataclasses import dataclass
import re

from pipeline_core.corpus.extraction.chunking import (
    ChunkSpec,
    _filter_parent_assets,
    count_tokens,
    first_tokens,
    last_tokens,
    make_chunk_id,
    split_chunk_in_half,
)
from pipeline_core.corpus.extraction.extraction_policy import ExtractionPolicy


@dataclass(frozen=True)
class RechunkResult:
    children: tuple[ChunkSpec, ChunkSpec]
    split_method: str
    parent_chunk_id: str
    reason: str


def _paragraph_split(text: str) -> tuple[str, str] | None:
    units = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    if len(units) < 2:
        return None

    total_tokens = sum(count_tokens(item) for item in units)
    target = total_tokens / 2
    running = 0
    best_index = 1
    best_distance = float("inf")

    for index in range(1, len(units)):
        running += count_tokens(units[index - 1])
        distance = abs(running - target)
        if distance < best_distance:
            best_distance = distance
            best_index = index

    left = "\n\n".join(units[:best_index]).strip()
    right = "\n\n".join(units[best_index:]).strip()
    if not left or not right:
        return None
    return left, right


def _build_children(
    *,
    chunk: ChunkSpec,
    policy: ExtractionPolicy,
    left_core: str,
    right_core: str,
) -> tuple[ChunkSpec, ChunkSpec]:
    left_meta = _filter_parent_assets(left_core, chunk)
    right_meta = _filter_parent_assets(right_core, chunk)

    left = ChunkSpec(
        paper_id=chunk.paper_id,
        document_id=chunk.document_id,
        document_role=chunk.document_role,
        section=chunk.section,
        index=chunk.index * 2,
        core_text=left_core,
        left_context=chunk.left_context,
        right_context=first_tokens(right_core, policy.right_context_tokens),
        page_ids=left_meta[0],
        asset_ids=left_meta[1],
        asset_paths=left_meta[2],
        asset_pages=left_meta[3],
        asset_locators=left_meta[4],
        asset_context=chunk.asset_context,
        chunk_id=make_chunk_id(
            chunk.paper_id,
            chunk.section,
            left_core,
            document_id=chunk.document_id,
        ),
        split_depth=chunk.split_depth + 1,
    )

    right = ChunkSpec(
        paper_id=chunk.paper_id,
        document_id=chunk.document_id,
        document_role=chunk.document_role,
        section=chunk.section,
        index=chunk.index * 2 + 1,
        core_text=right_core,
        left_context=last_tokens(left_core, policy.left_context_tokens),
        right_context=chunk.right_context,
        page_ids=right_meta[0],
        asset_ids=right_meta[1],
        asset_paths=right_meta[2],
        asset_pages=right_meta[3],
        asset_locators=right_meta[4],
        asset_context=chunk.asset_context,
        chunk_id=make_chunk_id(
            chunk.paper_id,
            chunk.section,
            right_core,
            document_id=chunk.document_id,
        ),
        split_depth=chunk.split_depth + 1,
    )
    return left, right


def split_chunk_structurally(
    chunk: ChunkSpec,
    policy: ExtractionPolicy,
    *,
    reason: str,
) -> RechunkResult:
    paragraph_split = _paragraph_split(chunk.core_text)
    if paragraph_split is not None:
        left_core, right_core = paragraph_split
        if min(count_tokens(left_core), count_tokens(right_core)) >= 150:
            return RechunkResult(
                children=_build_children(
                    chunk=chunk,
                    policy=policy,
                    left_core=left_core,
                    right_core=right_core,
                ),
                split_method="paragraph_boundary",
                parent_chunk_id=chunk.chunk_id,
                reason=reason,
            )

    children = split_chunk_in_half(chunk, policy)
    return RechunkResult(
        children=(children[0], children[1]),
        split_method="token_midpoint_fallback",
        parent_chunk_id=chunk.chunk_id,
        reason=reason,
    )
