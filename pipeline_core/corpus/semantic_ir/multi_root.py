from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.semantic_ir.corpus_audit import (
    DuplicatePaperPolicy,
    DiscoveredExtractionAttempt,
    discover_extraction_attempts,
)
from pipeline_core.corpus.semantic_ir.existing_extraction import (
    load_existing_extraction_semantic_ir,
)
from pipeline_core.corpus.semantic_ir.schema import SemanticIRBundle


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CrossRootDuplicatePolicy = Literal["error", "prefer_last_root"]


class MultiRootPaperSource(StrictModel):
    paper_id: str = Field(min_length=1)
    root: str = Field(min_length=1)
    attempt_directory: str = Field(min_length=1)
    active_chunks_path: str = Field(min_length=1)
    attempt_id: str | None = None


class MultiRootSemanticIRLoadResult(StrictModel):
    schema_version: Literal[
        "multi-root-semantic-ir-load-v1"
    ] = "multi-root-semantic-ir-load-v1"

    roots: list[str] = Field(min_length=1)
    requested_paper_ids: list[str] = Field(default_factory=list)
    available_paper_ids: list[str] = Field(default_factory=list)
    missing_paper_ids: list[str] = Field(default_factory=list)
    cross_root_duplicate_paper_ids: list[str] = Field(default_factory=list)
    paper_sources: list[MultiRootPaperSource] = Field(default_factory=list)

    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False
    negative_evidence_inferred: Literal[False] = False

    @model_validator(mode="after")
    def validate_sets(self) -> "MultiRootSemanticIRLoadResult":
        requested = set(self.requested_paper_ids)
        available = set(self.available_paper_ids)
        missing = set(self.missing_paper_ids)
        if available & missing:
            raise ValueError("paper cannot be both available and missing")
        if requested and requested != available | missing:
            raise ValueError(
                "requested paper IDs must equal available union missing"
            )
        if {row.paper_id for row in self.paper_sources} != available:
            raise ValueError(
                "paper_sources must contain exactly the available paper IDs"
            )
        return self


def _attempt_key(row: DiscoveredExtractionAttempt) -> tuple[str, str]:
    return (row.attempt_id or "", row.active_chunks_path)


def load_semantic_ir_from_roots(
    roots: list[str | Path],
    *,
    requested_paper_ids: set[str] | None = None,
    duplicate_paper_policy: DuplicatePaperPolicy = "error",
    cross_root_duplicate_policy: CrossRootDuplicatePolicy = "error",
) -> tuple[MultiRootSemanticIRLoadResult, dict[str, SemanticIRBundle]]:
    """Load only requested paper bundles from one or more extraction roots.

    Root order is explicit provenance. Cross-root duplicates fail closed unless
    the caller deliberately requests ``prefer_last_root``.
    """

    if not roots:
        raise ValueError("at least one extraction root is required")

    normalized_roots = [str(Path(root).resolve()) for root in roots]
    requested = set(requested_paper_ids or set())

    candidates: dict[str, list[tuple[int, str, DiscoveredExtractionAttempt]]] = {}
    for index, root in enumerate(normalized_roots):
        discovery = discover_extraction_attempts(
            root,
            duplicate_paper_policy=duplicate_paper_policy,
        )
        for attempt in discovery.attempts:
            if requested and attempt.paper_id not in requested:
                continue
            candidates.setdefault(attempt.paper_id, []).append(
                (index, root, attempt)
            )

    duplicate_ids = sorted(
        paper_id
        for paper_id, rows in candidates.items()
        if len(rows) > 1
    )
    if duplicate_ids and cross_root_duplicate_policy == "error":
        raise ValueError(
            "paper IDs occur in multiple extraction roots; resolve the root "
            "set or use cross_root_duplicate_policy='prefer_last_root': "
            + ", ".join(duplicate_ids[:30])
        )

    selected: dict[str, tuple[str, DiscoveredExtractionAttempt]] = {}
    for paper_id, rows in candidates.items():
        if len(rows) == 1:
            _, root, attempt = rows[0]
            selected[paper_id] = (root, attempt)
            continue
        # Root order wins first; within the same root prefer the latest attempt
        # deterministically, even though discover_extraction_attempts normally
        # already made that decision.
        _, root, attempt = max(
            rows,
            key=lambda item: (item[0], *_attempt_key(item[2])),
        )
        selected[paper_id] = (root, attempt)

    bundles: dict[str, SemanticIRBundle] = {}
    sources: list[MultiRootPaperSource] = []
    for paper_id in sorted(selected):
        root, attempt = selected[paper_id]
        imported = load_existing_extraction_semantic_ir(
            attempt.attempt_directory
        )
        if imported.bundle.paper_id != paper_id:
            raise ValueError(
                "loaded bundle paper_id does not match selected paper: "
                f"{imported.bundle.paper_id!r} != {paper_id!r}"
            )
        bundles[paper_id] = imported.bundle
        sources.append(
            MultiRootPaperSource(
                paper_id=paper_id,
                root=root,
                attempt_directory=attempt.attempt_directory,
                active_chunks_path=attempt.active_chunks_path,
                attempt_id=attempt.attempt_id,
            )
        )

    available = sorted(bundles)
    missing = sorted(requested - set(available)) if requested else []
    result = MultiRootSemanticIRLoadResult(
        roots=normalized_roots,
        requested_paper_ids=sorted(requested),
        available_paper_ids=available,
        missing_paper_ids=missing,
        cross_root_duplicate_paper_ids=duplicate_ids,
        paper_sources=sources,
    )
    return result, bundles
