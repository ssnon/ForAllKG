"""Loading boundary for already strict-validated chunk graphs."""

from __future__ import annotations

from pathlib import Path

from pipeline_core.corpus.graph.knowledge_graph_validation_context import (
    RELATION_SEMANTICS_ALREADY_VALIDATED_CONTEXT_KEY,
)
from pipeline_core.corpus.schemas import KnowledgeGraph


def load_strict_validated_chunk_graph(
    json_path: str | Path,
) -> KnowledgeGraph:
    """Reload a strict-valid chunk without replaying legacy relation policy.

    Structural KnowledgeGraph validation remains active. Only the historical
    no-contract relation compatibility policy is suppressed because relation
    semantics were already validated by the active extraction domain before
    serialization.
    """
    path = Path(json_path)

    return KnowledgeGraph.model_validate_json(
        path.read_text(encoding="utf-8"),
        context={
            RELATION_SEMANTICS_ALREADY_VALIDATED_CONTEXT_KEY: True,
        },
    )
