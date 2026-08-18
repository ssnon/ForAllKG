from __future__ import annotations

from typing import Iterable

import networkx as nx

from dac_her.metric_normalization_policy import (
    refine_distance_metric_id,
    refine_semantic_metric_id,
)
from dac_her.schemas import (
    KnowledgeGraph,
)
from dac_her.vocab_registry import (
    VocabularyRegistry,
)

from pipeline_core.graph_normalization_runtime import (
    VocabularyIssue,
    normalize_graph_vocabularies as _normalize_graph_vocabularies,
    normalize_networkx_metric_vocabularies as _normalize_networkx_metric_vocabularies,
)


PARAMETER_CONDITION_NAMES = {
    "analyte",
    "orbital",
    "site",
    "component",
    "isotope",
    "phase",
}


def normalize_graph_vocabularies(
    graph: KnowledgeGraph,
    *,
    experiment_registry: VocabularyRegistry,
    metric_registry: VocabularyRegistry,
    relation_semantics_already_validated: bool = False,
) -> tuple[
    KnowledgeGraph,
    list[VocabularyIssue],
]:
    def bound_metric_refiner(
        *,
        entry_id: str | None,
        label: str | None,
        source_texts: Iterable[
            str | None
        ],
    ) -> str | None:
        return refine_semantic_metric_id(
            entry_id=entry_id,
            label=label,
            source_texts=source_texts,
        )

    return _normalize_graph_vocabularies(
        graph,
        experiment_registry=experiment_registry,
        metric_registry=metric_registry,
        metric_refiner=bound_metric_refiner,
        parameter_condition_names=(
            PARAMETER_CONDITION_NAMES
        ),
        relation_semantics_already_validated=(
            relation_semantics_already_validated
        ),
    )


def normalize_networkx_metric_vocabularies(
    graph: nx.Graph,
    *,
    metric_registry: VocabularyRegistry,
) -> list[VocabularyIssue]:
    def bound_metric_refiner(
        *,
        entry_id: str | None,
        label: str | None,
        source_texts: Iterable[
            str | None
        ],
    ) -> str | None:
        return refine_semantic_metric_id(
            entry_id=entry_id,
            label=label,
            source_texts=source_texts,
        )

    return _normalize_networkx_metric_vocabularies(
        graph,
        metric_registry=metric_registry,
        metric_refiner=bound_metric_refiner,
        parameter_condition_names=(
            PARAMETER_CONDITION_NAMES
        ),
    )
