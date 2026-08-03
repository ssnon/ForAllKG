from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from dac_her.schemas import KnowledgeGraph
from dac_her.vocab_registry import VocabularyRegistry


@dataclass(frozen=True)
class VocabularyIssue:
    node_id: str
    vocabulary: str
    raw_id: str
    raw_label: str
    normalized_id: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_graph_vocabularies(
    graph: KnowledgeGraph,
    *,
    experiment_registry: VocabularyRegistry,
    metric_registry: VocabularyRegistry,
) -> tuple[KnowledgeGraph, list[VocabularyIssue]]:
    issues: list[VocabularyIssue] = []

    experiments = []
    for node in graph.experiments:
        canonical_id, canonical_label, registered = (
            experiment_registry.canonical_or_unregistered(
                entry_id=node.experiment_type,
                label=node.method_label or node.raw_method_name or node.name,
            )
        )
        entry = experiment_registry.resolve(canonical_id, canonical_label)
        family = (
            str(entry.metadata.get("family"))
            if entry is not None and entry.metadata.get("family")
            else node.experiment_family
        )
        experiments.append(node.model_copy(update={
            "experiment_type": canonical_id,
            "experiment_family": family,
            "method_label": canonical_label,
        }))
        if not registered:
            issues.append(VocabularyIssue(
                node_id=node.id,
                vocabulary="experiment_methods",
                raw_id=node.experiment_type,
                raw_label=node.method_label,
                normalized_id=canonical_id,
                status="unregistered",
            ))

    measurements = []
    for node in graph.measurements:
        canonical_id, canonical_label, registered = (
            metric_registry.canonical_or_unregistered(
                entry_id=node.metric_id,
                label=node.metric,
            )
        )
        measurements.append(node.model_copy(update={
            "metric_id": canonical_id,
            "metric": canonical_label,
        }))
        if not registered:
            issues.append(VocabularyIssue(
                node_id=node.id,
                vocabulary="metrics",
                raw_id=node.metric_id,
                raw_label=node.metric,
                normalized_id=canonical_id,
                status="unregistered",
            ))

    payload = graph.model_dump()
    payload["experiments"] = [node.model_dump() for node in experiments]
    payload["measurements"] = [node.model_dump() for node in measurements]
    return KnowledgeGraph.model_validate(payload), issues
