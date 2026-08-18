from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import networkx as nx

from pipeline_core.knowledge_graph_validation_context import (
    RELATION_SEMANTICS_ALREADY_VALIDATED_CONTEXT_KEY,
)

from dac_her.schemas import Condition, KnowledgeGraph
from dac_her.vocab_registry import VocabularyRegistry




from dac_her.metric_normalization_policy import (
    refine_distance_metric_id,
    refine_semantic_metric_id,
)


PARAMETER_CONDITION_NAMES = {
    "analyte",
    "orbital",
    "site",
    "component",
    "isotope",
    "phase",
}


@dataclass(frozen=True)
class VocabularyIssue:
    node_id: str
    vocabulary: str
    raw_id: str
    raw_label: str
    normalized_id: str
    status: str
    parameters: dict[str, str] | None = None
    matched_pattern: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _condition_key(condition: Condition) -> tuple[str, str, str, str, str]:
    return (
        condition.name.strip().lower(),
        "" if condition.value_numeric is None else str(condition.value_numeric),
        "" if condition.value_text is None else condition.value_text.strip().lower(),
        "" if condition.unit is None else condition.unit.strip().lower(),
        "" if condition.reference is None else condition.reference.strip().lower(),
    )


def _append_parameter_conditions(
    conditions: Iterable[Condition],
    parameters: dict[str, str],
) -> list[Condition]:
    result = list(conditions)
    seen = {_condition_key(condition) for condition in result}
    existing_parameter_names = {
        condition.name.strip().lower()
        for condition in result
        if condition.name.strip().lower() in PARAMETER_CONDITION_NAMES
    }

    for name, value in sorted(parameters.items()):
        normalized_name = str(name).strip().lower()
        normalized_value = str(value).strip()
        if not normalized_name or not normalized_value:
            continue
        if normalized_name in existing_parameter_names:
            continue
        condition = Condition(
            name=normalized_name,
            value_numeric=None,
            value_text=normalized_value,
            unit=None,
            reference=None,
        )
        key = _condition_key(condition)
        if key not in seen:
            result.append(condition)
            seen.add(key)
            existing_parameter_names.add(normalized_name)
    return result


def normalize_graph_vocabularies(
    graph: KnowledgeGraph,
    *,
    experiment_registry: VocabularyRegistry,
    metric_registry: VocabularyRegistry,
    relation_semantics_already_validated: bool = False,
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
        experiments.append(
            node.model_copy(
                update={
                    "experiment_type": canonical_id,
                    "experiment_family": family,
                    "method_label": canonical_label,
                }
            )
        )
        if not registered:
            issues.append(
                VocabularyIssue(
                    node_id=node.id,
                    vocabulary="experiment_methods",
                    raw_id=node.experiment_type,
                    raw_label=node.method_label,
                    normalized_id=canonical_id,
                    status="unregistered",
                )
            )

    measurements = []
    for node in graph.measurements:
        refined_metric_id = refine_semantic_metric_id(
            entry_id=node.metric_id,
            label=node.metric,
            source_texts=(
                node.source_expression,
                node.description,
                node.basis,
            ),
        )
        match = metric_registry.resolve_parameterized(
            entry_id=refined_metric_id,
            label=node.metric,
            source_texts=(
                node.source_expression,
                node.description,
                node.basis,
            ),
        )
        if match.entry is not None:
            canonical_id = match.entry.entry_id
            canonical_label = match.entry.label
            registered = True
        else:
            canonical_id, canonical_label, registered = (
                metric_registry.canonical_or_unregistered(
                    entry_id=node.metric_id,
                    label=node.metric,
                )
            )

        conditions = _append_parameter_conditions(
            node.conditions,
            dict(match.parameters),
        )
        measurements.append(
            node.model_copy(
                update={
                    "metric_id": canonical_id,
                    "metric": canonical_label,
                    "conditions": conditions,
                }
            )
        )

        if not registered:
            issues.append(
                VocabularyIssue(
                    node_id=node.id,
                    vocabulary="metrics",
                    raw_id=node.metric_id,
                    raw_label=node.metric,
                    normalized_id=canonical_id,
                    status="unregistered",
                    parameters=dict(match.parameters),
                    matched_pattern=match.matched_pattern,
                )
            )
        elif canonical_id != node.metric_id or match.parameters:
            issues.append(
                VocabularyIssue(
                    node_id=node.id,
                    vocabulary="metrics",
                    raw_id=node.metric_id,
                    raw_label=node.metric,
                    normalized_id=canonical_id,
                    status="normalized_parameterized",
                    parameters=dict(match.parameters),
                    matched_pattern=match.matched_pattern,
                )
            )

    payload = graph.model_dump()
    payload["experiments"] = [node.model_dump() for node in experiments]
    payload["measurements"] = [node.model_dump() for node in measurements]

    relation_validation_context = (
        {
            RELATION_SEMANTICS_ALREADY_VALIDATED_CONTEXT_KEY: True,
        }
        if relation_semantics_already_validated
        else None
    )

    return (
        KnowledgeGraph.model_validate(
            payload,
            context=relation_validation_context,
        ),
        issues,
    )


def _parse_conditions_json(value: Any) -> list[Condition]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    conditions: list[Condition] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            conditions.append(Condition.model_validate(item))
        except Exception:
            continue
    return conditions


def normalize_networkx_metric_vocabularies(
    graph: nx.Graph,
    *,
    metric_registry: VocabularyRegistry,
) -> list[VocabularyIssue]:
    """Migrate already-extracted Measurement nodes during paper build.

    This lets existing chunk JSON remain untouched while a rebuild adopts a
    generic metric ID plus structured analyte/orbital/site conditions.
    """
    issues: list[VocabularyIssue] = []
    for node_id, data in graph.nodes(data=True):
        if str(data.get("type", "")) != "Measurement":
            continue

        raw_id = str(data.get("metric_id", ""))
        raw_label = str(data.get("metric") or data.get("label") or "")
        source_expression = str(data.get("source_expression", ""))
        description = str(data.get("description", ""))
        basis = str(data.get("basis", ""))
        refined_metric_id = refine_semantic_metric_id(
            entry_id=raw_id,
            label=raw_label,
            source_texts=(source_expression, description, basis),
        )
        match = metric_registry.resolve_parameterized(
            entry_id=refined_metric_id,
            label=raw_label,
            source_texts=(source_expression, description, basis),
        )

        if match.entry is None:
            canonical_id, canonical_label, registered = (
                metric_registry.canonical_or_unregistered(
                    entry_id=raw_id,
                    label=raw_label,
                )
            )
        else:
            canonical_id = match.entry.entry_id
            canonical_label = match.entry.label
            registered = True

        conditions = _append_parameter_conditions(
            _parse_conditions_json(data.get("conditions_json")),
            dict(match.parameters),
        )
        parameter_payload = {
            condition.name: condition.value_text
            if condition.value_text is not None
            else condition.value_numeric
            for condition in conditions
            if condition.name.strip().lower() in PARAMETER_CONDITION_NAMES
        }

        data["metric_id"] = canonical_id
        data["metric"] = canonical_label
        data["label"] = canonical_label
        data["conditions_json"] = json.dumps(
            [condition.model_dump() for condition in conditions],
            ensure_ascii=False,
        )
        data["metric_parameters_json"] = json.dumps(
            parameter_payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        data["metric_registry_status"] = (
            "registered" if registered else "unregistered"
        )
        if match.matched_pattern:
            data["metric_matched_pattern"] = match.matched_pattern

        status = (
            "normalized_parameterized"
            if registered and (canonical_id != raw_id or match.parameters)
            else "registered"
            if registered
            else "unregistered"
        )
        if status != "registered":
            issues.append(
                VocabularyIssue(
                    node_id=str(node_id),
                    vocabulary="metrics",
                    raw_id=raw_id,
                    raw_label=raw_label,
                    normalized_id=canonical_id,
                    status=status,
                    parameters=dict(match.parameters),
                    matched_pattern=match.matched_pattern,
                )
            )
    return issues
