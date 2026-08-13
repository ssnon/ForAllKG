from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib

import networkx as nx

from dac_her.metric_definition_domain import (
    MetricDefinitionContext,
    MetricDefinitionDomainAdapter,
)


_ALLOWED_SOURCE_TYPES = frozenset({
    "Measurement",
    "MeasurementGroup",
    "Experiment",
    "Calculation",
})


def stable_metric_definition_id(
    *,
    paper_id: str,
    measurement_id: str,
    semantics_id: str,
) -> str:
    payload = "|".join((paper_id, measurement_id, semantics_id))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"metricdef:{digest}"


@dataclass(frozen=True)
class MetricDefinitionAudit:
    domain_profile_id: str
    metric_definition_semantics_id: str
    context_count: int
    known_count: int
    partial_count: int
    unknown_count: int
    observable_counts: dict[str, int]
    definition_family_counts: dict[str, int]
    aggregation_scope_counts: dict[str, int]
    explicit_formula_count: int
    issues: tuple[str, ...]
    structural_gate: bool

    def to_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["issues"] = list(self.issues)
        return row


def audit_metric_definition_contexts(
    *,
    contexts: list[MetricDefinitionContext],
    source_graphs: dict[str, nx.Graph],
    adapter: MetricDefinitionDomainAdapter,
) -> MetricDefinitionAudit:
    issues: list[str] = []
    seen_measurements: set[tuple[str, str]] = set()

    for item in contexts:
        graph = source_graphs.get(item.paper_id)
        if graph is None:
            issues.append(f"missing_source_graph:{item.paper_id}")
            continue
        key = (item.paper_id, item.measurement_id)
        if key in seen_measurements:
            issues.append(
                f"duplicate_measurement_context:{item.paper_id}:{item.measurement_id}"
            )
        seen_measurements.add(key)

        if item.measurement_id not in graph:
            issues.append(
                f"missing_measurement:{item.context_id}:{item.measurement_id}"
            )
        elif str(graph.nodes[item.measurement_id].get("type", "")) != "Measurement":
            issues.append(
                f"context_measurement_not_measurement:{item.context_id}:"
                f"{item.measurement_id}"
            )

        for node_id in item.source_node_ids:
            if node_id not in graph:
                issues.append(f"missing_source_node:{item.context_id}:{node_id}")
                continue
            node_type = str(graph.nodes[node_id].get("type", ""))
            if node_type not in _ALLOWED_SOURCE_TYPES:
                issues.append(
                    f"unsupported_source_type:{item.context_id}:"
                    f"{node_id}:{node_type}"
                )

        if item.definition_status == "known" and item.definition_family.endswith(
            "_unspecified"
        ):
            issues.append(
                f"known_definition_unspecified_family:{item.context_id}"
            )
        if item.formula_text.strip() and not (
            item.source_calculation_ids or item.source_measurement_ids
        ):
            issues.append(
                f"formula_without_grounded_source:{item.context_id}"
            )

    status_counts = Counter(item.definition_status for item in contexts)
    observable_counts = Counter(item.observable_key for item in contexts)
    family_counts = Counter(item.definition_family for item in contexts)
    aggregation_counts = Counter(item.aggregation_scope for item in contexts)
    return MetricDefinitionAudit(
        domain_profile_id=adapter.domain_profile_id,
        metric_definition_semantics_id=adapter.semantics_id,
        context_count=len(contexts),
        known_count=status_counts["known"],
        partial_count=status_counts["partial"],
        unknown_count=status_counts["unknown"],
        observable_counts=dict(sorted(observable_counts.items())),
        definition_family_counts=dict(sorted(family_counts.items())),
        aggregation_scope_counts=dict(sorted(aggregation_counts.items())),
        explicit_formula_count=sum(bool(item.formula_text.strip()) for item in contexts),
        issues=tuple(sorted(set(issues))),
        structural_gate=not issues,
    )
