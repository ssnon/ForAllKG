from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Literal

import networkx as nx


DefinitionStatus = Literal["known", "partial", "unknown"]


@dataclass(frozen=True)
class MetricDefinitionContext:
    context_id: str
    domain_profile_id: str
    metric_definition_semantics_id: str
    paper_id: str
    measurement_id: str
    observable_key: str
    definition_status: DefinitionStatus
    definition_family: str
    aggregation_scope: str
    normalization_basis: str = "unspecified"
    reference_basis: str = "unspecified"
    criterion: str = ""
    formula_text: str = ""
    raman_peak: str = ""
    source_expression: str = ""
    source_measurement_ids: tuple[str, ...] = ()
    source_measurement_group_ids: tuple[str, ...] = ()
    source_experiment_ids: tuple[str, ...] = ()
    source_calculation_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("context_id", self.context_id),
            ("domain_profile_id", self.domain_profile_id),
            ("metric_definition_semantics_id", self.metric_definition_semantics_id),
            ("paper_id", self.paper_id),
            ("measurement_id", self.measurement_id),
            ("observable_key", self.observable_key),
            ("definition_family", self.definition_family),
            ("aggregation_scope", self.aggregation_scope),
            ("normalization_basis", self.normalization_basis),
            ("reference_basis", self.reference_basis),
        ):
            if not str(value).strip():
                raise ValueError(f"MetricDefinitionContext {name} must not be empty.")
        if self.definition_status not in {"known", "partial", "unknown"}:
            raise ValueError(
                f"Unknown metric definition status: {self.definition_status!r}."
            )
        if self.definition_status == "unknown" and not self.definition_family.endswith(
            "_unspecified"
        ):
            raise ValueError(
                "Unknown metric-definition contexts must use an explicit "
                "*_unspecified definition family."
            )
        if self.definition_status == "unknown" and any(
            (
                self.criterion.strip(),
                self.formula_text.strip(),
                self.normalization_basis not in {"unspecified", "not_applicable"},
                self.reference_basis not in {"unspecified", "not_applicable"},
            )
        ):
            raise ValueError(
                "Unknown metric-definition contexts cannot carry definition evidence."
            )
        if not self.source_node_ids:
            raise ValueError(
                "MetricDefinitionContext requires at least one grounded source node."
            )
        if self.measurement_id not in self.source_measurement_ids:
            raise ValueError(
                "MetricDefinitionContext measurement_id must be included in "
                "source_measurement_ids."
            )
        typed = set(self.source_measurement_ids)
        typed.update(self.source_measurement_group_ids)
        typed.update(self.source_experiment_ids)
        typed.update(self.source_calculation_ids)
        if not typed.issubset(set(self.source_node_ids)):
            raise ValueError(
                "Typed metric-definition source IDs must be included in source_node_ids."
            )

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        for key in (
            "source_measurement_ids",
            "source_measurement_group_ids",
            "source_experiment_ids",
            "source_calculation_ids",
            "source_node_ids",
        ):
            row[key] = list(row[key])
        return row


ExtractMetricDefinitionContexts = Callable[
    [nx.Graph, str],
    list[MetricDefinitionContext],
]


@dataclass(frozen=True)
class MetricDefinitionDomainAdapter:
    adapter_id: str
    domain_profile_id: str
    semantics_id: str
    supported_observable_keys: frozenset[str]
    definition_families: frozenset[str]
    aggregation_scopes: frozenset[str]
    normalization_bases: frozenset[str]
    reference_bases: frozenset[str]
    extract_contexts_fn: ExtractMetricDefinitionContexts = field(repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("adapter_id", self.adapter_id),
            ("domain_profile_id", self.domain_profile_id),
            ("semantics_id", self.semantics_id),
        ):
            if not value.strip():
                raise ValueError(f"Metric-definition adapter {name} must not be empty.")
        for name, values in (
            ("supported_observable_keys", self.supported_observable_keys),
            ("definition_families", self.definition_families),
            ("aggregation_scopes", self.aggregation_scopes),
            ("normalization_bases", self.normalization_bases),
            ("reference_bases", self.reference_bases),
        ):
            if not values or any(not value.strip() for value in values):
                raise ValueError(
                    f"Metric-definition adapter {name} must contain non-empty values."
                )

    def extract_contexts(
        self,
        graph: nx.Graph,
        paper_id: str,
    ) -> list[MetricDefinitionContext]:
        contexts = self.extract_contexts_fn(graph, paper_id)
        seen_context_ids: set[str] = set()
        seen_measurements: set[str] = set()
        for context in contexts:
            if context.context_id in seen_context_ids:
                raise ValueError(
                    "Metric-definition context IDs must be unique: "
                    f"{context.context_id!r}."
                )
            seen_context_ids.add(context.context_id)
            if context.measurement_id in seen_measurements:
                raise ValueError(
                    "Each supported Measurement must have exactly one "
                    "MetricDefinitionContext: "
                    f"{context.measurement_id!r}."
                )
            seen_measurements.add(context.measurement_id)
            if context.domain_profile_id != self.domain_profile_id:
                raise ValueError(
                    "Metric-definition context/domain mismatch: "
                    f"{context.domain_profile_id!r} != {self.domain_profile_id!r}."
                )
            if context.metric_definition_semantics_id != self.semantics_id:
                raise ValueError(
                    "Metric-definition context/semantics mismatch: "
                    f"{context.metric_definition_semantics_id!r} != {self.semantics_id!r}."
                )
            if context.paper_id != paper_id:
                raise ValueError(
                    "Metric-definition context/paper mismatch: "
                    f"{context.paper_id!r} != {paper_id!r}."
                )
            if context.observable_key not in self.supported_observable_keys:
                raise ValueError(
                    "Unsupported metric-definition observable: "
                    f"{context.observable_key!r}."
                )
            if context.definition_family not in self.definition_families:
                raise ValueError(
                    "Unknown metric-definition family: "
                    f"{context.definition_family!r}."
                )
            if context.aggregation_scope not in self.aggregation_scopes:
                raise ValueError(
                    "Unknown metric aggregation scope: "
                    f"{context.aggregation_scope!r}."
                )
            if context.normalization_basis not in self.normalization_bases:
                raise ValueError(
                    "Unknown metric normalization basis: "
                    f"{context.normalization_basis!r}."
                )
            if context.reference_basis not in self.reference_bases:
                raise ValueError(
                    "Unknown metric reference basis: "
                    f"{context.reference_basis!r}."
                )
        return contexts
