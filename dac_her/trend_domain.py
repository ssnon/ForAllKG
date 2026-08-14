from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Mapping

import networkx as nx


TREND_EVIDENCE_CONTRACT_SEMANTICS_ID = "trend_evidence_contract_v1_alpha4c1"

TrendDirection = Literal[
    "positive",
    "negative",
 "non_monotonic",
    "unchanged",
    "unspecified",
]
TrendShape = Literal[
    "monotonic",
    "saturating",
    "single_optimum",
    "threshold",
    "u_shaped",
    "inverted_u",
    "unspecified",
]
TrendEvidenceBasis = Literal[
    "controlled_numeric_series",
    "controlled_numeric_pair",
    "reported_directional_claim",
    "reported_correlation",
]
TrendCausalStatus = Literal["not_asserted", "source_asserted"]

TREND_DIRECTIONS = frozenset({
    "positive", "negative", "non_monotonic", "unchanged", "unspecified",
})
TREND_SHAPES = frozenset({
    "monotonic", "saturating", "single_optimum", "threshold",
    "u_shaped", "inverted_u", "unspecified",
})
TREND_EVIDENCE_BASES = frozenset({
    "controlled_numeric_series", "controlled_numeric_pair",
    "reported_directional_claim", "reported_correlation",
})
TREND_SOURCE_INPUTS = frozenset({
    "canonical_graph", "measurement_result_identity",
    "method_context", "comparison_context",
})


def _validate_numeric_text_xor(*, numeric: float | None, text: str, label: str) -> None:
    has_numeric = numeric is not None
    has_text = bool(text.strip())
    if has_numeric == has_text:
        raise ValueError(f"{label} requires exactly one of numeric or text value.")
    if numeric is not None and not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite when populated.")


@dataclass(frozen=True)
class TrendSeriesPoint:
    point_id: str
    independent_value_numeric: float | None = None
    independent_value_text: str = ""
    independent_unit: str = ""
    dependent_value_numeric: float | None = None
    dependent_value_text: str = ""
    dependent_unit: str = ""
    source_measurement_result_ids: tuple[str, ...] = ()
    source_measurement_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.point_id.strip():
            raise ValueError("TrendSeriesPoint point_id must not be empty.")
        _validate_numeric_text_xor(
            numeric=self.independent_value_numeric,
            text=self.independent_value_text,
            label="TrendSeriesPoint independent value",
        )
        _validate_numeric_text_xor(
            numeric=self.dependent_value_numeric,
            text=self.dependent_value_text,
            label="TrendSeriesPoint dependent value",
        )
        if not self.source_node_ids:
            raise ValueError("TrendSeriesPoint requires at least one grounded source node.")
        if not set(self.source_measurement_ids).issubset(set(self.source_node_ids)):
            raise ValueError(
                "TrendSeriesPoint source_measurement_ids must be included in source_node_ids."
            )

    @property
    def is_fully_numeric(self) -> bool:
        return (
            self.independent_value_numeric is not None
            and self.dependent_value_numeric is not None
        )

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        for key in (
            "source_measurement_result_ids", "source_measurement_ids", "source_node_ids",
        ):
            row[key] = list(row[key])
        return row


@dataclass(frozen=True)
class TrendEvidence:
    trend_id: str
    domain_profile_id: str
    trend_semantics_id: str
    paper_id: str
    independent_variable_key: str
    independent_variable_label: str
    dependent_observable_key: str
    dependent_observable_label: str
    direction: TrendDirection
    shape: TrendShape
    evidence_basis: TrendEvidenceBasis
    causal_status: TrendCausalStatus = "not_asserted"
    varied_dimension: str = ""
    subject_ids: tuple[str, ...] = ()
    series_points: tuple[TrendSeriesPoint, ...] = ()
    source_expression: str = ""
    source_expressions: tuple[str, ...] = ()
    source_claim_ids: tuple[str, ...] = ()
    source_measurement_ids: tuple[str, ...] = ()
    source_measurement_group_ids: tuple[str, ...] = ()
    source_experiment_ids: tuple[str, ...] = ()
    source_calculation_ids: tuple[str, ...] = ()
    source_measurement_result_ids: tuple[str, ...] = ()
    source_method_context_ids: tuple[str, ...] = ()
    source_comparison_context_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()
    requires_verification: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("trend_id", self.trend_id),
            ("domain_profile_id", self.domain_profile_id),
            ("trend_semantics_id", self.trend_semantics_id),
            ("paper_id", self.paper_id),
            ("independent_variable_key", self.independent_variable_key),
            ("independent_variable_label", self.independent_variable_label),
            ("dependent_observable_key", self.dependent_observable_key),
            ("dependent_observable_label", self.dependent_observable_label),
        ):
            if not str(value).strip():
                raise ValueError(f"TrendEvidence {name} must not be empty.")
        if self.independent_variable_key == self.dependent_observable_key:
            raise ValueError("TrendEvidence independent and dependent keys must differ.")
        if self.direction not in TREND_DIRECTIONS:
            raise ValueError(f"Unknown TrendEvidence direction: {self.direction!r}.")
        if self.shape not in TREND_SHAPES:
            raise ValueError(f"Unknown TrendEvidence shape: {self.shape!r}.")
        if self.evidence_basis not in TREND_EVIDENCE_BASES:
            raise ValueError(f"Unknown TrendEvidence evidence_basis: {self.evidence_basis!r}.")
        if self.causal_status not in {"not_asserted", "source_asserted"}:
            raise ValueError(f"Unknown TrendEvidence causal_status: {self.causal_status!r}.")
        if not self.source_node_ids:
            raise ValueError("TrendEvidence requires at least one grounded source node.")

        typed_graph_nodes = set(self.source_claim_ids)
        typed_graph_nodes.update(self.source_measurement_ids)
        typed_graph_nodes.update(self.source_measurement_group_ids)
        typed_graph_nodes.update(self.source_experiment_ids)
        typed_graph_nodes.update(self.source_calculation_ids)
        if not typed_graph_nodes.issubset(set(self.source_node_ids)):
            raise ValueError(
                "Typed TrendEvidence graph-source IDs must be included in source_node_ids."
            )
        if self.source_expression.strip() and self.source_expressions:
            if self.source_expression not in self.source_expressions:
                raise ValueError(
                    "Primary source_expression must be included in source_expressions."
                )

        numeric_basis = self.evidence_basis in {
            "controlled_numeric_series", "controlled_numeric_pair",
        }
        claim_basis = self.evidence_basis in {
            "reported_directional_claim", "reported_correlation",
        }

        if numeric_basis:
            if not self.varied_dimension.strip():
                raise ValueError("Controlled numeric TrendEvidence requires varied_dimension.")
            required_points = 2 if self.evidence_basis == "controlled_numeric_pair" else 3
            if self.evidence_basis == "controlled_numeric_pair" and len(self.series_points) != 2:
                raise ValueError("controlled_numeric_pair requires exactly two TrendSeriesPoint values.")
            if self.evidence_basis == "controlled_numeric_series" and len(self.series_points) < 3:
                raise ValueError("controlled_numeric_series requires at least three TrendSeriesPoint values.")
            if any(not point.is_fully_numeric for point in self.series_points):
                raise ValueError(
                    "Controlled numeric TrendEvidence requires fully numeric independent/dependent points."
                )
            if not (self.source_measurement_group_ids or self.source_experiment_ids):
                raise ValueError(
                    "Controlled numeric TrendEvidence requires explicit MeasurementGroup or Experiment lineage."
                )
            if not (self.source_measurement_ids or self.source_measurement_result_ids):
                raise ValueError(
                    "Controlled numeric TrendEvidence requires Measurement or MeasurementResult provenance."
                )
            if self.causal_status != "not_asserted":
                raise ValueError(
                    "Controlled numeric trends do not establish causation; causal_status must be 'not_asserted'."
                )
            independent_values = [
                float(point.independent_value_numeric)
                for point in self.series_points
                if point.independent_value_numeric is not None
            ]
            if len(independent_values) != len(set(independent_values)):
                raise ValueError(
                    "Controlled numeric TrendEvidence requires distinct independent-variable values."
                )
            if len({point.independent_unit.strip() for point in self.series_points}) != 1:
                raise ValueError("Controlled numeric trend points must use one independent unit representation.")
            if len({point.dependent_unit.strip() for point in self.series_points}) != 1:
                raise ValueError("Controlled numeric trend points must use one dependent unit representation.")

        if claim_basis:
            if self.series_points:
                raise ValueError(
                    "Claim-based TrendEvidence must not masquerade as numeric series evidence."
                )
            if not self.source_claim_ids:
                raise ValueError("Claim-based TrendEvidence requires source_claim_ids.")
            if not (self.source_expression.strip() or self.source_expressions):
                raise ValueError("Claim-based TrendEvidence requires source text provenance.")
        if self.evidence_basis == "reported_correlation" and self.causal_status != "not_asserted":
            raise ValueError("reported_correlation cannot be upgraded to causal evidence.")

        if self.direction == "unchanged" and self.shape != "unspecified":
            raise ValueError("unchanged TrendEvidence must use shape='unspecified'.")
        if self.direction == "non_monotonic" and self.shape == "monotonic":
            raise ValueError("non_monotonic TrendEvidence cannot use shape='monotonic'.")
        if self.shape in {"single_optimum", "u_shaped", "inverted_u"} and self.direction != "non_monotonic":
            raise ValueError(f"shape={self.shape!r} requires direction='non_monotonic'.")

        point_source_nodes = {
            node_id for point in self.series_points for node_id in point.source_node_ids
        }
        if not point_source_nodes.issubset(set(self.source_node_ids)):
            raise ValueError(
                "TrendSeriesPoint source nodes must be included in parent TrendEvidence source_node_ids."
            )

    @property
    def is_quantitative(self) -> bool:
        return self.evidence_basis in {
            "controlled_numeric_series", "controlled_numeric_pair",
        }

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["series_points"] = [point.to_row() for point in self.series_points]
        for key in (
            "subject_ids", "source_expressions", "source_claim_ids",
            "source_measurement_ids", "source_measurement_group_ids",
            "source_experiment_ids", "source_calculation_ids",
            "source_measurement_result_ids", "source_method_context_ids",
            "source_comparison_context_ids", "source_node_ids",
        ):
            row[key] = list(row[key])
        return row


@dataclass(frozen=True)
class TrendEvidenceSource:
    graph: nx.Graph = field(repr=False)
    paper_id: str = ""
    measurement_result_rows: tuple[Mapping[str, Any], ...] = ()
    method_context_rows: tuple[Mapping[str, Any], ...] = ()
    comparison_context_rows: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("TrendEvidenceSource paper_id must not be empty.")

    @property
    def available_inputs(self) -> frozenset[str]:
        inputs = {"canonical_graph"}
        if self.measurement_result_rows:
            inputs.add("measurement_result_identity")
        if self.method_context_rows:
            inputs.add("method_context")
        if self.comparison_context_rows:
            inputs.add("comparison_context")
        return frozenset(inputs)


ExtractTrendEvidence = Callable[[TrendEvidenceSource], list[TrendEvidence]]


@dataclass(frozen=True)
class TrendDomainAdapter:
    adapter_id: str
    domain_profile_id: str
    semantics_id: str
    supported_evidence_bases: frozenset[str]
    required_inputs: frozenset[str]
    extract_evidence_fn: ExtractTrendEvidence = field(repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("adapter_id", self.adapter_id),
            ("domain_profile_id", self.domain_profile_id),
            ("semantics_id", self.semantics_id),
        ):
            if not value.strip():
                raise ValueError(f"Trend adapter {name} must not be empty.")
        if not self.supported_evidence_bases:
            raise ValueError("Trend adapters must declare at least one supported evidence basis.")
        unknown_bases = self.supported_evidence_bases - TREND_EVIDENCE_BASES
        if unknown_bases:
            raise ValueError(f"Trend adapter declares unknown evidence bases: {sorted(unknown_bases)!r}.")
        unknown_inputs = self.required_inputs - TREND_SOURCE_INPUTS
        if unknown_inputs:
            raise ValueError(f"Trend adapter declares unknown required inputs: {sorted(unknown_inputs)!r}.")
        if "canonical_graph" not in self.required_inputs:
            raise ValueError("Trend adapters must require the canonical_graph input.")

    def extract_evidence(self, source: TrendEvidenceSource) -> list[TrendEvidence]:
        missing_inputs = self.required_inputs - source.available_inputs
        if missing_inputs:
            raise ValueError(
                f"Trend adapter is missing required source inputs: {sorted(missing_inputs)!r}."
            )
        evidence = self.extract_evidence_fn(source)
        seen: set[str] = set()
        for item in evidence:
            if item.trend_id in seen:
                raise ValueError(f"Trend evidence IDs must be unique: {item.trend_id!r}.")
            seen.add(item.trend_id)
            if item.domain_profile_id != self.domain_profile_id:
                raise ValueError(
                    f"Trend evidence/domain mismatch: {item.domain_profile_id!r} != {self.domain_profile_id!r}."
                )
            if item.trend_semantics_id != self.semantics_id:
                raise ValueError(
                    f"Trend evidence/semantics mismatch: {item.trend_semantics_id!r} != {self.semantics_id!r}."
                )
            if item.paper_id != source.paper_id:
                raise ValueError(
                    f"Trend evidence/paper mismatch: {item.paper_id!r} != {source.paper_id!r}."
                )
            if item.evidence_basis not in self.supported_evidence_bases:
                raise ValueError(
                    f"Trend evidence basis is not supported by adapter {self.adapter_id!r}: {item.evidence_basis!r}."
                )
        return evidence
