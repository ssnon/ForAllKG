from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Literal

import networkx as nx

from dac_her.method_context import MethodContext, MethodContextSemantics


DimensionStatus = Literal["known", "unknown", "ambiguous"]
ComparisonCompatibility = Literal[
    "compatible",
    "partially_compatible",
    "incompatible",
    "unknown",
]
MeasurementUnitStatus = Literal["matched", "mismatched", "unknown"]
NumericRankingMode = Literal["allowed_if_complete", "disabled"]
RankingDirection = Literal["higher_better", "lower_better", "none"]


@dataclass(frozen=True)
class ObservableComparisonPolicy:
    policy_id: str
    family: str
    observable_keys: frozenset[str]
    applicable_dimensions: tuple[str, ...]
    ranking_required_dimensions: frozenset[str]
    numeric_ranking_mode: NumericRankingMode = "disabled"
    ranking_direction: RankingDirection = "none"

    def __post_init__(self) -> None:
        for name, value in (
            ("policy_id", self.policy_id),
            ("family", self.family),
        ):
            if not value.strip():
                raise ValueError(
                    f"ObservableComparisonPolicy {name} must not be empty."
                )
        if not self.observable_keys:
            raise ValueError(
                "ObservableComparisonPolicy observable_keys must not be empty."
            )
        if any(not key.strip() for key in self.observable_keys):
            raise ValueError("Observable keys must not be empty.")
        if len(self.applicable_dimensions) != len(
            set(self.applicable_dimensions)
        ):
            raise ValueError(
                "Observable applicable dimensions must be unique."
            )
        unknown_required = (
            self.ranking_required_dimensions
            - set(self.applicable_dimensions)
        )
        if unknown_required:
            raise ValueError(
                "ranking_required_dimensions must be a subset of "
                f"applicable_dimensions: {sorted(unknown_required)!r}"
            )
        if (
            self.numeric_ranking_mode == "allowed_if_complete"
            and not self.ranking_required_dimensions
        ):
            raise ValueError(
                "Rankable observable policy requires explicit ranking "
                "dimensions."
            )
        if (
            self.numeric_ranking_mode == "allowed_if_complete"
            and self.ranking_direction == "none"
        ):
            raise ValueError(
                "Rankable observable policy requires ranking_direction."
            )
        if (
            self.numeric_ranking_mode == "disabled"
            and self.ranking_direction != "none"
        ):
            raise ValueError(
                "Disabled numeric ranking must use ranking_direction='none'."
            )


@dataclass(frozen=True)
class ComparisonDimensionValue:
    name: str
    status: DimensionStatus
    normalized_value: str = ""
    source_values: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Comparison dimension name must not be empty.")
        if self.status not in {"known", "unknown", "ambiguous"}:
            raise ValueError(
                f"Unknown comparison dimension status: {self.status!r}"
            )
        if self.status == "known" and not self.normalized_value.strip():
            raise ValueError(
                f"Known comparison dimension {self.name!r} must have a value."
            )
        if self.status == "unknown" and self.normalized_value:
            raise ValueError(
                f"Unknown comparison dimension {self.name!r} cannot have a value."
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonContext:
    context_id: str
    domain_profile_id: str
    comparison_semantics_id: str
    paper_id: str
    measurement_id: str
    observable_key: str
    observable_label: str
    value_numeric: float | None
    value_text: str
    unit: str
    source_expression: str
    subject_ids: tuple[str, ...]
    dimensions: tuple[ComparisonDimensionValue, ...]
    source_node_ids: tuple[str, ...]
    method_context_id: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("context_id", self.context_id),
            ("domain_profile_id", self.domain_profile_id),
            ("comparison_semantics_id", self.comparison_semantics_id),
            ("paper_id", self.paper_id),
            ("measurement_id", self.measurement_id),
            ("observable_key", self.observable_key),
        ):
            if not str(value).strip():
                raise ValueError(
                    f"ComparisonContext {name} must not be empty."
                )
        names = [item.name for item in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError(
                "ComparisonContext dimensions must be unique by name."
            )
        if self.value_numeric is not None and self.value_text.strip():
            raise ValueError(
                "ComparisonContext preserves the strict measurement XOR: "
                "value_numeric and value_text cannot both be populated."
            )

    @property
    def dimension_map(self) -> dict[str, ComparisonDimensionValue]:
        return {item.name: item for item in self.dimensions}

    def to_row(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "domain_profile_id": self.domain_profile_id,
            "comparison_semantics_id": self.comparison_semantics_id,
            "paper_id": self.paper_id,
            "measurement_id": self.measurement_id,
            "observable_key": self.observable_key,
            "observable_label": self.observable_label,
            "value_numeric": self.value_numeric,
            "value_text": self.value_text,
            "unit": self.unit,
            "source_expression": self.source_expression,
            "subject_ids": list(self.subject_ids),
            "dimensions": [
                item.to_dict() for item in self.dimensions
            ],
            "source_node_ids": list(self.source_node_ids),
            "method_context_id": self.method_context_id,
        }


@dataclass(frozen=True)
class ComparisonAssessment:
    assessment_id: str
    comparison_semantics_id: str
    observable_key: str
    observable_policy_id: str
    observable_family: str
    applicable_dimensions: tuple[str, ...]
    ranking_required_dimensions: tuple[str, ...]
    numeric_ranking_mode: NumericRankingMode
    ranking_direction: RankingDirection
    left_context_id: str
    right_context_id: str
    left_paper_id: str
    right_paper_id: str
    compatibility: ComparisonCompatibility
    measurement_unit_status: MeasurementUnitStatus
    matched_dimensions: tuple[str, ...]
    mismatched_dimensions: tuple[str, ...]
    unknown_dimensions: tuple[str, ...]
    ambiguous_dimensions: tuple[str, ...]
    reasons: tuple[str, ...]
    numeric_ranking_allowed: bool
    protocol_assessment_id: str = ""
    protocol_comparability: str = "not_applicable"

    def to_row(self) -> dict[str, object]:
        return asdict(self)


ExtractComparisonContexts = Callable[
    [nx.Graph, str],
    list[ComparisonContext],
]
ExtractMethodContexts = Callable[
    [nx.Graph, str],
    list[MethodContext],
]


@dataclass(frozen=True)
class ComparisonDomainAdapter:
    adapter_id: str
    domain_profile_id: str
    semantics_id: str
    dimensions: tuple[str, ...]
    required_for_numeric_ranking: frozenset[str]
    extract_contexts_fn: ExtractComparisonContexts = field(repr=False)
    observable_policies: tuple[ObservableComparisonPolicy, ...] = ()
    method_semantics: MethodContextSemantics | None = None
    extract_method_contexts_fn: ExtractMethodContexts | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("adapter_id", self.adapter_id),
            ("domain_profile_id", self.domain_profile_id),
            ("semantics_id", self.semantics_id),
        ):
            if not value.strip():
                raise ValueError(
                    f"Comparison adapter {name} must not be empty."
                )
        if not self.dimensions:
            raise ValueError(
                "Comparison adapter dimensions must not be empty."
            )
        if any(not value.strip() for value in self.dimensions):
            raise ValueError(
                "Comparison adapter dimension names must not be empty."
            )
        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError(
                "Comparison adapter dimensions must be unique."
            )
        unknown = self.required_for_numeric_ranking - set(
            self.dimensions
        )
        if unknown:
            raise ValueError(
                "Comparison required_for_numeric_ranking must be a "
                f"subset of dimensions: {sorted(unknown)!r}"
            )

        if (
            self.method_semantics is None
            and self.extract_method_contexts_fn is not None
        ):
            raise ValueError(
                "Method extractor requires method_semantics."
            )
        if (
            self.method_semantics is not None
            and self.extract_method_contexts_fn is None
        ):
            raise ValueError(
                "method_semantics requires a method extractor."
            )

        seen_keys: set[str] = set()
        seen_policy_ids: set[str] = set()
        for policy in self.observable_policies:
            if policy.policy_id in seen_policy_ids:
                raise ValueError(
                    "Observable comparison policy IDs must be unique: "
                    f"{policy.policy_id!r}"
                )
            seen_policy_ids.add(policy.policy_id)
            unknown_dimensions = (
                set(policy.applicable_dimensions)
                - set(self.dimensions)
            )
            if unknown_dimensions:
                raise ValueError(
                    f"Policy {policy.policy_id!r} references unknown "
                    f"dimensions: {sorted(unknown_dimensions)!r}"
                )
            overlap = seen_keys & set(policy.observable_keys)
            if overlap:
                raise ValueError(
                    "Observable keys must belong to exactly one policy: "
                    f"{sorted(overlap)!r}"
                )
            seen_keys.update(policy.observable_keys)

    def policy_for(
        self,
        observable_key: str,
    ) -> ObservableComparisonPolicy | None:
        for policy in self.observable_policies:
            if observable_key in policy.observable_keys:
                return policy
        return None

    def extract_contexts(
        self,
        graph: nx.Graph,
        paper_id: str,
    ) -> list[ComparisonContext]:
        contexts = self.extract_contexts_fn(graph, paper_id)
        expected_dimensions = set(self.dimensions)
        for context in contexts:
            if context.domain_profile_id != self.domain_profile_id:
                raise ValueError(
                    "Comparison context/domain mismatch: "
                    f"{context.domain_profile_id!r} != "
                    f"{self.domain_profile_id!r}"
                )
            if context.comparison_semantics_id != self.semantics_id:
                raise ValueError(
                    "Comparison context/semantics mismatch: "
                    f"{context.comparison_semantics_id!r} != "
                    f"{self.semantics_id!r}"
                )
            found = {item.name for item in context.dimensions}
            if found != expected_dimensions:
                raise ValueError(
                    "Comparison context dimension contract mismatch: "
                    f"expected={sorted(expected_dimensions)!r}, "
                    f"found={sorted(found)!r}"
                )
        return contexts

    def extract_method_contexts(
        self,
        graph: nx.Graph,
        paper_id: str,
    ) -> list[MethodContext]:
        if (
            self.method_semantics is None
            or self.extract_method_contexts_fn is None
        ):
            return []
        contexts = self.extract_method_contexts_fn(graph, paper_id)
        expected_dimensions = set(self.method_semantics.dimensions)
        for context in contexts:
            if context.domain_profile_id != self.domain_profile_id:
                raise ValueError(
                    "Method context/domain mismatch: "
                    f"{context.domain_profile_id!r} != "
                    f"{self.domain_profile_id!r}"
                )
            if (
                context.method_semantics_id
                != self.method_semantics.semantics_id
            ):
                raise ValueError(
                    "Method context/semantics mismatch: "
                    f"{context.method_semantics_id!r} != "
                    f"{self.method_semantics.semantics_id!r}"
                )
            found = {item.name for item in context.dimensions}
            if found != expected_dimensions:
                raise ValueError(
                    "Method context dimension contract mismatch: "
                    f"expected={sorted(expected_dimensions)!r}, "
                    f"found={sorted(found)!r}"
                )
        return contexts
