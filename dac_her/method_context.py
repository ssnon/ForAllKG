from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


MethodDimensionStatus = Literal["known", "unknown", "ambiguous"]
ProtocolComparability = Literal[
    "same_protocol",
    "harmonized_protocol",
    "partially_matched",
    "different_protocol",
    "unknown",
]


@dataclass(frozen=True)
class MethodDimensionValue:
    name: str
    status: MethodDimensionStatus
    normalized_value: str = ""
    source_values: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()
    provenance_scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Method dimension name must not be empty.")
        if self.status not in {"known", "unknown", "ambiguous"}:
            raise ValueError(
                f"Unknown method dimension status: {self.status!r}"
            )
        if self.status == "known" and not self.normalized_value.strip():
            raise ValueError(
                f"Known method dimension {self.name!r} must have a value."
            )
        if self.status == "unknown" and self.normalized_value:
            raise ValueError(
                f"Unknown method dimension {self.name!r} cannot have a value."
            )
        if self.status != "unknown" and not self.provenance_scopes:
            raise ValueError(
                f"Non-unknown method dimension {self.name!r} requires "
                "explicit provenance_scopes."
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MethodContext:
    method_context_id: str
    domain_profile_id: str
    method_semantics_id: str
    paper_id: str
    measurement_id: str
    producer_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    dimensions: tuple[MethodDimensionValue, ...]
    source_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("method_context_id", self.method_context_id),
            ("domain_profile_id", self.domain_profile_id),
            ("method_semantics_id", self.method_semantics_id),
            ("paper_id", self.paper_id),
            ("measurement_id", self.measurement_id),
        ):
            if not str(value).strip():
                raise ValueError(f"MethodContext {name} must not be empty.")
        names = [dimension.name for dimension in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError(
                "MethodContext dimensions must be unique by name."
            )

    @property
    def dimension_map(self) -> dict[str, MethodDimensionValue]:
        return {dimension.name: dimension for dimension in self.dimensions}

    def to_row(self) -> dict[str, object]:
        return {
            "method_context_id": self.method_context_id,
            "domain_profile_id": self.domain_profile_id,
            "method_semantics_id": self.method_semantics_id,
            "paper_id": self.paper_id,
            "measurement_id": self.measurement_id,
            "producer_ids": list(self.producer_ids),
            "subject_ids": list(self.subject_ids),
            "dimensions": [
                dimension.to_dict() for dimension in self.dimensions
            ],
            "source_node_ids": list(self.source_node_ids),
        }


@dataclass(frozen=True)
class MethodContextSemantics:
    semantics_id: str
    dimensions: tuple[str, ...]
    critical_dimensions: frozenset[str]
    numeric_ranking_allowed_protocols: frozenset[ProtocolComparability] = (
        frozenset({"same_protocol"})
    )

    def __post_init__(self) -> None:
        if not self.semantics_id.strip():
            raise ValueError("Method semantics_id must not be empty.")
        if not self.dimensions:
            raise ValueError("Method semantics dimensions must not be empty.")
        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError("Method semantics dimensions must be unique.")
        unknown = self.critical_dimensions - set(self.dimensions)
        if unknown:
            raise ValueError(
                "Method critical dimensions must be a subset of dimensions: "
                f"{sorted(unknown)!r}"
            )
        allowed_states = {
            "same_protocol",
            "harmonized_protocol",
            "partially_matched",
            "different_protocol",
            "unknown",
        }
        bad_states = (
            set(self.numeric_ranking_allowed_protocols) - allowed_states
        )
        if bad_states:
            raise ValueError(
                "Unknown numeric protocol states: "
                f"{sorted(bad_states)!r}"
            )


@dataclass(frozen=True)
class ProtocolAssessment:
    protocol_assessment_id: str
    method_semantics_id: str
    observable_key: str
    left_context_id: str
    right_context_id: str
    left_method_context_id: str
    right_method_context_id: str
    left_paper_id: str
    right_paper_id: str
    comparability: ProtocolComparability
    matched_dimensions: tuple[str, ...]
    mismatched_dimensions: tuple[str, ...]
    unknown_dimensions: tuple[str, ...]
    ambiguous_dimensions: tuple[str, ...]
    critical_mismatches: tuple[str, ...]
    reasons: tuple[str, ...]
    numeric_protocol_gate: bool

    def to_row(self) -> dict[str, object]:
        return asdict(self)
