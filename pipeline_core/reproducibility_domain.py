from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Callable, Literal

import networkx as nx


ResultIdentityStatus = Literal["single_mention", "consolidated_exact"]


@dataclass(frozen=True)
class ReproducibilityEvidence:
    evidence_id: str
    domain_profile_id: str
    reproducibility_semantics_id: str
    paper_id: str
    evidence_kind: str
    reproducibility_scope: str
    value_numeric: float | None = None
    value_text: str = ""
    unit: str = ""
    n_spots: int | None = None
    n_substrates: int | None = None
    n_batches: int | None = None
    n_replicates: int | None = None
    n_particles: int | None = None
    mapping_area: str = ""
    internal_standard: str = ""
    result_identity_status: ResultIdentityStatus = "single_mention"
    source_expression: str = ""
    source_expressions: tuple[str, ...] = ()
    source_mention_node_ids: tuple[str, ...] = ()
    source_measurement_ids: tuple[str, ...] = ()
    source_measurement_group_ids: tuple[str, ...] = ()
    source_experiment_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_id", self.evidence_id),
            ("domain_profile_id", self.domain_profile_id),
            ("reproducibility_semantics_id", self.reproducibility_semantics_id),
            ("paper_id", self.paper_id),
            ("evidence_kind", self.evidence_kind),
            ("reproducibility_scope", self.reproducibility_scope),
        ):
            if not str(value).strip():
                raise ValueError(f"ReproducibilityEvidence {name} must not be empty.")
        if self.value_numeric is not None and self.value_text.strip():
            raise ValueError(
                "ReproducibilityEvidence preserves numeric/text XOR: "
                "value_numeric and value_text cannot both be populated."
            )
        if self.value_numeric is not None and not math.isfinite(self.value_numeric):
            raise ValueError("ReproducibilityEvidence numeric value must be finite.")
        for name in (
            "n_spots",
            "n_substrates",
            "n_batches",
            "n_replicates",
            "n_particles",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 when reported.")
        if self.result_identity_status not in {
            "single_mention",
            "consolidated_exact",
        }:
            raise ValueError(
                "Unknown reproducibility result_identity_status: "
                f"{self.result_identity_status!r}."
            )
        if not self.source_node_ids:
            raise ValueError(
                "ReproducibilityEvidence requires at least one grounded source node."
            )
        # Backward-compatible construction for existing tests/callers that
        # predate alpha4b.3b.4a.1. New providers always set the mention IDs.
        if not self.source_mention_node_ids:
            object.__setattr__(
                self,
                "source_mention_node_ids",
                (self.source_node_ids[0],),
            )
        if not set(self.source_mention_node_ids).issubset(
            set(self.source_node_ids)
        ):
            raise ValueError(
                "Reproducibility source mention IDs must be included in "
                "source_node_ids."
            )
        if self.result_identity_status == "single_mention" and len(
            self.source_mention_node_ids
        ) != 1:
            raise ValueError(
                "single_mention evidence must have exactly one source mention."
            )
        if self.result_identity_status == "consolidated_exact" and len(
            self.source_mention_node_ids
        ) < 2:
            raise ValueError(
                "consolidated_exact evidence must have at least two source mentions."
            )
        if self.source_expression.strip() and self.source_expressions:
            if self.source_expression not in self.source_expressions:
                raise ValueError(
                    "Primary source_expression must be included in "
                    "source_expressions when the latter is populated."
                )
        combined = set(self.source_measurement_ids)
        combined.update(self.source_measurement_group_ids)
        combined.update(self.source_experiment_ids)
        if not combined.issubset(set(self.source_node_ids)):
            raise ValueError(
                "Typed reproducibility source IDs must be included in source_node_ids."
            )

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        for key in (
            "source_expressions",
            "source_mention_node_ids",
            "source_measurement_ids",
            "source_measurement_group_ids",
            "source_experiment_ids",
            "source_node_ids",
        ):
            row[key] = list(row[key])
        row["source_mention_count"] = len(self.source_mention_node_ids)
        return row


ExtractReproducibilityEvidence = Callable[
    [nx.Graph, str],
    list[ReproducibilityEvidence],
]


@dataclass(frozen=True)
class ReproducibilityDomainAdapter:
    adapter_id: str
    domain_profile_id: str
    semantics_id: str
    scope_labels: frozenset[str]
    extract_evidence_fn: ExtractReproducibilityEvidence = field(repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("adapter_id", self.adapter_id),
            ("domain_profile_id", self.domain_profile_id),
            ("semantics_id", self.semantics_id),
        ):
            if not value.strip():
                raise ValueError(f"Reproducibility adapter {name} must not be empty.")
        if "unknown" not in self.scope_labels:
            raise ValueError(
                "Reproducibility adapters must expose an explicit 'unknown' scope."
            )
        if any(not scope.strip() for scope in self.scope_labels):
            raise ValueError("Reproducibility scope labels must not be empty.")

    def extract_evidence(
        self,
        graph: nx.Graph,
        paper_id: str,
    ) -> list[ReproducibilityEvidence]:
        evidence = self.extract_evidence_fn(graph, paper_id)
        seen: set[str] = set()
        for item in evidence:
            if item.evidence_id in seen:
                raise ValueError(
                    "Reproducibility evidence IDs must be unique: "
                    f"{item.evidence_id!r}"
                )
            seen.add(item.evidence_id)
            if item.domain_profile_id != self.domain_profile_id:
                raise ValueError(
                    "Reproducibility evidence/domain mismatch: "
                    f"{item.domain_profile_id!r} != {self.domain_profile_id!r}."
                )
            if item.reproducibility_semantics_id != self.semantics_id:
                raise ValueError(
                    "Reproducibility evidence/semantics mismatch: "
                    f"{item.reproducibility_semantics_id!r} != {self.semantics_id!r}."
                )
            if item.paper_id != paper_id:
                raise ValueError(
                    "Reproducibility evidence/paper mismatch: "
                    f"{item.paper_id!r} != {paper_id!r}."
                )
            if item.reproducibility_scope not in self.scope_labels:
                raise ValueError(
                    "Unknown reproducibility scope for adapter "
                    f"{self.adapter_id!r}: {item.reproducibility_scope!r}."
                )
        return evidence
