"""Shared evidence/claim graph relation constraints."""

from __future__ import annotations

from pipeline_core.graph_domain import RelationConstraint


COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS = (
    RelationConstraint(
        "HAS_MEASUREMENT",
        source_types=frozenset({"Experiment", "Calculation"}),
        target_types=frozenset({"Measurement"}),
    ),
    RelationConstraint(
        "MEASURED_FOR",
        source_types=frozenset({"Measurement"}),
        target_types=frozenset({"Entity"}),
    ),
    RelationConstraint(
        "IN_MEASUREMENT_GROUP",
        source_types=frozenset({"Measurement"}),
        target_types=frozenset({"MeasurementGroup"}),
    ),
    RelationConstraint(
        "SUPPORTS_CLAIM",
        source_types=frozenset({"Measurement", "Experiment", "Calculation"}),
        target_types=frozenset({"ObservationClaim", "MechanismClaim"}),
    ),
    RelationConstraint(
        "INTERPRETED_AS",
        source_types=frozenset({"ObservationClaim"}),
        target_types=frozenset({"MechanismClaim"}),
    ),
    RelationConstraint(
        "APPLIES_TO",
        source_types=frozenset({"ObservationClaim", "MechanismClaim"}),
        target_types=frozenset({"Entity"}),
    ),
)

