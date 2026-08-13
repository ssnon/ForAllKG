from __future__ import annotations

from dac_her.graph_domain import RelationConstraint


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


DAC_LEGACY_STRICT_RELATION_CONSTRAINTS = (
    RelationConstraint(
        "EVALUATED_IN",
        source_types=frozenset({"Catalyst", "CatalystModel", "Material"}),
        target_types=frozenset({"Experiment"}),
    ),
    RelationConstraint(
        "CHARACTERIZED_BY",
        source_types=frozenset(
            {"Catalyst", "Support", "Material", "CoordinationMotif"}
        ),
        target_types=frozenset({"Experiment"}),
    ),
    RelationConstraint(
        "MODELED_BY",
        source_types=frozenset({"CatalystModel"}),
        target_types=frozenset({"Calculation"}),
    ),
    RelationConstraint(
        "SYNTHESIZED_BY",
        source_types=frozenset({"Catalyst"}),
        target_types=frozenset({"SynthesisMethod"}),
    ),
    RelationConstraint(
        "USES_PRECURSOR",
        source_types=frozenset({"SynthesisMethod"}),
        target_types=frozenset({"Precursor"}),
    ),
    *COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
    RelationConstraint(
        "MODEL_OF",
        source_types=frozenset({"CatalystModel"}),
        target_types=frozenset({"Catalyst"}),
    ),
    RelationConstraint(
        "HAS_METAL",
        source_types=frozenset({"Catalyst", "CatalystModel"}),
        target_types=frozenset({"Metal"}),
    ),
    RelationConstraint(
        "SUPPORTED_ON",
        source_types=frozenset({"Catalyst", "CatalystModel"}),
        target_types=frozenset({"Support"}),
    ),
    RelationConstraint(
        "CATALYZES",
        source_types=frozenset({"Catalyst"}),
        target_types=frozenset({"Reaction"}),
    ),
)

DAC_HER_STRICT_RELATION_CONSTRAINTS = DAC_LEGACY_STRICT_RELATION_CONSTRAINTS
CATALYSIS_MECHANISM_STRICT_RELATION_CONSTRAINTS = (
    DAC_LEGACY_STRICT_RELATION_CONSTRAINTS
)
SERS_AU_AG_STRICT_RELATION_CONSTRAINTS = (
    COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS
)
