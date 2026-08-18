from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyRelationEndpointPolicy:
    """Historical relation endpoint contract kept for DAC compatibility."""

    relation: str
    source_types: frozenset[str]
    target_types: frozenset[str]


# Historical DAC-HER compatibility payload.
#
# This is intentionally named as legacy DAC policy rather than shared
# scientific semantics. New runtime domain validation remains owned by
# explicit domain RelationConstraint adapters.
LEGACY_DAC_RELATION_ENDPOINT_POLICY = (
    LegacyRelationEndpointPolicy(
        relation="EVALUATED_IN",
        source_types=frozenset(
            {
                "Catalyst",
                "CatalystModel",
                "Material",
            }
        ),
        target_types=frozenset({"Experiment"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="CHARACTERIZED_BY",
        source_types=frozenset(
            {
                "Catalyst",
                "Support",
                "Material",
                "CoordinationMotif",
            }
        ),
        target_types=frozenset({"Experiment"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="MODELED_BY",
        source_types=frozenset({"CatalystModel"}),
        target_types=frozenset({"Calculation"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="SYNTHESIZED_BY",
        source_types=frozenset({"Catalyst"}),
        target_types=frozenset({"SynthesisMethod"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="USES_PRECURSOR",
        source_types=frozenset({"SynthesisMethod"}),
        target_types=frozenset({"Precursor"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="HAS_MEASUREMENT",
        source_types=frozenset(
            {
                "Experiment",
                "Calculation",
            }
        ),
        target_types=frozenset({"Measurement"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="MEASURED_FOR",
        source_types=frozenset({"Measurement"}),
        target_types=frozenset({"Entity"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="IN_MEASUREMENT_GROUP",
        source_types=frozenset({"Measurement"}),
        target_types=frozenset({"MeasurementGroup"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="SUPPORTS_CLAIM",
        source_types=frozenset(
            {
                "Measurement",
                "Experiment",
                "Calculation",
            }
        ),
        target_types=frozenset(
            {
                "ObservationClaim",
                "MechanismClaim",
            }
        ),
    ),
    LegacyRelationEndpointPolicy(
        relation="INTERPRETED_AS",
        source_types=frozenset({"ObservationClaim"}),
        target_types=frozenset({"MechanismClaim"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="APPLIES_TO",
        source_types=frozenset(
            {
                "ObservationClaim",
                "MechanismClaim",
            }
        ),
        target_types=frozenset({"Entity"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="MODEL_OF",
        source_types=frozenset({"CatalystModel"}),
        target_types=frozenset({"Catalyst"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="HAS_METAL",
        source_types=frozenset(
            {
                "Catalyst",
                "CatalystModel",
            }
        ),
        target_types=frozenset({"Metal"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="SUPPORTED_ON",
        source_types=frozenset(
            {
                "Catalyst",
                "CatalystModel",
            }
        ),
        target_types=frozenset({"Support"}),
    ),
    LegacyRelationEndpointPolicy(
        relation="CATALYZES",
        source_types=frozenset({"Catalyst"}),
        target_types=frozenset({"Reaction"}),
    ),
)


LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION = {
    item.relation: item
    for item in LEGACY_DAC_RELATION_ENDPOINT_POLICY
}
