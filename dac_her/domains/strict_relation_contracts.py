from __future__ import annotations

from dac_her.graph_domain import RelationConstraint


from pipeline_core.evidence_relation_constraints import (
    COMMON_EVIDENCE_STRICT_RELATION_CONSTRAINTS,
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
