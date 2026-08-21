from __future__ import annotations

import networkx as nx

from pipeline_core.graph_domain import GraphDomainAdapter, RelationConstraint
from domains.sers.graph_diagnostics import (
    collect_sers_graph_diagnostics,
)


def _preserve_sers_semantic_roles(
    graph: nx.MultiDiGraph,
    *,
    chunk_id: str,
):
    """Preserve strict SERS roles; do not infer electrocatalyst roles."""
    del chunk_id
    return graph, []


_SCIENTIFIC_TARGETS = frozenset({
    "PlasmonicSubstrate",
    "Nanostructure",
    "Metal",
    "Material",
    "Support",
    "StructuralMotif",
    "Morphology",
    "Analyte",
    "RamanReporter",
    "OpticalCondition",
    "SynthesisMethod",
    "Precursor",
})

_SUBSTRATE_LIKE = frozenset({
    "PlasmonicSubstrate",
    "Nanostructure",
    "Material",
})

_STRUCTURE_BEARING = frozenset({
    "PlasmonicSubstrate",
    "Nanostructure",
    "StructuralMotif",
    "Morphology",
})

_CLAIMS = frozenset({"ObservationClaim", "MechanismClaim"})


SERS_RELATION_CONSTRAINTS = (
    RelationConstraint(
        "STUDIES",
        source_types=frozenset({"Paper"}),
        target_types=_SCIENTIFIC_TARGETS,
    ),
    RelationConstraint(
        "HAS_COMPONENT",
        source_types=frozenset({
            "PlasmonicSubstrate", "Nanostructure", "Support",
        }),
        target_types=frozenset({"Metal", "Material", "Nanostructure", "Support"}),
    ),
    RelationConstraint(
        "HAS_ARCHITECTURE",
        source_types=frozenset({"PlasmonicSubstrate", "Nanostructure"}),
        target_types=frozenset({"StructuralMotif", "Morphology"}),
    ),
    RelationConstraint(
        "HAS_STRUCTURAL_MOTIF",
        source_types=frozenset({"PlasmonicSubstrate", "Nanostructure"}),
        target_types=frozenset({"StructuralMotif"}),
    ),
    RelationConstraint(
        "HAS_MORPHOLOGY",
        source_types=frozenset({"PlasmonicSubstrate", "Nanostructure"}),
        target_types=frozenset({"Morphology"}),
    ),
    RelationConstraint(
        "HAS_SUPPORT",
        source_types=frozenset({"PlasmonicSubstrate", "Nanostructure"}),
        target_types=frozenset({"Support", "Material"}),
    ),
    RelationConstraint(
        "PREPARED_BY",
        source_types=_SUBSTRATE_LIKE | frozenset({"Support"}),
        target_types=frozenset({"SynthesisMethod"}),
    ),
    RelationConstraint(
        "USES_PRECURSOR",
        source_types=frozenset({"SynthesisMethod"}),
        target_types=frozenset({"Precursor"}),
    ),
    RelationConstraint(
        "USES_MATERIAL",
        source_types=frozenset({"SynthesisMethod"}),
        target_types=frozenset({"Material"}),
    ),
    RelationConstraint(
        "TESTED_IN",
        source_types=_SUBSTRATE_LIKE | frozenset({"Support"}),
        target_types=frozenset({"Experiment"}),
    ),
    RelationConstraint(
        "CHARACTERIZED_IN",
        source_types=_STRUCTURE_BEARING | frozenset({"Material", "Support", "Metal"}),
        target_types=frozenset({"Experiment"}),
    ),
    RelationConstraint(
        "SIMULATED_BY",
        source_types=_STRUCTURE_BEARING | frozenset({"Material", "Metal"}),
        target_types=frozenset({"Calculation"}),
    ),
    RelationConstraint(
        "USES_ANALYTE",
        source_types=frozenset({"Experiment"}),
        target_types=frozenset({"Analyte"}),
    ),
    RelationConstraint(
        "USES_REPORTER",
        source_types=frozenset({"Experiment", "Calculation", "SynthesisMethod"}),
        target_types=frozenset({"RamanReporter"}),
    ),
    RelationConstraint(
        "USES_OPTICAL_CONDITION",
        source_types=frozenset({
            "Experiment", "Calculation", "Measurement", "SynthesisMethod",
        }),
        target_types=frozenset({"OpticalCondition"}),
    ),
    RelationConstraint(
        "HAS_MEASUREMENT",
        source_types=frozenset({"Experiment", "Calculation"}),
        target_types=frozenset({"Measurement"}),
    ),
    RelationConstraint(
        "MEASURED_FOR",
        source_types=frozenset({"Measurement"}),
        target_types=_SCIENTIFIC_TARGETS,
    ),
    RelationConstraint(
        "IN_MEASUREMENT_GROUP",
        source_types=frozenset({"Measurement"}),
        target_types=frozenset({"MeasurementGroup"}),
    ),
    RelationConstraint(
        "SUPPORTS_CLAIM",
        source_types=frozenset({"Experiment", "Calculation", "Measurement", "MeasurementGroup"}),
        target_types=_CLAIMS,
    ),
    RelationConstraint(
        "INTERPRETED_AS",
        source_types=frozenset({"ObservationClaim"}),
        target_types=frozenset({"MechanismClaim"}),
    ),
    RelationConstraint(
        "PROPOSES_CLAIM",
        source_types=frozenset({"Paper"}),
        target_types=_CLAIMS,
    ),
    RelationConstraint(
        "APPLIES_TO",
        source_types=_CLAIMS,
        target_types=_SCIENTIFIC_TARGETS,
    ),
    RelationConstraint(
        "DERIVED_FROM",
        source_types=frozenset({"Experiment", "Calculation", "Measurement"}),
        target_types=frozenset({"SynthesisMethod", "Experiment", "Calculation", "Measurement"}),
    ),
)


SERS_AU_AG_GRAPH_ADAPTER = GraphDomainAdapter(
    adapter_id="sers_au_ag",
    domain_profile_id="sers_au_ag",
    semantic_role_policy=(
        "No electrocatalysis-specific semantic-role coercion is applied. "
        "Strict SERS entity types are preserved through paper-graph merging."
    ),
    semantic_role_normalizer=_preserve_sers_semantic_roles,
    relation_constraints=SERS_RELATION_CONSTRAINTS,
    primary_subject_types=frozenset({
        "PlasmonicSubstrate",
        "Nanostructure",
    }),
    duplicate_review_types=frozenset({
        "PlasmonicSubstrate",
        "Nanostructure",
        "StructuralMotif",
        "Morphology",
        "Analyte",
        "RamanReporter",
        "OpticalCondition",
        "Paper",
    }),
    diagnostics_collector=collect_sers_graph_diagnostics,
)
