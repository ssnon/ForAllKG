from __future__ import annotations
from collections import defaultdict
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.corpus.extraction.evidence_schema import (
    ConfidenceLevel,
    DocumentRole,
    EvidencePointer,
    EvidenceStrength,
    EvidenceType,
    KGEdge,
    RelationType,
)

from pipeline_core.corpus.extraction.experiment_schema import (
    ExperimentFamily,
    ExperimentNode,
    ExperimentType,
)

from pipeline_core.corpus.extraction.measurement_schema import (
    Condition,
    MeasurementGroupNode,
    MeasurementGroupType,
    MeasurementNode,
)

from pipeline_core.corpus.extraction.scientific_node_schema import (
    CalculationNode,
    CalculationType,
    EntityNode,
    EntityType,
    MechanismBasis,
    MechanismClaimNode,
    MechanismClaimType,
    ObservationClaimNode,
    ObservationClaimType,
)

from pipeline_core.corpus.graph.knowledge_graph_schema import KnowledgeGraph


# ============================================================
# Controlled vocabularies
# ============================================================

KnownEntityType = Literal[
    "Paper",
    "Catalyst",
    "CatalystModel",
    "Metal",
    "Support",
    "CoordinationMotif",
    "SynthesisMethod",
    "Precursor",
    "Reaction",
    "ReactionStep",
    "Intermediate",
    "Material",
]


KnownCalculationType = Literal[
    "dft",
    "adsorption_energy",
    "gibbs_free_energy",
    "pdos",
    "charge_analysis",
    "fpmd",
    "xanes_simulation",
    "exafs_fitting",
    "other",
]

KnownObservationClaimType = Literal[
    "performance_comparison",
    "stability_observation",
    "structural_observation",
    "adsorption_energy_comparison",
    "adsorption_site_preference",
    "other",
]

KnownMechanismClaimType = Literal[
    "active_site",
    "reaction_pathway",
    "adsorption_mechanism",
    "electronic_structure",
    "formation_preference",
    "stability_mechanism",
    "performance_mechanism",
    "other",
]




KnownRelationType = Literal[
    "STUDIES",
    "HAS_METAL",
    "SUPPORTED_ON",
    "HAS_MOTIF",
    "SYNTHESIZED_BY",
    "USES_PRECURSOR",
    "CATALYZES",
    "EVALUATED_IN",
    "CHARACTERIZED_BY",
    "MODELED_BY",
    "HAS_MEASUREMENT",
    "MEASURED_FOR",
    "IN_MEASUREMENT_GROUP",
    "MODEL_OF",
    "HAS_DESCRIPTOR",
    "CALCULATES",
    "SUPPORTS_CLAIM",
    "INTERPRETED_AS",
    "PROPOSES_CLAIM",
    "APPLIES_TO",
    "INVOLVES_STEP",
    "INVOLVES_INTERMEDIATE",
    "ADSORBS",
    "FACILITATES_STEP",
    "COMPARED_WITH",
    "DERIVED_FROM",
]

# ============================================================
# Graph node models
# ============================================================
