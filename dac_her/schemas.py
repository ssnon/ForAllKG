from __future__ import annotations
from collections import defaultdict
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.measurement_schema import (
    Condition,
    MeasurementGroupNode,
    MeasurementGroupType,
    MeasurementNode,
)


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
EntityType = str


ExperimentFamily = Literal[
    "electrochemistry",
    "microscopy",
    "spectroscopy",
    "diffraction",
    "composition_analysis",
    "surface_area_analysis",
    "thermal_analysis",
    "synthesis",
    "stability_test",
    "other",
]

# Registry-backed method identifier. The YAML vocabulary, rather than this
# Python type alias, controls which method IDs are registered.
ExperimentType = str


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
CalculationType = str

KnownObservationClaimType = Literal[
    "performance_comparison",
    "stability_observation",
    "structural_observation",
    "adsorption_energy_comparison",
    "adsorption_site_preference",
    "other",
]
ObservationClaimType = str

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
MechanismClaimType = str


MechanismBasis = Literal[
    "experimental",
    "computational",
    "mixed",
]


EvidenceType = Literal[
    "bibliographic_metadata",
    "synthesis_procedure",
    "experimental_setup",
    "experimental_observation",
    "structural_characterization",
    "computational_method",
    "computational_result",
    "author_interpretation",
]


EvidenceStrength = Literal[
    "direct",
    "indirect",
    "interpretive",
]


ConfidenceLevel = Literal[
    "high",
    "medium",
    "low",
]


DocumentRole = Literal[
    "main",
    "supporting_information",
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
RelationType = str

# ============================================================
# Graph node models
# ============================================================

class EntityNode(BaseModel):
    """
    Ordinary scientific entities.

    Measurements, experiments, calculations, and
    mechanism claims should not be placed here.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        ...,
        description=(
            "Unique canonical graph node identifier."
        ),
    )

    type: EntityType

    label: str = Field(
        ...,
        description=(
            "Human-readable canonical node label."
        ),
    )

    description: str | None = Field(
        ...,
        description=(
            "Brief source-grounded description. "
            "Use null when unnecessary."
        ),
    )


class ExperimentNode(BaseModel):
    """
    Experimental or characterization setup.

    Results from the experiment belong in
    MeasurementNode objects.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        ...,
        description=(
            "Unique experiment identifier."
        ),
    )

    name: str = Field(
        ...,
        description=(
            "Concise human-readable experiment name."
        ),
    )

    experiment_type: ExperimentType = Field(
        ...,
        description=(
            "Registry-backed method ID, such as xps, haadf_stem, "
            "or electrochemical_impedance_spectroscopy. Use an "
            "unregistered_<slug> ID only when no registry method fits."
        ),
    )

    experiment_family: ExperimentFamily = Field(
        ...,
        description="Broad experiment family from the controlled taxonomy.",
    )

    method_label: str = Field(
        ...,
        description="Preferred human-readable method label.",
    )

    raw_method_name: str | None = Field(
        ...,
        description=(
            "Source wording when it differs from the preferred label. "
            "Use null when unnecessary."
        ),
    )

    conditions: list[Condition] = Field(
        ...,
        description=(
            "All explicitly reported experimental "
            "conditions. Use an empty list when none "
            "are stated."
        ),
    )

    description: str | None = Field(
        ...,
        description=(
            "Brief description of what was performed."
        ),
    )


    @model_validator(mode="before")
    @classmethod
    def backfill_legacy_registry_fields(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            method_id = str(value.get("experiment_type", "other"))
            family_map = {
                "cyclic_voltammetry": "electrochemistry",
                "linear_sweep_voltammetry": "electrochemistry",
                "tafel_analysis": "electrochemistry",
                "accelerated_degradation_test": "stability_test",
                "extended_electrolysis": "stability_test",
                "chronoamperometry": "electrochemistry",
                "chronopotentiometry": "electrochemistry",
                "haadf_stem": "microscopy",
                "tem": "microscopy",
                "xanes": "spectroscopy",
                "exafs": "spectroscopy",
                "xas": "spectroscopy",
                "icp_oes": "composition_analysis",
            }
            value.setdefault("experiment_family", family_map.get(method_id, "other"))
            value.setdefault("method_label", value.get("name") or method_id)
            value.setdefault("raw_method_name", None)
        return value


class CalculationNode(BaseModel):
    """
    Computational procedure such as DFT, PDOS,
    adsorption-energy calculation, or FPMD.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        ...,
        description=(
            "Unique calculation identifier."
        ),
    )

    name: str = Field(
        ...,
        description=(
            "Concise human-readable calculation name."
        ),
    )

    calculation_type: CalculationType

    conditions: list[Condition] = Field(
        ...,
        description=(
            "Explicit computational settings or "
            "coverage conditions. Use an empty list "
            "when none are stated."
        ),
    )

    method_details: str | None = Field(
        ...,
        description=(
            "Brief method description, such as DFT "
            "model or functional. Use null when the "
            "chunk does not provide this information."
        ),
    )


class ObservationClaimNode(BaseModel):
    """
    Measurements or calculations summarized into a
    directly evidence-supported scientific conclusion.

    This node must not contain a causal explanation.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        ...,
        description=(
            "Unique observation-claim identifier."
        ),
    )

    claim_type: ObservationClaimType

    statement: str = Field(
        ...,
        description=(
            "A concise conclusion directly supported "
            "by reported measurements, calculations, "
            "or characterization results."
        ),
    )

    basis: MechanismBasis = Field(
        ...,
        description=(
            "Whether the observation is based on "
            "experimental, computational, or mixed "
            "evidence."
        ),
    )

    description: str | None = Field(
        ...,
        description=(
            "Additional source-grounded clarification. "
            "Use null when unnecessary."
        ),
    )

class MechanismClaimNode(BaseModel):
    """
    A causal, mechanistic, or explanatory interpretation.

    Directly reported numerical comparisons belong in
    ObservationClaimNode, not here.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    id: str = Field(
        ...,
        description=(
            "Unique mechanism-claim identifier."
        ),
    )

    claim_type: MechanismClaimType

    statement: str = Field(
        ...,
        description=(
            "Concise faithful statement of the authors' "
            "causal or mechanistic interpretation."
        ),
    )

    basis: MechanismBasis

    description: str | None = Field(
        ...,
        description=(
            "Additional clarification. "
            "Use null when unnecessary."
        ),
    )


# ============================================================
# Edge and graph models
# ============================================================

class EvidencePointer(BaseModel):
    """Locator from a graph edge back to a source document or asset."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        ...,
        description="Document identifier supplied in the prompt.",
    )
    document_role: DocumentRole
    page_id: int | None = Field(
        ...,
        description=(
            "Marker page identifier when available. Use null when the "
            "source block has no reliable page locator."
        ),
    )
    asset_ids: list[str] = Field(
        ...,
        description=(
            "Subset of the chunk-level asset IDs that directly support "
            "this edge. Use an empty list for text-only evidence."
        ),
    )
    locator_text: str | None = Field(
        ...,
        description=(
            "Figure/table label, subsection locator, or concise source "
            "locator. Use null when unavailable."
        ),
    )


class KGEdge(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    source: str = Field(
        ...,
        description=(
            "ID of an existing source node."
        ),
    )

    relation: RelationType

    target: str = Field(
        ...,
        description=(
            "ID of an existing target node."
        ),
    )

    evidence_type: EvidenceType

    evidence_strength: EvidenceStrength = Field(
        ...,
        description=(
            "direct for directly reported observations "
            "or calculations; indirect for evidence-based "
            "support; interpretive for author explanations."
        ),
    )

    evidence_text: str = Field(
        ...,
        description=(
            "Short source-supported evidence span or "
            "faithful paraphrase."
        ),
    )

    confidence: ConfidenceLevel

    evidence_pointers: list[EvidencePointer] = Field(
        ...,
        description=(
            "One or more source locators. Every edge must retain at least "
            "one text/document pointer; asset_ids may be empty."
        ),
    )

    subsection: str | None = Field(
        ...,
        description=(
            "More specific subsection title when the "
            "chunk contains multiple subsections. "
            "Use null when unavailable."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def backfill_legacy_pointer(cls, value):
        # Old chunk caches predate document/asset provenance. Keep them
        # readable while requiring the fields in new structured outputs.
        if isinstance(value, dict) and "evidence_pointers" not in value:
            value = dict(value)
            value["evidence_pointers"] = [{
                "document_id": "main",
                "document_role": "main",
                "page_id": None,
                "asset_ids": [],
                "locator_text": value.get("subsection"),
            }]
        return value


class KnowledgeGraph(BaseModel):
    """
    Provenance-preserving graph extracted from one chunk.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    paper_id: str = Field(
        ...,
        description=(
            "Paper identifier supplied in the prompt."
        ),
    )

    chunk_id: str = Field(
        ...,
        description=(
            "Chunk identifier supplied in the prompt."
        ),
    )

    section: str = Field(
        ...,
        description=(
            "Parent section supplied in the prompt."
        ),
    )

    document_id: str = Field(
        ...,
        description="Source document identifier supplied in the prompt.",
    )

    document_role: DocumentRole

    page_ids: list[int] = Field(
        ...,
        description="Marker page identifiers associated with the core chunk.",
    )

    asset_ids: list[str] = Field(
        ...,
        description="Figure/table asset IDs linked to the core chunk.",
    )

    entities: list[EntityNode] = Field(
        ...,
        description=(
            "Scientific entity nodes."
        ),
    )

    experiments: list[ExperimentNode] = Field(
        ...,
        description=(
            "Experimental and characterization setups."
        ),
    )

    calculations: list[CalculationNode] = Field(
        ...,
        description=(
            "Computational procedures."
        ),
    )

    measurements: list[MeasurementNode] = Field(
        ...,
        description=(
            "Individual scalar experimental or computational results."
        ),
    )

    measurement_groups: list[MeasurementGroupNode] = Field(
        ...,
        description=(
            "Comparison/series containers whose members remain separate "
            "scalar Measurement nodes."
        ),
    )

    observation_claims: list[ObservationClaimNode] = Field(
        ...,
        description=(
            "Direct evidence-supported observational "
            "or comparative conclusions."
        ),
    )

    mechanism_claims: list[MechanismClaimNode] = Field(
        ...,
        description=(
            "Author-proposed mechanistic explanations."
        ),
    )

    edges: list[KGEdge] = Field(
        ...,
        description=(
            "Directed relationships between all nodes."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def backfill_legacy_document_provenance(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            value.setdefault("document_id", "main")
            value.setdefault("document_role", "main")
            value.setdefault("page_ids", [])
            value.setdefault("asset_ids", [])
            value.setdefault("measurement_groups", [])
        return value

    def all_node_ids(self) -> set[str]:
        return {
            node.id
            for group in (
                self.entities,
                self.experiments,
                self.calculations,
                self.measurements,
                self.measurement_groups,
                self.observation_claims,
                self.mechanism_claims,
            )
            for node in group
        }

    @model_validator(mode="after")
    def validate_graph_integrity(
        self,
    ) -> "KnowledgeGraph":
        node_id_list = [
            node.id
            for group in (
                self.entities,
                self.experiments,
                self.calculations,
                self.measurements,
                self.measurement_groups,
                self.observation_claims,
                self.mechanism_claims,
            )
            for node in group
        ]

        node_ids = set(node_id_list)

        if len(node_ids) != len(node_id_list):
            duplicates = {
                node_id
                for node_id in node_id_list
                if node_id_list.count(node_id) > 1
            }

            raise ValueError(
                "Duplicate node IDs were found: "
                f"{sorted(duplicates)}"
            )

        if len(self.asset_ids) != len(set(self.asset_ids)):
            raise ValueError("Duplicate asset IDs were found at graph level.")

        provenance_errors: list[str] = []
        allowed_asset_ids = set(self.asset_ids)
        allowed_page_ids = set(self.page_ids)

        for edge_index, edge in enumerate(self.edges):
            if not edge.evidence_pointers:
                provenance_errors.append(
                    f"Edge {edge_index} has no evidence_pointers."
                )
                continue
            for pointer in edge.evidence_pointers:
                if pointer.document_id != self.document_id:
                    provenance_errors.append(
                        f"Edge {edge_index} pointer document_id "
                        f"{pointer.document_id!r} does not match "
                        f"{self.document_id!r}."
                    )
                if pointer.document_role != self.document_role:
                    provenance_errors.append(
                        f"Edge {edge_index} pointer document_role "
                        f"{pointer.document_role!r} does not match "
                        f"{self.document_role!r}."
                    )
                unknown_assets = set(pointer.asset_ids) - allowed_asset_ids
                if unknown_assets:
                    provenance_errors.append(
                        f"Edge {edge_index} references unknown asset IDs: "
                        f"{sorted(unknown_assets)}"
                    )
                if (
                    pointer.page_id is not None
                    and allowed_page_ids
                    and pointer.page_id not in allowed_page_ids
                ):
                    provenance_errors.append(
                        f"Edge {edge_index} references page_id "
                        f"{pointer.page_id}, not present in page_ids."
                    )

        if provenance_errors:
            raise ValueError(
                "Graph provenance validation failed:\n"
                + "\n".join(f"- {message}" for message in provenance_errors)
            )

        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(
                    "Edge references undefined source: "
                    f"{edge.source!r}"
                )

            if edge.target not in node_ids:
                raise ValueError(
                    "Edge references undefined target: "
                    f"{edge.target!r}"
                )

        incoming: dict[
            str,
            list[KGEdge],
        ] = defaultdict(list)

        outgoing: dict[
            str,
            list[KGEdge],
        ] = defaultdict(list)

        for edge in self.edges:
            outgoing[edge.source].append(edge)
            incoming[edge.target].append(edge)

        graph_errors: list[str] = []

        experiment_ids = {
            node.id
            for node in self.experiments
        }

        calculation_ids = {
            node.id
            for node in self.calculations
        }

        valid_measurement_sources = (
            experiment_ids
            | calculation_ids
        )

        entity_ids_for_measurement = {node.id for node in self.entities}
        measurement_ids_for_group = {node.id for node in self.measurements}
        measurement_group_by_id = {
            node.id: node for node in self.measurement_groups
        }

        # ----------------------------------------------------
        # Claim-like objects must not be ordinary entities
        # ----------------------------------------------------

        claim_id_prefixes = (
            "claim_",
            "obs_",
            "oc_",
            "mech_",
        )

        for entity in self.entities:
            if entity.id.lower().startswith(
                claim_id_prefixes
            ):
                graph_errors.append(
                    "Claim-like node was placed in "
                    "entities instead of a claim array: "
                    f"{entity.id!r} "
                    f"(entity type={entity.type!r})"
                )

        # ----------------------------------------------------
        # Every measurement must come from an experiment
        # or calculation
        # ----------------------------------------------------

        for measurement in self.measurements:
            producer_edges = [
                edge
                for edge in incoming[measurement.id]
                if (
                    edge.relation == "HAS_MEASUREMENT"
                    and edge.source in valid_measurement_sources
                )
            ]
            if not producer_edges:
                graph_errors.append(
                    "Measurement has no incoming HAS_MEASUREMENT edge from "
                    f"an Experiment or Calculation: {measurement.id!r}"
                )

            measured_for_edges = [
                edge for edge in outgoing[measurement.id]
                if edge.relation == "MEASURED_FOR"
            ]
            if len(measured_for_edges) != 1:
                graph_errors.append(
                    "Measurement must have exactly one MEASURED_FOR edge: "
                    f"{measurement.id!r}"
                )
            elif measured_for_edges[0].target != measurement.subject_id:
                graph_errors.append(
                    "Measurement subject_id does not match MEASURED_FOR target: "
                    f"{measurement.id!r}"
                )
            if measurement.subject_id not in entity_ids_for_measurement:
                graph_errors.append(
                    "Measurement subject_id must reference a scientific Entity: "
                    f"{measurement.id!r} -> {measurement.subject_id!r}"
                )

            membership_edges = [
                edge for edge in outgoing[measurement.id]
                if edge.relation == "IN_MEASUREMENT_GROUP"
            ]
            if measurement.group_id is None and membership_edges:
                graph_errors.append(
                    f"Measurement {measurement.id!r} has a group edge but group_id is null."
                )
            if measurement.group_id is not None:
                if measurement.group_id not in measurement_group_by_id:
                    graph_errors.append(
                        f"Measurement {measurement.id!r} references unknown group "
                        f"{measurement.group_id!r}."
                    )
                if len(membership_edges) != 1 or membership_edges[0].target != measurement.group_id:
                    graph_errors.append(
                        f"Measurement {measurement.id!r} must have one matching "
                        "IN_MEASUREMENT_GROUP edge."
                    )

        for group in self.measurement_groups:
            unknown_members = set(group.member_measurement_ids) - measurement_ids_for_group
            if unknown_members:
                graph_errors.append(
                    f"MeasurementGroup {group.id!r} has unknown members: "
                    f"{sorted(unknown_members)}"
                )
            for member_id in group.member_measurement_ids:
                member = next((item for item in self.measurements if item.id == member_id), None)
                if member is not None and member.group_id != group.id:
                    graph_errors.append(
                        f"MeasurementGroup {group.id!r} and member {member_id!r} "
                        "do not agree on group_id."
                    )

        # ----------------------------------------------------
        # No isolated nodes
        # ----------------------------------------------------

        isolated_node_ids = sorted(
            node_id
            for node_id in node_ids
            if (
                not incoming[node_id]
                and not outgoing[node_id]
            )
        )

        if isolated_node_ids:
            graph_errors.append(
                "Isolated nodes were found: "
                + ", ".join(isolated_node_ids)
            )

        if graph_errors:
            raise ValueError(
                "Graph structural validation failed:\n"
                + "\n".join(
                    f"- {message}"
                    for message in graph_errors
                )
            )        

        claim_errors: list[str] = []

        # Observation claims:
        # evidence support + application target required
        for claim in self.observation_claims:
            support_edges = [
                edge
                for edge in incoming[claim.id]
                if edge.relation == "SUPPORTS_CLAIM"
            ]

            application_edges = [
                edge
                for edge in outgoing[claim.id]
                if edge.relation == "APPLIES_TO"
            ]

            if not support_edges:
                claim_errors.append(
                    "Observation claim has no "
                    "SUPPORTS_CLAIM evidence: "
                    f"{claim.id}"
                )

            if not application_edges:
                claim_errors.append(
                    "Observation claim has no "
                    "APPLIES_TO target: "
                    f"{claim.id}"
                )

        # Mechanism claims:
        # direct support or interpretation source required
        for claim in self.mechanism_claims:
            direct_support = [
                edge
                for edge in incoming[claim.id]
                if edge.relation == "SUPPORTS_CLAIM"
            ]

            interpretation_sources = [
                edge
                for edge in incoming[claim.id]
                if edge.relation == "INTERPRETED_AS"
            ]

            application_edges = [
                edge
                for edge in outgoing[claim.id]
                if edge.relation == "APPLIES_TO"
            ]

            if (
                not direct_support
                and not interpretation_sources
            ):
                claim_errors.append(
                    "Mechanism claim has neither "
                    "SUPPORTS_CLAIM nor INTERPRETED_AS "
                    f"evidence: {claim.id}"
                )

            if not application_edges:
                claim_errors.append(
                    "Mechanism claim has no "
                    "APPLIES_TO target: "
                    f"{claim.id}"
                )

        if claim_errors:
            raise ValueError(
                "Claim validation failed:\n"
                + "\n".join(
                    f"- {message}"
                    for message in claim_errors
                )
            )

        entity_by_id = {
            node.id: node
            for node in self.entities
        }

        entity_ids = set(entity_by_id)

        experiment_ids = {
            node.id
            for node in self.experiments
        }

        calculation_ids = {
            node.id
            for node in self.calculations
        }

        measurement_ids = {
            node.id
            for node in self.measurements
        }

        measurement_group_ids = {
            node.id
            for node in self.measurement_groups
        }

        observation_claim_ids = {
            node.id
            for node in self.observation_claims
        }

        mechanism_claim_ids = {
            node.id
            for node in self.mechanism_claims
        }

        all_claim_ids = (
            observation_claim_ids
            | mechanism_claim_ids
        )

        semantic_errors: list[str] = []

        def entity_has_type(
            node_id: str,
            allowed_types: set[str],
        ) -> bool:
            node = entity_by_id.get(node_id)

            return (
                node is not None
                and node.type in allowed_types
            )

        for edge in self.edges:
            relation = edge.relation
            source = edge.source
            target = edge.target

            # Catalyst -> Experiment
            if relation == "EVALUATED_IN":
                if not entity_has_type(
                    source,
                    {
                        "Catalyst",
                        "CatalystModel",
                        "Material",
                    },
                ):
                    semantic_errors.append(
                        "EVALUATED_IN source must be "
                        "a Catalyst, CatalystModel, "
                        "or Material: "
                        f"{source!r}"
                    )

                if target not in experiment_ids:
                    semantic_errors.append(
                        "EVALUATED_IN target must be "
                        f"an Experiment: {target!r}"
                    )

            # Physical scientific object -> characterization
            elif relation == "CHARACTERIZED_BY":
                if not entity_has_type(
                    source,
                    {
                        "Catalyst",
                        "Support",
                        "Material",
                        "CoordinationMotif",
                    },
                ):
                    semantic_errors.append(
                        "CHARACTERIZED_BY source must be "
                        "a physical scientific entity: "
                        f"{source!r}"
                    )

                if target not in experiment_ids:
                    semantic_errors.append(
                        "CHARACTERIZED_BY target must be "
                        f"an Experiment: {target!r}"
                    )

            # CatalystModel -> Calculation
            elif relation == "MODELED_BY":
                if not entity_has_type(
                    source,
                    {"CatalystModel"},
                ):
                    semantic_errors.append(
                        "MODELED_BY source must be "
                        f"a CatalystModel: {source!r}"
                    )

                if target not in calculation_ids:
                    semantic_errors.append(
                        "MODELED_BY target must be "
                        f"a Calculation: {target!r}"
                    )

            # Catalyst -> SynthesisMethod
            elif relation == "SYNTHESIZED_BY":
                if not entity_has_type(
                    source,
                    {"Catalyst"},
                ):
                    semantic_errors.append(
                        "SYNTHESIZED_BY source must be "
                        f"a Catalyst: {source!r}"
                    )

                if not entity_has_type(
                    target,
                    {"SynthesisMethod"},
                ):
                    semantic_errors.append(
                        "SYNTHESIZED_BY target must be "
                        f"a SynthesisMethod: {target!r}"
                    )

            # SynthesisMethod -> Precursor
            elif relation == "USES_PRECURSOR":
                if not entity_has_type(
                    source,
                    {"SynthesisMethod"},
                ):
                    semantic_errors.append(
                        "USES_PRECURSOR source must be "
                        f"a SynthesisMethod: {source!r}"
                    )

                if not entity_has_type(
                    target,
                    {"Precursor"},
                ):
                    semantic_errors.append(
                        "USES_PRECURSOR target must be "
                        f"a Precursor: {target!r}"
                    )

            # Experiment/Calculation -> Measurement
            elif relation == "HAS_MEASUREMENT":
                if (
                    source not in experiment_ids
                    and source not in calculation_ids
                ):
                    semantic_errors.append(
                        "HAS_MEASUREMENT source must be "
                        "an Experiment or Calculation: "
                        f"{source!r}"
                    )

                if target not in measurement_ids:
                    semantic_errors.append(
                        "HAS_MEASUREMENT target must be "
                        f"a Measurement: {target!r}"
                    )

            elif relation == "MEASURED_FOR":
                if source not in measurement_ids:
                    semantic_errors.append(
                        f"MEASURED_FOR source must be a Measurement: {source!r}"
                    )
                if target not in entity_ids:
                    semantic_errors.append(
                        f"MEASURED_FOR target must be an Entity: {target!r}"
                    )

            elif relation == "IN_MEASUREMENT_GROUP":
                if source not in measurement_ids:
                    semantic_errors.append(
                        "IN_MEASUREMENT_GROUP source must be a Measurement: "
                        f"{source!r}"
                    )
                if target not in measurement_group_ids:
                    semantic_errors.append(
                        "IN_MEASUREMENT_GROUP target must be a MeasurementGroup: "
                        f"{target!r}"
                    )

            elif relation == "MODEL_OF":
                if not entity_has_type(source, {"CatalystModel"}):
                    semantic_errors.append(
                        f"MODEL_OF source must be a CatalystModel: {source!r}"
                    )
                if not entity_has_type(target, {"Catalyst"}):
                    semantic_errors.append(
                        f"MODEL_OF target must be a Catalyst: {target!r}"
                    )

            # Evidence -> claim
            elif relation == "SUPPORTS_CLAIM":
                valid_sources = (
                    measurement_ids
                    | experiment_ids
                    | calculation_ids
                )

                if source not in valid_sources:
                    semantic_errors.append(
                        "SUPPORTS_CLAIM source must be "
                        "a Measurement, Experiment, or "
                        f"Calculation: {source!r}"
                    )

                if target not in all_claim_ids:
                    semantic_errors.append(
                        "SUPPORTS_CLAIM target must be "
                        f"a claim: {target!r}"
                    )

            # ObservationClaim -> MechanismClaim
            elif relation == "INTERPRETED_AS":
                if source not in observation_claim_ids:
                    semantic_errors.append(
                        "INTERPRETED_AS source must be "
                        "an ObservationClaim: "
                        f"{source!r}"
                    )

                if target not in mechanism_claim_ids:
                    semantic_errors.append(
                        "INTERPRETED_AS target must be "
                        "a MechanismClaim: "
                        f"{target!r}"
                    )

            # Claim -> scientific entity
            elif relation == "APPLIES_TO":
                if source not in all_claim_ids:
                    semantic_errors.append(
                        "APPLIES_TO source must be "
                        f"a claim: {source!r}"
                    )

                if target not in entity_ids:
                    semantic_errors.append(
                        "APPLIES_TO target must be "
                        f"an Entity: {target!r}"
                    )

            # SynthesisMethod -> Precursor already handled;
            # catalyst composition
            elif relation == "HAS_METAL":
                if not entity_has_type(
                    source,
                    {
                        "Catalyst",
                        "CatalystModel",
                    },
                ):
                    semantic_errors.append(
                        "HAS_METAL source must be "
                        "a Catalyst or CatalystModel: "
                        f"{source!r}"
                    )

                if not entity_has_type(
                    target,
                    {"Metal"},
                ):
                    semantic_errors.append(
                        "HAS_METAL target must be "
                        f"a Metal: {target!r}"
                    )

            elif relation == "SUPPORTED_ON":
                if not entity_has_type(
                    source,
                    {
                        "Catalyst",
                        "CatalystModel",
                    },
                ):
                    semantic_errors.append(
                        "SUPPORTED_ON source must be "
                        "a Catalyst or CatalystModel: "
                        f"{source!r}"
                    )

                if not entity_has_type(
                    target,
                    {"Support"},
                ):
                    semantic_errors.append(
                        "SUPPORTED_ON target must be "
                        f"a Support: {target!r}"
                    )

            elif relation == "CATALYZES":
                if not entity_has_type(
                    source,
                    {"Catalyst"},
                ):
                    semantic_errors.append(
                        "CATALYZES source must be "
                        f"a Catalyst: {source!r}"
                    )

                if not entity_has_type(
                    target,
                    {"Reaction"},
                ):
                    semantic_errors.append(
                        "CATALYZES target must be "
                        f"a Reaction: {target!r}"
                    )

        if semantic_errors:
            raise ValueError(
                "Graph relation validation failed:\n"
                + "\n".join(
                    f"- {message}"
                    for message in semantic_errors
                )
            )
        return self