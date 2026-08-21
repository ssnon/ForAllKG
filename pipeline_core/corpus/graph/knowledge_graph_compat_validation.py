from __future__ import annotations

from collections import defaultdict
from typing import Any

from pipeline_core.corpus.graph.knowledge_graph_legacy_relation_compat import (
    validate_legacy_relation_semantics_compat,
)


def validate_graph_integrity_compat(
    graph: Any,
    *,
    validate_legacy_relations: bool = True,
) -> Any:
    """Preserve historical direct KnowledgeGraph validation behavior."""

    self = graph
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

    if validate_legacy_relations:
        validate_legacy_relation_semantics_compat(self)

    return self
