from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import networkx as nx


CATALYST_ROLE_SOURCE_TYPES = {"Material", "Support"}
CATALYTIC_RELATIONS = {"EVALUATED_IN", "CATALYZES"}


@dataclass(frozen=True)
class SemanticRoleAdjustment:
    chunk_id: str
    source_node_id: str
    original_type: str
    resolved_type: str
    action: str
    role_node_id: str
    measurement_ids: tuple[str, ...]
    experiment_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["measurement_ids"] = list(self.measurement_ids)
        payload["experiment_ids"] = list(self.experiment_ids)
        return payload


def _relation(data: dict[str, Any]) -> str:
    return str(data.get("relation", ""))


def _catalytic_measurements(
    graph: nx.MultiDiGraph,
    node_id: str,
    evaluated_experiments: set[str],
) -> set[str]:
    measurements: set[str] = set()
    for measurement_id, _, _, edge_data in graph.in_edges(
        node_id, keys=True, data=True
    ):
        if _relation(edge_data) != "MEASURED_FOR":
            continue
        producer_experiments = {
            str(source)
            for source, _, _, producer_edge in graph.in_edges(
                measurement_id, keys=True, data=True
            )
            if _relation(producer_edge) == "HAS_MEASUREMENT"
            and str(graph.nodes[source].get("type", "")) == "Experiment"
        }
        if evaluated_experiments & producer_experiments:
            measurements.add(str(measurement_id))
    return measurements


def _has_support_role(graph: nx.MultiDiGraph, node_id: str) -> bool:
    return any(
        _relation(data) == "SUPPORTED_ON"
        for _, _, _, data in graph.in_edges(node_id, keys=True, data=True)
    )


def _role_node_id(chunk_id: str, node_id: str) -> str:
    digest = hashlib.sha256(
        f"{chunk_id}|{node_id}|catalyst_role".encode("utf-8")
    ).hexdigest()[:12]
    return f"{node_id}__role_catalyst_{digest}"


def normalize_measurement_subject_roles(
    graph: nx.MultiDiGraph,
    *,
    chunk_id: str,
) -> tuple[nx.MultiDiGraph, list[SemanticRoleAdjustment]]:
    """Infer catalyst-role mentions from graph structure, not paper names.

    A Material/Support is acting as a catalyst when it is evaluated in an
    experiment and measurements produced by that same experiment are explicitly
    MEASURED_FOR the node. When the same node also serves as a support, a
    separate Catalyst role mention is created instead of destroying the support
    identity.
    """
    normalized = nx.MultiDiGraph()
    normalized.graph.update(graph.graph)
    normalized.add_nodes_from(
        (str(node_id), dict(data))
        for node_id, data in graph.nodes(data=True)
    )
    normalized.add_edges_from(
        (
            str(source),
            str(target),
            str(key),
            dict(data),
        )
        for source, target, key, data in graph.edges(
            keys=True, data=True
        )
    )

    adjustments: list[SemanticRoleAdjustment] = []

    for node_id, data in list(normalized.nodes(data=True)):
        node_id = str(node_id)
        original_type = str(data.get("type", ""))
        if original_type not in CATALYST_ROLE_SOURCE_TYPES:
            continue

        evaluated_experiments = {
            str(target)
            for _, target, _, edge_data in normalized.out_edges(
                node_id, keys=True, data=True
            )
            if _relation(edge_data) in CATALYTIC_RELATIONS
            and str(normalized.nodes[target].get("type", "")) == "Experiment"
        }
        if not evaluated_experiments:
            continue

        measurement_ids = _catalytic_measurements(
            normalized,
            node_id,
            evaluated_experiments,
        )
        has_catalyzes_edge = any(
            _relation(edge_data) == "CATALYZES"
            for _, _, _, edge_data in normalized.out_edges(
                node_id, keys=True, data=True
            )
        )
        if not measurement_ids and not has_catalyzes_edge:
            continue

        reason = (
            "Material/Support is evaluated in an experiment and is the "
            "MEASURED_FOR target of measurements produced by that experiment."
        )

        if not _has_support_role(normalized, node_id):
            normalized.nodes[node_id].update(
                {
                    "type": "Catalyst",
                    "semantic_role_original_type": original_type,
                    "semantic_role_inference": "catalyst",
                    "semantic_role_confidence": "high",
                    "semantic_role_reason": reason,
                }
            )
            adjustments.append(
                SemanticRoleAdjustment(
                    chunk_id=chunk_id,
                    source_node_id=node_id,
                    original_type=original_type,
                    resolved_type="Catalyst",
                    action="retyped_role_pure_mention",
                    role_node_id=node_id,
                    measurement_ids=tuple(sorted(measurement_ids)),
                    experiment_ids=tuple(sorted(evaluated_experiments)),
                    reason=reason,
                )
            )
            continue

        role_node_id = _role_node_id(chunk_id, node_id)
        suffix = 1
        base_role_id = role_node_id
        while role_node_id in normalized:
            role_node_id = f"{base_role_id}_{suffix}"
            suffix += 1

        role_data = dict(normalized.nodes[node_id])
        role_data.update(
            {
                "type": "Catalyst",
                "semantic_role_original_type": original_type,
                "semantic_role_source_node_id": node_id,
                "semantic_role_inference": "catalyst",
                "semantic_role_confidence": "high",
                "semantic_role_reason": reason,
            }
        )
        normalized.add_node(role_node_id, **role_data)

        edges_to_move: list[tuple[str, str, str, dict[str, Any]]] = []
        for source, target, key, edge_data in list(
            normalized.edges(keys=True, data=True)
        ):
            relation = _relation(edge_data)
            source_s, target_s = str(source), str(target)
            move = False
            new_source, new_target = source_s, target_s

            if source_s == node_id and relation in CATALYTIC_RELATIONS:
                move = True
                new_source = role_node_id
            elif (
                target_s == node_id
                and relation == "MEASURED_FOR"
                and source_s in measurement_ids
            ):
                move = True
                new_target = role_node_id

            if move:
                edges_to_move.append(
                    (source_s, target_s, str(key), dict(edge_data))
                )
                normalized.remove_edge(source, target, key=key)
                normalized.add_edge(
                    new_source,
                    new_target,
                    key=str(key),
                    **dict(edge_data),
                )

        for measurement_id in measurement_ids:
            if measurement_id in normalized:
                normalized.nodes[measurement_id]["subject_id"] = role_node_id

        adjustments.append(
            SemanticRoleAdjustment(
                chunk_id=chunk_id,
                source_node_id=node_id,
                original_type=original_type,
                resolved_type="Catalyst",
                action="split_support_and_catalyst_roles",
                role_node_id=role_node_id,
                measurement_ids=tuple(sorted(measurement_ids)),
                experiment_ids=tuple(sorted(evaluated_experiments)),
                reason=reason,
            )
        )

    return normalized, adjustments
