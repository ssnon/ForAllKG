from __future__ import annotations

from typing import Any, Callable


def append_legacy_dac_relation_compat_issues(
    *,
    graph: Any,
    node_ids: set[str],
    by_id: dict[str, list[tuple[str, Any]]],
    claim_ids: set[str],
    observation_ids: set[str],
    mechanism_ids: set[str],
    issues: list[Any],
    entity_types_fn: Callable[
        [Any],
        dict[str, str],
    ],
    relation_type_issue_fn: Callable[..., Any],
) -> None:
    """Preserve historical no-contract DAC relation validation."""

    _entity_types = entity_types_fn
    _relation_type_issue = relation_type_issue_fn

    entity_type_by_id = _entity_types(graph)
    collection_by_id = {
        node_id: entries[0][0]
        for node_id, entries in by_id.items()
        if len(entries) == 1
    }

    def entity_type(node_id: str) -> str | None:
        return entity_type_by_id.get(node_id)

    def collection(node_id: str) -> str | None:
        return collection_by_id.get(node_id)

    for edge_index, edge in enumerate(graph.edges):
        if edge.source not in node_ids or edge.target not in node_ids:
            continue

        relation = edge.relation
        source = edge.source
        target = edge.target

        if relation == "EVALUATED_IN":
            if entity_type(source) not in {"Catalyst", "CatalystModel", "Material"}:
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"Catalyst", "CatalystModel", "Material"}, actual=entity_type(source)))
            if collection(target) != "experiments":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"Experiment"}, actual=collection(target)))

        elif relation == "CHARACTERIZED_BY":
            if entity_type(source) not in {"Catalyst", "Support", "Material", "CoordinationMotif"}:
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"Catalyst", "Support", "Material", "CoordinationMotif"}, actual=entity_type(source)))
            if collection(target) != "experiments":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"Experiment"}, actual=collection(target)))

        elif relation == "MODELED_BY":
            if entity_type(source) != "CatalystModel":
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"CatalystModel"}, actual=entity_type(source)))
            if collection(target) != "calculations":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"Calculation"}, actual=collection(target)))

        elif relation == "SYNTHESIZED_BY":
            if entity_type(source) != "Catalyst":
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"Catalyst"}, actual=entity_type(source)))
            if entity_type(target) != "SynthesisMethod":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"SynthesisMethod"}, actual=entity_type(target)))

        elif relation == "USES_PRECURSOR":
            if entity_type(source) != "SynthesisMethod":
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"SynthesisMethod"}, actual=entity_type(source)))
            if entity_type(target) != "Precursor":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"Precursor"}, actual=entity_type(target)))

        elif relation == "HAS_MEASUREMENT":
            if collection(source) not in {"experiments", "calculations"}:
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"Experiment", "Calculation"}, actual=collection(source)))
            if collection(target) != "measurements":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"Measurement"}, actual=collection(target)))

        elif relation == "MEASURED_FOR":
            if collection(source) != "measurements":
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"Measurement"}, actual=collection(source)))
            if collection(target) != "entities":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"Entity"}, actual=collection(target)))

        elif relation == "IN_MEASUREMENT_GROUP":
            if collection(source) != "measurements":
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"Measurement"}, actual=collection(source)))
            if collection(target) != "measurement_groups":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"MeasurementGroup"}, actual=collection(target)))

        elif relation == "MODEL_OF":
            if entity_type(source) != "CatalystModel":
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"CatalystModel"}, actual=entity_type(source)))
            if entity_type(target) != "Catalyst":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"Catalyst"}, actual=entity_type(target)))

        elif relation == "SUPPORTS_CLAIM":
            if collection(source) not in {"measurements", "experiments", "calculations"}:
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"Measurement", "Experiment", "Calculation"}, actual=collection(source)))
            if target not in claim_ids:
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"ObservationClaim", "MechanismClaim"}, actual=collection(target)))

        elif relation == "INTERPRETED_AS":
            if source not in observation_ids:
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"ObservationClaim"}, actual=collection(source)))
            if target not in mechanism_ids:
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"MechanismClaim"}, actual=collection(target)))

        elif relation == "APPLIES_TO":
            if source not in claim_ids:
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"ObservationClaim", "MechanismClaim"}, actual=collection(source)))
            if collection(target) != "entities":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"Entity"}, actual=collection(target)))

        elif relation == "HAS_METAL":
            if entity_type(source) not in {"Catalyst", "CatalystModel"}:
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"Catalyst", "CatalystModel"}, actual=entity_type(source)))
            if entity_type(target) != "Metal":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"Metal"}, actual=entity_type(target)))

        elif relation == "SUPPORTED_ON":
            if entity_type(source) not in {"Catalyst", "CatalystModel"}:
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"Catalyst", "CatalystModel"}, actual=entity_type(source)))
            if entity_type(target) != "Support":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"Support"}, actual=entity_type(target)))

        elif relation == "CATALYZES":
            if entity_type(source) != "Catalyst":
                issues.append(_relation_type_issue(source=True, edge_index=edge_index, relation=relation, endpoint_id=source, expected={"Catalyst"}, actual=entity_type(source)))
            if entity_type(target) != "Reaction":
                issues.append(_relation_type_issue(source=False, edge_index=edge_index, relation=relation, endpoint_id=target, expected={"Reaction"}, actual=entity_type(target)))
