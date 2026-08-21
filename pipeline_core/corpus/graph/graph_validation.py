from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from pipeline_core.corpus.extraction.draft_schema import KnowledgeGraphDraft
from pipeline_core.corpus.graph.graph_domain import RelationConstraint
from pipeline_core.runtime.validation_issues import (
    IssueCode,
    IssueStage,
    ValidationIssue,
    ValidationReport,
    issue,
)

from pipeline_core.corpus.graph.graph_validation_legacy_relation_compat import (
    append_legacy_dac_relation_compat_issues,
)


def _node_index(graph: KnowledgeGraphDraft):
    collections = graph.node_collections()
    by_id: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for collection, nodes in collections.items():
        for node in nodes:
            by_id[node.id].append((collection, node))
    return collections, by_id


def _entity_types(graph: KnowledgeGraphDraft) -> dict[str, str]:
    return {node.id: node.type for node in graph.entities}


def _relation_type_issue(
    *,
    source: bool,
    edge_index: int,
    relation: str,
    endpoint_id: str,
    expected: Iterable[str],
    actual: str | None,
) -> ValidationIssue:
    side = "source" if source else "target"
    code = (
        IssueCode.RELATION_SOURCE_TYPE_MISMATCH
        if source
        else IssueCode.RELATION_TARGET_TYPE_MISMATCH
    )
    return issue(
        code=code,
        stage=IssueStage.RELATION,
        message=(
            f"{relation} {side} {endpoint_id!r} has type {actual!r}; "
            f"expected one of {sorted(expected)!r}."
        ),
        edge_index=edge_index,
        source_id=endpoint_id if source else None,
        target_id=endpoint_id if not source else None,
        relation=relation,
        expected={"types": sorted(expected)},
        actual={"type": actual},
    )


def collect_graph_issues(
    graph: KnowledgeGraphDraft,
    *,
    relation_constraints: tuple[RelationConstraint, ...] | None = None,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    collections, by_id = _node_index(graph)
    node_ids = set(by_id)

    # --------------------------------------------------------
    # Duplicate IDs and graph-level metadata
    # --------------------------------------------------------
    for node_id, entries in sorted(by_id.items()):
        if len(entries) > 1:
            issues.append(
                issue(
                    code=IssueCode.DUPLICATE_NODE_ID,
                    stage=IssueStage.STRUCTURAL,
                    message=f"Duplicate node ID {node_id!r} appears {len(entries)} times.",
                    node_id=node_id,
                    actual={"collections": [collection for collection, _ in entries]},
                )
            )

    duplicate_assets = sorted(
        asset_id
        for asset_id, count in Counter(graph.asset_ids).items()
        if count > 1
    )
    for asset_id in duplicate_assets:
        issues.append(
            issue(
                code=IssueCode.DUPLICATE_GRAPH_ASSET_ID,
                stage=IssueStage.PROVENANCE,
                message=f"Duplicate graph-level asset ID: {asset_id!r}.",
                node_id=asset_id,
            )
        )

    claim_prefixes = ("claim_", "obs_", "oc_", "mech_")
    for entity in graph.entities:
        if entity.id.lower().startswith(claim_prefixes):
            issues.append(
                issue(
                    code=IssueCode.CLAIM_LIKE_ENTITY,
                    stage=IssueStage.STRUCTURAL,
                    message=(
                        f"Claim-like node {entity.id!r} was placed in entities "
                        f"with type {entity.type!r}."
                    ),
                    node_id=entity.id,
                    node_collection="entities",
                    actual={"type": entity.type},
                )
            )

    # --------------------------------------------------------
    # Provenance and undefined endpoints
    # --------------------------------------------------------
    allowed_assets = set(graph.asset_ids)
    allowed_pages = set(graph.page_ids)
    for edge_index, edge in enumerate(graph.edges):
        if not edge.evidence_pointers:
            issues.append(
                issue(
                    code=IssueCode.EDGE_MISSING_EVIDENCE_POINTER,
                    stage=IssueStage.PROVENANCE,
                    message=f"Edge {edge_index} has no evidence pointer.",
                    edge_index=edge_index,
                    source_id=edge.source,
                    target_id=edge.target,
                    relation=edge.relation,
                )
            )

        for pointer in edge.evidence_pointers:
            if pointer.document_id != graph.document_id:
                issues.append(
                    issue(
                        code=IssueCode.POINTER_DOCUMENT_ID_MISMATCH,
                        stage=IssueStage.PROVENANCE,
                        message=(
                            f"Edge {edge_index} pointer document_id "
                            f"{pointer.document_id!r} does not match {graph.document_id!r}."
                        ),
                        edge_index=edge_index,
                        source_id=edge.source,
                        target_id=edge.target,
                        relation=edge.relation,
                        expected={"document_id": graph.document_id},
                        actual={"document_id": pointer.document_id},
                    )
                )
            if pointer.document_role != graph.document_role:
                issues.append(
                    issue(
                        code=IssueCode.POINTER_DOCUMENT_ROLE_MISMATCH,
                        stage=IssueStage.PROVENANCE,
                        message=(
                            f"Edge {edge_index} pointer role {pointer.document_role!r} "
                            f"does not match {graph.document_role!r}."
                        ),
                        edge_index=edge_index,
                        source_id=edge.source,
                        target_id=edge.target,
                        relation=edge.relation,
                        expected={"document_role": graph.document_role},
                        actual={"document_role": pointer.document_role},
                    )
                )
            unknown_assets = sorted(set(pointer.asset_ids) - allowed_assets)
            if unknown_assets:
                issues.append(
                    issue(
                        code=IssueCode.POINTER_UNKNOWN_ASSET,
                        stage=IssueStage.PROVENANCE,
                        message=f"Edge {edge_index} points to unknown assets {unknown_assets!r}.",
                        edge_index=edge_index,
                        source_id=edge.source,
                        target_id=edge.target,
                        relation=edge.relation,
                        actual={"unknown_asset_ids": unknown_assets},
                    )
                )
            if (
                pointer.page_id is not None
                and allowed_pages
                and pointer.page_id not in allowed_pages
            ):
                issues.append(
                    issue(
                        code=IssueCode.POINTER_UNKNOWN_PAGE,
                        stage=IssueStage.PROVENANCE,
                        message=(
                            f"Edge {edge_index} points to page {pointer.page_id}, "
                            "which is not present in graph.page_ids."
                        ),
                        edge_index=edge_index,
                        source_id=edge.source,
                        target_id=edge.target,
                        relation=edge.relation,
                        actual={"page_id": pointer.page_id},
                    )
                )

        if edge.source not in node_ids:
            issues.append(
                issue(
                    code=IssueCode.UNDEFINED_EDGE_SOURCE,
                    stage=IssueStage.STRUCTURAL,
                    message=f"Edge {edge_index} references undefined source {edge.source!r}.",
                    edge_index=edge_index,
                    source_id=edge.source,
                    target_id=edge.target,
                    relation=edge.relation,
                )
            )
        if edge.target not in node_ids:
            issues.append(
                issue(
                    code=IssueCode.UNDEFINED_EDGE_TARGET,
                    stage=IssueStage.STRUCTURAL,
                    message=f"Edge {edge_index} references undefined target {edge.target!r}.",
                    edge_index=edge_index,
                    source_id=edge.source,
                    target_id=edge.target,
                    relation=edge.relation,
                )
            )

    # Continue collecting only with defined endpoints in adjacency maps.
    incoming: dict[str, list[tuple[int, object]]] = defaultdict(list)
    outgoing: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for edge_index, edge in enumerate(graph.edges):
        if edge.source in node_ids:
            outgoing[edge.source].append((edge_index, edge))
        if edge.target in node_ids:
            incoming[edge.target].append((edge_index, edge))

    entity_ids = {node.id for node in graph.entities}
    experiment_ids = {node.id for node in graph.experiments}
    calculation_ids = {node.id for node in graph.calculations}
    measurement_ids = {node.id for node in graph.measurements}
    group_ids = {node.id for node in graph.measurement_groups}
    observation_ids = {node.id for node in graph.observation_claims}
    mechanism_ids = {node.id for node in graph.mechanism_claims}
    claim_ids = observation_ids | mechanism_ids

    # --------------------------------------------------------
    # Measurement and group bookkeeping
    # --------------------------------------------------------
    valid_measurement_sources = experiment_ids | calculation_ids
    group_by_id = {node.id: node for node in graph.measurement_groups}
    measurement_by_id = {node.id: node for node in graph.measurements}

    for measurement in graph.measurements:
        producer_edges = [
            edge
            for _, edge in incoming[measurement.id]
            if edge.relation == "HAS_MEASUREMENT" and edge.source in valid_measurement_sources
        ]
        if not producer_edges:
            issues.append(
                issue(
                    code=IssueCode.MISSING_MEASUREMENT_PRODUCER,
                    stage=IssueStage.MEASUREMENT,
                    message=(
                        f"Measurement {measurement.id!r} has no incoming "
                        "HAS_MEASUREMENT from an Experiment or Calculation."
                    ),
                    node_id=measurement.id,
                    node_collection="measurements",
                )
            )

        measured_for = [
            edge for _, edge in outgoing[measurement.id] if edge.relation == "MEASURED_FOR"
        ]
        if len(measured_for) != 1:
            issues.append(
                issue(
                    code=IssueCode.INVALID_MEASURED_FOR_COUNT,
                    stage=IssueStage.MEASUREMENT,
                    message=(
                        f"Measurement {measurement.id!r} has {len(measured_for)} "
                        "MEASURED_FOR edges; expected exactly one."
                    ),
                    node_id=measurement.id,
                    node_collection="measurements",
                    expected={"count": 1, "target": measurement.subject_id},
                    actual={"count": len(measured_for), "targets": [edge.target for edge in measured_for]},
                )
            )
        elif measured_for[0].target != measurement.subject_id:
            issues.append(
                issue(
                    code=IssueCode.MEASURED_FOR_TARGET_MISMATCH,
                    stage=IssueStage.MEASUREMENT,
                    message=(
                        f"Measurement {measurement.id!r} subject_id "
                        f"{measurement.subject_id!r} does not match MEASURED_FOR "
                        f"target {measured_for[0].target!r}."
                    ),
                    node_id=measurement.id,
                    node_collection="measurements",
                    source_id=measurement.id,
                    target_id=measured_for[0].target,
                    relation="MEASURED_FOR",
                    expected={"target": measurement.subject_id},
                    actual={"target": measured_for[0].target},
                )
            )

        if measurement.subject_id not in entity_ids:
            issues.append(
                issue(
                    code=IssueCode.MEASUREMENT_SUBJECT_NOT_ENTITY,
                    stage=IssueStage.MEASUREMENT,
                    message=(
                        f"Measurement {measurement.id!r} subject_id "
                        f"{measurement.subject_id!r} is not an Entity."
                    ),
                    node_id=measurement.id,
                    node_collection="measurements",
                    target_id=measurement.subject_id,
                    expected={"target_collection": "entities"},
                )
            )

        membership_edges = [
            edge
            for _, edge in outgoing[measurement.id]
            if edge.relation == "IN_MEASUREMENT_GROUP"
        ]
        if measurement.group_id is None:
            if membership_edges:
                issues.append(
                    issue(
                        code=IssueCode.UNEXPECTED_MEASUREMENT_GROUP_EDGE,
                        stage=IssueStage.MEASUREMENT,
                        message=(
                            f"Measurement {measurement.id!r} has group edges while group_id is null."
                        ),
                        node_id=measurement.id,
                        node_collection="measurements",
                        actual={"targets": [edge.target for edge in membership_edges]},
                    )
                )
        else:
            if measurement.group_id not in group_ids:
                issues.append(
                    issue(
                        code=IssueCode.UNKNOWN_MEASUREMENT_GROUP,
                        stage=IssueStage.MEASUREMENT,
                        message=(
                            f"Measurement {measurement.id!r} references unknown group "
                            f"{measurement.group_id!r}."
                        ),
                        node_id=measurement.id,
                        node_collection="measurements",
                        target_id=measurement.group_id,
                    )
                )
            if len(membership_edges) != 1 or membership_edges[0].target != measurement.group_id:
                issues.append(
                    issue(
                        code=IssueCode.MISSING_MEASUREMENT_GROUP_EDGE,
                        stage=IssueStage.MEASUREMENT,
                        message=(
                            f"Measurement {measurement.id!r} must have one "
                            f"IN_MEASUREMENT_GROUP edge to {measurement.group_id!r}."
                        ),
                        node_id=measurement.id,
                        node_collection="measurements",
                        source_id=measurement.id,
                        target_id=measurement.group_id,
                        relation="IN_MEASUREMENT_GROUP",
                        expected={"count": 1, "target": measurement.group_id},
                        actual={"count": len(membership_edges), "targets": [edge.target for edge in membership_edges]},
                    )
                )

    for group in graph.measurement_groups:
        if len(group.member_measurement_ids) < 2:
            issues.append(
                issue(
                    code=IssueCode.SINGLETON_MEASUREMENT_GROUP,
                    stage=IssueStage.MEASUREMENT,
                    message=(
                        f"MeasurementGroup {group.id!r} has fewer than two members."
                    ),
                    node_id=group.id,
                    node_collection="measurement_groups",
                    actual={"member_ids": list(group.member_measurement_ids)},
                )
            )
        if len(group.member_measurement_ids) != len(set(group.member_measurement_ids)):
            issues.append(
                issue(
                    code=IssueCode.DUPLICATE_MEASUREMENT_GROUP_MEMBER,
                    stage=IssueStage.MEASUREMENT,
                    message=f"MeasurementGroup {group.id!r} contains duplicate member IDs.",
                    node_id=group.id,
                    node_collection="measurement_groups",
                    actual={"member_ids": list(group.member_measurement_ids)},
                )
            )
        unknown_members = sorted(set(group.member_measurement_ids) - measurement_ids)
        if unknown_members:
            issues.append(
                issue(
                    code=IssueCode.MEASUREMENT_GROUP_UNKNOWN_MEMBER,
                    stage=IssueStage.MEASUREMENT,
                    message=(
                        f"MeasurementGroup {group.id!r} has unknown members {unknown_members!r}."
                    ),
                    node_id=group.id,
                    node_collection="measurement_groups",
                    actual={"unknown_member_ids": unknown_members},
                )
            )
        for member_id in group.member_measurement_ids:
            member = measurement_by_id.get(member_id)
            if member is not None and member.group_id != group.id:
                issues.append(
                    issue(
                        code=IssueCode.MEASUREMENT_GROUP_MEMBER_MISMATCH,
                        stage=IssueStage.MEASUREMENT,
                        message=(
                            f"MeasurementGroup {group.id!r} and member {member_id!r} "
                            "do not agree on group_id."
                        ),
                        node_id=member_id,
                        node_collection="measurements",
                        target_id=group.id,
                        expected={"group_id": group.id},
                        actual={"group_id": member.group_id},
                    )
                )

    # --------------------------------------------------------
    # Isolated nodes and claims
    # --------------------------------------------------------
    for collection, nodes in collections.items():
        for node in nodes:
            if not incoming[node.id] and not outgoing[node.id]:
                issues.append(
                    issue(
                        code=IssueCode.ISOLATED_NODE,
                        stage=IssueStage.STRUCTURAL,
                        message=f"Node {node.id!r} is isolated.",
                        node_id=node.id,
                        node_collection=collection,
                    )
                )

    for claim in graph.observation_claims:
        support = [edge for _, edge in incoming[claim.id] if edge.relation == "SUPPORTS_CLAIM"]
        targets = [edge for _, edge in outgoing[claim.id] if edge.relation == "APPLIES_TO"]
        if not support:
            issues.append(
                issue(
                    code=IssueCode.OBSERVATION_MISSING_SUPPORT,
                    stage=IssueStage.CLAIM,
                    message=f"Observation claim {claim.id!r} has no SUPPORTS_CLAIM evidence.",
                    node_id=claim.id,
                    node_collection="observation_claims",
                )
            )
        if not targets:
            issues.append(
                issue(
                    code=IssueCode.CLAIM_MISSING_APPLICATION_TARGET,
                    stage=IssueStage.CLAIM,
                    message=f"Observation claim {claim.id!r} has no APPLIES_TO target.",
                    node_id=claim.id,
                    node_collection="observation_claims",
                )
            )

    for claim in graph.mechanism_claims:
        direct = [edge for _, edge in incoming[claim.id] if edge.relation == "SUPPORTS_CLAIM"]
        interpreted = [edge for _, edge in incoming[claim.id] if edge.relation == "INTERPRETED_AS"]
        targets = [edge for _, edge in outgoing[claim.id] if edge.relation == "APPLIES_TO"]
        if not direct and not interpreted:
            issues.append(
                issue(
                    code=IssueCode.MECHANISM_MISSING_SUPPORT,
                    stage=IssueStage.CLAIM,
                    message=(
                        f"Mechanism claim {claim.id!r} has neither SUPPORTS_CLAIM "
                        "nor INTERPRETED_AS evidence."
                    ),
                    node_id=claim.id,
                    node_collection="mechanism_claims",
                )
            )
        if not targets:
            issues.append(
                issue(
                    code=IssueCode.CLAIM_MISSING_APPLICATION_TARGET,
                    stage=IssueStage.CLAIM,
                    message=f"Mechanism claim {claim.id!r} has no APPLIES_TO target.",
                    node_id=claim.id,
                    node_collection="mechanism_claims",
                )
            )

    # --------------------------------------------------------
    # Relation endpoint types
    # --------------------------------------------------------
    entity_type_by_id = _entity_types(graph)
    collection_by_id = {
        node_id: entries[0][0]
        for node_id, entries in by_id.items()
        if len(entries) == 1
    }

    def semantic_type(node_id: str) -> str | None:
        entity_type = entity_type_by_id.get(node_id)
        if entity_type is not None:
            return entity_type
        collection_name = collection_by_id.get(node_id)
        return {
            "experiments": "Experiment",
            "calculations": "Calculation",
            "measurements": "Measurement",
            "measurement_groups": "MeasurementGroup",
            "observation_claims": "ObservationClaim",
            "mechanism_claims": "MechanismClaim",
        }.get(collection_name)

    def endpoint_matches(
        node_id: str,
        expected: frozenset[str],
    ) -> bool:
        if not expected:
            return True
        if (
            "Entity" in expected
            and collection_by_id.get(node_id) == "entities"
        ):
            return True
        return semantic_type(node_id) in expected

    if relation_constraints is not None:
        constraints_by_relation = {
            item.relation: item
            for item in relation_constraints
        }
        for edge_index, edge in enumerate(graph.edges):
            if edge.source not in node_ids or edge.target not in node_ids:
                continue
            constraint = constraints_by_relation.get(edge.relation)
            if constraint is None:
                continue
            if not endpoint_matches(edge.source, constraint.source_types):
                issues.append(
                    _relation_type_issue(
                        source=True,
                        edge_index=edge_index,
                        relation=edge.relation,
                        endpoint_id=edge.source,
                        expected=constraint.source_types,
                        actual=semantic_type(edge.source),
                    )
                )
            if not endpoint_matches(edge.target, constraint.target_types):
                issues.append(
                    _relation_type_issue(
                        source=False,
                        edge_index=edge_index,
                        relation=edge.relation,
                        endpoint_id=edge.target,
                        expected=constraint.target_types,
                        actual=semantic_type(edge.target),
                    )
                )
        return ValidationReport.from_issues(issues)

    # Legacy direct-call fallback. Runtime strict extraction now passes the
    # active domain adapter's explicit relation contract.
    # Legacy direct-call fallback. Runtime strict extraction now passes the
    # active domain adapter's explicit relation contract.
    append_legacy_dac_relation_compat_issues(
        graph=graph,
        node_ids=node_ids,
        by_id=by_id,
        claim_ids=claim_ids,
        observation_ids=observation_ids,
        mechanism_ids=mechanism_ids,
        issues=issues,
        entity_types_fn=_entity_types,
        relation_type_issue_fn=_relation_type_issue,
    )

    return ValidationReport.from_issues(issues)
