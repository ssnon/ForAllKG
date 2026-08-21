from __future__ import annotations

from typing import Any, Callable

from pipeline_core.corpus.graph.legacy_dac_relation_policy import (
    LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION,
)


_COLLECTION_BY_SEMANTIC_TYPE = {
    "Entity": "entities",
    "Experiment": "experiments",
    "Calculation": "calculations",
    "Measurement": "measurements",
    "MeasurementGroup": "measurement_groups",
    "ObservationClaim": "observation_claims",
    "MechanismClaim": "mechanism_claims",
}


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

    entity_type_by_id = entity_types_fn(graph)
    collection_by_id = {
        node_id: entries[0][0]
        for node_id, entries in by_id.items()
        if len(entries) == 1
    }

    def entity_type(
        node_id: str,
    ) -> str | None:
        return entity_type_by_id.get(node_id)

    def collection(
        node_id: str,
    ) -> str | None:
        return collection_by_id.get(node_id)

    def membership_override(
        *,
        relation: str,
        side: str,
        node_id: str,
    ) -> bool | None:
        if (
            relation == "SUPPORTS_CLAIM"
            and side == "target"
        ):
            return node_id in claim_ids

        if relation == "INTERPRETED_AS":
            if side == "source":
                return node_id in observation_ids
            return node_id in mechanism_ids

        if (
            relation == "APPLIES_TO"
            and side == "source"
        ):
            return node_id in claim_ids

        return None

    def endpoint_matches(
        *,
        relation: str,
        side: str,
        node_id: str,
        expected: frozenset[str],
    ) -> tuple[bool, str | None]:
        override = membership_override(
            relation=relation,
            side=side,
            node_id=node_id,
        )

        if override is not None:
            return override, collection(node_id)

        expected_collections = {
            _COLLECTION_BY_SEMANTIC_TYPE[
                semantic_type
            ]
            for semantic_type in expected
            if semantic_type
            in _COLLECTION_BY_SEMANTIC_TYPE
        }

        if len(expected_collections) == len(
            expected
        ):
            actual = collection(node_id)
            return (
                actual in expected_collections,
                actual,
            )

        actual = entity_type(node_id)
        return actual in expected, actual

    for edge_index, edge in enumerate(
        graph.edges
    ):
        if (
            edge.source not in node_ids
            or edge.target not in node_ids
        ):
            continue

        policy = (
            LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION.get(
                edge.relation
            )
        )

        if policy is None:
            continue

        source_ok, source_actual = (
            endpoint_matches(
                relation=edge.relation,
                side="source",
                node_id=edge.source,
                expected=policy.source_types,
            )
        )

        if not source_ok:
            issues.append(
                relation_type_issue_fn(
                    source=True,
                    edge_index=edge_index,
                    relation=edge.relation,
                    endpoint_id=edge.source,
                    expected=policy.source_types,
                    actual=source_actual,
                )
            )

        target_ok, target_actual = (
            endpoint_matches(
                relation=edge.relation,
                side="target",
                node_id=edge.target,
                expected=policy.target_types,
            )
        )

        if not target_ok:
            issues.append(
                relation_type_issue_fn(
                    source=False,
                    edge_index=edge_index,
                    relation=edge.relation,
                    endpoint_id=edge.target,
                    expected=policy.target_types,
                    actual=target_actual,
                )
            )
