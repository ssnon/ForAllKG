from __future__ import annotations

from typing import Any

from pipeline_core.corpus.graph.legacy_dac_relation_policy import (
    LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION,
)


def validate_legacy_relation_semantics_compat(
    graph: Any,
) -> Any:
    """Preserve historical direct-call relation validation semantics."""

    self = graph
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

    collection_ids_by_semantic_type = {
        "Entity": entity_ids,
        "Experiment": experiment_ids,
        "Calculation": calculation_ids,
        "Measurement": measurement_ids,
        "MeasurementGroup": measurement_group_ids,
        "ObservationClaim": observation_claim_ids,
        "MechanismClaim": mechanism_claim_ids,
    }

    def legacy_endpoint_matches(
        *,
        relation: str,
        side: str,
        node_id: str,
    ) -> bool:
        policy = (
            LEGACY_DAC_RELATION_ENDPOINT_POLICY_BY_RELATION.get(
                relation
            )
        )

        if policy is None:
            raise RuntimeError(
                "Missing legacy DAC relation policy "
                f"for {relation!r}."
            )

        if side == "source":
            expected_types = policy.source_types
        elif side == "target":
            expected_types = policy.target_types
        else:
            raise ValueError(
                f"Unknown relation endpoint side: {side!r}"
            )

        collection_types = {
            semantic_type
            for semantic_type in expected_types
            if semantic_type
            in collection_ids_by_semantic_type
        }

        if collection_types:
            if len(collection_types) != len(
                expected_types
            ):
                raise RuntimeError(
                    "Mixed entity/collection legacy relation "
                    f"policy is unsupported for {relation!r} "
                    f"{side}."
                )

            return any(
                node_id
                in collection_ids_by_semantic_type[
                    semantic_type
                ]
                for semantic_type
                in collection_types
            )

        return entity_has_type(
            node_id,
            set(expected_types),
        )

    for edge in self.edges:
        relation = edge.relation
        source = edge.source
        target = edge.target

        # Catalyst -> Experiment
        if relation == "EVALUATED_IN":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "EVALUATED_IN source must be "
                    "a Catalyst, CatalystModel, "
                    "or Material: "
                    f"{source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "EVALUATED_IN target must be "
                    f"an Experiment: {target!r}"
                )

        # Physical scientific object -> characterization
        elif relation == "CHARACTERIZED_BY":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "CHARACTERIZED_BY source must be "
                    "a physical scientific entity: "
                    f"{source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "CHARACTERIZED_BY target must be "
                    f"an Experiment: {target!r}"
                )

        # CatalystModel -> Calculation
        elif relation == "MODELED_BY":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "MODELED_BY source must be "
                    f"a CatalystModel: {source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "MODELED_BY target must be "
                    f"a Calculation: {target!r}"
                )

        # Catalyst -> SynthesisMethod
        elif relation == "SYNTHESIZED_BY":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "SYNTHESIZED_BY source must be "
                    f"a Catalyst: {source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "SYNTHESIZED_BY target must be "
                    f"a SynthesisMethod: {target!r}"
                )

        # SynthesisMethod -> Precursor
        elif relation == "USES_PRECURSOR":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "USES_PRECURSOR source must be "
                    f"a SynthesisMethod: {source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "USES_PRECURSOR target must be "
                    f"a Precursor: {target!r}"
                )

        # Experiment/Calculation -> Measurement
        elif relation == "HAS_MEASUREMENT":
            if (
                not legacy_endpoint_matches(relation=relation, side="source", node_id=source)
            ):
                semantic_errors.append(
                    "HAS_MEASUREMENT source must be "
                    "an Experiment or Calculation: "
                    f"{source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "HAS_MEASUREMENT target must be "
                    f"a Measurement: {target!r}"
                )

        elif relation == "MEASURED_FOR":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    f"MEASURED_FOR source must be a Measurement: {source!r}"
                )
            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    f"MEASURED_FOR target must be an Entity: {target!r}"
                )

        elif relation == "IN_MEASUREMENT_GROUP":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "IN_MEASUREMENT_GROUP source must be a Measurement: "
                    f"{source!r}"
                )
            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "IN_MEASUREMENT_GROUP target must be a MeasurementGroup: "
                    f"{target!r}"
                )

        elif relation == "MODEL_OF":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    f"MODEL_OF source must be a CatalystModel: {source!r}"
                )
            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
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

            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "SUPPORTS_CLAIM source must be "
                    "a Measurement, Experiment, or "
                    f"Calculation: {source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "SUPPORTS_CLAIM target must be "
                    f"a claim: {target!r}"
                )

        # ObservationClaim -> MechanismClaim
        elif relation == "INTERPRETED_AS":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "INTERPRETED_AS source must be "
                    "an ObservationClaim: "
                    f"{source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "INTERPRETED_AS target must be "
                    "a MechanismClaim: "
                    f"{target!r}"
                )

        # Claim -> scientific entity
        elif relation == "APPLIES_TO":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "APPLIES_TO source must be "
                    f"a claim: {source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "APPLIES_TO target must be "
                    f"an Entity: {target!r}"
                )

        # SynthesisMethod -> Precursor already handled;
        # catalyst composition
        elif relation == "HAS_METAL":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "HAS_METAL source must be "
                    "a Catalyst or CatalystModel: "
                    f"{source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "HAS_METAL target must be "
                    f"a Metal: {target!r}"
                )

        elif relation == "SUPPORTED_ON":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "SUPPORTED_ON source must be "
                    "a Catalyst or CatalystModel: "
                    f"{source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
                semantic_errors.append(
                    "SUPPORTED_ON target must be "
                    f"a Support: {target!r}"
                )

        elif relation == "CATALYZES":
            if not legacy_endpoint_matches(relation=relation, side="source", node_id=source):
                semantic_errors.append(
                    "CATALYZES source must be "
                    f"a Catalyst: {source!r}"
                )

            if not legacy_endpoint_matches(relation=relation, side="target", node_id=target):
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
