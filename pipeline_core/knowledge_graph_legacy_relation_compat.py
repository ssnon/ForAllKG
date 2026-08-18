from __future__ import annotations

from typing import Any


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
