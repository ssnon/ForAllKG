from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_experiment_repair import (
    ExperimentRepairSet,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


WitnessRequirementStatus = Literal[
    "unresolved_requires_measurement_witness",
    "no_witness_required",
]


class OperationalizationWitnessRequirement(StrictModel):
    schema_version: Literal[
        "operationalization-witness-requirement-v1"
    ] = "operationalization-witness-requirement-v1"

    requirement_id: str
    source_experiment_id: str
    output_experiment_id: str
    issue_code: Literal[
        "MEASUREMENT_INDEPENDENCE_UNRESOLVED"
    ]
    status: WitnessRequirementStatus

    requested_target_observable: str
    anchor_observable: str
    observable_pair: list[str] = Field(min_length=2, max_length=2)

    required_graph_topology: list[str] = Field(min_length=1)
    minimum_candidate_evidence: list[str] = Field(min_length=1)
    independence_acceptance_rule: str
    independence_rejection_rule: str

    corpus_lookup_required: Literal[True] = True
    distinct_measurement_provider_required: Literal[True] = True
    provenance_required: Literal[True] = True
    explicit_method_or_calculation_identity_required: Literal[True] = True

    measurement_independence_verified: Literal[False] = False
    witness_candidate_count: Literal[0] = 0

    candidate_inspiration_involved: bool = False
    requires_verification: Literal[True] = True
    epistemic_status: Literal[
        "inspiration_only"
    ] = "inspiration_only"

    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    shadow_only: Literal[True] = True


class OperationalizationWitnessRequirementSet(StrictModel):
    schema_version: Literal[
        "operationalization-witness-requirement-set-v1"
    ] = "operationalization-witness-requirement-set-v1"

    repaired_experiment_count: int = Field(ge=0)
    requirement_count: int = Field(ge=0)
    candidate_inspiration_requirement_count: int = Field(ge=0)
    requirements: list[OperationalizationWitnessRequirement] = Field(
        default_factory=list
    )

    corpus_lookup_performed: Literal[False] = False
    measurement_independence_verified_count: Literal[0] = 0
    witness_candidate_count: Literal[0] = 0

    diagnostic_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    experiment_selection_performed: Literal[False] = False
    rejection_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    gap_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    external_novelty_review_bypass_authorized: Literal[False] = False
    shadow_only: Literal[True] = True


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def build_operationalization_witness_requirements(
    repairs: ExperimentRepairSet,
) -> OperationalizationWitnessRequirementSet:
    experiments_by_id = {
        row.experiment_id: row
        for row in repairs.repaired_experiments.experiments
    }

    requirements = []

    for record in repairs.records:
        experiment = experiments_by_id.get(record.output_experiment_id)
        if experiment is None:
            raise ValueError(
                "repair record output experiment missing: "
                + record.output_experiment_id
            )

        for action in record.actions:
            if (
                action.issue_code
                != "MEASUREMENT_INDEPENDENCE_UNRESOLVED"
                or action.disposition != "deferred_requires_witness"
            ):
                continue

            if len(experiment.observables) < 2:
                raise ValueError(
                    "measurement-independence requirement needs two "
                    "operational observables"
                )

            target = str(experiment.observables[0]).strip()
            anchor = str(experiment.observables[1]).strip()

            if not target or not anchor:
                raise ValueError(
                    "measurement-independence requirement has blank observable"
                )

            requirements.append(
                OperationalizationWitnessRequirement(
                    requirement_id=_stable_id(
                        "operationalization_witness_requirement",
                        record.output_experiment_id,
                        target,
                        anchor,
                    ),
                    source_experiment_id=record.source_experiment_id,
                    output_experiment_id=record.output_experiment_id,
                    issue_code="MEASUREMENT_INDEPENDENCE_UNRESOLVED",
                    status="unresolved_requires_measurement_witness",
                    requested_target_observable=target,
                    anchor_observable=anchor,
                    observable_pair=[target, anchor],
                    required_graph_topology=[
                        (
                            "Experiment or Calculation "
                            "--HAS_MEASUREMENT--> Measurement(target)"
                        ),
                        (
                            "Experiment or Calculation "
                            "--HAS_MEASUREMENT--> Measurement(anchor)"
                        ),
                        (
                            "Each Measurement should preserve MEASURED_FOR "
                            "and source provenance when available"
                        ),
                    ],
                    minimum_candidate_evidence=[
                        (
                            "At least one corpus-grounded measurement candidate "
                            "whose metric/source expression operationalizes the "
                            "requested target observable"
                        ),
                        (
                            "At least one corpus-grounded measurement candidate "
                            "whose metric/source expression operationalizes the "
                            "anchor observable"
                        ),
                        (
                            "The two candidates must retain identifiable "
                            "Experiment/Calculation providers and provenance"
                        ),
                    ],
                    independence_acceptance_rule=(
                        "Do not mark measurement independence verified merely "
                        "because both observables have Measurement nodes. A "
                        "later resolver must show distinct operationalization "
                        "or an explicit disentangling model/method, with "
                        "provenance sufficient to audit the claim."
                    ),
                    independence_rejection_rule=(
                        "Remain unresolved when only one observable has a "
                        "measurement candidate, when both map only to the same "
                        "undifferentiated measurement representation, when "
                        "provider/method identity is missing, or when the "
                        "source does not support independent/disentangled "
                        "measurement."
                    ),
                    candidate_inspiration_involved=(
                        experiment.candidate_inspiration_involved
                    ),
                )
            )

    return OperationalizationWitnessRequirementSet(
        repaired_experiment_count=(
            repairs.repaired_experiments.experiment_count
        ),
        requirement_count=len(requirements),
        candidate_inspiration_requirement_count=sum(
            row.candidate_inspiration_involved
            for row in requirements
        ),
        requirements=requirements,
    )
