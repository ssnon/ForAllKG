from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.higher_order_discriminating_experiments import (
    DiscriminatingExperimentCandidate,
    DiscriminatingExperimentSet,
)
from pipeline_core.discovery.higher_order_experiment_critic import (
    DiscriminatingExperimentCriticReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


RepairDisposition = Literal[
    "applied_deterministic_repair",
    "deferred_requires_witness",
    "no_repair_needed",
]


class ExperimentRepairAction(StrictModel):
    issue_code: str
    disposition: RepairDisposition
    rationale: str
    before_text: str | None = None
    after_text: str | None = None
    required_witness: str | None = None

    scientific_claim_created: Literal[False] = False
    authority_created: Literal[False] = False


class ExperimentRepairRecord(StrictModel):
    schema_version: Literal[
        "discriminating-experiment-repair-record-v1"
    ] = "discriminating-experiment-repair-record-v1"

    source_experiment_id: str
    output_experiment_id: str
    changed: bool
    actions: list[ExperimentRepairAction] = Field(
        default_factory=list
    )

    shadow_only: Literal[True] = True
    diagnostic_only: Literal[True] = True
    rejection_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


class ExperimentRepairSet(StrictModel):
    schema_version: Literal[
        "discriminating-experiment-repair-set-v1"
    ] = "discriminating-experiment-repair-set-v1"

    input_experiment_count: int = Field(ge=0)
    output_experiment_count: int = Field(ge=0)
    changed_experiment_count: int = Field(ge=0)
    deferred_experiment_count: int = Field(ge=0)

    records: list[ExperimentRepairRecord] = Field(default_factory=list)
    repaired_experiments: DiscriminatingExperimentSet

    deterministic_repair_only: Literal[True] = True
    llm_repair_used: Literal[False] = False
    original_artifact_mutated: Literal[False] = False
    scientific_claim_created: Literal[False] = False
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


def _repair_tradeoff_intervention(
    experiment: DiscriminatingExperimentCandidate,
) -> tuple[DiscriminatingExperimentCandidate, ExperimentRepairAction]:
    if (
        experiment.experiment_mode
        != "matched_joint_tradeoff_sweep"
        or len(experiment.observables) < 2
        or not str(experiment.requested_source).strip()
    ):
        return (
            experiment,
            ExperimentRepairAction(
                issue_code="INTERVENTION_UNDERSPECIFIED",
                disposition="deferred_requires_witness",
                rationale=(
                    "The intervention is underspecified, but the deterministic "
                    "tradeoff repair preconditions are not satisfied."
                ),
                required_witness=(
                    "An explicit intervention variable or sweep variable "
                    "grounded in the source task or experiment design."
                ),
            ),
        )

    before = experiment.intervention_or_sweep
    after = (
        f"Perform a matched sweep over {experiment.requested_source} while "
        f"jointly measuring {experiment.observables[0]} and "
        f"{experiment.observables[1]}."
    )

    repaired = experiment.model_copy(
        update={
            "experiment_id": _stable_id(
                "repaired_discriminating_experiment",
                experiment.experiment_id,
                after,
            ),
            "intervention_or_sweep": after,
        }
    )

    return (
        repaired,
        ExperimentRepairAction(
            issue_code="INTERVENTION_UNDERSPECIFIED",
            disposition="applied_deterministic_repair",
            rationale=(
                "The original text already identified requested_source as a "
                "valid sweep variable but also added an unresolved alternative. "
                "The repair removes only that alternative and does not invent "
                "a new intervention."
            ),
            before_text=before,
            after_text=after,
        ),
    )


def _deferred_action(issue_code: str) -> ExperimentRepairAction:
    witness_by_code = {
        "MEASUREMENT_INDEPENDENCE_UNRESOLVED": (
            "A measurement-method or instrumentation witness showing that "
            "the target and anchor observables can be measured independently "
            "or disentangled with an identified model."
        ),
        "PREDICTION_SEPARATION_UNRESOLVED": (
            "A domain-grounded distinction between the A and B outcome "
            "patterns; deterministic text rewriting is not sufficient."
        ),
        "OBSERVABLE_SET_DEGENERATE": (
            "An additional non-redundant observable or an explicit witness "
            "that the existing observables are operationally distinct."
        ),
        "BASIS_ALIGNMENT_UNRESOLVED": (
            "An explicit intervention-to-basis relation witness."
        ),
    }
    return ExperimentRepairAction(
        issue_code=issue_code,
        disposition="deferred_requires_witness",
        rationale=(
            "No authority-safe deterministic repair is available for this "
            "issue type."
        ),
        required_witness=witness_by_code.get(
            issue_code,
            "Additional domain-grounded evidence or experimental-design witness.",
        ),
    )


def repair_discriminating_experiments(
    *,
    experiments: DiscriminatingExperimentSet,
    critique: DiscriminatingExperimentCriticReport,
) -> ExperimentRepairSet:
    critique_by_id = {
        row.experiment_id: row
        for row in critique.experiments
    }

    repaired_rows = []
    records = []

    for experiment in experiments.experiments:
        row_critique = critique_by_id.get(experiment.experiment_id)
        if row_critique is None:
            raise ValueError(
                "missing experiment critique for: "
                + experiment.experiment_id
            )

        current = experiment
        actions: list[ExperimentRepairAction] = []

        for issue in row_critique.issues:
            if issue.code == "INTERVENTION_UNDERSPECIFIED":
                current, action = _repair_tradeoff_intervention(current)
                actions.append(action)
            else:
                actions.append(_deferred_action(issue.code))

        if not actions:
            actions.append(
                ExperimentRepairAction(
                    issue_code="",
                    disposition="no_repair_needed",
                    rationale=(
                        "The shadow experiment critic reported no issue for "
                        "this experiment."
                    ),
                )
            )

        changed = current.experiment_id != experiment.experiment_id
        repaired_rows.append(current)
        records.append(
            ExperimentRepairRecord(
                source_experiment_id=experiment.experiment_id,
                output_experiment_id=current.experiment_id,
                changed=changed,
                actions=actions,
            )
        )

    repaired_set = experiments.model_copy(
        update={
            "experiments": repaired_rows,
            "experiment_count": len(repaired_rows),
        }
    )

    return ExperimentRepairSet(
        input_experiment_count=experiments.experiment_count,
        output_experiment_count=repaired_set.experiment_count,
        changed_experiment_count=sum(
            record.changed
            for record in records
        ),
        deferred_experiment_count=sum(
            any(
                action.disposition == "deferred_requires_witness"
                for action in record.actions
            )
            for record in records
        ),
        records=records,
        repaired_experiments=repaired_set,
    )
