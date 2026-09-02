from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisType,
)
from pipeline_core.discovery.nonobviousness_mechanism_semantics_contracts import (
    MechanismSearchOperator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


N11PredictionDirection = Literal[
    "shift",
    "qualitative_change",
    "unspecified",
]


class N11OperatorPredictionDraft(
    StrictModel
):
    local_id: str = Field(
        min_length=1
    )

    observable: str = Field(
        min_length=1
    )

    expected_direction: N11PredictionDirection

    rationale: str = Field(
        min_length=1
    )


class N11OperatorFalsificationDraft(
    StrictModel
):
    local_id: str = Field(
        min_length=1
    )

    # Exact structural link to the predicted observable being falsified.
    # Do not restate the observable in free text.
    prediction_local_id: str = Field(
        min_length=1
    )

    falsifying_outcome: str = Field(
        min_length=1
    )


class N11OperatorCandidateDraft(
    StrictModel
):
    local_id: str = Field(
        min_length=1
    )

    title: str = Field(
        min_length=1
    )

    hypothesis_statement: str = Field(
        min_length=1
    )

    operator: MechanismSearchOperator

    hypothesis_type: HypothesisType

    # Existing HypothesisContext positive evidence only.
    baseline_premise_statement_ids: list[str] = Field(
        min_length=1
    )

    # Separate B0 grounded-component lane.
    # These IDs MUST NOT be treated as HypothesisContext premises.
    supplemental_mechanism_node_ids: list[str] = Field(
        min_length=1
    )

    # Unresolved evidence only.
    gap_statement_ids: list[str] = Field(
        min_length=1
    )

    # IDs assigned by the C1 prompt assembler to the semantic
    # decomposition produced by B1.
    shared_component_ids: list[str] = Field(
        min_length=1
    )

    supplemental_only_component_ids: list[str] = Field(
        min_length=1
    )

    # The new scientific relation being proposed.
    relative_contribution_claim: str = Field(
        min_length=1
    )

    inferential_bridge: str = Field(
        min_length=1
    )

    predicted_observations: list[
        N11OperatorPredictionDraft
    ] = Field(
        min_length=1
    )

    # One prediction must specifically discriminate the new
    # operator-level relation from a simple magnitude-only effect.
    discriminating_observation_local_id: str = Field(
        min_length=1
    )

    falsification_criteria: list[
        N11OperatorFalsificationDraft
    ] = Field(
        min_length=1
    )

    # Required even when empty so OpenAI strict structured output
    # receives every property in `required`.
    assumptions: list[str]

    generated_relation_status: Literal[
        "INFERENCE_NOT_REPORTED"
    ]

    task_to_supplemental_relation_grounded: Literal[
        False
    ]

    @model_validator(
        mode="after"
    )
    def _local_id_consistency(
        self,
    ) -> "N11OperatorCandidateDraft":
        prediction_ids = [
            row.local_id
            for row
            in self.predicted_observations
        ]

        falsifier_ids = [
            row.local_id
            for row
            in self.falsification_criteria
        ]

        if (
            len(prediction_ids)
            != len(set(prediction_ids))
        ):
            raise ValueError(
                "duplicate prediction local_id"
            )

        if (
            len(falsifier_ids)
            != len(set(falsifier_ids))
        ):
            raise ValueError(
                "duplicate falsification local_id"
            )

        if (
            self.discriminating_observation_local_id
            not in prediction_ids
        ):
            raise ValueError(
                "discriminating_observation_local_id "
                "must reference a predicted observation"
            )

        for falsifier in self.falsification_criteria:
            if (
                falsifier.prediction_local_id
                not in prediction_ids
            ):
                raise ValueError(
                    "falsification prediction_local_id "
                    "must reference a predicted observation"
                )

        return self


class N11OperatorGenerationDraft(
    StrictModel
):
    # Both properties are intentionally required.
    # Exactly one semantic branch is active.
    candidate: N11OperatorCandidateDraft | None

    abstention_reason: str | None

    @model_validator(
        mode="after"
    )
    def _candidate_or_abstention(
        self,
    ) -> "N11OperatorGenerationDraft":
        if (
            self.candidate is None
            and not str(
                self.abstention_reason
                or ""
            ).strip()
        ):
            raise ValueError(
                "abstention_reason is required "
                "when candidate is null"
            )

        if (
            self.candidate is not None
            and self.abstention_reason
            is not None
        ):
            raise ValueError(
                "abstention_reason must be null "
                "when candidate exists"
            )

        return self
