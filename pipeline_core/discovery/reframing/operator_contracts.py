from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.corpus.semantic_ir.capability import CapabilityRequirement
from pipeline_core.corpus.semantic_ir.task_gap import TaskLocalInspectableCapability


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ReframingOperatorId = Literal[
    "LATENT_VARIABLE",
    "REGIME_BOUNDARY",
    "PROXY_CHALLENGE",
]

ExplicitGapTargetSelector = Literal[
    "none",
    "measurement_bearing_chunks",
]


class TaskLocalConditionPolicy(StrictModel):
    capabilities: list[TaskLocalInspectableCapability] = Field(min_length=1)
    minimum_supported_families: int = Field(ge=1)
    desired_state: Literal["complete"] = "complete"
    incomplete_coverage_is_optional_enrichment: Literal[True] = True


class ReframingOperatorContract(StrictModel):
    schema_version: Literal[
        "reframing-operator-contract-v1"
    ] = "reframing-operator-contract-v1"

    operator_id: ReframingOperatorId
    scientific_role: str = Field(min_length=1)
    hard_requirements: list[CapabilityRequirement] = Field(default_factory=list)
    mandatory_explicit_gap_capabilities: list[str] = Field(default_factory=list)
    explicit_gap_target_selector: ExplicitGapTargetSelector = "none"
    task_local_condition_policy: TaskLocalConditionPolicy | None = None

    shadow_only: Literal[True] = True
    scientific_trigger_evaluated: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    production_selection_authority: Literal[False] = False


def get_reframing_operator_contracts() -> list[ReframingOperatorContract]:
    common = [
        CapabilityRequirement(
            capability="relation_structure",
            minimum_state="complete",
            reason="Reframing must preserve the grounded scientific relation structure.",
        ),
        CapabilityRequirement(
            capability="edge_provenance",
            minimum_state="complete",
            reason="Every reframe premise must remain traceable to source evidence pointers.",
        ),
        CapabilityRequirement(
            capability="source_chunk_recovery",
            minimum_state="complete",
            reason="Every grounded semantic object used for reframing must remain recoverable to source text.",
        ),
    ]

    return [
        ReframingOperatorContract(
            operator_id="LATENT_VARIABLE",
            scientific_role=(
                "Test whether a shared unmodeled construct could explain multiple grounded evidence families."
            ),
            hard_requirements=[
                *common,
                CapabilityRequirement(
                    capability="claim_support_linkage",
                    minimum_state="partial",
                    reason="Latent-variable reframing needs at least some evidence-linked scientific claims.",
                ),
                CapabilityRequirement(
                    capability="claim_application_target",
                    minimum_state="partial",
                    reason="Claims used by a latent-variable reframe need identified scientific targets.",
                ),
            ],
        ),
        ReframingOperatorContract(
            operator_id="REGIME_BOUNDARY",
            scientific_role=(
                "Test whether the response model changes across experimental or computational conditions rather than merely changing magnitude."
            ),
            hard_requirements=[
                *common,
                CapabilityRequirement(
                    capability="experiment_representation",
                    minimum_state="complete",
                    reason="Regime analysis must retain experimental contexts as explicit objects.",
                ),
                CapabilityRequirement(
                    capability="measurement_representation",
                    minimum_state="complete",
                    reason="Regime analysis must retain measured outcomes separately from methods.",
                ),
            ],
            task_local_condition_policy=TaskLocalConditionPolicy(
                capabilities=[
                    "measurement_condition_coverage",
                    "experiment_condition_coverage",
                    "calculation_condition_coverage",
                ],
                minimum_supported_families=1,
            ),
        ),
        ReframingOperatorContract(
            operator_id="PROXY_CHALLENGE",
            scientific_role=(
                "Test whether a reported observable is being treated as an interchangeable proxy for a distinct scientific construct."
            ),
            hard_requirements=[
                *common,
                CapabilityRequirement(
                    capability="measurement_identity",
                    minimum_state="complete",
                    reason="Proxy challenge requires explicit observable/metric identity.",
                ),
                CapabilityRequirement(
                    capability="measurement_provider_linkage",
                    minimum_state="complete",
                    reason="Proxy challenge must retain how each measurement was produced.",
                ),
            ],
            mandatory_explicit_gap_capabilities=["proxy_semantics"],
            explicit_gap_target_selector="measurement_bearing_chunks",
        ),
    ]
