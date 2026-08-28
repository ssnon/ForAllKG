from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.realization_search_cohort import (
    AxisRealizationCohort,
)
from pipeline_core.discovery.realization_search_shadow import (
    RealizationSearchPolicy,
    RealizationSemanticObservation,
    RealizationShadowCandidateDecision,
    select_realization_shadow_winner,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


DeterminateTier = Literal[
    "LOW",
    "MODERATE",
    "HIGH",
]


class ProductionRealizationSelectionReport(
    StrictModel
):
    """Authoritative best-of-k realization selection.

    Ranking semantics are deliberately inherited from the frozen
    realization-search shadow selector:

      * exactly ``search_width`` independent realizations,
      * two semantic passes per realization,
      * unstable semantic tiers are ineligible,
      * INDETERMINATE is ineligible,
      * stable determinate tiers rank HIGH > MODERATE > LOW,
      * ties resolve to the lowest realization slot,
      * no stable determinate realization means fail closed.

    This contract does not itself rewrite a portfolio.  It is the
    production-authoritative decision consumed by the orchestration
    layer that performs winner retention.
    """

    schema_version: Literal[
        "realization-search-production-selection-v1"
    ] = (
        "realization-search-production-selection-v1"
    )

    search_width: int = Field(
        ge=1,
        le=4,
    )

    retained_hypotheses_per_axis: Literal[
        1
    ] = 1

    selection_objective: Literal[
        "STABLE_DETERMINATE_SEMANTIC_TIER"
    ] = (
        "STABLE_DETERMINATE_SEMANTIC_TIER"
    )

    status: Literal[
        "WINNER_SELECTED",
        "NO_STABLE_DETERMINATE_CANDIDATE",
    ]

    winner_slot_index: int | None = None
    winner_hypothesis_id: str | None = None
    winner_tier: DeterminateTier | None = None

    candidates: list[
        RealizationShadowCandidateDecision
    ]

    served_model: str | None = None

    source_shadow_schema_version: Literal[
        "realization-search-shadow-selection-v1"
    ] = (
        "realization-search-shadow-selection-v1"
    )

    production_authority: Literal[
        True
    ] = True

    production_selection_changed: Literal[
        True
    ] = True

    semantic_diagnostic_contract_preserved: Literal[
        True
    ] = True

    semantic_observation_count: int = Field(
        default=0,
        ge=0,
    )

    missing_slot_count: int = Field(
        default=0,
        ge=0,
    )

    missing_slot_indices: list[int] = Field(
        default_factory=list
    )

    partial_cohort_selection_allowed: Literal[
        True
    ] = True


def select_realization_production_winner(
    observations: list[
        RealizationSemanticObservation
    ],
    *,
    policy: RealizationSearchPolicy,
) -> ProductionRealizationSelectionReport:
    """Elevate the frozen shadow ranking rule to production authority.

    The existing shadow selector remains the single source of truth for
    candidate eligibility, stability checks, model-consistency checks,
    tier ordering, and tie breaking.  This wrapper changes authority,
    not ranking semantics.
    """

    if (
        policy.retained_hypotheses_per_axis
        != 1
    ):
        raise ValueError(
            "production realization selector currently supports "
            "exactly one retained hypothesis per axis"
        )

    shadow = select_realization_shadow_winner(
        observations,
        policy=policy,
    )

    return ProductionRealizationSelectionReport(
        search_width=shadow.search_width,
        status=shadow.status,
        winner_slot_index=(
            shadow.winner_slot_index
        ),
        winner_hypothesis_id=(
            shadow.winner_hypothesis_id
        ),
        winner_tier=shadow.winner_tier,
        candidates=list(
            shadow.candidates
        ),
        served_model=shadow.served_model,
        semantic_observation_count=len(
            observations
        ),
        missing_slot_count=0,
        missing_slot_indices=[],
    )



def select_axis_realization_production_winner(
    cohort: AxisRealizationCohort,
    *,
    policy: RealizationSearchPolicy,
) -> ProductionRealizationSelectionReport:
    """Select the production winner for one frozen discovery axis.

    Production search width means *attempted* realization width, not
    "all attempts must successfully reach semantic evaluation".

    Missing/failed realization slots are explicit in ``cohort`` and
    are never eligible.  Among realizations that do reach two-pass
    semantic evaluation, the exact frozen shadow semantics remain the
    authority for quality:

      stable HIGH > stable MODERATE > stable LOW
      unstable -> ineligible
      INDETERMINATE -> ineligible
      tier ties -> earliest original realization slot

    If zero successful semantic observations exist, or if all
    successful observations are unstable/INDETERMINATE, selection
    fails closed.

    To avoid duplicating ranking logic, successful observations are
    compacted in original slot order and passed through the frozen
    shadow selector.  Compact slot IDs are then mapped back to their
    original realization slots.
    """

    if (
        cohort.search_width
        != policy.search_width
    ):
        raise ValueError(
            "axis realization cohort search_width does not match "
            "production policy: "
            f"cohort={cohort.search_width}, "
            f"policy={policy.search_width}"
        )

    if (
        policy.retained_hypotheses_per_axis
        != 1
    ):
        raise ValueError(
            "production realization selection currently supports "
            "exactly one retained hypothesis per axis"
        )

    semantic_rows = [
        row
        for row in sorted(
            cohort.slots,
            key=lambda item:
                item.slot_index,
        )
        if (
            row.status
            == "SEMANTIC_EVALUATED"
        )
    ]

    missing_rows = [
        row
        for row in sorted(
            cohort.slots,
            key=lambda item:
                item.slot_index,
        )
        if (
            row.status
            != "SEMANTIC_EVALUATED"
        )
    ]

    missing_indices = [
        row.slot_index
        for row in missing_rows
    ]

    if not semantic_rows:
        return (
            ProductionRealizationSelectionReport(
                search_width=(
                    cohort.search_width
                ),
                status=(
                    "NO_STABLE_DETERMINATE_CANDIDATE"
                ),
                candidates=[],
                served_model=None,
                semantic_observation_count=0,
                missing_slot_count=len(
                    missing_rows
                ),
                missing_slot_indices=(
                    missing_indices
                ),
            )
        )

    original_slot_by_compact: dict[
        int,
        int,
    ] = {}

    compact_observations = []

    for compact_index, row in enumerate(
        semantic_rows
    ):
        observation = (
            row.semantic_observation
        )

        if observation is None:
            raise ValueError(
                "SEMANTIC_EVALUATED slot lacks "
                "semantic_observation"
            )

        original_slot_by_compact[
            compact_index
        ] = row.slot_index

        compact_observations.append(
            observation.model_copy(
                update={
                    "slot_index":
                        compact_index,
                }
            )
        )

    compact_policy = (
        RealizationSearchPolicy(
            search_width=len(
                compact_observations
            ),
            retained_hypotheses_per_axis=(
                policy
                .retained_hypotheses_per_axis
            ),
            semantic_passes_per_candidate=(
                policy
                .semantic_passes_per_candidate
            ),
            require_stable_semantic_tier=(
                policy
                .require_stable_semantic_tier
            ),
            require_single_served_model=(
                policy
                .require_single_served_model
            ),
            selection_objective=(
                policy.selection_objective
            ),
        )
    )

    shadow = (
        select_realization_shadow_winner(
            compact_observations,
            policy=compact_policy,
        )
    )

    restored_candidates = [
        row.model_copy(
            update={
                "slot_index":
                    original_slot_by_compact[
                        row.slot_index
                    ],
            }
        )
        for row in shadow.candidates
    ]

    winner_original_slot = None

    if (
        shadow.winner_slot_index
        is not None
    ):
        winner_original_slot = (
            original_slot_by_compact[
                shadow.winner_slot_index
            ]
        )

    return (
        ProductionRealizationSelectionReport(
            search_width=(
                cohort.search_width
            ),
            status=shadow.status,
            winner_slot_index=(
                winner_original_slot
            ),
            winner_hypothesis_id=(
                shadow.winner_hypothesis_id
            ),
            winner_tier=(
                shadow.winner_tier
            ),
            candidates=(
                restored_candidates
            ),
            served_model=(
                shadow.served_model
            ),
            semantic_observation_count=(
                len(
                    semantic_rows
                )
            ),
            missing_slot_count=(
                len(
                    missing_rows
                )
            ),
            missing_slot_indices=(
                missing_indices
            ),
        )
    )
