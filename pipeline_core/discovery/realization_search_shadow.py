from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.discovery.semantic_distinctiveness import (
    SEMANTIC_DISTINCTIVENESS_AGGREGATION_VERSION,
)
from pipeline_core.discovery.semantic_distinctiveness_contracts import (
    SemanticDistinctivenessTier,
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


class RealizationSearchPolicy(StrictModel):
    """Experimental realization-search policy.

    Search width and retained portfolio width are separate concepts.

    `search_width`:
        number of independent realization trajectories generated from
        the same frozen discovery axis.

    `retained_hypotheses_per_axis`:
        number of hypotheses intended to survive downstream after
        quality/stability/diversity selection.

    Default behavior:
        search_width = 3
        retained_hypotheses_per_axis = 1

    Width 1 remains available explicitly as the single-realization
    control.

    Multi-retention (>1) is represented by this contract but is not
    implemented by the current single-winner shadow selector. It must
    pass through a diversity-aware retention layer instead of silently
    degrading to top-N-by-tier selection.
    """

    search_width: int = Field(
        default=3,
        ge=1,
        le=4,
    )

    retained_hypotheses_per_axis: int = Field(
        default=1,
        ge=1,
        le=3,
    )

    semantic_passes_per_candidate: Literal[
        2
    ] = 2

    require_stable_semantic_tier: Literal[
        True
    ] = True

    require_single_served_model: bool = True

    selection_objective: Literal[
        "STABLE_DETERMINATE_SEMANTIC_TIER"
    ] = (
        "STABLE_DETERMINATE_SEMANTIC_TIER"
    )

    shadow_only: Literal[
        True
    ] = True


    @model_validator(mode="after")
    def _validate_retention_width(
        self,
    ) -> "RealizationSearchPolicy":
        if (
            self.retained_hypotheses_per_axis
            > self.search_width
        ):
            raise ValueError(
                "retained_hypotheses_per_axis may not exceed "
                "search_width"
            )

        return self


class RealizationSemanticObservation(StrictModel):
    """Frozen two-pass semantic observation for one realization."""

    slot_index: int = Field(
        ge=0
    )

    hypothesis_id: str = Field(
        min_length=1
    )

    pass_tiers: tuple[
        SemanticDistinctivenessTier,
        SemanticDistinctivenessTier,
    ]

    pass_aggregation_versions: tuple[
        str,
        str,
    ]

    pass_served_models: tuple[
        str,
        str,
    ]

    pass_diagnostic_only: tuple[
        bool,
        bool,
    ] = (
        True,
        True,
    )

    pass_action_policy_applied: tuple[
        bool,
        bool,
    ] = (
        False,
        False,
    )

    pass_scientific_selection_changed: tuple[
        bool,
        bool,
    ] = (
        False,
        False,
    )


class RealizationShadowCandidateDecision(
    StrictModel
):
    slot_index: int
    hypothesis_id: str

    stable: bool

    stable_tier: (
        SemanticDistinctivenessTier
        | None
    ) = None

    determinate: bool
    eligible: bool

    reason_codes: list[str] = Field(
        default_factory=list,
        max_length=8,
    )


ShadowSelectionStatus = Literal[
    "WINNER_SELECTED",
    "NO_STABLE_DETERMINATE_CANDIDATE",
]


class RealizationShadowSelectionReport(
    StrictModel
):
    schema_version: Literal[
        "realization-search-shadow-selection-v1"
    ] = (
        "realization-search-shadow-selection-v1"
    )

    search_width: int = Field(
        ge=1,
        le=4,
    )

    selection_objective: Literal[
        "STABLE_DETERMINATE_SEMANTIC_TIER"
    ] = (
        "STABLE_DETERMINATE_SEMANTIC_TIER"
    )

    status: ShadowSelectionStatus

    winner_slot_index: int | None = None
    winner_hypothesis_id: str | None = None
    winner_tier: DeterminateTier | None = None

    candidates: list[
        RealizationShadowCandidateDecision
    ]

    served_model: str | None = None

    # Critical architecture boundary:
    # the selector is observational/shadow only.
    shadow_only: Literal[True] = True
    production_selection_changed: Literal[
        False
    ] = False

    semantic_diagnostic_contract_preserved: Literal[
        True
    ] = True


_TIER_SCORE: dict[
    DeterminateTier,
    int,
] = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
}


def _candidate_decision(
    observation: RealizationSemanticObservation,
) -> RealizationShadowCandidateDecision:
    first, second = observation.pass_tiers

    stable = (
        first == second
    )

    stable_tier = (
        first
        if stable
        else None
    )

    determinate = (
        stable
        and stable_tier
        in _TIER_SCORE
    )

    reasons: list[str] = []

    if not stable:
        reasons.append(
            "SEMANTIC_TIER_UNSTABLE"
        )

    elif stable_tier == "INDETERMINATE":
        reasons.append(
            "SEMANTIC_TIER_INDETERMINATE"
        )

    elif determinate:
        reasons.append(
            "STABLE_DETERMINATE_TIER"
        )

    return RealizationShadowCandidateDecision(
        slot_index=observation.slot_index,
        hypothesis_id=observation.hypothesis_id,
        stable=stable,
        stable_tier=stable_tier,
        determinate=determinate,
        eligible=determinate,
        reason_codes=reasons,
    )


def select_realization_shadow_winner(
    observations: list[
        RealizationSemanticObservation
    ],
    *,
    policy: RealizationSearchPolicy,
) -> RealizationShadowSelectionReport:
    """Select a shadow best-of-k realization deterministically.

    Eligibility exactly matches the retrospective experiment:
      - two semantic passes,
      - identical pass tier,
      - determinate tier in LOW/MODERATE/HIGH,
      - semantic reviews remain diagnostic-only,
      - no action/scientific selection mutation.

    Tie-break:
      highest tier, then lowest slot index.

    The returned winner is shadow metadata only. This function does not
    mutate a hypothesis portfolio or scientific selection.
    """

    if (
        policy.retained_hypotheses_per_axis
        != 1
    ):
        raise ValueError(
            "current shadow winner selector supports exactly one "
            "retained hypothesis; retained_hypotheses_per_axis > 1 "
            "requires the diversity-aware retention selector"
        )

    if len(
        observations
    ) != policy.search_width:
        raise ValueError(
            "observation count must equal search_width: "
            f"observations={len(observations)}, "
            f"search_width={policy.search_width}"
        )

    slots = [
        row.slot_index
        for row
        in observations
    ]

    expected_slots = list(
        range(
            policy.search_width
        )
    )

    if sorted(slots) != expected_slots:
        raise ValueError(
            "realization slots must be exactly "
            f"{expected_slots}; got {sorted(slots)}"
        )

    hypothesis_ids = [
        row.hypothesis_id
        for row
        in observations
    ]

    if (
        len(
            set(
                hypothesis_ids
            )
        )
        != len(
            hypothesis_ids
        )
    ):
        raise ValueError(
            "realization hypotheses must be unique"
        )

    expected_aggregation = (
        SEMANTIC_DISTINCTIVENESS_AGGREGATION_VERSION
    )

    served_models: set[str] = set()

    for observation in observations:
        if any(
            version
            != expected_aggregation
            for version
            in observation.pass_aggregation_versions
        ):
            raise ValueError(
                "semantic aggregation version mismatch"
            )

        if observation.pass_diagnostic_only != (
            True,
            True,
        ):
            raise ValueError(
                "semantic reviews must remain diagnostic-only"
            )

        if observation.pass_action_policy_applied != (
            False,
            False,
        ):
            raise ValueError(
                "semantic action policy must remain unapplied"
            )

        if (
            observation
            .pass_scientific_selection_changed
            != (
                False,
                False,
            )
        ):
            raise ValueError(
                "semantic review may not change scientific selection"
            )

        first_model, second_model = (
            observation.pass_served_models
        )

        if (
            not first_model
            or not second_model
        ):
            raise ValueError(
                "served model may not be empty"
            )

        if first_model != second_model:
            raise ValueError(
                "served model changed across semantic passes"
            )

        served_models.add(
            first_model
        )

    if (
        policy.require_single_served_model
        and len(
            served_models
        )
        != 1
    ):
        raise ValueError(
            "realization cohort must use one served model"
        )

    decisions = [
        _candidate_decision(
            row
        )
        for row
        in sorted(
            observations,
            key=lambda item:
                item.slot_index,
        )
    ]

    eligible = [
        row
        for row
        in decisions
        if row.eligible
    ]

    served_model = (
        next(
            iter(
                served_models
            )
        )
        if len(
            served_models
        )
        == 1
        else None
    )

    if not eligible:
        return RealizationShadowSelectionReport(
            search_width=policy.search_width,
            status=(
                "NO_STABLE_DETERMINATE_CANDIDATE"
            ),
            candidates=decisions,
            served_model=served_model,
        )

    winner = max(
        eligible,
        key=lambda row: (
            _TIER_SCORE[
                row.stable_tier
            ],
            -row.slot_index,
        ),
    )

    return RealizationShadowSelectionReport(
        search_width=policy.search_width,
        status="WINNER_SELECTED",
        winner_slot_index=winner.slot_index,
        winner_hypothesis_id=(
            winner.hypothesis_id
        ),
        winner_tier=winner.stable_tier,
        candidates=decisions,
        served_model=served_model,
    )
