from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyStatus,
)
from pipeline_core.discovery.semantic_distinctiveness_contracts import (
    SemanticDistinctivenessTier,
)


_POLICY_VERSION = (
    "scientific-novelty-action-shadow-v1"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


ScientificNoveltyAction = Literal[
    "KEEP_ELIGIBLE",
    "REFINE_OR_REAXIS",
    "REAXIS_REQUIRED",
    "UNRESOLVED",
    "EVIDENCE_REQUIRED",
]


SelectionClass = Literal[
    "ELIGIBLE",
    "CONDITIONAL",
    "INELIGIBLE",
]


class ScientificNoveltyActionDecision(
    StrictModel
):
    schema_version: Literal[
        "scientific-novelty-action-decision-v1"
    ] = (
        "scientific-novelty-action-decision-v1"
    )

    policy_version: Literal[
        "scientific-novelty-action-shadow-v1"
    ] = _POLICY_VERSION

    external_status: ExternalNoveltyStatus

    semantic_pass_tiers: tuple[
        SemanticDistinctivenessTier,
        SemanticDistinctivenessTier,
    ]

    semantic_stable: bool

    stable_semantic_tier: (
        SemanticDistinctivenessTier
        | None
    ) = None

    action: ScientificNoveltyAction

    selection_class: SelectionClass

    reason_codes: list[str] = Field(
        default_factory=list,
        max_length=8,
    )

    # N1 is deliberately observational only.
    shadow_only: Literal[True] = True
    action_policy_applied: Literal[False] = False
    scientific_selection_changed: Literal[
        False
    ] = False
    production_selection_changed: Literal[
        False
    ] = False


class ScientificNoveltyActionPolicy:
    """Deterministic shadow policy over frozen novelty signals.

    This policy does not:
    - change external novelty status;
    - mutate semantic reviews;
    - regenerate hypotheses;
    - alter Alpha6 behavior;
    - change production selection.

    It only answers what action WOULD be preferred if the
    external novelty status and two semantic review passes were
    granted production authority.
    """

    policy_version = _POLICY_VERSION

    destructive_external_statuses = frozenset({
        "WELL_ESTABLISHED",
        "CONFLICTING_PRIOR_ART",
    })

    evidence_limited_external_statuses = (
        frozenset({
            "INSUFFICIENT_SEARCH_EVIDENCE",
        })
    )

    def evaluate(
        self,
        *,
        external_status: ExternalNoveltyStatus,
        semantic_pass_1: SemanticDistinctivenessTier,
        semantic_pass_2: SemanticDistinctivenessTier,
    ) -> ScientificNoveltyActionDecision:
        passes = (
            semantic_pass_1,
            semantic_pass_2,
        )

        stable = (
            semantic_pass_1
            == semantic_pass_2
        )

        stable_tier = (
            semantic_pass_1
            if stable
            else None
        )

        # ----------------------------------------------------------
        # External destructive evidence remains authoritative.
        # ----------------------------------------------------------
        if (
            external_status
            in self.destructive_external_statuses
        ):
            return ScientificNoveltyActionDecision(
                external_status=external_status,
                semantic_pass_tiers=passes,
                semantic_stable=stable,
                stable_semantic_tier=stable_tier,
                action="REAXIS_REQUIRED",
                selection_class="INELIGIBLE",
                reason_codes=[
                    "DESTRUCTIVE_EXTERNAL_STATUS",
                ],
            )

        # ----------------------------------------------------------
        # Two-pass semantic instability cannot support selection.
        # ----------------------------------------------------------
        if not stable:
            return ScientificNoveltyActionDecision(
                external_status=external_status,
                semantic_pass_tiers=passes,
                semantic_stable=False,
                stable_semantic_tier=None,
                action="UNRESOLVED",
                selection_class="INELIGIBLE",
                reason_codes=[
                    "SEMANTIC_TIER_UNSTABLE",
                ],
            )

        # ----------------------------------------------------------
        # Semantic LOW is the central N1 intervention.
        #
        # LOW does not mean false. It means insufficient scientific
        # distinctiveness for a final discovery survivor.
        # ----------------------------------------------------------
        if stable_tier == "LOW":
            return ScientificNoveltyActionDecision(
                external_status=external_status,
                semantic_pass_tiers=passes,
                semantic_stable=True,
                stable_semantic_tier="LOW",
                action="REAXIS_REQUIRED",
                selection_class="INELIGIBLE",
                reason_codes=[
                    "STABLE_SEMANTIC_LOW",
                    "FINAL_DISCOVERY_SURVIVOR_DISALLOWED",
                ],
            )

        # ----------------------------------------------------------
        # Indeterminate semantic evidence is not positive evidence.
        # ----------------------------------------------------------
        if stable_tier == "INDETERMINATE":
            return ScientificNoveltyActionDecision(
                external_status=external_status,
                semantic_pass_tiers=passes,
                semantic_stable=True,
                stable_semantic_tier=(
                    "INDETERMINATE"
                ),
                action="EVIDENCE_REQUIRED",
                selection_class="INELIGIBLE",
                reason_codes=[
                    "STABLE_SEMANTIC_INDETERMINATE",
                ],
            )

        # ----------------------------------------------------------
        # Search-bounded uncertainty remains unresolved even when
        # semantic structure itself looks moderate/high.
        # ----------------------------------------------------------
        if (
            external_status
            in self.evidence_limited_external_statuses
        ):
            return ScientificNoveltyActionDecision(
                external_status=external_status,
                semantic_pass_tiers=passes,
                semantic_stable=True,
                stable_semantic_tier=stable_tier,
                action="EVIDENCE_REQUIRED",
                selection_class="INELIGIBLE",
                reason_codes=[
                    "EXTERNAL_SEARCH_EVIDENCE_INSUFFICIENT",
                ],
            )

        # ----------------------------------------------------------
        # LITERATURE_SUPPORTED_EXTENSION + MODERATE:
        # scientifically usable, but refinement/re-axis is preferred
        # before treating it as a preferred discovery survivor.
        # ----------------------------------------------------------
        if (
            external_status
            == "LITERATURE_SUPPORTED_EXTENSION"
            and stable_tier == "MODERATE"
        ):
            return ScientificNoveltyActionDecision(
                external_status=external_status,
                semantic_pass_tiers=passes,
                semantic_stable=True,
                stable_semantic_tier="MODERATE",
                action="REFINE_OR_REAXIS",
                selection_class="CONDITIONAL",
                reason_codes=[
                    "LITERATURE_SUPPORTED_EXTENSION",
                    "STABLE_SEMANTIC_MODERATE",
                ],
            )

        # ----------------------------------------------------------
        # Remaining stable MODERATE cases are eligible but are not
        # semantically as preferred as HIGH candidates.
        # ----------------------------------------------------------
        if stable_tier == "MODERATE":
            return ScientificNoveltyActionDecision(
                external_status=external_status,
                semantic_pass_tiers=passes,
                semantic_stable=True,
                stable_semantic_tier="MODERATE",
                action="KEEP_ELIGIBLE",
                selection_class="ELIGIBLE",
                reason_codes=[
                    "STABLE_SEMANTIC_MODERATE",
                    "NON_DESTRUCTIVE_EXTERNAL_STATUS",
                ],
            )

        # ----------------------------------------------------------
        # Stable HIGH + non-destructive, adequately searched
        # external status is eligible.
        # ----------------------------------------------------------
        if stable_tier == "HIGH":
            return ScientificNoveltyActionDecision(
                external_status=external_status,
                semantic_pass_tiers=passes,
                semantic_stable=True,
                stable_semantic_tier="HIGH",
                action="KEEP_ELIGIBLE",
                selection_class="ELIGIBLE",
                reason_codes=[
                    "STABLE_SEMANTIC_HIGH",
                    "NON_DESTRUCTIVE_EXTERNAL_STATUS",
                ],
            )

        raise RuntimeError(
            "unhandled scientific novelty action state: "
            f"external={external_status!r}, "
            f"semantic={passes!r}"
        )
