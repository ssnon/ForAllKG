from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxisPlan,
    DiscoveryAxisSynthesisReport,
    DiscoveryHypothesisLineage,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisPortfolio,
)
from pipeline_core.discovery.realization_search_cohort import (
    RealizationSearchCohortReport,
)
from pipeline_core.discovery.realization_search_production import (
    ProductionRealizationSelectionReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


def _canonical_json(
    value: Any,
) -> str:
    if hasattr(
        value,
        "model_dump",
    ):
        value = value.model_dump(
            mode="json"
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(
    value: Any,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            value
        ).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(x)
        for x in parts
    ).encode("utf-8")

    return (
        f"{prefix}:"
        + hashlib.sha256(
            raw
        ).hexdigest()[:length]
    )


class AxisWinnerMaterializationRecord(
    StrictModel
):
    axis_id: str

    status: Literal[
        "WINNER_MATERIALIZED",
        "ELIGIBLE_NOT_GLOBALLY_SELECTED",
        "NO_ELIGIBLE_REALIZATION",
    ]

    winner_slot_index: int | None = None

    winner_hypothesis_id: (
        str
        | None
    ) = None

    winner_tier: Literal[
        "LOW",
        "MODERATE",
        "HIGH",
    ] | None = None


class RealizationWinnerMaterializationReport(
    StrictModel
):
    schema_version: Literal[
        "realization-winner-materialization-v1"
    ] = (
        "realization-winner-materialization-v1"
    )

    search_width: int = Field(
        ge=1,
        le=4,
    )

    attempted_axis_count: int = Field(
        ge=0
    )

    materialized_winner_count: int = Field(
        ge=0
    )

    selections: list[
        AxisWinnerMaterializationRecord
    ]

    winner_portfolio_id: str

    winner_lineage_report_id: str

    production_selection_applied: Literal[
        True
    ] = True

    production_selection_changed: Literal[
        True
    ] = True


@dataclass(
    frozen=True
)
class RealizationWinnerMaterialization:
    portfolio: HypothesisPortfolio
    lineage_report: DiscoveryAxisSynthesisReport
    report: RealizationWinnerMaterializationReport


def _portfolio_header(
    portfolio: HypothesisPortfolio,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
]:
    return (
        portfolio.domain_profile_id,
        portfolio.source_context_id,
        portfolio.source_context_sha256,
        portfolio.source_report_id,
        portfolio.source_report_sha256,
    )


def materialize_realization_winners(
    *,
    plan: DiscoveryAxisPlan,
    slot_portfolios: dict[
        int,
        HypothesisPortfolio,
    ],
    slot_lineage_reports: dict[
        int,
        DiscoveryAxisSynthesisReport,
    ],
    cohort_report: RealizationSearchCohortReport,
    selections_by_axis: dict[
        str,
        ProductionRealizationSelectionReport,
    ],
    global_selection_enforced: bool = False,
    global_winner_axis_id: str | None = None,
) -> RealizationWinnerMaterialization:
    """Materialize authoritative per-axis best-of-k winners.

    Winner ordering follows the frozen discovery-axis plan, never the
    order in which realization slots happened to finish.

    Selected hypothesis cards and their original discovery lineages are
    preserved verbatim.  Only the enclosing portfolio/report provenance
    is regenerated because the winner set is a new production
    selection.
    """

    search_width = (
        cohort_report.search_width
    )

    expected_slots = set(
        range(
            search_width
        )
    )

    if (
        set(
            slot_portfolios
        )
        != expected_slots
    ):
        raise ValueError(
            "slot_portfolios must contain exactly "
            f"{sorted(expected_slots)}"
        )

    if (
        set(
            slot_lineage_reports
        )
        != expected_slots
    ):
        raise ValueError(
            "slot_lineage_reports must contain exactly "
            f"{sorted(expected_slots)}"
        )

    plan_axis_ids = [
        axis.axis_id
        for axis in plan.axes
    ]

    cohort_axis_ids = [
        axis.axis_id
        for axis in cohort_report.axes
    ]

    if (
        cohort_axis_ids
        != plan_axis_ids
    ):
        raise ValueError(
            "cohort axis order must exactly match "
            "the frozen discovery-axis plan"
        )

    if (
        set(
            selections_by_axis
        )
        != set(
            plan_axis_ids
        )
    ):
        raise ValueError(
            "selections_by_axis must contain "
            "exactly the frozen plan axes"
        )

    # All realization portfolios originate from the same dual/context
    # provenance.  The production winner portfolio must not silently
    # combine unrelated contexts.
    baseline_header = (
        _portfolio_header(
            slot_portfolios[0]
        )
    )

    for slot_index in sorted(
        slot_portfolios
    ):
        current_header = (
            _portfolio_header(
                slot_portfolios[
                    slot_index
                ]
            )
        )

        if (
            current_header
            != baseline_header
        ):
            raise ValueError(
                "realization portfolio provenance mismatch "
                f"at slot {slot_index}"
            )

    card_by_slot_and_id: dict[
        tuple[int, str],
        HypothesisCard,
    ] = {}

    lineage_by_slot_and_id: dict[
        tuple[int, str],
        DiscoveryHypothesisLineage,
    ] = {}

    for slot_index in sorted(
        expected_slots
    ):
        portfolio = (
            slot_portfolios[
                slot_index
            ]
        )

        lineage_report = (
            slot_lineage_reports[
                slot_index
            ]
        )

        if (
            lineage_report.axis_plan_id
            != plan.plan_id
        ):
            raise ValueError(
                "realization lineage axis_plan_id mismatch "
                f"at slot {slot_index}"
            )

        if (
            lineage_report.axis_plan_sha256
            != plan.plan_sha256
        ):
            raise ValueError(
                "realization lineage axis_plan_sha256 mismatch "
                f"at slot {slot_index}"
            )

        portfolio_ids = {
            card.hypothesis_id
            for card
            in portfolio.hypotheses
        }

        lineage_ids = {
            row.hypothesis_id
            for row
            in lineage_report.lineages
        }

        if (
            portfolio_ids
            != lineage_ids
        ):
            raise ValueError(
                "realization portfolio/lineage hypothesis IDs "
                f"do not match at slot {slot_index}"
            )

        for card in (
            portfolio.hypotheses
        ):
            key = (
                slot_index,
                card.hypothesis_id,
            )

            if (
                key
                in card_by_slot_and_id
            ):
                raise ValueError(
                    "duplicate realization hypothesis key"
                )

            card_by_slot_and_id[
                key
            ] = card

        for lineage in (
            lineage_report.lineages
        ):
            key = (
                slot_index,
                lineage.hypothesis_id,
            )

            lineage_by_slot_and_id[
                key
            ] = lineage

    cohort_by_axis = {
        axis.axis_id:
            axis
        for axis
        in cohort_report.axes
    }

    selected_cards: list[
        HypothesisCard
    ] = []

    selected_lineages: list[
        DiscoveryHypothesisLineage
    ] = []

    materialization_records: list[
        AxisWinnerMaterializationRecord
    ] = []

    selected_hypothesis_ids: set[
        str
    ] = set()

    for axis in plan.axes:
        axis_id = (
            axis.axis_id
        )

        cohort = (
            cohort_by_axis[
                axis_id
            ]
        )

        selection = (
            selections_by_axis[
                axis_id
            ]
        )

        if (
            selection.search_width
            != search_width
        ):
            raise ValueError(
                "selection search_width mismatch "
                f"for axis {axis_id}"
            )

        if (
            selection.status
            == (
                "NO_STABLE_DETERMINATE_CANDIDATE"
            )
        ):
            if (
                selection.winner_slot_index
                is not None
                or
                selection.winner_hypothesis_id
                is not None
                or
                selection.winner_tier
                is not None
            ):
                raise ValueError(
                    "no-winner selection unexpectedly "
                    "contains winner metadata"
                )

            materialization_records.append(
                AxisWinnerMaterializationRecord(
                    axis_id=axis_id,
                    status=(
                        "NO_ELIGIBLE_REALIZATION"
                    ),
                )
            )

            continue

        if global_selection_enforced:
            if global_winner_axis_id is None:
                raise ValueError(
                    "global production selection has no winner, "
                    "but an axis-local eligible winner exists"
                )

            if axis_id != global_winner_axis_id:
                if (
                    selection.status
                    != "WINNER_SELECTED"
                    or selection.winner_slot_index
                    is None
                    or not selection.winner_hypothesis_id
                    or selection.winner_tier
                    is None
                ):
                    raise ValueError(
                        "axis-local eligible winner lacks complete "
                        "winner metadata before global filtering"
                    )

                materialization_records.append(
                    AxisWinnerMaterializationRecord(
                        axis_id=axis_id,
                        status=(
                            "ELIGIBLE_NOT_GLOBALLY_SELECTED"
                        ),
                        winner_slot_index=(
                            selection.winner_slot_index
                        ),
                        winner_hypothesis_id=(
                            selection.winner_hypothesis_id
                        ),
                        winner_tier=(
                            selection.winner_tier
                        ),
                    )
                )

                continue

        winner_slot = (
            selection.winner_slot_index
        )

        winner_hypothesis_id = (
            selection.winner_hypothesis_id
        )

        winner_tier = (
            selection.winner_tier
        )

        if (
            winner_slot
            is None
            or not winner_hypothesis_id
            or winner_tier
            is None
        ):
            raise ValueError(
                "WINNER_SELECTED is missing winner metadata "
                f"for axis {axis_id}"
            )

        if (
            winner_slot
            not in expected_slots
        ):
            raise ValueError(
                "winner slot is outside production "
                "search width"
            )

        cohort_slot = next(
            (
                row
                for row
                in cohort.slots
                if (
                    row.slot_index
                    == winner_slot
                )
            ),
            None,
        )

        if (
            cohort_slot
            is None
        ):
            raise ValueError(
                "winner slot is absent from axis cohort"
            )

        if (
            cohort_slot.status
            != "SEMANTIC_EVALUATED"
        ):
            raise ValueError(
                "winner slot was not semantically evaluated"
            )

        if (
            cohort_slot.hypothesis_id
            != winner_hypothesis_id
        ):
            raise ValueError(
                "winner hypothesis does not match "
                "axis cohort"
            )

        key = (
            winner_slot,
            winner_hypothesis_id,
        )

        card = (
            card_by_slot_and_id.get(
                key
            )
        )

        lineage = (
            lineage_by_slot_and_id.get(
                key
            )
        )

        if card is None:
            raise ValueError(
                "winner hypothesis card not found "
                f"for axis {axis_id}"
            )

        if lineage is None:
            raise ValueError(
                "winner hypothesis lineage not found "
                f"for axis {axis_id}"
            )

        if (
            lineage.axis_id
            != axis_id
        ):
            raise ValueError(
                "winner lineage axis mismatch"
            )

        if (
            winner_hypothesis_id
            in selected_hypothesis_ids
        ):
            raise ValueError(
                "same hypothesis selected for "
                "multiple axes"
            )

        selected_hypothesis_ids.add(
            winner_hypothesis_id
        )

        selected_cards.append(
            card
        )

        selected_lineages.append(
            lineage
        )

        materialization_records.append(
            AxisWinnerMaterializationRecord(
                axis_id=axis_id,
                status=(
                    "WINNER_MATERIALIZED"
                ),
                winner_slot_index=(
                    winner_slot
                ),
                winner_hypothesis_id=(
                    winner_hypothesis_id
                ),
                winner_tier=(
                    winner_tier
                ),
            )
        )

    (
        domain_profile_id,
        source_context_id,
        source_context_sha256,
        source_report_id,
        source_report_sha256,
    ) = baseline_header

    winner_signature = "|".join(
        (
            f"{row.axis_id}:"
            f"{row.winner_slot_index}:"
            f"{row.winner_hypothesis_id}:"
            f"{row.winner_tier}"
        )
        for row
        in materialization_records
        if (
            row.status
            == "WINNER_MATERIALIZED"
        )
    )

    if global_selection_enforced:
        if (
            global_winner_axis_id
            is not None
            and global_winner_axis_id
            not in plan_axis_ids
        ):
            raise ValueError(
                "global_winner_axis_id is not present "
                "in the frozen discovery-axis plan"
            )

        if global_winner_axis_id is None:
            if selected_cards:
                raise ValueError(
                    "global winner is null but canonical "
                    "winner cards were materialized"
                )
        elif len(selected_cards) != 1:
            raise ValueError(
                "global production selection must materialize "
                "exactly one canonical winner"
            )

    winner_portfolio_id = (
        _stable_id(
            "hypothesis_portfolio",
            plan.plan_sha256,
            search_width,
            winner_signature,
        )
    )

    winner_portfolio = (
        HypothesisPortfolio(
            portfolio_id=(
                winner_portfolio_id
            ),
            domain_profile_id=(
                domain_profile_id
            ),
            source_context_id=(
                source_context_id
            ),
            source_context_sha256=(
                source_context_sha256
            ),
            source_report_id=(
                source_report_id
            ),
            source_report_sha256=(
                source_report_sha256
            ),
            hypotheses=(
                selected_cards
            ),
            abstention_reason=(
                None
                if selected_cards
                else (
                    "No stable determinate realization "
                    "survived production best-of-k "
                    "selection."
                )
            ),
        )
    )

    portfolio_sha = (
        _sha256_json(
            winner_portfolio
        )
    )

    winner_lineage_report_id = (
        _stable_id(
            "discovery_axis_synthesis_report",
            plan.source_dual_context_sha256,
            plan.plan_sha256,
            winner_portfolio.portfolio_id,
            portfolio_sha,
        )
    )

    report_payload = {
        "schema_version":
            "discovery-axis-synthesis-report-v1",

        "report_id":
            winner_lineage_report_id,

        "source_dual_context_id":
            plan.source_dual_context_id,

        "source_dual_context_sha256":
            plan.source_dual_context_sha256,

        "axis_plan_id":
            plan.plan_id,

        "axis_plan_sha256":
            plan.plan_sha256,

        "final_portfolio_id":
            winner_portfolio.portfolio_id,

        "final_portfolio_sha256":
            portfolio_sha,

        "attempted_axis_count":
            len(
                plan.axes
            ),

        "accepted_hypothesis_count":
            len(
                selected_cards
            ),

        "lineages":
            [
                row.model_dump(
                    mode="json"
                )
                for row
                in selected_lineages
            ],

        # Realization-level failed attempts are preserved in the
        # realization cohort/materialization artifacts rather than
        # misrepresented as ordinary Alpha4 repair attempts.
        "attempts":
            [],

        "external_novelty_status":
            "not_assessed",

        "policy_version":
            "discovery-axis-synthesis-policy-v2",
    }

    winner_lineage_report = (
        DiscoveryAxisSynthesisReport(
            **report_payload,
            report_sha256=(
                _sha256_json(
                    report_payload
                )
            ),
        )
    )

    materialization_report = (
        RealizationWinnerMaterializationReport(
            search_width=(
                search_width
            ),
            attempted_axis_count=(
                len(
                    plan.axes
                )
            ),
            materialized_winner_count=(
                len(
                    selected_cards
                )
            ),
            selections=(
                materialization_records
            ),
            winner_portfolio_id=(
                winner_portfolio.portfolio_id
            ),
            winner_lineage_report_id=(
                winner_lineage_report.report_id
            ),
        )
    )

    return (
        RealizationWinnerMaterialization(
            portfolio=(
                winner_portfolio
            ),
            lineage_report=(
                winner_lineage_report
            ),
            report=(
                materialization_report
            ),
        )
    )
