from __future__ import annotations

import hashlib
import json
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ExternalNoveltyCard,
    ExternalNoveltyReport,
)
from pipeline_core.discovery.hypothesis_action_contracts import (
    G1FindingAuthority,
    G1FindingRef,
    G1FindingScope,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyRefinementReport,
)


_POLICY_VERSION = (
    "external-novelty-g1-normalization-v1"
)


class ExternalNoveltyActionBindingError(
    ValueError
):
    pass


def _canonical_json(
    value: object,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(
    prefix: str,
    payload: object,
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            payload
        ).encode("utf-8")
    ).hexdigest()[:20]

    return f"{prefix}:{digest}"


# ----------------------------------------------------------------------
# Aggregate authority
# ----------------------------------------------------------------------

_AGGREGATE_AUTHORITY: dict[
    str,
    G1FindingAuthority,
] = {
    "PLAUSIBLY_NOVEL":
        "informational",

    "NEW_COMBINATION_OF_KNOWN_EFFECTS":
        "informational",

    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP":
        "informational",

    # Search-bounded literature support substantially reduces
    # distinctiveness, but does not make the hypothesis scientifically
    # false. It creates downgrade pressure rather than rejection.
    "LITERATURE_SUPPORTED_EXTENSION":
        "actionable",

    "INSUFFICIENT_SEARCH_EVIDENCE":
        "advisory",

    "WELL_ESTABLISHED":
        "terminal_candidate",

    "CONFLICTING_PRIOR_ART":
        "terminal_candidate",
}


def _claim_authority(
    *,
    status: str,
    importance: str,
) -> G1FindingAuthority:
    if status in {
        "COMPONENTS_ONLY",
        "NO_DIRECT_MATCH_FOUND",
        "TITLE_ONLY_NEIGHBORS",
    }:
        return "informational"

    if status == "INSUFFICIENT_METADATA":
        return "advisory"

    if status == "PARTIAL_PRIOR_ART":
        return (
            "actionable"
            if importance == "core"
            else "advisory"
        )

    if status in {
        "DIRECT_PRIOR_ART",
        "CONFLICTING_PRIOR_ART",
    }:
        return (
            "terminal_candidate"
            if importance == "core"
            else "actionable"
        )

    raise ExternalNoveltyActionBindingError(
        "unsupported claim prior-art status: "
        f"{status!r}"
    )


def _corrected_cards(
    corrected: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if (
        corrected.get("schema_version")
        !=
        "n1-4b-relation-nucleus-hybrid-reaggregation-corrected-v1"
    ):
        raise ExternalNoveltyActionBindingError(
            "unexpected calibrated N1 correction schema"
        )

    rows = corrected.get(
        "cards",
        [],
    )

    if not isinstance(
        rows,
        list,
    ):
        raise ExternalNoveltyActionBindingError(
            "corrected N1 cards must be a list"
        )

    result = {}

    for row in rows:
        if not isinstance(
            row,
            dict,
        ):
            raise ExternalNoveltyActionBindingError(
                "corrected N1 card must be an object"
            )

        hypothesis_id = str(
            row.get(
                "hypothesis_id",
                "",
            )
        )

        if not hypothesis_id:
            raise ExternalNoveltyActionBindingError(
                "corrected N1 card missing hypothesis_id"
            )

        if hypothesis_id in result:
            raise ExternalNoveltyActionBindingError(
                "duplicate corrected N1 hypothesis_id"
            )

        result[
            hypothesis_id
        ] = row

    return result


def _corrected_claim_index(
    row: dict[str, Any],
) -> dict[str, dict[str, str]]:
    claims = row.get(
        "claims",
        [],
    )

    if not isinstance(
        claims,
        list,
    ):
        raise ExternalNoveltyActionBindingError(
            "corrected N1 claims must be a list"
        )

    result = {}

    for claim in claims:
        if not isinstance(
            claim,
            dict,
        ):
            raise ExternalNoveltyActionBindingError(
                "corrected N1 claim must be an object"
            )

        claim_id = str(
            claim.get(
                "claim_id",
                "",
            )
        )

        importance = str(
            claim.get(
                "importance",
                "",
            )
        )

        status = str(
            claim.get(
                "status",
                "",
            )
        )

        if (
            not claim_id
            or importance
            not in {
                "core",
                "supporting",
            }
            or not status
        ):
            raise ExternalNoveltyActionBindingError(
                "malformed corrected N1 claim"
            )

        if claim_id in result:
            raise ExternalNoveltyActionBindingError(
                "duplicate corrected N1 claim_id"
            )

        result[
            claim_id
        ] = {
            "importance":
                importance,

            "status":
                status,
        }

    return result


class ExternalNoveltyFindingActionAdapter:
    """Bind calibrated external-novelty findings to final hypotheses.

    The full ExternalNoveltyReport remains the source for claim text,
    coverage, matches, and search-bounded epistemic provenance.

    The N1 relation-nucleus corrected artifact is an authoritative
    status overlay only.

    A pre-refinement novelty assessment may cross the refinement
    generation boundary only when the hypothesis was kept_original.
    Accepted refinements require a fresh final external-novelty review.
    """

    policy_version = (
        _POLICY_VERSION
    )

    def normalize(
        self,
        *,
        base_report:
            ExternalNoveltyReport,

        corrected_overlay:
            dict[str, Any],

        refinement_report:
            NoveltyRefinementReport,

        final_portfolio:
            HypothesisPortfolio,

        base_artifact_id:
            str | None = None,

        corrected_artifact_id:
            str | None = None,
    ) -> tuple[
        G1FindingRef,
        ...
    ]:
        if (
            base_report.epistemic_usage
            !=
            "prior_art_only_not_positive_premise"
        ):
            raise ExternalNoveltyActionBindingError(
                "external prior art must remain "
                "non-positive-premise evidence"
            )

        if (
            base_report.external_novelty_claim_scope
            !=
            "search-bounded_prior-art_assessment_not_literature-wide_proof"
        ):
            raise ExternalNoveltyActionBindingError(
                "unexpected external novelty epistemic scope"
            )

        if (
            refinement_report.source_portfolio_id
            != base_report.source_portfolio_id
        ):
            raise ExternalNoveltyActionBindingError(
                "external novelty / refinement source "
                "portfolio mismatch"
            )

        if (
            refinement_report.final_portfolio_id
            != final_portfolio.portfolio_id
        ):
            raise ExternalNoveltyActionBindingError(
                "refinement / final portfolio mismatch"
            )

        corrected = _corrected_cards(
            corrected_overlay
        )

        base_cards = {
            card.hypothesis_id:
                card
            for card
            in base_report.cards
        }

        if (
            set(corrected)
            != set(base_cards)
        ):
            raise ExternalNoveltyActionBindingError(
                "corrected N1 hypothesis set must "
                "exactly match base external report"
            )

        attempts = {
            row.original_hypothesis_id:
                row
            for row
            in refinement_report.attempts
            if row.decision
            in {
                "kept_original",
                "accepted_refinement",
            }
        }

        if (
            set(attempts)
            != set(base_cards)
        ):
            raise ExternalNoveltyActionBindingError(
                "external novelty source hypotheses must "
                "exactly match surviving refinement lineage"
            )

        final_ids = {
            card.hypothesis_id
            for card
            in final_portfolio.hypotheses
        }

        output: list[
            G1FindingRef
        ] = []

        base_source_id = (
            base_artifact_id
            or base_report.report_id
        )

        corrected_source_id = (
            corrected_artifact_id
            or str(
                corrected_overlay.get(
                    "schema_version"
                )
            )
        )

        for source_id in sorted(
            base_cards
        ):
            base_card = base_cards[
                source_id
            ]

            corrected_card = corrected[
                source_id
            ]

            attempt = attempts[
                source_id
            ]

            if (
                attempt.decision
                != "kept_original"
            ):
                raise ExternalNoveltyActionBindingError(
                    "pre-refinement external novelty "
                    "cannot be rebound across accepted "
                    "scientific refinement; fresh final "
                    "novelty assessment is required: "
                    + source_id
                )

            final_id = (
                attempt.final_hypothesis_id
            )

            if (
                final_id is None
                or final_id
                not in final_ids
            ):
                raise ExternalNoveltyActionBindingError(
                    "invalid final hypothesis lineage "
                    "for external novelty source"
                )

            corrected_status = str(
                corrected_card.get(
                    "new_status",
                    "",
                )
            )

            if (
                corrected_status
                not in _AGGREGATE_AUTHORITY
            ):
                raise ExternalNoveltyActionBindingError(
                    "unsupported corrected aggregate status: "
                    + repr(
                        corrected_status
                    )
                )

            lineage_refs = [
                refinement_report.report_id,
                corrected_source_id,
            ]

            # ------------------------------------------------------
            # Hypothesis-level calibrated aggregate finding
            # ------------------------------------------------------

            aggregate_ref_id = _stable_id(
                "g1_finding_ref",
                {
                    "policy":
                        self.policy_version,

                    "level":
                        "hypothesis_aggregate",

                    "base_report_id":
                        base_report.report_id,

                    "corrected_source":
                        corrected_source_id,

                    "source_hypothesis_id":
                        source_id,

                    "target_hypothesis_id":
                        final_id,

                    "status":
                        corrected_status,
                },
            )

            output.append(
                G1FindingRef(
                    finding_ref_id=
                        aggregate_ref_id,

                    source_kind=
                        "external_novelty",

                    source_artifact_id=
                        base_source_id,

                    source_finding_id=(
                        "external_novelty_aggregate:"
                        + source_id
                    ),

                    source_status=
                        corrected_status,

                    source_attributes={
                        "assessment_level":
                            "hypothesis_aggregate",

                        "base_status":
                            base_card.status,

                        "calibration_source":
                            corrected_source_id,

                        "epistemic_usage":
                            base_report.epistemic_usage,

                        "claim_scope":
                            base_report
                            .external_novelty_claim_scope,
                    },

                    authority=
                        _AGGREGATE_AUTHORITY[
                            corrected_status
                        ],

                    source_portfolio_id=
                        base_report.source_portfolio_id,

                    source_hypothesis_ids=[
                        source_id
                    ],

                    source_scope=
                        G1FindingScope(
                            kind=
                                "hypothesis",

                            hypothesis_ids=[
                                source_id
                            ],
                        ),

                    target_portfolio_id=
                        final_portfolio
                        .portfolio_id,

                    target_hypothesis_id=
                        final_id,

                    target_scope=
                        G1FindingScope(
                            kind=
                                "hypothesis",

                            hypothesis_ids=[
                                final_id
                            ],
                        ),

                    lineage_ref_ids=
                        lineage_refs,

                    rationale=(
                        "Calibrated N1 hypothesis-level "
                        "external-novelty status. External "
                        "prior art remains search-bounded "
                        "and cannot become a positive premise."
                    ),
                )
            )

            # ------------------------------------------------------
            # Claim-level calibrated findings
            # ------------------------------------------------------

            corrected_claims = (
                _corrected_claim_index(
                    corrected_card
                )
            )

            base_claims = {
                claim.claim_id:
                    claim
                for claim
                in base_card.claim_reviews
            }

            if (
                set(corrected_claims)
                != set(base_claims)
            ):
                raise ExternalNoveltyActionBindingError(
                    "corrected N1 claim set must exactly "
                    "match base claim-review set for "
                    + source_id
                )

            for claim_id in sorted(
                base_claims
            ):
                claim = base_claims[
                    claim_id
                ]

                correction = (
                    corrected_claims[
                        claim_id
                    ]
                )

                if (
                    correction[
                        "importance"
                    ]
                    != claim.importance
                ):
                    raise ExternalNoveltyActionBindingError(
                        "corrected N1 claim importance "
                        "drift: "
                        + claim_id
                    )

                corrected_claim_status = (
                    correction[
                        "status"
                    ]
                )

                authority = _claim_authority(
                    status=
                        corrected_claim_status,

                    importance=
                        claim.importance,
                )

                finding_ref_id = _stable_id(
                    "g1_finding_ref",
                    {
                        "policy":
                            self.policy_version,

                        "level":
                            "claim",

                        "source_hypothesis_id":
                            source_id,

                        "target_hypothesis_id":
                            final_id,

                        "claim_id":
                            claim_id,

                        "importance":
                            claim.importance,

                        "base_status":
                            claim.status,

                        "corrected_status":
                            corrected_claim_status,
                    },
                )

                output.append(
                    G1FindingRef(
                        finding_ref_id=
                            finding_ref_id,

                        source_kind=
                            "external_novelty",

                        source_artifact_id=
                            base_source_id,

                        source_finding_id=
                            claim_id,

                        source_status=
                            corrected_claim_status,

                        source_attributes={
                            "assessment_level":
                                "claim",

                            "importance":
                                claim.importance,

                            "base_claim_status":
                                claim.status,

                            "calibration_source":
                                corrected_source_id,

                            "epistemic_usage":
                                base_report.epistemic_usage,

                            "claim_scope":
                                base_report
                                .external_novelty_claim_scope,
                        },

                        authority=
                            authority,

                        source_portfolio_id=
                            base_report.source_portfolio_id,

                        source_hypothesis_ids=[
                            source_id
                        ],

                        source_scope=
                            G1FindingScope(
                                kind=
                                    "external_novelty_claim",

                                hypothesis_ids=[
                                    source_id
                                ],

                                assertion_ids=[
                                    claim_id
                                ],
                            ),

                        target_portfolio_id=
                            final_portfolio
                            .portfolio_id,

                        target_hypothesis_id=
                            final_id,

                        # Claim decomposition is an external-analysis
                        # scope attached to the final hypothesis. The
                        # claim ID is preserved only because R6 says
                        # kept_original; accepted refinement is blocked.
                        target_scope=
                            G1FindingScope(
                                kind=
                                    "external_novelty_claim",

                                hypothesis_ids=[
                                    final_id
                                ],

                                assertion_ids=[
                                    claim_id
                                ],
                            ),

                        lineage_ref_ids=
                            lineage_refs,

                        rationale=(
                            f"Search-bounded external prior-art "
                            f"assessment for {claim.importance} "
                            f"claim: {claim.claim_text}"
                        ),
                    )
                )

        finding_ids = [
            row.finding_ref_id
            for row
            in output
        ]

        if (
            len(finding_ids)
            != len(
                set(
                    finding_ids
                )
            )
        ):
            raise ExternalNoveltyActionBindingError(
                "duplicate normalized external-novelty "
                "finding_ref_id"
            )

        return tuple(
            sorted(
                output,
                key=lambda row:
                    row.finding_ref_id,
            )
        )
