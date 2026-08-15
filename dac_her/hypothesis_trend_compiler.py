from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from pydantic import BaseModel, ConfigDict

from dac_her.hypothesis_trend_contracts import (
    CompiledTrendReference,
    HYPOTHESIS_TREND_REFERENCE_CONTRACT_SEMANTICS_ID,
    TrendAwareFalsificationCriterion,
    TrendAwareHypothesisCard,
    TrendAwareHypothesisEvidenceProfile,
    TrendAwareHypothesisPortfolio,
    TrendAwareHypothesisPortfolioDraft,
    TrendAwarePredictedObservation,
    TrendReferenceUse,
)
from dac_her.hypothesis_trend_input import (
    HypothesisTrendInputView,
    TrendAwareHypothesisInput,
    verify_trend_aware_input_sources,
)


HYPOTHESIS_TREND_COMPILER_SEMANTICS_ID = (
    "hypothesis_trend_compiler_v1_alpha4c5c"
)


class TrendHypothesisCompileIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    location: str
    message: str


class TrendHypothesisCompileError(ValueError):
    def __init__(
        self,
        issues: list[TrendHypothesisCompileIssue],
    ) -> None:
        self.issues = issues
        super().__init__(
            "; ".join(
                f"{row.code}: {row.message}"
                for row in issues
            )
        )


USE_TO_LANE: dict[TrendReferenceUse, str] = {
    "positive_empirical_support":
        "local_empirical_support",
    "cross_paper_empirical_support":
        "cross_paper_replicated_support",
    "context_qualification":
        "context_dependency_signal",
    "counterevidence_boundary":
        "reversal_boundary",
    "replication_gap":
        "replication_gap",
}

POSITIVE_USES = {
    "positive_empirical_support",
    "cross_paper_empirical_support",
}

CONTEXT_USES = {
    "context_qualification",
    "counterevidence_boundary",
}

GAP_USES = {"replication_gap"}


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    payload = "|".join(str(part) for part in parts).encode(
        "utf-8"
    )
    return (
        f"{prefix}:"
        f"{hashlib.sha256(payload).hexdigest()[:length]}"
    )


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({
        str(value)
        for value in values
        if str(value).strip()
    })


def required_companion_uses(
    view: HypothesisTrendInputView,
) -> set[TrendReferenceUse]:
    if view.cross_context_status == "insufficient":
        return {"replication_gap"}
    if view.cross_context_status == "context_specific":
        return {"context_qualification"}
    if view.cross_context_status == "reversed":
        return {
            "context_qualification",
            "counterevidence_boundary",
        }
    return set()


class TrendAwareHypothesisCompiler:
    semantics_id = HYPOTHESIS_TREND_COMPILER_SEMANTICS_ID
    reference_contract_semantics_id = (
        HYPOTHESIS_TREND_REFERENCE_CONTRACT_SEMANTICS_ID
    )

    def compile(
        self,
        source: TrendAwareHypothesisInput,
        draft: TrendAwareHypothesisPortfolioDraft,
    ) -> TrendAwareHypothesisPortfolio:
        verify_trend_aware_input_sources(source)

        context = source.grounded_context
        statement_index = {
            row.statement_id: row
            for row in context.evidence_statements
        }
        view_index = {
            row.view_id: row for row in source.trend_views
        }

        issues: list[TrendHypothesisCompileIssue] = []

        for h_index, hypothesis in enumerate(
            draft.hypotheses
        ):
            base = f"draft.hypotheses[{h_index}]"

            for statement_id in hypothesis.premise_statement_ids:
                statement = statement_index.get(statement_id)
                if statement is None:
                    issues.append(
                        TrendHypothesisCompileIssue(
                            code="UNKNOWN_PREMISE_STATEMENT",
                            location=(
                                base + ".premise_statement_ids"
                            ),
                            message=(
                                "Unknown Explorer premise statement "
                                f"ID: {statement_id}"
                            ),
                        )
                    )
                elif not statement.eligible_as_premise:
                    issues.append(
                        TrendHypothesisCompileIssue(
                            code="INELIGIBLE_POSITIVE_PREMISE",
                            location=(
                                base + ".premise_statement_ids"
                            ),
                            message=(
                                f"{statement_id} is not eligible "
                                "as an Explorer positive premise."
                            ),
                        )
                    )

            for statement_id in hypothesis.gap_statement_ids:
                statement = statement_index.get(statement_id)
                if statement is None:
                    issues.append(
                        TrendHypothesisCompileIssue(
                            code="UNKNOWN_GAP_STATEMENT",
                            location=base + ".gap_statement_ids",
                            message=(
                                "Unknown Explorer gap statement ID: "
                                f"{statement_id}"
                            ),
                        )
                    )
                elif not statement.eligible_as_gap:
                    issues.append(
                        TrendHypothesisCompileIssue(
                            code="INELIGIBLE_GAP_STATEMENT",
                            location=base + ".gap_statement_ids",
                            message=(
                                f"{statement_id} is not an "
                                "Explorer gap statement."
                            ),
                        )
                    )

            resolved: list[
                tuple[HypothesisTrendInputView, TrendReferenceUse]
            ] = []
            uses_by_grounding: dict[
                str, set[TrendReferenceUse]
            ] = defaultdict(set)

            for r_index, reference in enumerate(
                hypothesis.trend_references
            ):
                location = (
                    base
                    + f".trend_references[{r_index}]"
                )
                view = view_index.get(reference.view_id)
                if view is None:
                    issues.append(
                        TrendHypothesisCompileIssue(
                            code="UNKNOWN_TREND_VIEW",
                            location=location + ".view_id",
                            message=(
                                "Unknown Trend input view ID: "
                                f"{reference.view_id}"
                            ),
                        )
                    )
                    continue

                expected_lane = USE_TO_LANE[
                    reference.use_role
                ]
                if view.lane != expected_lane:
                    issues.append(
                        TrendHypothesisCompileIssue(
                            code="TREND_USE_LANE_MISMATCH",
                            location=location + ".use_role",
                            message=(
                                f"use_role={reference.use_role!r} "
                                f"requires lane={expected_lane!r}, "
                                f"but view {view.view_id} has "
                                f"lane={view.lane!r}."
                            ),
                        )
                    )
                    continue

                if (
                    reference.use_role
                    == "cross_paper_empirical_support"
                    and len(view.paper_ids) < 2
                ):
                    issues.append(
                        TrendHypothesisCompileIssue(
                            code=(
                                "CROSS_PAPER_TREND_SUPPORT_"
                                "LACKS_TWO_PAPERS"
                            ),
                            location=location,
                            message=(
                                "Cross-paper Trend support requires "
                                "at least two source papers."
                            ),
                        )
                    )

                resolved.append(
                    (view, reference.use_role)
                )
                uses_by_grounding[
                    view.grounding_id
                ].add(reference.use_role)

            # Positive local Trend support must preserve the limitation
            # implied by its cross-context status.
            for view, use_role in resolved:
                if use_role not in POSITIVE_USES:
                    continue
                required = required_companion_uses(view)
                missing = required - uses_by_grounding[
                    view.grounding_id
                ]
                for missing_use in sorted(missing):
                    code = {
                        "replication_gap":
                            "MISSING_REPLICATION_GAP_COMPANION",
                        "context_qualification":
                            "MISSING_CONTEXT_QUALIFICATION_COMPANION",
                        "counterevidence_boundary":
                            "MISSING_REVERSAL_BOUNDARY_COMPANION",
                    }[missing_use]
                    issues.append(
                        TrendHypothesisCompileIssue(
                            code=code,
                            location=base + ".trend_references",
                            message=(
                                f"Positive Trend support from "
                                f"{view.grounding_id} with status "
                                f"{view.cross_context_status!r} "
                                f"requires companion use "
                                f"{missing_use!r} from the same "
                                "grounding."
                            ),
                        )
                    )

        if issues:
            raise TrendHypothesisCompileError(issues)

        cards: list[TrendAwareHypothesisCard] = []

        for hypothesis in draft.hypotheses:
            premise_ids = _sorted_unique(
                hypothesis.premise_statement_ids
            )
            gap_ids = _sorted_unique(
                hypothesis.gap_statement_ids
            )
            explorer_premises = [
                statement_index[value]
                for value in premise_ids
            ]
            explorer_gaps = [
                statement_index[value] for value in gap_ids
            ]

            compiled_refs: list[
                CompiledTrendReference
            ] = []
            for reference in sorted(
                hypothesis.trend_references,
                key=lambda row: (
                    row.use_role,
                    row.view_id,
                ),
            ):
                view = view_index[reference.view_id]
                compiled_refs.append(
                    CompiledTrendReference(
                        reference_id=_stable_id(
                            "hypothesis_trend_reference",
                            source.input_sha256,
                            hypothesis.local_id,
                            view.view_id,
                            reference.use_role,
                        ),
                        view_id=view.view_id,
                        grounding_id=view.grounding_id,
                        relation_id=view.relation_id,
                        lane=view.lane,
                        use_role=reference.use_role,
                        cross_context_status=
                            view.cross_context_status,
                        paper_ids=list(view.paper_ids),
                        directions=list(view.directions),
                        shapes=list(view.shapes),
                        requires_context_qualification=
                            view.requires_context_qualification,
                        requires_verification=
                            view.requires_verification,
                        association_only=bool(
                            view.association_only_result_ids
                        ),
                        directional_cross_paper_premise_allowed=
                            view.
                            directional_cross_paper_premise_allowed,
                        trend_causal_authorization=False,
                        trend_universal_authorization=False,
                    )
                )

            positive_refs = [
                row
                for row in compiled_refs
                if row.use_role in POSITIVE_USES
            ]
            trend_gap_refs = [
                row
                for row in compiled_refs
                if row.use_role in GAP_USES
            ]
            context_refs = [
                row
                for row in compiled_refs
                if row.use_role in CONTEXT_USES
            ]

            explorer_support_papers = _sorted_unique(
                paper_id
                for statement in explorer_premises
                for paper_id in statement.paper_ids
            )
            trend_positive_papers = _sorted_unique(
                paper_id
                for row in positive_refs
                for paper_id in row.paper_ids
            )
            support_papers = _sorted_unique(
                [
                    *explorer_support_papers,
                    *trend_positive_papers,
                ]
            )

            explorer_gap_papers = _sorted_unique(
                paper_id
                for statement in explorer_gaps
                for paper_id in statement.paper_ids
            )
            trend_gap_papers = _sorted_unique(
                paper_id
                for row in trend_gap_refs
                for paper_id in row.paper_ids
            )
            context_papers = _sorted_unique(
                paper_id
                for row in context_refs
                for paper_id in row.paper_ids
            )

            explorer_verification = sum(
                bool(row.requires_verification)
                for row in explorer_premises
            )
            trend_verification = sum(
                bool(row.requires_verification)
                for row in positive_refs
            )
            positive_source_count = (
                len(explorer_premises)
                + len(positive_refs)
            )
            verification_count = (
                explorer_verification
                + trend_verification
            )
            if verification_count == 0:
                verification_dependency = "none"
            elif verification_count == positive_source_count:
                verification_dependency = "essential"
            else:
                verification_dependency = "supporting"

            hypothesis_id = _stable_id(
                "trend_aware_hypothesis",
                source.domain_profile_id,
                source.input_sha256,
                hypothesis.local_id,
                hypothesis.hypothesis_statement,
                ",".join(premise_ids),
                ",".join(
                    f"{row.view_id}:{row.use_role}"
                    for row in compiled_refs
                ),
            )

            predictions = [
                TrendAwarePredictedObservation(
                    observation_id=_stable_id(
                        "trend_aware_prediction",
                        hypothesis_id,
                        row.local_id,
                        row.observable,
                        row.expected_direction,
                    ),
                    observable=row.observable,
                    expected_direction=
                        row.expected_direction,
                    rationale=row.rationale,
                )
                for row in hypothesis.predicted_observations
            ]
            falsifiers = [
                TrendAwareFalsificationCriterion(
                    criterion_id=_stable_id(
                        "trend_aware_falsifier",
                        hypothesis_id,
                        row.local_id,
                        row.observable,
                        row.falsifying_outcome,
                    ),
                    observable=row.observable,
                    falsifying_outcome=
                        row.falsifying_outcome,
                )
                for row in hypothesis.falsification_criteria
            ]

            profile = TrendAwareHypothesisEvidenceProfile(
                explorer_premise_count=len(
                    explorer_premises
                ),
                explorer_gap_count=len(explorer_gaps),
                trend_reference_count=len(compiled_refs),
                trend_positive_support_count=len(
                    positive_refs
                ),
                trend_cross_paper_support_count=sum(
                    row.use_role
                    == "cross_paper_empirical_support"
                    for row in compiled_refs
                ),
                trend_context_qualification_count=sum(
                    row.use_role == "context_qualification"
                    for row in compiled_refs
                ),
                trend_counterevidence_count=sum(
                    row.use_role
                    == "counterevidence_boundary"
                    for row in compiled_refs
                ),
                trend_gap_count=len(trend_gap_refs),
                support_paper_count=len(support_papers),
                verification_required_support_count=
                    verification_count,
                association_only_support_count=sum(
                    row.association_only
                    for row in positive_refs
                ),
                reported_explorer_premise_count=sum(
                    row.epistemic_role == "reported"
                    for row in explorer_premises
                ),
                synthesis_explorer_premise_count=sum(
                    row.epistemic_role
                    == "evidence_synthesis"
                    for row in explorer_premises
                ),
            )

            cards.append(
                TrendAwareHypothesisCard(
                    hypothesis_id=hypothesis_id,
                    domain_profile_id=
                        source.domain_profile_id,
                    source_context_id=
                        context.context_id,
                    source_context_sha256=
                        context.context_sha256,
                    source_report_id=
                        context.source_report_id,
                    source_report_sha256=
                        context.source_report_sha256,
                    source_trend_input_id=source.input_id,
                    source_trend_input_sha256=
                        source.input_sha256,
                    title=hypothesis.title,
                    hypothesis_statement=
                        hypothesis.hypothesis_statement,
                    hypothesis_type=
                        hypothesis.hypothesis_type,
                    premise_statement_ids=premise_ids,
                    gap_statement_ids=gap_ids,
                    trend_references=compiled_refs,
                    inferential_bridge=
                        hypothesis.inferential_bridge,
                    predicted_observations=predictions,
                    falsification_criteria=falsifiers,
                    assumptions=list(
                        hypothesis.assumptions
                    ),
                    explorer_source_paper_ids=
                        explorer_support_papers,
                    trend_positive_source_paper_ids=
                        trend_positive_papers,
                    support_paper_ids=support_papers,
                    explorer_gap_paper_ids=
                        explorer_gap_papers,
                    trend_gap_paper_ids=trend_gap_papers,
                    context_and_counterevidence_paper_ids=
                        context_papers,
                    cross_paper_synthesis=(
                        len(support_papers) >= 2
                    ),
                    verification_dependency=
                        verification_dependency,
                    evidence_profile=profile,
                    trend_causal_authorization=False,
                    trend_universal_authorization=False,
                )
            )

        portfolio_id = _stable_id(
            "trend_aware_hypothesis_portfolio",
            source.domain_profile_id,
            source.input_sha256,
            *[
                card.hypothesis_id for card in cards
            ],
            draft.abstention_reason or "",
        )

        return TrendAwareHypothesisPortfolio(
            portfolio_id=portfolio_id,
            domain_profile_id=source.domain_profile_id,
            source_context_id=context.context_id,
            source_context_sha256=
                context.context_sha256,
            source_report_id=context.source_report_id,
            source_report_sha256=
                context.source_report_sha256,
            source_trend_input_id=source.input_id,
            source_trend_input_sha256=
                source.input_sha256,
            hypotheses=cards,
            abstention_reason=draft.abstention_reason,
        )
