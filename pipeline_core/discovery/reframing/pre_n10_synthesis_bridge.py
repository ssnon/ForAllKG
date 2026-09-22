from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.reframing.cross_lane_synthesis import (
    CrossLaneSynthesisShadowReport,
    CrossLaneSynthesizedHypothesis,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_SYNTHESIS_TYPE_MAP = {
    "complementary_mechanism_integration": "cross_evidence_synthesis",
    "conditional_relation_refinement": "context_dependency",
    "competing_model_formulation": "mechanistic_extension",
    "measurement_model_integration": "descriptor_mediation",
    "cross_lane_explanatory_synthesis": "cross_evidence_synthesis",
}


class SynthesisN10LineageEntry(StrictModel):
    projected_hypothesis_id: str
    source_synthesis_hypothesis_id: str
    source_candidate_ids: list[str]
    source_object_ids: list[str]
    source_lanes: list[str]
    synthesis_kind: str
    discriminating_test: dict[str, Any]
    unresolved_questions: list[str]
    prediction_direction_projection: Literal["unspecified"] = "unspecified"
    production_authority: Literal[False] = False


class SynthesisN10ProjectionReport(StrictModel):
    schema_version: Literal[
        "scientific-synthesis-pre-n10-projection-v1"
    ] = "scientific-synthesis-pre-n10-projection-v1"

    report_id: str
    source_context_id: str
    source_alpha6_portfolio_id: str
    source_synthesis_report_id: str
    output_portfolio_id: str
    projected_hypothesis_count: int = Field(ge=0)
    entries: list[SynthesisN10LineageEntry]
    abstention_reason: str | None = None

    source_statement_ids_validated: Literal[True] = True
    positive_premise_eligibility_validated: Literal[True] = True
    gap_eligibility_validated: Literal[True] = True
    source_report_lineage_inherited_from_context: Literal[True] = True
    synthesis_lineage_preserved_in_sidecar: Literal[True] = True
    prediction_direction_inferred_from_free_text: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    n10_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


def _stable_id(prefix: str, *parts: object) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _unique(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _validate_parent_lineage(
    *,
    context: HypothesisContext,
    alpha6: HypothesisPortfolio,
    synthesis: CrossLaneSynthesisShadowReport,
) -> None:
    if alpha6.source_context_id != context.context_id:
        raise ValueError("Alpha6 portfolio/context ID mismatch")
    if alpha6.source_context_sha256 != context.context_sha256:
        raise ValueError("Alpha6 portfolio/context SHA mismatch")
    if alpha6.domain_profile_id != context.domain_profile_id:
        raise ValueError("Alpha6 portfolio/context domain mismatch")
    if alpha6.source_report_id != context.source_report_id:
        raise ValueError("Alpha6 portfolio/context source report ID mismatch")
    if alpha6.source_report_sha256 != context.source_report_sha256:
        raise ValueError("Alpha6 portfolio/context source report SHA mismatch")
    if synthesis.source_context_id != context.context_id:
        raise ValueError("synthesis/context ID mismatch")
    if synthesis.source_task_id != context.task_id:
        raise ValueError("synthesis/context task mismatch")
    if synthesis.question != context.question:
        raise ValueError("synthesis/context question mismatch")


def _project_one(
    *,
    row: CrossLaneSynthesizedHypothesis,
    context: HypothesisContext,
) -> tuple[HypothesisCard, SynthesisN10LineageEntry]:
    statement_by_id = {
        statement.statement_id: statement
        for statement in context.evidence_statements
    }
    research_gap_statement_ids = {
        gap.statement_id for gap in context.research_gaps
    }

    premise_ids = _unique(list(row.premise_statement_ids))
    gap_ids = _unique(list(row.gap_statement_ids))
    if not premise_ids:
        raise ValueError(
            f"synthesis {row.hypothesis_id} has no grounded positive premises"
        )
    overlap = sorted(set(premise_ids) & set(gap_ids))
    if overlap:
        raise ValueError(
            f"synthesis uses statement IDs as both premise and gap: {overlap}"
        )

    missing_premises = [
        sid for sid in premise_ids if sid not in statement_by_id
    ]
    missing_gaps = [
        sid for sid in gap_ids if sid not in statement_by_id
    ]
    if missing_premises:
        raise ValueError(
            f"synthesis cites unknown premise statement IDs: {missing_premises}"
        )
    if missing_gaps:
        raise ValueError(
            f"synthesis cites unknown gap statement IDs: {missing_gaps}"
        )

    premise_rows = [statement_by_id[sid] for sid in premise_ids]
    invalid_premises = [
        row.statement_id for row in premise_rows if not row.eligible_as_premise
    ]
    if invalid_premises:
        raise ValueError(
            "synthesis cites statements not eligible as positive premises: "
            + ", ".join(invalid_premises)
        )

    gap_rows = [statement_by_id[sid] for sid in gap_ids]
    invalid_gaps = [
        gap.statement_id
        for gap in gap_rows
        if not (
            gap.eligible_as_gap
            or gap.statement_id in research_gap_statement_ids
        )
    ]
    if invalid_gaps:
        raise ValueError(
            "synthesis cites statements not eligible as research gaps: "
            + ", ".join(invalid_gaps)
        )

    source_papers = sorted(
        {
            paper_id
            for statement in premise_rows
            for paper_id in statement.paper_ids
        }
    )
    gap_papers = sorted(
        {
            paper_id
            for statement in gap_rows
            for paper_id in statement.paper_ids
        }
    )

    candidate_premise_count = sum(
        1 for statement in premise_rows if statement.requires_verification
    )
    if candidate_premise_count == 0:
        candidate_dependency = "none"
    elif candidate_premise_count == len(premise_rows):
        candidate_dependency = "essential"
    else:
        candidate_dependency = "supporting"

    projected_id = _stable_id(
        "hypothesis",
        "scientific_cross_lane_synthesis",
        context.context_id,
        row.hypothesis_id,
    )

    predictions = [
        PredictedObservation(
            observation_id=_stable_id(
                "prediction",
                projected_id,
                index,
                prediction.observable,
                prediction.expected_result,
            ),
            observable=prediction.observable,
            expected_direction="unspecified",
            rationale=(
                "Expected result: "
                + prediction.expected_result.strip()
                + " | "
                + prediction.rationale.strip()
            ),
        )
        for index, prediction in enumerate(row.predictions, start=1)
    ]
    falsifiers = [
        FalsificationCriterion(
            criterion_id=_stable_id(
                "falsifier",
                projected_id,
                index,
                falsifier.observable or "",
                falsifier.falsifying_outcome,
            ),
            observable=(
                falsifier.observable
                or "response under the proposed discriminating conditions"
            ),
            falsifying_outcome=falsifier.falsifying_outcome,
        )
        for index, falsifier in enumerate(row.falsifiers, start=1)
    ]

    hypothesis_type = _SYNTHESIS_TYPE_MAP.get(row.synthesis_kind)
    if hypothesis_type is None:
        raise ValueError(
            f"unsupported synthesis kind for N10 projection: {row.synthesis_kind}"
        )

    card = HypothesisCard(
        hypothesis_id=projected_id,
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        title=row.title,
        hypothesis_statement=row.hypothesis_statement,
        hypothesis_type=hypothesis_type,
        premise_statement_ids=premise_ids,
        gap_statement_ids=gap_ids,
        inferential_bridge=row.synthesis_rationale,
        predicted_observations=predictions,
        falsification_criteria=falsifiers,
        assumptions=list(row.assumptions),
        source_paper_ids=source_papers,
        gap_paper_ids=gap_papers,
        cross_paper_synthesis=len(source_papers) > 1,
        candidate_dependency=candidate_dependency,
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=len(premise_rows),
            gap_count=len(gap_rows),
            source_paper_count=len(source_papers),
            candidate_premise_count=candidate_premise_count,
            reported_premise_count=sum(
                1
                for statement in premise_rows
                if statement.epistemic_role == "reported"
            ),
            synthesis_premise_count=sum(
                1
                for statement in premise_rows
                if statement.epistemic_role == "evidence_synthesis"
            ),
        ),
    )
    lineage = SynthesisN10LineageEntry(
        projected_hypothesis_id=projected_id,
        source_synthesis_hypothesis_id=row.hypothesis_id,
        source_candidate_ids=list(row.source_candidate_ids),
        source_object_ids=list(row.source_object_ids),
        source_lanes=list(row.source_lanes),
        synthesis_kind=row.synthesis_kind,
        discriminating_test=row.discriminating_test.model_dump(mode="json"),
        unresolved_questions=list(row.unresolved_questions),
    )
    return card, lineage


def build_pre_n10_synthesis_portfolio(
    *,
    context: HypothesisContext,
    alpha6_portfolio: HypothesisPortfolio,
    synthesis_report: CrossLaneSynthesisShadowReport,
) -> tuple[HypothesisPortfolio, SynthesisN10ProjectionReport]:
    _validate_parent_lineage(
        context=context,
        alpha6=alpha6_portfolio,
        synthesis=synthesis_report,
    )

    projected: list[HypothesisCard] = []
    lineage: list[SynthesisN10LineageEntry] = []
    for row in synthesis_report.hypotheses:
        card, entry = _project_one(row=row, context=context)
        projected.append(card)
        lineage.append(entry)

    hypothesis_ids = [row.hypothesis_id for row in projected]
    if len(hypothesis_ids) != len(set(hypothesis_ids)):
        raise ValueError("duplicate projected synthesis hypothesis IDs")

    abstention = None
    if not projected:
        abstention = (
            synthesis_report.abstention_reason
            or "cross_lane_synthesis_abstained_before_n10"
        )

    portfolio_id = _stable_id(
        "hypothesis_portfolio",
        "scientific_synthesis_pre_n10",
        context.context_id,
        synthesis_report.report_id,
        *hypothesis_ids,
        abstention or "",
    )
    portfolio = HypothesisPortfolio(
        portfolio_id=portfolio_id,
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        hypotheses=projected,
        abstention_reason=abstention,
    )
    report = SynthesisN10ProjectionReport(
        report_id=_stable_id(
            "scientific_synthesis_n10_projection",
            portfolio_id,
            synthesis_report.report_id,
        ),
        source_context_id=context.context_id,
        source_alpha6_portfolio_id=alpha6_portfolio.portfolio_id,
        source_synthesis_report_id=synthesis_report.report_id,
        output_portfolio_id=portfolio.portfolio_id,
        projected_hypothesis_count=len(projected),
        entries=lineage,
        abstention_reason=abstention,
    )
    return portfolio, report


__all__ = [
    "SynthesisN10LineageEntry",
    "SynthesisN10ProjectionReport",
    "build_pre_n10_synthesis_portfolio",
]
