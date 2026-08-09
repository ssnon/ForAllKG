from __future__ import annotations

import hashlib
from typing import Iterable

from pydantic import BaseModel, ConfigDict

from dac_her.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
    PredictedObservation,
)


class HypothesisCompileIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    location: str
    message: str


class HypothesisCompileError(ValueError):
    def __init__(self, issues: list[HypothesisCompileIssue]) -> None:
        self.issues = issues
        super().__init__("; ".join(f"{x.code}: {x.message}" for x in issues))


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(v) for v in values if str(v).strip()})


class HypothesisCompiler:
    """Deterministically enrich an LLM-owned HypothesisPortfolioDraft.

    v2.6.0 deliberately assigns provenance-derived metadata here rather than in
    the future LLM layer: stable IDs, paper scope, cross-paper status, candidate
    dependency, source hashes, and evidence-profile counts.
    """

    def compile(
        self,
        context: HypothesisContext,
        draft: HypothesisPortfolioDraft,
    ) -> HypothesisPortfolio:
        statement_index = {x.statement_id: x for x in context.evidence_statements}
        issues: list[HypothesisCompileIssue] = []

        for h_index, hypothesis in enumerate(draft.hypotheses):
            location = f"draft.hypotheses[{h_index}]"
            for statement_id in hypothesis.premise_statement_ids:
                statement = statement_index.get(statement_id)
                if statement is None:
                    issues.append(
                        HypothesisCompileIssue(
                            code="UNKNOWN_PREMISE_STATEMENT",
                            location=location + ".premise_statement_ids",
                            message=f"Unknown premise statement ID: {statement_id}",
                        )
                    )
                elif not statement.eligible_as_premise:
                    issues.append(
                        HypothesisCompileIssue(
                            code="INELIGIBLE_POSITIVE_PREMISE",
                            location=location + ".premise_statement_ids",
                            message=(
                                f"Statement {statement_id} is not eligible as a positive premise; "
                                f"restrictions={statement.premise_restrictions}"
                            ),
                        )
                    )
            for statement_id in hypothesis.gap_statement_ids:
                statement = statement_index.get(statement_id)
                if statement is None:
                    issues.append(
                        HypothesisCompileIssue(
                            code="UNKNOWN_GAP_STATEMENT",
                            location=location + ".gap_statement_ids",
                            message=f"Unknown gap statement ID: {statement_id}",
                        )
                    )
                elif not statement.eligible_as_gap:
                    issues.append(
                        HypothesisCompileIssue(
                            code="INELIGIBLE_GAP_STATEMENT",
                            location=location + ".gap_statement_ids",
                            message=f"Statement {statement_id} is not an unresolved/gap statement.",
                        )
                    )

        if issues:
            raise HypothesisCompileError(issues)

        cards: list[HypothesisCard] = []
        for hypothesis in draft.hypotheses:
            premise_ids = _sorted_unique(hypothesis.premise_statement_ids)
            gap_ids = _sorted_unique(hypothesis.gap_statement_ids)
            premises = [statement_index[x] for x in premise_ids]
            gaps = [statement_index[x] for x in gap_ids]

            source_papers = _sorted_unique(
                paper_id for statement in premises for paper_id in statement.paper_ids
            )
            gap_papers = _sorted_unique(
                paper_id for statement in gaps for paper_id in statement.paper_ids
            )
            candidate_count = sum(bool(x.requires_verification) for x in premises)
            if candidate_count == 0:
                candidate_dependency = "none"
            elif candidate_count == len(premises):
                candidate_dependency = "essential"
            else:
                candidate_dependency = "supporting"

            hypothesis_id = _stable_id(
                "hypothesis",
                context.context_sha256,
                hypothesis.local_id,
                hypothesis.hypothesis_statement,
                ",".join(premise_ids),
            )
            predictions = [
                PredictedObservation(
                    observation_id=_stable_id(
                        "prediction",
                        hypothesis_id,
                        row.local_id,
                        row.observable,
                        row.expected_direction,
                    ),
                    observable=row.observable,
                    expected_direction=row.expected_direction,
                    rationale=row.rationale,
                )
                for row in hypothesis.predicted_observations
            ]
            falsifiers = [
                FalsificationCriterion(
                    criterion_id=_stable_id(
                        "falsifier",
                        hypothesis_id,
                        row.local_id,
                        row.observable,
                        row.falsifying_outcome,
                    ),
                    observable=row.observable,
                    falsifying_outcome=row.falsifying_outcome,
                )
                for row in hypothesis.falsification_criteria
            ]
            cards.append(
                HypothesisCard(
                    hypothesis_id=hypothesis_id,
                    source_context_id=context.context_id,
                    source_context_sha256=context.context_sha256,
                    source_report_id=context.source_report_id,
                    source_report_sha256=context.source_report_sha256,
                    title=hypothesis.title,
                    hypothesis_statement=hypothesis.hypothesis_statement,
                    hypothesis_type=hypothesis.hypothesis_type,
                    premise_statement_ids=premise_ids,
                    gap_statement_ids=gap_ids,
                    inferential_bridge=hypothesis.inferential_bridge,
                    predicted_observations=predictions,
                    falsification_criteria=falsifiers,
                    assumptions=list(hypothesis.assumptions),
                    source_paper_ids=source_papers,
                    gap_paper_ids=gap_papers,
                    cross_paper_synthesis=len(source_papers) >= 2,
                    candidate_dependency=candidate_dependency,
                    evidence_profile=HypothesisEvidenceProfile(
                        premise_count=len(premises),
                        gap_count=len(gaps),
                        source_paper_count=len(source_papers),
                        candidate_premise_count=candidate_count,
                        reported_premise_count=sum(
                            x.epistemic_role == "reported" for x in premises
                        ),
                        synthesis_premise_count=sum(
                            x.epistemic_role == "evidence_synthesis" for x in premises
                        ),
                    ),
                )
            )

        portfolio_id = _stable_id(
            "hypothesis_portfolio",
            context.context_sha256,
            *(card.hypothesis_id for card in cards),
            draft.abstention_reason or "",
        )
        return HypothesisPortfolio(
            portfolio_id=portfolio_id,
            source_context_id=context.context_id,
            source_context_sha256=context.context_sha256,
            source_report_id=context.source_report_id,
            source_report_sha256=context.source_report_sha256,
            hypotheses=cards,
            abstention_reason=draft.abstention_reason,
        )
