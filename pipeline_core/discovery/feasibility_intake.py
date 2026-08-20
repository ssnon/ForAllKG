from __future__ import annotations

import hashlib
import json
from typing import Any

from pipeline_core.feasibility_contracts import (
    FeasibilityFalsifier,
    FeasibilityHypothesis,
    FeasibilityIntake,
    FeasibilityPrediction,
    FeasibilityPremise,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.hypothesis_semantic_contracts import HypothesisSemanticReview


CRITICAL_SEMANTIC_DIMENSIONS = {
    "premise_fidelity",
    "gap_discipline",
    "candidate_calibration",
    "inferential_proportionality",
    "causal_strengthening",
    "prediction_linkage",
    "falsifier_informativeness",
    "cross_paper_discipline",
    "abstention_appropriateness",
}


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class FeasibilityIntakeBuilder:
    """Create a narrow downstream API surface from accepted hypothesis artifacts.

    This builder does not decide chemistry. It only freezes hypothesis lineage,
    propagates semantic-review outcomes, and exposes exact positive premises to
    later physics/experimental evaluators.
    """

    def build(
        self,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
        semantic_review: HypothesisSemanticReview,
    ) -> FeasibilityIntake:
        if portfolio.source_context_id != context.context_id:
            raise ValueError("portfolio/context ID mismatch")
        if portfolio.source_context_sha256 != context.context_sha256:
            raise ValueError("portfolio/context SHA mismatch")
        if semantic_review.source_context_id != context.context_id:
            raise ValueError("semantic review/context ID mismatch")
        if semantic_review.source_context_sha256 != context.context_sha256:
            raise ValueError("semantic review/context SHA mismatch")
        if semantic_review.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("semantic review/portfolio ID mismatch")

        portfolio_sha = _sha256(portfolio)
        if semantic_review.source_portfolio_sha256 != portfolio_sha:
            raise ValueError("semantic review/portfolio SHA mismatch")

        statement_index = {row.statement_id: row for row in context.evidence_statements}
        review_by_hypothesis: dict[str, list[object]] = {
            row.hypothesis_id: [] for row in portfolio.hypotheses
        }
        portfolio_level_rows: list[object] = []

        for row in semantic_review.dimensions:
            if row.hypothesis_ids:
                for hypothesis_id in row.hypothesis_ids:
                    if hypothesis_id in review_by_hypothesis:
                        review_by_hypothesis[hypothesis_id].append(row)
            else:
                portfolio_level_rows.append(row)

        hypotheses: list[FeasibilityHypothesis] = []
        for card in portfolio.hypotheses:
            rows = [*review_by_hypothesis.get(card.hypothesis_id, []), *portfolio_level_rows]
            warnings = sorted({
                row.dimension for row in rows if row.verdict == "warning"
            })
            failures = sorted({
                row.dimension for row in rows if row.verdict == "fail"
            })
            critical_failures = sorted(set(failures) & CRITICAL_SEMANTIC_DIMENSIONS)

            if critical_failures:
                gate_status = "human_review_required"
            elif warnings or failures:
                gate_status = "eligible_with_warnings"
            else:
                gate_status = "eligible"

            premises: list[FeasibilityPremise] = []
            for statement_id in card.premise_statement_ids:
                statement = statement_index.get(statement_id)
                if statement is None:
                    raise ValueError(
                        f"{card.hypothesis_id}: premise missing from context: {statement_id}"
                    )
                if not statement.eligible_as_premise:
                    raise ValueError(
                        f"{card.hypothesis_id}: ineligible premise reached feasibility intake: "
                        f"{statement_id}"
                    )
                premises.append(
                    FeasibilityPremise(
                        statement_id=statement.statement_id,
                        text=statement.text,
                        epistemic_role=statement.epistemic_role,
                        claim_kind=statement.claim_kind,
                        paper_ids=list(statement.paper_ids),
                        requires_verification=statement.requires_verification,
                    )
                )

            hypotheses.append(
                FeasibilityHypothesis(
                    hypothesis_id=card.hypothesis_id,
                    title=card.title,
                    statement=card.hypothesis_statement,
                    hypothesis_type=card.hypothesis_type,
                    inferential_bridge=card.inferential_bridge,
                    assumptions=list(card.assumptions),
                    premises=premises,
                    source_paper_ids=list(card.source_paper_ids),
                    candidate_dependency=card.candidate_dependency,
                    predictions=[
                        FeasibilityPrediction(
                            observation_id=row.observation_id,
                            observable=row.observable,
                            expected_direction=row.expected_direction,
                            rationale=row.rationale,
                        )
                        for row in card.predicted_observations
                    ],
                    falsifiers=[
                        FeasibilityFalsifier(
                            criterion_id=row.criterion_id,
                            observable=row.observable,
                            falsifying_outcome=row.falsifying_outcome,
                        )
                        for row in card.falsification_criteria
                    ],
                    semantic_gate_status=gate_status,
                    semantic_warning_dimensions=warnings,
                    semantic_fail_dimensions=failures,
                )
            )

        intake_id = _stable_id(
            "feasibility_intake",
            context.context_sha256,
            portfolio_sha,
            semantic_review.review_id,
        )
        payload = {
            "schema_version": "feasibility-intake-v0",
            "intake_id": intake_id,
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "source_portfolio_id": portfolio.portfolio_id,
            "source_portfolio_sha256": portfolio_sha,
            "source_semantic_review_id": semantic_review.review_id,
            "task_id": context.task_id,
            "question": context.question,
            "corpus_id": context.corpus_id,
            "abstention_reason": portfolio.abstention_reason,
            "hypotheses": [row.model_dump(mode="json") for row in hypotheses],
        }
        return FeasibilityIntake(
            **payload,
            intake_sha256=_sha256(payload),
        )
