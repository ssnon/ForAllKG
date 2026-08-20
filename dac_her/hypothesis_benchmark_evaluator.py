from __future__ import annotations

import hashlib
import json
from typing import Any

from pipeline_core.discovery.hypothesis_benchmark_contracts import (
    HypothesisBenchmarkIssue,
    HypothesisEvaluationReport,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from dac_her.hypothesis_semantic_checks import semantic_diagnostics
from dac_her.hypothesis_validation import HypothesisValidator


EVALUATOR_VERSION = "hypothesis-benchmark-evaluator-v2.6.2-a1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class HypothesisBenchmarkEvaluator:
    """Development/benchmark evaluator, not a production Evidence Auditor.

    Hard-gate authority is deterministic. Semantic checks are diagnostic-only
    and cannot reject a portfolio in v2.6.2-a.
    """

    def __init__(self, *, validator: HypothesisValidator | None = None) -> None:
        self.validator = validator or HypothesisValidator()

    def evaluate(
        self,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
    ) -> HypothesisEvaluationReport:
        baseline = self.validator.validate(context, portfolio)
        hard_issues: list[HypothesisBenchmarkIssue] = [
            HypothesisBenchmarkIssue(
                severity=issue.severity,
                layer="hard_gate",
                code=issue.code,
                location=issue.location,
                message=issue.message,
                source="hypothesis_validation_v260",
            )
            for issue in baseline.issues
        ]

        def hard_error(
            code: str,
            location: str,
            message: str,
            *,
            hypothesis_id: str | None = None,
        ) -> None:
            hard_issues.append(
                HypothesisBenchmarkIssue(
                    severity="error",
                    layer="hard_gate",
                    code=code,
                    location=location,
                    message=message,
                    hypothesis_id=hypothesis_id,
                    source=EVALUATOR_VERSION,
                )
            )

        eligible_ids = {
            row.statement_id
            for row in context.evidence_statements
            if row.eligible_as_premise
        }

        for index, card in enumerate(portfolio.hypotheses):
            location = f"hypotheses[{index}].premise_statement_ids"
            if not card.premise_statement_ids:
                hard_error(
                    "HYPOTHESIS_WITHOUT_ELIGIBLE_PREMISE",
                    location,
                    "A non-abstaining hypothesis must reference at least one eligible "
                    "positive premise.",
                    hypothesis_id=card.hypothesis_id,
                )
                continue
            if not any(sid in eligible_ids for sid in card.premise_statement_ids):
                hard_error(
                    "HYPOTHESIS_WITHOUT_ELIGIBLE_PREMISE",
                    location,
                    "Hypothesis references no statement that the context marks "
                    "eligible_as_premise=true.",
                    hypothesis_id=card.hypothesis_id,
                )

        if portfolio.hypotheses and not eligible_ids:
            # Portfolio-level guard; de-duplicate only if a card-level issue with the
            # same code/location does not already exist.
            hard_error(
                "NON_ABSTENTION_WITH_ZERO_ELIGIBLE_CONTEXT",
                "portfolio.hypotheses",
                "Context contains zero eligible positive premises, but the portfolio "
                "contains one or more hypotheses.",
            )

        errors = sum(issue.severity == "error" for issue in hard_issues)
        warnings = sum(issue.severity == "warning" for issue in hard_issues)
        diagnostics = semantic_diagnostics(context, portfolio)

        return HypothesisEvaluationReport(
            evaluator_version=EVALUATOR_VERSION,
            context_id=context.context_id,
            context_sha256=context.context_sha256,
            portfolio_id=portfolio.portfolio_id,
            portfolio_sha256=_sha256_json(portfolio),
            hard_gate_passed=errors == 0,
            hard_gate_errors=errors,
            hard_gate_warnings=warnings,
            hard_gate_issues=hard_issues,
            diagnostics=diagnostics,
        )
