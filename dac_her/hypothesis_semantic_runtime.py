from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from dac_her.hypothesis_benchmark_evaluator import HypothesisBenchmarkEvaluator
from pipeline_core.discovery.hypothesis_benchmark_contracts import HypothesisEvaluationReport
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.hypothesis_semantic_contracts import (
    HypothesisSemanticReview,
    HypothesisSemanticReviewDraft,
    HypothesisSemanticRunRecord,
)
from dac_her.hypothesis_semantic_llm import (
    HypothesisSemanticBackend,
    HypothesisSemanticGeneration,
)
from dac_her.hypothesis_semantic_prompt import (
    HypothesisSemanticPrompt,
    HypothesisSemanticPromptAssembler,
)
from pipeline_core.discovery.hypothesis_semantic_reference import (
    HypothesisSemanticReferenceSanitizer,
    SemanticReferenceAudit,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


class SemanticReviewValidationError(ValueError):
    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


class HypothesisSemanticReviewCompiler:
    """Attach deterministic lineage and reject invented references."""

    def compile(
        self,
        *,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
        evaluation: HypothesisEvaluationReport,
        prompt: HypothesisSemanticPrompt,
        draft: HypothesisSemanticReviewDraft,
    ) -> HypothesisSemanticReview:
        valid_hypothesis_ids = {row.hypothesis_id for row in portfolio.hypotheses}
        valid_statement_ids = {row.statement_id for row in context.evidence_statements}
        issues: list[str] = []

        for index, row in enumerate(draft.dimensions):
            unknown_h = sorted(set(row.hypothesis_ids) - valid_hypothesis_ids)
            unknown_s = sorted(set(row.statement_ids) - valid_statement_ids)
            if unknown_h:
                issues.append(
                    f"dimensions[{index}].hypothesis_ids contains unknown IDs: {unknown_h}"
                )
            if unknown_s:
                issues.append(
                    f"dimensions[{index}].statement_ids contains unknown IDs: {unknown_s}"
                )
        if issues:
            raise SemanticReviewValidationError(issues)

        review_id = _stable_id(
            "hypothesis_semantic_review",
            context.context_sha256,
            evaluation.portfolio_sha256,
            prompt.prompt_sha256,
            *(f"{row.dimension}:{row.verdict}" for row in draft.dimensions),
        )
        return HypothesisSemanticReview(
            review_id=review_id,
            source_context_id=context.context_id,
            source_context_sha256=context.context_sha256,
            source_portfolio_id=portfolio.portfolio_id,
            source_portfolio_sha256=evaluation.portfolio_sha256,
            source_evaluator_version=evaluation.evaluator_version,
            source_hard_gate_passed=True,
            critic_prompt_version=prompt.prompt_version,
            critic_prompt_sha256=prompt.prompt_sha256,
            dimensions=draft.dimensions,
            overall_summary=draft.overall_summary,
        )


@dataclass(frozen=True)
class HypothesisSemanticOutcome:
    evaluation: HypothesisEvaluationReport
    prompt: HypothesisSemanticPrompt
    generation: HypothesisSemanticGeneration | None
    review: HypothesisSemanticReview | None
    sanitized_draft: HypothesisSemanticReviewDraft | None
    reference_audit: SemanticReferenceAudit
    run_record: HypothesisSemanticRunRecord
    review_validation_issues: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.review is not None and self.run_record.accepted


class HypothesisSemanticCriticRuntime:
    def __init__(
        self,
        backend: HypothesisSemanticBackend,
        *,
        evaluator: HypothesisBenchmarkEvaluator | None = None,
        prompt_assembler: HypothesisSemanticPromptAssembler | None = None,
        compiler: HypothesisSemanticReviewCompiler | None = None,
        reference_sanitizer: HypothesisSemanticReferenceSanitizer | None = None,
    ) -> None:
        self.backend = backend
        self.evaluator = evaluator or HypothesisBenchmarkEvaluator()
        self.prompt_assembler = prompt_assembler or HypothesisSemanticPromptAssembler()
        self.compiler = compiler or HypothesisSemanticReviewCompiler()
        self.reference_sanitizer = (
            reference_sanitizer
            or HypothesisSemanticReferenceSanitizer()
        )

    def run(
        self,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
    ) -> HypothesisSemanticOutcome:
        evaluation = self.evaluator.evaluate(context, portfolio)

        # Build a deterministic placeholder prompt even for hard-gate failure so
        # the run record remains auditable. Do not call the LLM.
        if not evaluation.hard_gate_passed:
            prompt = HypothesisSemanticPrompt.create(
                system_prompt="SEMANTIC CRITIC NOT RUN: deterministic hard gate failed.",
                user_prompt=json.dumps(
                    {
                        "context_id": context.context_id,
                        "portfolio_id": portfolio.portfolio_id,
                        "hard_gate_issues": [
                            row.model_dump(mode="json")
                            for row in evaluation.hard_gate_issues
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            reference_audit = SemanticReferenceAudit()
            record = self._record(
                context=context,
                evaluation=evaluation,
                prompt=prompt,
                generation=None,
                review=None,
                accepted=False,
                reference_audit=reference_audit,
                failure_stage="hard_gate",
                elapsed_seconds=0.0,
            )
            return HypothesisSemanticOutcome(
                evaluation=evaluation,
                prompt=prompt,
                generation=None,
                review=None,
                sanitized_draft=None,
                reference_audit=reference_audit,
                run_record=record,
                review_validation_issues=(),
            )

        prompt = self.prompt_assembler.build(context, portfolio, evaluation)
        started = time.perf_counter()
        try:
            generation = self.backend.review(prompt)
        except Exception:
            raise
        elapsed = time.perf_counter() - started

        valid_hypothesis_ids = {
            row.hypothesis_id
            for row in portfolio.hypotheses
        }
        valid_statement_ids = {
            row.statement_id
            for row in context.evidence_statements
        }
        sanitized = self.reference_sanitizer.sanitize(
            generation.draft,
            valid_hypothesis_ids=valid_hypothesis_ids,
            valid_statement_ids=valid_statement_ids,
        )
        reference_audit = sanitized.audit
        sanitized_draft = sanitized.draft

        if reference_audit.fatal:
            review = None
            issues = tuple(reference_audit.fatal_reasons)
            accepted = False
            failure_stage = "review_validation"
        else:
            try:
                review = self.compiler.compile(
                    context=context,
                    portfolio=portfolio,
                    evaluation=evaluation,
                    prompt=prompt,
                    draft=sanitized_draft,
                )
                issues = ()
                accepted = True
                failure_stage = "none"
            except SemanticReviewValidationError as exc:
                review = None
                issues = tuple(exc.issues)
                accepted = False
                failure_stage = "review_validation"

        record = self._record(
            context=context,
            evaluation=evaluation,
            prompt=prompt,
            generation=generation,
            review=review,
            accepted=accepted,
            reference_audit=reference_audit,
            failure_stage=failure_stage,
            elapsed_seconds=elapsed,
        )
        return HypothesisSemanticOutcome(
            evaluation=evaluation,
            prompt=prompt,
            generation=generation,
            review=review,
            sanitized_draft=sanitized_draft,
            reference_audit=reference_audit,
            run_record=record,
            review_validation_issues=issues,
        )

    def _record(
        self,
        *,
        context: HypothesisContext,
        evaluation: HypothesisEvaluationReport,
        prompt: HypothesisSemanticPrompt,
        generation: HypothesisSemanticGeneration | None,
        review: HypothesisSemanticReview | None,
        accepted: bool,
        reference_audit: SemanticReferenceAudit,
        failure_stage: str,
        elapsed_seconds: float,
    ) -> HypothesisSemanticRunRecord:
        backend_name = str(getattr(self.backend, "backend_name", type(self.backend).__name__))
        model_name = str(getattr(self.backend, "model_name", "unknown"))
        temperature_raw = getattr(self.backend, "temperature", None)
        temperature = float(temperature_raw) if temperature_raw is not None else None
        run_id = _stable_id(
            "hypothesis_semantic_run",
            context.context_sha256,
            evaluation.portfolio_sha256,
            prompt.prompt_sha256,
            backend_name,
            model_name,
            review.review_id if review is not None else failure_stage,
        )
        return HypothesisSemanticRunRecord(
            run_id=run_id,
            context_id=context.context_id,
            context_sha256=context.context_sha256,
            portfolio_id=evaluation.portfolio_id,
            portfolio_sha256=evaluation.portfolio_sha256,
            hard_gate_passed=evaluation.hard_gate_passed,
            review_id=(review.review_id if review is not None else None),
            critic_prompt_version=prompt.prompt_version,
            critic_prompt_sha256=prompt.prompt_sha256,
            backend=backend_name,
            model=model_name,
            generated=generation is not None,
            accepted=accepted,
            failure_stage=failure_stage,  # type: ignore[arg-type]
            reference_sanitization_applied=reference_audit.applied,
            reference_drop_count=reference_audit.drop_count,
            reference_integrity_failure=reference_audit.fatal,
            input_tokens=(generation.input_tokens if generation is not None else None),
            output_tokens=(generation.output_tokens if generation is not None else None),
            elapsed_seconds=elapsed_seconds,
            temperature=temperature,
            backend_mode=(str(getattr(self.backend, "instructor_mode", "")) or None),
            base_url=(str(getattr(self.backend, "base_url", "")) or None),
            parse_retries=(
                int(getattr(self.backend, "parse_retries"))
                if getattr(self.backend, "parse_retries", None) is not None
                else None
            ),
        )
