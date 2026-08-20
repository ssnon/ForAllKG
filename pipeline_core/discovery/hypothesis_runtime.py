from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from pipeline_core.discovery.hypothesis_compiler import (
    HypothesisCompileError,
    HypothesisCompileIssue,
    HypothesisCompiler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftBackend, HypothesisDraftGeneration
from pipeline_core.discovery.hypothesis_prompt import HypothesisPrompt, HypothesisPromptAssembler
from pipeline_core.discovery.hypothesis_run_record import HypothesisMakerRunRecord
from pipeline_core.discovery.hypothesis_validation import HypothesisValidationResult, HypothesisValidator


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


@dataclass(frozen=True)
class HypothesisMakerRunOutcome:
    accepted_portfolio: HypothesisPortfolio | None
    last_portfolio: HypothesisPortfolio | None
    validation: HypothesisValidationResult | None
    final_draft: HypothesisPortfolioDraft | None
    draft_history: tuple[HypothesisPortfolioDraft, ...]
    compile_issues: tuple[HypothesisCompileIssue, ...]
    run_record: HypothesisMakerRunRecord
    prompt: HypothesisPrompt

    @property
    def accepted(self) -> bool:
        return (
            self.accepted_portfolio is not None
            and self.run_record.final_validation_passed
        )


class CompiledHypothesisMakerBackend:
    """One-shot backend-compatible adapter without bounded repair."""

    def __init__(
        self,
        draft_backend: HypothesisDraftBackend,
        *,
        prompt_assembler: HypothesisPromptAssembler | None = None,
        compiler: HypothesisCompiler | None = None,
    ) -> None:
        self.draft_backend = draft_backend
        self.prompt_assembler = prompt_assembler or HypothesisPromptAssembler()
        self.compiler = compiler or HypothesisCompiler()

    def propose(self, context: HypothesisContext) -> HypothesisPortfolio:
        prompt = self.prompt_assembler.build(context)
        generation = self.draft_backend.generate(prompt)
        return self.compiler.compile(context, generation.draft)


class HypothesisMakerAgentRuntime:
    def __init__(
        self,
        draft_backend: HypothesisDraftBackend,
        *,
        prompt_assembler: HypothesisPromptAssembler | None = None,
        compiler: HypothesisCompiler | None = None,
        validator: HypothesisValidator | None = None,
        max_repairs: int = 1,
    ) -> None:
        if max_repairs not in {0, 1}:
            raise ValueError("Hypothesis Maker v2.6.1 supports max_repairs of 0 or 1 only.")
        self.draft_backend = draft_backend
        self.prompt_assembler = prompt_assembler or HypothesisPromptAssembler()
        self.compiler = compiler or HypothesisCompiler()
        self.validator = validator or HypothesisValidator()
        self.max_repairs = int(max_repairs)

    def run(self, context: HypothesisContext) -> HypothesisMakerRunOutcome:
        prompt = self.prompt_assembler.build(context)
        started = time.perf_counter()
        generations: list[HypothesisDraftGeneration] = []
        drafts: list[HypothesisPortfolioDraft] = []
        compile_issues: list[HypothesisCompileIssue] = []
        last_portfolio: HypothesisPortfolio | None = None
        validation: HypothesisValidationResult | None = None
        failure_stage = "none"

        try:
            generation = self.draft_backend.generate(prompt)
        except Exception:
            # Preserve the Explorer runtime convention: generation failures are
            # fail-closed and re-raised to the caller/CLI.
            raise

        generations.append(generation)
        drafts.append(generation.draft)
        current_draft = generation.draft
        repair_attempts = 0

        for cycle in range(self.max_repairs + 1):
            try:
                last_portfolio = self.compiler.compile(context, current_draft)
                compile_issues = []
            except HypothesisCompileError as exc:
                last_portfolio = None
                validation = None
                compile_issues = list(exc.issues)
                failure_stage = "compile"
                if cycle >= self.max_repairs:
                    break
                feedback = self.prompt_assembler.repair_feedback(
                    previous_draft=current_draft,
                    issues=compile_issues,
                )
                repaired = self.draft_backend.repair(
                    prompt,
                    current_draft,
                    feedback,
                )
                generations.append(repaired)
                drafts.append(repaired.draft)
                current_draft = repaired.draft
                repair_attempts += 1
                continue

            validation = self.validator.validate(context, last_portfolio)
            if validation.passes:
                failure_stage = "none"
                break

            failure_stage = "validation"
            if cycle >= self.max_repairs:
                break
            feedback = self.prompt_assembler.repair_feedback(
                previous_draft=current_draft,
                issues=validation.issues,
            )
            repaired = self.draft_backend.repair(
                prompt,
                current_draft,
                feedback,
            )
            generations.append(repaired)
            drafts.append(repaired.draft)
            current_draft = repaired.draft
            repair_attempts += 1

        accepted_portfolio = (
            last_portfolio
            if validation is not None and validation.passes
            else None
        )
        elapsed = time.perf_counter() - started
        record = self._run_record(
            context=context,
            prompt=prompt,
            portfolio=last_portfolio,
            validation=validation,
            generations=generations,
            repair_attempts=repair_attempts,
            compile_issues=compile_issues,
            failure_stage=("none" if accepted_portfolio is not None else failure_stage),
            elapsed_seconds=elapsed,
        )
        return HypothesisMakerRunOutcome(
            accepted_portfolio=accepted_portfolio,
            last_portfolio=last_portfolio,
            validation=validation,
            final_draft=current_draft,
            draft_history=tuple(drafts),
            compile_issues=tuple(compile_issues),
            run_record=record,
            prompt=prompt,
        )

    def _run_record(
        self,
        *,
        context: HypothesisContext,
        prompt: HypothesisPrompt,
        portfolio: HypothesisPortfolio | None,
        validation: HypothesisValidationResult | None,
        generations: list[HypothesisDraftGeneration],
        repair_attempts: int,
        compile_issues: list[HypothesisCompileIssue],
        failure_stage: str,
        elapsed_seconds: float,
    ) -> HypothesisMakerRunRecord:
        portfolio_sha = _sha256_json(portfolio) if portfolio is not None else None
        input_token_values = [g.input_tokens for g in generations if g.input_tokens is not None]
        output_token_values = [g.output_tokens for g in generations if g.output_tokens is not None]

        backend_name = str(
            getattr(self.draft_backend, "backend_name", type(self.draft_backend).__name__)
        )
        model_name = str(getattr(self.draft_backend, "model_name", "unknown"))
        temperature_raw = getattr(self.draft_backend, "temperature", None)
        temperature = float(temperature_raw) if temperature_raw is not None else None

        run_id = _stable_id(
            "hypothesis_run",
            context.context_sha256,
            prompt.prompt_sha256,
            backend_name,
            model_name,
            portfolio_sha or "rejected",
            len(generations),
        )

        return HypothesisMakerRunRecord(
            run_id=run_id,
            context_id=context.context_id,
            context_sha256=context.context_sha256,
            source_packet_id=context.source_packet_id,
            source_packet_sha256=context.source_packet_sha256,
            source_report_id=context.source_report_id,
            source_report_sha256=context.source_report_sha256,
            portfolio_id=(portfolio.portfolio_id if portfolio is not None else None),
            portfolio_sha256=portfolio_sha,
            prompt_version=prompt.prompt_version,
            prompt_sha256=prompt.prompt_sha256,
            backend=backend_name,
            model=model_name,
            generation_attempts=len(generations),
            repair_attempts=repair_attempts,
            final_validation_passed=bool(validation and validation.passes),
            failure_stage=failure_stage,  # type: ignore[arg-type]
            validation_errors=(validation.errors if validation is not None else 0),
            validation_warnings=(validation.warnings if validation is not None else 0),
            compile_issue_count=len(compile_issues),
            input_tokens=(sum(input_token_values) if input_token_values else None),
            output_tokens=(sum(output_token_values) if output_token_values else None),
            elapsed_seconds=elapsed_seconds,
            temperature=temperature,
            backend_mode=(str(getattr(self.draft_backend, "instructor_mode", "")) or None),
            base_url=(str(getattr(self.draft_backend, "base_url", "")) or None),
            parse_retries=(
                int(getattr(self.draft_backend, "parse_retries"))
                if getattr(self.draft_backend, "parse_retries", None) is not None
                else None
            ),
            max_repairs=self.max_repairs,
        )
