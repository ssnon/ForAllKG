from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from dac_her.hypothesis_trend_compiler import (
    TrendAwareHypothesisCompiler,
    TrendHypothesisCompileError,
    TrendHypothesisCompileIssue,
)
from dac_her.hypothesis_trend_contracts import (
    TrendAwareHypothesisPortfolio,
    TrendAwareHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_input import (
    TrendAwareHypothesisInput,
    verify_trend_aware_input_sources,
)
from dac_her.hypothesis_trend_llm import (
    TrendAwareHypothesisDraftBackend,
    TrendAwareHypothesisDraftGeneration,
)
from dac_her.hypothesis_trend_maker_exposure import (
    TrendMakerExposure,
    build_trend_maker_exposure,
)
from dac_her.hypothesis_trend_prompt import (
    TrendAwareHypothesisPrompt,
    TrendAwareHypothesisPromptAssembler,
)
from dac_her.hypothesis_trend_run_record import (
    HYPOTHESIS_TREND_MAKER_RUNTIME_SEMANTICS_ID,
    TrendAwareHypothesisMakerRunRecord,
)
from dac_her.hypothesis_trend_validator import (
    TrendAwareHypothesisValidator,
    TrendHypothesisValidationResult,
)


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


@dataclass(frozen=True)
class TrendAwareHypothesisMakerRunOutcome:
    accepted_portfolio: TrendAwareHypothesisPortfolio | None
    last_portfolio: TrendAwareHypothesisPortfolio | None
    validation: TrendHypothesisValidationResult | None
    final_draft: TrendAwareHypothesisPortfolioDraft | None
    draft_history: tuple[TrendAwareHypothesisPortfolioDraft, ...]
    compile_issues: tuple[TrendHypothesisCompileIssue, ...]
    run_record: TrendAwareHypothesisMakerRunRecord
    prompt: TrendAwareHypothesisPrompt
    exposure: TrendMakerExposure

    @property
    def accepted(self) -> bool:
        return (
            self.accepted_portfolio is not None
            and self.run_record.final_validation_passed
        )


class TrendAwareHypothesisMakerAgentRuntime:
    semantics_id = HYPOTHESIS_TREND_MAKER_RUNTIME_SEMANTICS_ID

    def __init__(
        self,
        draft_backend: TrendAwareHypothesisDraftBackend,
        *,
        prompt_assembler: TrendAwareHypothesisPromptAssembler | None = None,
        compiler: TrendAwareHypothesisCompiler | None = None,
        validator: TrendAwareHypothesisValidator | None = None,
        max_repairs: int = 1,
    ) -> None:
        if max_repairs not in {0, 1}:
            raise ValueError(
                "alpha4c.5d supports max_repairs of 0 or 1 only."
            )
        self.draft_backend = draft_backend
        self.prompt_assembler = (
            prompt_assembler or TrendAwareHypothesisPromptAssembler()
        )
        self.compiler = compiler or TrendAwareHypothesisCompiler()
        self.validator = validator or TrendAwareHypothesisValidator()
        self.max_repairs = int(max_repairs)

    def run(
        self,
        source: TrendAwareHypothesisInput,
    ) -> TrendAwareHypothesisMakerRunOutcome:
        verify_trend_aware_input_sources(source)
        exposure = build_trend_maker_exposure(source)
        prompt = self.prompt_assembler.build(
            source,
            exposure=exposure,
        )

        started = time.perf_counter()
        generations: list[TrendAwareHypothesisDraftGeneration] = []
        drafts: list[TrendAwareHypothesisPortfolioDraft] = []
        compile_issues: list[TrendHypothesisCompileIssue] = []
        last_portfolio: TrendAwareHypothesisPortfolio | None = None
        validation: TrendHypothesisValidationResult | None = None
        failure_stage = "none"

        try:
            generation = self.draft_backend.generate(prompt)
        except Exception:
            raise

        generations.append(generation)
        drafts.append(generation.draft)
        current_draft = generation.draft
        repair_attempts = 0

        for cycle in range(self.max_repairs + 1):
            try:
                last_portfolio = self.compiler.compile(
                    source,
                    current_draft,
                )
                compile_issues = []
            except TrendHypothesisCompileError as exc:
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

            validation = self.validator.validate(
                source,
                last_portfolio,
            )
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
            source=source,
            exposure=exposure,
            prompt=prompt,
            portfolio=last_portfolio,
            validation=validation,
            generations=generations,
            repair_attempts=repair_attempts,
            compile_issues=compile_issues,
            failure_stage=(
                "none" if accepted_portfolio is not None else failure_stage
            ),
            elapsed_seconds=elapsed,
        )
        return TrendAwareHypothesisMakerRunOutcome(
            accepted_portfolio=accepted_portfolio,
            last_portfolio=last_portfolio,
            validation=validation,
            final_draft=current_draft,
            draft_history=tuple(drafts),
            compile_issues=tuple(compile_issues),
            run_record=record,
            prompt=prompt,
            exposure=exposure,
        )

    def _run_record(
        self,
        *,
        source: TrendAwareHypothesisInput,
        exposure: TrendMakerExposure,
        prompt: TrendAwareHypothesisPrompt,
        portfolio: TrendAwareHypothesisPortfolio | None,
        validation: TrendHypothesisValidationResult | None,
        generations: list[TrendAwareHypothesisDraftGeneration],
        repair_attempts: int,
        compile_issues: list[TrendHypothesisCompileIssue],
        failure_stage: str,
        elapsed_seconds: float,
    ) -> TrendAwareHypothesisMakerRunRecord:
        portfolio_sha = (
            _sha256_json(portfolio) if portfolio is not None else None
        )
        input_token_values = [
            row.input_tokens
            for row in generations
            if row.input_tokens is not None
        ]
        output_token_values = [
            row.output_tokens
            for row in generations
            if row.output_tokens is not None
        ]

        backend_name = str(
            getattr(
                self.draft_backend,
                "backend_name",
                type(self.draft_backend).__name__,
            )
        )
        model_name = str(
            getattr(self.draft_backend, "model_name", "unknown")
        )
        temperature_raw = getattr(
            self.draft_backend,
            "temperature",
            None,
        )
        temperature = (
            float(temperature_raw)
            if temperature_raw is not None
            else None
        )

        run_id = _stable_id(
            "trend_hypothesis_run",
            source.input_sha256,
            exposure.exposure_sha256,
            prompt.prompt_sha256,
            backend_name,
            model_name,
            portfolio_sha or "rejected",
            len(generations),
        )
        context = source.grounded_context

        return TrendAwareHypothesisMakerRunRecord(
            runtime_semantics_id=self.semantics_id,
            run_id=run_id,
            context_id=context.context_id,
            context_sha256=context.context_sha256,
            source_packet_id=context.source_packet_id,
            source_packet_sha256=context.source_packet_sha256,
            source_report_id=context.source_report_id,
            source_report_sha256=context.source_report_sha256,
            source_trend_input_id=source.input_id,
            source_trend_input_sha256=source.input_sha256,
            trend_exposure_id=exposure.exposure_id,
            trend_exposure_sha256=exposure.exposure_sha256,
            portfolio_id=(
                portfolio.portfolio_id if portfolio is not None else None
            ),
            portfolio_sha256=portfolio_sha,
            compiler_semantics_id=str(self.compiler.semantics_id),
            validator_semantics_id=str(self.validator.semantics_id),
            prompt_version=prompt.prompt_version,
            prompt_sha256=prompt.prompt_sha256,
            backend=backend_name,
            model=model_name,
            generation_attempts=len(generations),
            repair_attempts=repair_attempts,
            final_validation_passed=bool(
                validation and validation.passes
            ),
            failure_stage=failure_stage,  # type: ignore[arg-type]
            validation_errors=(
                validation.errors if validation is not None else 0
            ),
            validation_warnings=(
                validation.warnings if validation is not None else 0
            ),
            compile_issue_count=len(compile_issues),
            input_tokens=(
                sum(input_token_values) if input_token_values else None
            ),
            output_tokens=(
                sum(output_token_values) if output_token_values else None
            ),
            elapsed_seconds=elapsed_seconds,
            temperature=temperature,
            backend_mode=(
                str(getattr(self.draft_backend, "instructor_mode", ""))
                or None
            ),
            base_url=(
                str(getattr(self.draft_backend, "base_url", "")) or None
            ),
            parse_retries=(
                int(getattr(self.draft_backend, "parse_retries"))
                if getattr(
                    self.draft_backend,
                    "parse_retries",
                    None,
                ) is not None
                else None
            ),
            max_repairs=self.max_repairs,
        )
