from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from dac_her.hypothesis_trend_compiler import (
    TrendHypothesisCompileError,
    TrendHypothesisCompileIssue,
)
from dac_her.hypothesis_trend_contract_hardened_compiler import (
    ContractHardenedTrendHypothesisCompiler,
)
from dac_her.hypothesis_trend_contract_hardened_contracts import (
    ContractHardenedTrendHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_contract_hardened_exposure import (
    ContractHardenedTrendMakerExposure,
    build_contract_hardened_trend_maker_exposure,
)
from dac_her.hypothesis_trend_contract_hardened_llm import (
    ContractHardenedTrendHypothesisDraftBackend,
    ContractHardenedTrendHypothesisDraftGeneration,
)
from dac_her.hypothesis_trend_contract_hardened_prompt import (
    ContractHardenedTrendHypothesisPrompt,
    ContractHardenedTrendHypothesisPromptAssembler,
)
from dac_her.hypothesis_trend_contract_hardened_renderer import (
    HYPOTHESIS_TREND_HARDENED_RENDERER_SEMANTICS_ID,
)
from dac_her.hypothesis_trend_contract_hardened_run_record import (
    HYPOTHESIS_TREND_HARDENED_RUNTIME_SEMANTICS_ID,
    ContractHardenedTrendHypothesisMakerRunRecord,
)
from dac_her.hypothesis_trend_contract_hardened_validator import (
    ContractHardenedTrendHypothesisValidator,
)
from dac_her.hypothesis_trend_directional_contracts import (
    DirectionAwareTrendHypothesisPortfolio,
)
from dac_her.hypothesis_trend_input import (
    TrendAwareHypothesisInput,
    verify_trend_aware_input_sources,
)
from dac_her.hypothesis_trend_validator import (
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
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


@dataclass(frozen=True)
class ContractHardenedTrendHypothesisMakerRunOutcome:
    accepted_portfolio: DirectionAwareTrendHypothesisPortfolio | None
    last_portfolio: DirectionAwareTrendHypothesisPortfolio | None
    validation: TrendHypothesisValidationResult | None
    final_draft: ContractHardenedTrendHypothesisPortfolioDraft | None
    draft_history: tuple[
        ContractHardenedTrendHypothesisPortfolioDraft, ...
    ]
    compile_issues: tuple[TrendHypothesisCompileIssue, ...]
    run_record: ContractHardenedTrendHypothesisMakerRunRecord
    prompt: ContractHardenedTrendHypothesisPrompt
    exposure: ContractHardenedTrendMakerExposure

    @property
    def accepted(self) -> bool:
        return (
            self.accepted_portfolio is not None
            and self.run_record.final_validation_passed
        )


class ContractHardenedTrendHypothesisMakerAgentRuntime:
    semantics_id = HYPOTHESIS_TREND_HARDENED_RUNTIME_SEMANTICS_ID

    def __init__(
        self,
        draft_backend: ContractHardenedTrendHypothesisDraftBackend,
        *,
        prompt_assembler:
            ContractHardenedTrendHypothesisPromptAssembler | None = None,
        compiler:
            ContractHardenedTrendHypothesisCompiler | None = None,
        validator:
            ContractHardenedTrendHypothesisValidator | None = None,
        max_repairs: int = 1,
    ) -> None:
        if max_repairs not in {0, 1}:
            raise ValueError(
                "alpha4c.5i supports max_repairs of 0 or 1 only."
            )
        self.draft_backend = draft_backend
        self.prompt_assembler = (
            prompt_assembler
            or ContractHardenedTrendHypothesisPromptAssembler()
        )
        self.compiler = (
            compiler or ContractHardenedTrendHypothesisCompiler()
        )
        self.validator = (
            validator or ContractHardenedTrendHypothesisValidator()
        )
        self.max_repairs = int(max_repairs)

    def run(
        self,
        source: TrendAwareHypothesisInput,
    ) -> ContractHardenedTrendHypothesisMakerRunOutcome:
        verify_trend_aware_input_sources(source)
        exposure = build_contract_hardened_trend_maker_exposure(
            source
        )
        prompt = self.prompt_assembler.build(
            source,
            exposure=exposure,
        )

        started = time.perf_counter()
        generations: list[
            ContractHardenedTrendHypothesisDraftGeneration
        ] = []
        drafts: list[
            ContractHardenedTrendHypothesisPortfolioDraft
        ] = []
        compile_issues: list[TrendHypothesisCompileIssue] = []
        last_portfolio = None
        validation = None
        failure_stage = "none"

        generation = self.draft_backend.generate(prompt)
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

        accepted = (
            last_portfolio
            if validation is not None and validation.passes
            else None
        )
        elapsed = time.perf_counter() - started
        record = self._record(
            source=source,
            exposure=exposure,
            prompt=prompt,
            portfolio=last_portfolio,
            validation=validation,
            generations=generations,
            repair_attempts=repair_attempts,
            compile_issues=compile_issues,
            failure_stage=(
                "none" if accepted is not None else failure_stage
            ),
            elapsed_seconds=elapsed,
        )
        return ContractHardenedTrendHypothesisMakerRunOutcome(
            accepted_portfolio=accepted,
            last_portfolio=last_portfolio,
            validation=validation,
            final_draft=current_draft,
            draft_history=tuple(drafts),
            compile_issues=tuple(compile_issues),
            run_record=record,
            prompt=prompt,
            exposure=exposure,
        )

    def _record(
        self,
        *,
        source: TrendAwareHypothesisInput,
        exposure: ContractHardenedTrendMakerExposure,
        prompt: ContractHardenedTrendHypothesisPrompt,
        portfolio: DirectionAwareTrendHypothesisPortfolio | None,
        validation: TrendHypothesisValidationResult | None,
        generations: list[
            ContractHardenedTrendHypothesisDraftGeneration
        ],
        repair_attempts: int,
        compile_issues: list[TrendHypothesisCompileIssue],
        failure_stage: str,
        elapsed_seconds: float,
    ) -> ContractHardenedTrendHypothesisMakerRunRecord:
        portfolio_sha = (
            _sha256_json(portfolio)
            if portfolio is not None
            else None
        )
        input_tokens = [
            row.input_tokens
            for row in generations
            if row.input_tokens is not None
        ]
        output_tokens = [
            row.output_tokens
            for row in generations
            if row.output_tokens is not None
        ]

        return ContractHardenedTrendHypothesisMakerRunRecord(
            runtime_semantics_id=self.semantics_id,
            run_id=_stable_id(
                "contract_hardened_trend_hypothesis_run",
                source.input_sha256,
                exposure.exposure_sha256,
                prompt.prompt_sha256,
                str(
                    getattr(
                        self.draft_backend,
                        "model_name",
                        "unknown",
                    )
                ),
                len(generations),
                repair_attempts,
                portfolio_sha or failure_stage,
            ),
            context_id=source.grounded_context.context_id,
            context_sha256=source.grounded_context.context_sha256,
            source_report_id=source.grounded_context.source_report_id,
            source_report_sha256=
                source.grounded_context.source_report_sha256,
            source_trend_input_id=source.input_id,
            source_trend_input_sha256=source.input_sha256,
            source_directional_exposure_id=
                exposure.source_directional_exposure_id,
            source_directional_exposure_sha256=
                exposure.source_directional_exposure_sha256,
            hardened_exposure_id=exposure.exposure_id,
            hardened_exposure_sha256=exposure.exposure_sha256,
            portfolio_id=(
                portfolio.portfolio_id
                if portfolio is not None
                else None
            ),
            portfolio_sha256=portfolio_sha,
            hardened_compiler_semantics_id=self.compiler.semantics_id,
            hardened_validator_semantics_id=self.validator.semantics_id,
            renderer_semantics_id=
                HYPOTHESIS_TREND_HARDENED_RENDERER_SEMANTICS_ID,
            downstream_directional_compiler_semantics_id=(
                self.compiler.directional_compiler.semantics_id
            ),
            prompt_version=prompt.prompt_version,
            prompt_sha256=prompt.prompt_sha256,
            backend=str(
                getattr(
                    self.draft_backend,
                    "backend_name",
                    type(self.draft_backend).__name__,
                )
            ),
            model=str(
                getattr(
                    self.draft_backend,
                    "model_name",
                    "unknown",
                )
            ),
            generation_attempts=len(generations),
            repair_attempts=repair_attempts,
            final_validation_passed=bool(
                validation is not None and validation.passes
            ),
            failure_stage=failure_stage,  # type: ignore[arg-type]
            validation_errors=(
                validation.errors
                if validation is not None
                else 0
            ),
            validation_warnings=(
                validation.warnings
                if validation is not None
                else 0
            ),
            compile_issue_count=len(compile_issues),
            input_tokens=(
                sum(input_tokens) if input_tokens else None
            ),
            output_tokens=(
                sum(output_tokens) if output_tokens else None
            ),
            elapsed_seconds=elapsed_seconds,
            temperature=getattr(
                self.draft_backend,
                "temperature",
                None,
            ),
            backend_mode=getattr(
                self.draft_backend,
                "instructor_mode",
                None,
            ),
            base_url=getattr(
                self.draft_backend,
                "base_url",
                None,
            ),
            parse_retries=getattr(
                self.draft_backend,
                "parse_retries",
                None,
            ),
            max_repairs=self.max_repairs,
        )
