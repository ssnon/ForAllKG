from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from pipeline_core.discovery.frontier_exploration_audit import FrontierExplorationAudit
from pipeline_core.discovery.frontier_idea_population import FrontierIdeaPopulation
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.idea_evolution import IdeaEvolutionReport
from pipeline_core.discovery.scientific_portfolio_materialization import (
    ScientificPortfolioMaterializationBatchDraft,
    ScientificPortfolioMaterializationReport,
    compile_scientific_portfolio_materialization,
)
from pipeline_core.discovery.scientific_portfolio_prompt import (
    ScientificPortfolioPrompt,
    build_evaluation_prompt,
    build_materialization_prompt,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
    ScientificPortfolioEvaluationBatchDraft,
    ScientificPortfolioEvaluationReport,
    ScientificPortfolioSelectionReport,
    build_scientific_portfolio_candidate_pool,
    build_scientific_portfolio_selection,
    compile_scientific_portfolio_evaluation,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


@dataclass(frozen=True)
class ScientificPortfolioEvaluationGeneration:
    draft: ScientificPortfolioEvaluationBatchDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class ScientificPortfolioMaterializationGeneration:
    draft: ScientificPortfolioMaterializationBatchDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class ScientificPortfolioBackend(Protocol):
    backend_name: str
    model_name: str

    def evaluate(
        self, prompt: ScientificPortfolioPrompt
    ) -> ScientificPortfolioEvaluationGeneration: ...

    def materialize(
        self, prompt: ScientificPortfolioPrompt
    ) -> ScientificPortfolioMaterializationGeneration: ...


class InstructorOpenAICompatibleScientificPortfolioBackend:
    backend_name = "instructor_openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 1,
        timeout: float | None = 180.0,
        extra_headers: Mapping[str, str] | None = None,
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key_env = str(api_key_env)
        self.api_key = os.getenv(self.api_key_env)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(telemetry_context or {})
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(f"No API key available. Set {self.api_key_env}.")
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Scientific Portfolio generation requires installed openai and instructor."
            ) from exc
        mode = getattr(instructor.Mode, self.instructor_mode, None)
        if mode is None:
            raise ValueError(f"Unknown Instructor mode: {self.instructor_mode}")
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers
        self._client = instructor.from_openai(OpenAI(**kwargs), mode=mode)
        return self._client

    def _call(self, *, prompt: ScientificPortfolioPrompt, response_model, stage: str):
        draft, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=response_model,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "scientific_portfolio_selection_shadow",
                "stage": stage,
            },
            semantic_components={
                "authority": "SHADOW_PORTFOLIO_ONLY",
                "truth_authority": False,
                "novelty_authority": False,
            },
        )
        if not isinstance(draft, response_model):
            draft = response_model.model_validate(draft)
        return draft, event

    def evaluate(
        self, prompt: ScientificPortfolioPrompt
    ) -> ScientificPortfolioEvaluationGeneration:
        draft, event = self._call(
            prompt=prompt,
            response_model=ScientificPortfolioEvaluationBatchDraft,
            stage="profile_evaluation",
        )
        return ScientificPortfolioEvaluationGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=event.elapsed_seconds,
        )

    def materialize(
        self, prompt: ScientificPortfolioPrompt
    ) -> ScientificPortfolioMaterializationGeneration:
        draft, event = self._call(
            prompt=prompt,
            response_model=ScientificPortfolioMaterializationBatchDraft,
            stage="grounded_hypothesis_materialization",
        )
        return ScientificPortfolioMaterializationGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=event.elapsed_seconds,
        )


@dataclass(frozen=True)
class ScientificPortfolioOutcome:
    pool: ScientificPortfolioCandidatePool
    evaluation: ScientificPortfolioEvaluationReport
    selection: ScientificPortfolioSelectionReport
    materialization_draft: ScientificPortfolioMaterializationBatchDraft
    materialization_report: ScientificPortfolioMaterializationReport
    materialized_portfolio: HypothesisPortfolio
    evaluation_prompt: ScientificPortfolioPrompt
    materialization_prompt: ScientificPortfolioPrompt | None
    materialization_generation_error: str | None
    llm_calls_performed: int


class ScientificPortfolioRuntime:
    def __init__(self, backend: ScientificPortfolioBackend) -> None:
        self.backend = backend

    def run(
        self,
        *,
        context: HypothesisContext,
        population: FrontierIdeaPopulation,
        exploration_audit: FrontierExplorationAudit,
        evolution_report: IdeaEvolutionReport,
        task_source: str,
        task_target: str,
        max_evaluation_candidates: int = 48,
        max_retained_candidates: int = 8,
        max_retained_per_profile: int = 2,
    ) -> ScientificPortfolioOutcome:
        if context.context_id != population.source_context_id:
            raise ValueError("context/Frontier source context mismatch")
        if context.context_sha256 != population.source_context_sha256:
            raise ValueError("context/Frontier source context SHA mismatch")

        pool = build_scientific_portfolio_candidate_pool(
            population=population,
            exploration_audit=exploration_audit,
            evolution_report=evolution_report,
            max_evaluation_candidates=max_evaluation_candidates,
        )
        eval_prompt = build_evaluation_prompt(
            pool=pool,
            task_source=task_source,
            task_target=task_target,
        )
        eval_generation = self.backend.evaluate(eval_prompt)
        evaluation = compile_scientific_portfolio_evaluation(
            pool=pool,
            draft=eval_generation.draft,
            evaluator_model=self.backend.model_name,
        )
        selection = build_scientific_portfolio_selection(
            pool=pool,
            evaluation=evaluation,
            max_retained_candidates=max_retained_candidates,
            max_retained_per_profile=max_retained_per_profile,
        )

        calls = 1
        materialization_generation_error: str | None = None
        if selection.retained_count:
            materialization_prompt = build_materialization_prompt(
                context=context,
                pool=pool,
                evaluation=evaluation,
                selection=selection,
            )
            calls += 1
            try:
                materialization_generation = self.backend.materialize(
                    materialization_prompt
                )
                materialization_draft = materialization_generation.draft
            except Exception as exc:
                # Selection is a distinct exploration-lane result and must not
                # disappear because one structured materialization call failed.
                # Preserve the exploration portfolio and record the failed
                # promotion attempt explicitly.
                materialization_generation_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                materialization_draft = (
                    ScientificPortfolioMaterializationBatchDraft()
                )
        else:
            materialization_prompt = None
            materialization_draft = ScientificPortfolioMaterializationBatchDraft()

        materialization_report, portfolio = compile_scientific_portfolio_materialization(
            context=context,
            pool=pool,
            selection=selection,
            draft=materialization_draft,
            generation_error=materialization_generation_error,
        )
        return ScientificPortfolioOutcome(
            pool=pool,
            evaluation=evaluation,
            selection=selection,
            materialization_draft=materialization_draft,
            materialization_report=materialization_report,
            materialized_portfolio=portfolio,
            evaluation_prompt=eval_prompt,
            materialization_prompt=materialization_prompt,
            materialization_generation_error=materialization_generation_error,
            llm_calls_performed=calls,
        )


__all__ = [
    "ScientificPortfolioBackend",
    "InstructorOpenAICompatibleScientificPortfolioBackend",
    "ScientificPortfolioOutcome",
    "ScientificPortfolioRuntime",
]
