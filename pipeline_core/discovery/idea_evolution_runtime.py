from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from pipeline_core.discovery.frontier_exploration_audit import (
    FrontierExplorationAudit,
)
from pipeline_core.discovery.frontier_idea_population import (
    FrontierIdeaPopulation,
)
from pipeline_core.discovery.idea_evolution import (
    IdeaEvolutionBatchDraft,
    IdeaEvolutionIdea,
    IdeaEvolutionPlan,
    IdeaEvolutionReport,
    IdeaEvolutionRunRecord,
    NativeEvolutionOperatorId,
    build_idea_evolution_report,
    compile_native_evolution_draft,
    import_proxy_challenge_ideas,
    import_scientific_reframe_ideas,
    operator_plan_by_id,
)
from pipeline_core.discovery.idea_evolution_prompt import (
    IdeaEvolutionPrompt,
    build_idea_evolution_prompt,
)
from pipeline_core.discovery.reframing.proxy_challenge import (
    ProxyChallengeRunReport,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframingShadowReport,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


@dataclass(frozen=True)
class IdeaEvolutionGeneration:
    draft: IdeaEvolutionBatchDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class IdeaEvolutionBackend(Protocol):
    backend_name: str
    model_name: str

    def generate(self, prompt: IdeaEvolutionPrompt) -> IdeaEvolutionGeneration: ...


class InstructorOpenAICompatibleIdeaEvolutionBackend:
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
                "Idea Evolution generation requires installed openai and instructor."
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

    def generate(self, prompt: IdeaEvolutionPrompt) -> IdeaEvolutionGeneration:
        draft, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=IdeaEvolutionBatchDraft,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "idea_evolution_shadow",
                "stage": prompt.operator_id,
                "call_kind": "scientific_idea_evolution",
            },
            semantic_components={
                "idea_evolution_operator": prompt.operator_id,
                "authority": "INSPIRATION_ONLY",
            },
        )
        if not isinstance(draft, IdeaEvolutionBatchDraft):
            draft = IdeaEvolutionBatchDraft.model_validate(draft)
        if draft.operator_id != prompt.operator_id:
            raise ValueError("backend returned a batch for the wrong evolution operator")
        return IdeaEvolutionGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=event.elapsed_seconds,
        )


@dataclass(frozen=True)
class IdeaEvolutionOutcome:
    report: IdeaEvolutionReport
    prompts: tuple[IdeaEvolutionPrompt, ...]


class IdeaEvolutionRuntime:
    _ORDER: tuple[NativeEvolutionOperatorId, ...] = (
        "CROSS_SOURCE_BRIDGE",
        "BACKBONE_MUTATION",
        "CANDIDATE_INTERPRETATION",
    )

    def __init__(self, backend: IdeaEvolutionBackend) -> None:
        self.backend = backend

    def run(
        self,
        *,
        population: FrontierIdeaPopulation,
        exploration_audit: FrontierExplorationAudit,
        plan: IdeaEvolutionPlan,
        population_artifact: str,
        population_artifact_sha256: str,
        scientific_reframe_report: ScientificReframingShadowReport | None = None,
        scientific_reframe_artifact: str = "",
        scientific_reframe_artifact_sha256: str = "",
        proxy_challenge_report: ProxyChallengeRunReport | None = None,
        proxy_challenge_artifact: str = "",
        proxy_challenge_artifact_sha256: str = "",
    ) -> IdeaEvolutionOutcome:
        plans = operator_plan_by_id(plan)
        native_runs: list[IdeaEvolutionRunRecord] = []
        native_ideas: list[IdeaEvolutionIdea] = []
        prompts: list[IdeaEvolutionPrompt] = []

        for operator_id in self._ORDER:
            operator_plan = plans[operator_id]
            if not operator_plan.enabled:
                native_runs.append(
                    IdeaEvolutionRunRecord(
                        operator_id=operator_id,
                        decision="skipped_no_inputs",
                        parent_pool_idea_ids=list(operator_plan.parent_pool_idea_ids),
                    )
                )
                continue

            prompt = build_idea_evolution_prompt(
                population=population,
                operator_plan=operator_plan,
            )
            prompts.append(prompt)
            try:
                generation = self.backend.generate(prompt)
            except Exception as exc:
                native_runs.append(
                    IdeaEvolutionRunRecord(
                        operator_id=operator_id,
                        decision="generation_failed",
                        parent_pool_idea_ids=list(operator_plan.parent_pool_idea_ids),
                        generation_error_type=type(exc).__name__,
                        compile_issues=[str(exc)],
                    )
                )
                continue

            if not generation.draft.candidates:
                native_runs.append(
                    IdeaEvolutionRunRecord(
                        operator_id=operator_id,
                        decision="abstained",
                        parent_pool_idea_ids=list(operator_plan.parent_pool_idea_ids),
                        abstention_reason=generation.draft.abstention_reason,
                        input_tokens=generation.input_tokens,
                        output_tokens=generation.output_tokens,
                        response_id=generation.response_id,
                        elapsed_seconds=generation.elapsed_seconds,
                    )
                )
                continue

            accepted: list[IdeaEvolutionIdea] = []
            issues: list[str] = []
            for index, draft in enumerate(generation.draft.candidates, start=1):
                if index > operator_plan.max_output_count:
                    issues.append(
                        f"candidate {draft.local_id}: exceeds frozen max_output_count"
                    )
                    continue
                try:
                    accepted.append(
                        compile_native_evolution_draft(
                            draft=draft,
                            operator_plan=operator_plan,
                            population=population,
                            population_artifact=population_artifact,
                            population_artifact_sha256=population_artifact_sha256,
                        )
                    )
                except Exception as exc:
                    issues.append(
                        f"candidate {draft.local_id}: {type(exc).__name__}: {exc}"
                    )

            native_ideas.extend(accepted)
            native_runs.append(
                IdeaEvolutionRunRecord(
                    operator_id=operator_id,
                    decision=("generated" if accepted else "rejected_invalid_draft"),
                    parent_pool_idea_ids=list(operator_plan.parent_pool_idea_ids),
                    candidate_ids=[row.evolution_id for row in accepted],
                    rejected_candidate_count=(
                        len(generation.draft.candidates) - len(accepted)
                    ),
                    compile_issues=issues,
                    input_tokens=generation.input_tokens,
                    output_tokens=generation.output_tokens,
                    response_id=generation.response_id,
                    elapsed_seconds=generation.elapsed_seconds,
                )
            )

        imported_reframes: list[IdeaEvolutionIdea] = []
        if scientific_reframe_report is not None:
            imported_reframes = import_scientific_reframe_ideas(
                report=scientific_reframe_report,
                population=population,
                source_artifact=scientific_reframe_artifact,
                source_artifact_sha256=scientific_reframe_artifact_sha256,
            )

        imported_proxy: list[IdeaEvolutionIdea] = []
        if proxy_challenge_report is not None:
            imported_proxy = import_proxy_challenge_ideas(
                report=proxy_challenge_report,
                population=population,
                source_artifact=proxy_challenge_artifact,
                source_artifact_sha256=proxy_challenge_artifact_sha256,
            )

        report = build_idea_evolution_report(
            population=population,
            exploration_audit=exploration_audit,
            plan=plan,
            native_runs=native_runs,
            native_ideas=native_ideas,
            imported_reframe_ideas=imported_reframes,
            imported_proxy_ideas=imported_proxy,
        )
        return IdeaEvolutionOutcome(report=report, prompts=tuple(prompts))


__all__ = [
    "IdeaEvolutionBackend",
    "IdeaEvolutionGeneration",
    "IdeaEvolutionOutcome",
    "IdeaEvolutionRuntime",
    "InstructorOpenAICompatibleIdeaEvolutionBackend",
]
