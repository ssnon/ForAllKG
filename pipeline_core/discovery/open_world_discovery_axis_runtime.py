from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxisPlan,
)
from pipeline_core.discovery.dual_hypothesis_context import (
    DualHypothesisContext,
)
from pipeline_core.discovery.node_mapping import NodeMapper
from pipeline_core.discovery.open_world_discovery_axis import (
    ExternalAxisDraftBatch,
    ExternalAxisValidationResult,
    OpenWorldAxisPlanResult,
    OpenWorldRankedWork,
    OpenWorldRetrievalResult,
    build_external_axis_plan,
    build_external_axis_prompt_payload,
    build_outcome_blind_retrieval_seeds,
    retrieve_open_world_sources,
    select_open_world_abstracts,
    validate_external_axis_drafts,
)
from pipeline_core.discovery.prior_art_retrieval import (
    LiteratureSearchProvider,
)
from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)


OPEN_WORLD_AXIS_SYNTHESIS_PROMPT_VERSION = (
    "open-world-discovery-axis-synthesis-v1"
)

_SYSTEM_PROMPT = """You are the open-world discovery-axis synthesizer for an evidence-grounded
scientific hypothesis system.

You are NOT generating hypotheses and you are NOT deciding novelty.

Your task is to propose a small set of discovery directions that would be
useful for hypothesis generation but are not supplied by the current
persistent-KG discovery-axis cohort.

You may use only:
1. the frozen research question;
2. the frozen grounded positive-premise and gap statements supplied to you;
3. the retrieved paper titles and abstracts supplied to you;
4. the current control-axis signatures only for duplicate avoidance.

Epistemic rules:
- Retrieved literature is inspiration-only in this stage.
- A retrieved relation MUST NOT be promoted to a positive premise.
- Every proposed external axis must set requires_verification=true.
- Every scientific relation in an external axis must be supported by at least
  one supplied abstract span or be explicitly marked as a bounded synthesis
  between such spans. Do not use remembered literature or outside knowledge.
- Absence from the persistent KG is not novelty.
- Absence from retrieval is not novelty.
- Do not make literature-wide novelty claims.
- Do not infer causal direction merely because components co-occur.
- Do not create an axis only by paraphrasing a current control axis.
- Prefer axes that expose a relation, moderator, coupling, descriptor
  interaction, mechanistic competition, or regime distinction that could
  connect to at least one supplied grounded premise while expanding the
  current search space.

Return at most four axes. It is valid to return fewer or zero.

For every axis return:
- local_id
- label
- proposed_subject
- proposed_relation
- proposed_object
- source_work_ids
- source_evidence_spans: exact short substrings from supplied title/abstract
- compatible_grounded_statement_ids
- bounded_synthesis_note
- requires_verification (must be true)

Do not return a hypothesis, prediction, falsifier, experimental protocol, or
novelty verdict.""".strip()


def build_external_axis_messages(
    payload: Mapping[str, Any],
) -> list[dict[str, str]]:
    user_prompt = (
        "OPEN-WORLD DISCOVERY-AXIS INPUT\n"
        "================================\n"
        + json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n\nOUTPUT DISCIPLINE\n"
        "=================\n"
        "- Return ExternalAxisDraftBatch only.\n"
        "- Return at most four axes; zero is allowed.\n"
        "- Every source_work_id must be one of the supplied retrieved work IDs.\n"
        "- Every source_evidence_span must be an exact contiguous substring "
        "from the supplied title or abstract of a cited source work.\n"
        "- Every compatible_grounded_statement_id must be an eligible positive "
        "premise statement ID supplied above.\n"
        "- requires_verification must be true for every axis.\n"
        "- Do not use research-gap IDs as positive-premise IDs.\n"
        "- Do not generate hypotheses or novelty verdicts.\n"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


@dataclass(frozen=True)
class ExternalAxisDraftGeneration:
    draft: ExternalAxisDraftBatch
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class ExternalAxisDraftBackend(Protocol):
    backend_name: str
    model_name: str

    def generate(
        self,
        payload: Mapping[str, Any],
    ) -> ExternalAxisDraftGeneration: ...


class InstructorOpenAICompatibleExternalAxisBackend:
    """One-call structured external-axis synthesis via OpenAI-compatible API.

    Parse retry defaults to zero because the validated S17 1B1E treatment used
    exactly one structured synthesis call with temperature=0 and no Instructor
    retry. Scientific validation remains deterministic after this call.
    """

    backend_name = "instructor_openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 0,
        timeout: float | None = 180.0,
        extra_headers: dict[str, str] | None = None,
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(api_key_env)
        )
        self.api_key_env = api_key_env
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or None
        )
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        if self.parse_retries != 0:
            raise ValueError(
                "open-world axis synthesis parse_retries must remain 0"
            )
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(
            telemetry_context or {}
        )
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env} "
                "or pass api_key explicitly."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Open-world discovery-axis synthesis requires "
                "the installed 'openai' and 'instructor' packages."
            ) from exc

        mode = getattr(
            instructor.Mode,
            self.instructor_mode,
            None,
        )
        if mode is None:
            available = sorted(
                name
                for name in dir(instructor.Mode)
                if name.isupper()
            )
            raise ValueError(
                f"Unknown Instructor mode "
                f"{self.instructor_mode!r}. "
                f"Available modes include: {available}"
            )

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "max_retries": 0,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers

        raw_client = OpenAI(**kwargs)
        self._client = instructor.from_openai(
            raw_client,
            mode=mode,
        )
        return self._client

    def generate(
        self,
        payload: Mapping[str, Any],
    ) -> ExternalAxisDraftGeneration:
        context = {
            **self.telemetry_context,
            "pipeline": "open_world_discovery_axis",
            "stage": "axis_synthesis",
            "call_kind": "axis_synthesis",
            "prompt_version":
                OPEN_WORLD_AXIS_SYNTHESIS_PROMPT_VERSION,
        }
        draft, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=ExternalAxisDraftBatch,
            messages=build_external_axis_messages(payload),
            temperature=self.temperature,
            max_retries=0,
            telemetry_path=self.telemetry_path,
            telemetry_context=context,
            semantic_components={
                "source_kind":
                    "frozen_grounding_plus_retrieved_abstracts",
                "external_literature_authority":
                    "inspiration_only",
            },
        )
        if not isinstance(draft, ExternalAxisDraftBatch):
            draft = ExternalAxisDraftBatch.model_validate(
                draft
            )
        return ExternalAxisDraftGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=event.elapsed_seconds,
        )


class NodeMapperAxisSimilarity:
    """Cosine similarity adapter using the current NodeMapper encoder."""

    def __init__(self, mapper: NodeMapper) -> None:
        self.mapper = mapper
        self._cache: dict[str, np.ndarray] = {}

    def _vector(self, text: str) -> np.ndarray:
        if text not in self._cache:
            vector = np.asarray(
                self.mapper.encoder.encode_query(text),
                dtype=np.float32,
            )
            if vector.ndim != 1:
                raise ValueError(
                    "axis similarity encoder must return 1-D vectors"
                )
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise ValueError(
                    "axis similarity encoder returned zero vector"
                )
            self._cache[text] = vector / norm
        return self._cache[text]

    def __call__(self, left: str, right: str) -> float:
        return float(
            np.dot(
                self._vector(left),
                self._vector(right),
            )
        )


@dataclass(frozen=True)
class OpenWorldDiscoveryAxisOutcome:
    retrieval: OpenWorldRetrievalResult
    selected_works: tuple[OpenWorldRankedWork, ...]
    generation: ExternalAxisDraftGeneration
    validation: ExternalAxisValidationResult
    axis_plan: OpenWorldAxisPlanResult


class OpenWorldDiscoveryAxisRuntime:
    """Bounded open-world discovery lane that produces a DiscoveryAxisPlan.

    The runtime does not generate hypotheses and does not judge novelty. It
    consumes a frozen DualHypothesisContext plus the existing control axis plan,
    runs the fixed retrieval matrix, performs exactly one axis-synthesis call,
    validates exact source provenance, and rejects control-like axes before the
    ordinary DiscoveryAxisSynthesisRuntime consumes the resulting plan.

    Provider-plan resolution is intentionally outside this class. Callers must
    construct and freeze providers before invoking `run`.
    """

    def __init__(
        self,
        *,
        providers: list[LiteratureSearchProvider],
        backend: ExternalAxisDraftBackend,
        mapper: NodeMapper,
        results_per_query: int = 20,
        max_selected_works: int = 12,
        max_control_similarity: float = 0.85,
        max_source_span_words: int = 40,
    ) -> None:
        if not providers:
            raise ValueError(
                "at least one frozen literature provider is required"
            )
        if results_per_query <= 0:
            raise ValueError(
                "results_per_query must be positive"
            )
        if max_selected_works <= 0:
            raise ValueError(
                "max_selected_works must be positive"
            )
        if not 0.0 <= max_control_similarity <= 1.0:
            raise ValueError(
                "max_control_similarity must be within [0, 1]"
            )
        if max_source_span_words <= 0:
            raise ValueError(
                "max_source_span_words must be positive"
            )

        self.providers = list(providers)
        self.backend = backend
        self.mapper = mapper
        self.results_per_query = int(results_per_query)
        self.max_selected_works = int(
            max_selected_works
        )
        self.max_control_similarity = float(
            max_control_similarity
        )
        self.max_source_span_words = int(
            max_source_span_words
        )

    def run(
        self,
        *,
        dual: DualHypothesisContext,
        control_plan: DiscoveryAxisPlan,
    ) -> OpenWorldDiscoveryAxisOutcome:
        if (
            control_plan.source_dual_context_id
            != dual.dual_context_id
        ):
            raise ValueError(
                "control plan dual_context_id mismatch"
            )
        if (
            control_plan.source_dual_context_sha256
            != dual.dual_context_sha256
        ):
            raise ValueError(
                "control plan dual_context_sha256 mismatch"
            )

        seeds = build_outcome_blind_retrieval_seeds(
            dual
        )
        retrieval = retrieve_open_world_sources(
            seeds=seeds,
            providers=self.providers,
            results_per_query=self.results_per_query,
        )
        selected = select_open_world_abstracts(
            retrieval,
            max_selected=self.max_selected_works,
        )

        payload = build_external_axis_prompt_payload(
            dual=dual,
            control_plan=control_plan,
            selected_works=selected,
        )
        generation = self.backend.generate(payload)

        validation = validate_external_axis_drafts(
            batch=generation.draft,
            dual=dual,
            selected_works=selected,
            max_source_span_words=
                self.max_source_span_words,
        )

        axis_plan = build_external_axis_plan(
            dual=dual,
            control_plan=control_plan,
            selected_works=selected,
            validated_axes=validation.accepted_axes,
            similarity=NodeMapperAxisSimilarity(
                self.mapper
            ),
            max_control_similarity=
                self.max_control_similarity,
        )

        return OpenWorldDiscoveryAxisOutcome(
            retrieval=retrieval,
            selected_works=tuple(selected),
            generation=generation,
            validation=validation,
            axis_plan=axis_plan,
        )
