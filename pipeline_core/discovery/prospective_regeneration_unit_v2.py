from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_compiler import (
    HypothesisCompileError,
    HypothesisCompiler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftBackend
from pipeline_core.discovery.hypothesis_prompt import (
    PROMPT_VERSION,
    HypothesisPromptAssembler,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


class ProspectiveRegenerationUnitV2Policy(StrictModel):
    schema_version: Literal[
        "prospective-regeneration-unit-v2-policy"
    ] = "prospective-regeneration-unit-v2-policy"

    semantic_name: Literal[
        "single_structured_hypothesis_generation_then_deterministic_compile"
    ] = "single_structured_hypothesis_generation_then_deterministic_compile"

    input_context_authority: Literal[
        "frozen_hypothesis_context_only"
    ] = "frozen_hypothesis_context_only"
    previous_hypothesis_text_allowed_as_input: Literal[False] = False
    novelty_outcome_allowed_as_input: Literal[False] = False
    verifier_outcome_allowed_as_input: Literal[False] = False
    external_novelty_result_allowed_as_input: Literal[False] = False

    prompt_version: str = PROMPT_VERSION
    max_hypotheses_per_generation_call: Literal[1] = 1

    structured_generation_calls_per_hypothesis_max: Literal[1] = 1
    structured_repair_calls_allowed: Literal[0] = 0
    generation_backend_method: Literal[
        "HypothesisDraftBackend.generate"
    ] = "HypothesisDraftBackend.generate"

    deterministic_compiler: Literal[
        "HypothesisCompiler.compile"
    ] = "HypothesisCompiler.compile"
    deterministic_compile_attempts_max: Literal[1] = 1

    retrieval_inside_regeneration_allowed: Literal[False] = False
    semantic_critic_inside_regeneration_allowed: Literal[False] = False
    external_novelty_inside_regeneration_allowed: Literal[False] = False
    n9_inside_regeneration_allowed: Literal[False] = False
    n10_inside_regeneration_allowed: Literal[False] = False
    endpoint_binding_inside_regeneration_allowed: Literal[False] = False
    verifier_inside_regeneration_allowed: Literal[False] = False

    downstream_evaluation_is_separate_stage: Literal[True] = True
    downstream_evaluation_may_not_mutate_regenerated_portfolio: Literal[
        True
    ] = True
    downstream_evaluation_requires_separately_frozen_budget: Literal[
        True
    ] = True

    new_lineage_required: Literal[True] = True
    full_e2e_rerun_is_regeneration: Literal[False] = False
    full_e2e_rerun_forbidden_inside_regeneration_unit: Literal[True] = True

    @model_validator(mode="after")
    def validate_policy(self) -> "ProspectiveRegenerationUnitV2Policy":
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(
                "regeneration-unit prompt version must match "
                "the repository Hypothesis Maker prompt version"
            )
        return self


RegenerationUnitStatus = Literal[
    "GENERATED_AND_COMPILED",
    "GENERATED_ABSTENTION_AND_COMPILED",
    "DETERMINISTIC_COMPILE_REJECTED",
    "GENERATION_CALL_FAILED",
]


class ProspectiveRegenerationUnitV2Result(StrictModel):
    schema_version: Literal[
        "prospective-regeneration-unit-v2-result"
    ] = "prospective-regeneration-unit-v2-result"

    result_id: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_context_id: str
    source_context_sha256: str
    prompt_version: str
    prompt_sha256: str

    backend_name: str
    model_name: str

    status: RegenerationUnitStatus
    generated_draft: HypothesisPortfolioDraft | None = None
    compiled_portfolio: HypothesisPortfolio | None = None
    compile_issue_codes: list[str] = Field(default_factory=list)
    failure_type: str | None = None
    failure_message: str | None = None

    generation_calls_attempted: Literal[1] = 1
    repair_calls_attempted: Literal[0] = 0
    deterministic_compile_attempts: int = Field(ge=0, le=1)

    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    provider_response_id: str | None = None

    retrieval_performed: Literal[False] = False
    semantic_critic_performed: Literal[False] = False
    external_novelty_performed: Literal[False] = False
    n9_performed: Literal[False] = False
    n10_performed: Literal[False] = False
    endpoint_binding_performed: Literal[False] = False
    verifier_performed: Literal[False] = False
    downstream_evaluation_performed: Literal[False] = False

    previous_hypothesis_text_consumed: Literal[False] = False
    novelty_outcome_consumed: Literal[False] = False
    verifier_outcome_consumed: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "ProspectiveRegenerationUnitV2Result":
        if self.status in {
            "GENERATED_AND_COMPILED",
            "GENERATED_ABSTENTION_AND_COMPILED",
        }:
            if self.generated_draft is None or self.compiled_portfolio is None:
                raise ValueError(
                    "successful regeneration requires draft and portfolio"
                )
            if self.deterministic_compile_attempts != 1:
                raise ValueError(
                    "successful regeneration requires one deterministic compile"
                )
        if self.status == "DETERMINISTIC_COMPILE_REJECTED":
            if self.generated_draft is None:
                raise ValueError(
                    "compile rejection must preserve generated draft"
                )
            if self.compiled_portfolio is not None:
                raise ValueError(
                    "compile rejection cannot contain compiled portfolio"
                )
            if not self.compile_issue_codes:
                raise ValueError(
                    "compile rejection requires deterministic issue codes"
                )
        if self.status == "GENERATION_CALL_FAILED":
            if self.generated_draft is not None:
                raise ValueError(
                    "failed generation cannot contain a generated draft"
                )
            if self.deterministic_compile_attempts != 0:
                raise ValueError(
                    "failed generation cannot attempt deterministic compile"
                )

        body = self.model_dump(mode="json")
        observed_id = body.pop("result_id")
        observed_sha = body.pop("result_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("regeneration-unit result SHA mismatch")
        if observed_id != (
            "prospective_regeneration_unit_v2_result:"
            + expected_sha[:20]
        ):
            raise ValueError("regeneration-unit result ID mismatch")
        return self


class ProspectiveRegenerationUnitV2Freeze(StrictModel):
    schema_version: Literal[
        "prospective-regeneration-unit-v2-freeze"
    ] = "prospective-regeneration-unit-v2-freeze"

    freeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository_head_sha: str
    repository_tracked_worktree_dirty: Literal[False] = False
    policy: ProspectiveRegenerationUnitV2Policy

    frozen_before_new_cohort_generation: Literal[True] = True
    prior_s135_scientific_outputs_used_to_change_generation_content: Literal[
        False
    ] = False
    prior_s135_protocol_failure_used_to_correct_execution_contract: Literal[
        True
    ] = True

    full_e2e_rerun_redefinition_allowed_after_freeze: Literal[False] = False
    downstream_budget_must_be_frozen_separately: Literal[True] = True
    production_selection_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(self) -> "ProspectiveRegenerationUnitV2Freeze":
        body = self.model_dump(mode="json")
        observed_id = body.pop("freeze_id")
        observed_sha = body.pop("freeze_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("regeneration-unit freeze SHA mismatch")
        if observed_id != (
            "prospective_regeneration_unit_v2_freeze:"
            + expected_sha[:20]
        ):
            raise ValueError("regeneration-unit freeze ID mismatch")
        return self


def build_regeneration_unit_v2_freeze(
    *,
    repository_head_sha: str,
    repository_tracked_worktree_dirty: bool,
) -> ProspectiveRegenerationUnitV2Freeze:
    if repository_tracked_worktree_dirty:
        raise ValueError(
            "regeneration-unit freeze requires a clean tracked worktree"
        )

    body = {
        "schema_version": "prospective-regeneration-unit-v2-freeze",
        "repository_head_sha": repository_head_sha,
        "repository_tracked_worktree_dirty": False,
        "policy": ProspectiveRegenerationUnitV2Policy().model_dump(
            mode="json"
        ),
        "frozen_before_new_cohort_generation": True,
        "prior_s135_scientific_outputs_used_to_change_generation_content":
            False,
        "prior_s135_protocol_failure_used_to_correct_execution_contract":
            True,
        "full_e2e_rerun_redefinition_allowed_after_freeze": False,
        "downstream_budget_must_be_frozen_separately": True,
        "production_selection_authority": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRegenerationUnitV2Freeze(
        **body,
        freeze_id=(
            "prospective_regeneration_unit_v2_freeze:"
            + digest[:20]
        ),
        freeze_sha256=digest,
    )


def execute_regeneration_generation_unit_v2(
    *,
    context: HypothesisContext,
    backend: HypothesisDraftBackend,
    policy: ProspectiveRegenerationUnitV2Policy | None = None,
) -> ProspectiveRegenerationUnitV2Result:
    resolved = policy or ProspectiveRegenerationUnitV2Policy()
    prompt = HypothesisPromptAssembler(
        max_hypotheses=resolved.max_hypotheses_per_generation_call
    ).build(context)

    try:
        generation = backend.generate(prompt)
    except Exception as exc:
        body = {
            "schema_version": "prospective-regeneration-unit-v2-result",
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "prompt_version": prompt.prompt_version,
            "prompt_sha256": prompt.prompt_sha256,
            "backend_name": str(getattr(backend, "backend_name", "unknown")),
            "model_name": str(getattr(backend, "model_name", "unknown")),
            "status": "GENERATION_CALL_FAILED",
            "generated_draft": None,
            "compiled_portfolio": None,
            "compile_issue_codes": [],
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "generation_calls_attempted": 1,
            "repair_calls_attempted": 0,
            "deterministic_compile_attempts": 0,
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "provider_response_id": None,
            "retrieval_performed": False,
            "semantic_critic_performed": False,
            "external_novelty_performed": False,
            "n9_performed": False,
            "n10_performed": False,
            "endpoint_binding_performed": False,
            "verifier_performed": False,
            "downstream_evaluation_performed": False,
            "previous_hypothesis_text_consumed": False,
            "novelty_outcome_consumed": False,
            "verifier_outcome_consumed": False,
        }
        digest = _sha256_json(body)
        return ProspectiveRegenerationUnitV2Result(
            **body,
            result_id=(
                "prospective_regeneration_unit_v2_result:"
                + digest[:20]
            ),
            result_sha256=digest,
        )

    draft = generation.draft
    try:
        portfolio = HypothesisCompiler().compile(context, draft)
    except HypothesisCompileError as exc:
        body = {
            "schema_version": "prospective-regeneration-unit-v2-result",
            "source_context_id": context.context_id,
            "source_context_sha256": context.context_sha256,
            "prompt_version": prompt.prompt_version,
            "prompt_sha256": prompt.prompt_sha256,
            "backend_name": str(getattr(backend, "backend_name", "unknown")),
            "model_name": str(getattr(backend, "model_name", "unknown")),
            "status": "DETERMINISTIC_COMPILE_REJECTED",
            "generated_draft": draft.model_dump(mode="json"),
            "compiled_portfolio": None,
            "compile_issue_codes": [
                issue.code for issue in exc.issues
            ],
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "generation_calls_attempted": 1,
            "repair_calls_attempted": 0,
            "deterministic_compile_attempts": 1,
            "provider_input_tokens": generation.input_tokens,
            "provider_output_tokens": generation.output_tokens,
            "provider_response_id": generation.response_id,
            "retrieval_performed": False,
            "semantic_critic_performed": False,
            "external_novelty_performed": False,
            "n9_performed": False,
            "n10_performed": False,
            "endpoint_binding_performed": False,
            "verifier_performed": False,
            "downstream_evaluation_performed": False,
            "previous_hypothesis_text_consumed": False,
            "novelty_outcome_consumed": False,
            "verifier_outcome_consumed": False,
        }
        digest = _sha256_json(body)
        return ProspectiveRegenerationUnitV2Result(
            **body,
            result_id=(
                "prospective_regeneration_unit_v2_result:"
                + digest[:20]
            ),
            result_sha256=digest,
        )

    status: RegenerationUnitStatus = (
        "GENERATED_AND_COMPILED"
        if portfolio.hypotheses
        else "GENERATED_ABSTENTION_AND_COMPILED"
    )
    body = {
        "schema_version": "prospective-regeneration-unit-v2-result",
        "source_context_id": context.context_id,
        "source_context_sha256": context.context_sha256,
        "prompt_version": prompt.prompt_version,
        "prompt_sha256": prompt.prompt_sha256,
        "backend_name": str(getattr(backend, "backend_name", "unknown")),
        "model_name": str(getattr(backend, "model_name", "unknown")),
        "status": status,
        "generated_draft": draft.model_dump(mode="json"),
        "compiled_portfolio": portfolio.model_dump(mode="json"),
        "compile_issue_codes": [],
        "failure_type": None,
        "failure_message": None,
        "generation_calls_attempted": 1,
        "repair_calls_attempted": 0,
        "deterministic_compile_attempts": 1,
        "provider_input_tokens": generation.input_tokens,
        "provider_output_tokens": generation.output_tokens,
        "provider_response_id": generation.response_id,
        "retrieval_performed": False,
        "semantic_critic_performed": False,
        "external_novelty_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "downstream_evaluation_performed": False,
        "previous_hypothesis_text_consumed": False,
        "novelty_outcome_consumed": False,
        "verifier_outcome_consumed": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRegenerationUnitV2Result(
        **body,
        result_id=(
            "prospective_regeneration_unit_v2_result:"
            + digest[:20]
        ),
        result_sha256=digest,
    )


__all__ = [
    "ProspectiveRegenerationUnitV2Freeze",
    "ProspectiveRegenerationUnitV2Policy",
    "ProspectiveRegenerationUnitV2Result",
    "RegenerationUnitStatus",
    "build_regeneration_unit_v2_freeze",
    "execute_regeneration_generation_unit_v2",
]
