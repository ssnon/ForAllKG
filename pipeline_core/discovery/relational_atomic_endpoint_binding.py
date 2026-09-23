from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingClaimPlan,
    RelationalAtomicBindingPlan,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


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


def _normalize(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐-‒–—−-]+", " ", text)
    text = re.sub(r"[^\w\s+*/().,]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _surface_contains(text: str, phrase: str) -> bool:
    needle = _normalize(phrase)
    haystack = _normalize(text)
    return bool(needle and needle in haystack)


_GENERIC_ENDPOINT_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "change",
        "changes",
        "changed",
        "changing",
        "condition",
        "conditions",
        "decrease",
        "decreases",
        "decreased",
        "depend",
        "depends",
        "dependence",
        "dependent",
        "determine",
        "determines",
        "difference",
        "differences",
        "differ",
        "differs",
        "effect",
        "effects",
        "for",
        "from",
        "greater",
        "higher",
        "in",
        "increase",
        "increases",
        "increased",
        "is",
        "less",
        "lower",
        "matched",
        "more",
        "of",
        "on",
        "or",
        "relation",
        "relationship",
        "same",
        "than",
        "the",
        "to",
        "under",
        "with",
        "within",
    }
)


def _endpoint_content_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _normalize(value).split()
        if token not in _GENERIC_ENDPOINT_TOKENS
        and len(token) >= 2
    )


def _validate_literal_span(
    *,
    claim: RelationalAtomicBindingClaimPlan,
    label: str,
    value: str,
) -> None:
    if not value.strip():
        raise ValueError(
            f"{claim.claim_id} {label} cannot be empty"
        )
    if not _surface_contains(claim.claim_text, value):
        raise ValueError(
            f"{claim.claim_id} {label} absent from claim text: {value}"
        )
    if not _surface_contains(claim.required_bridge, value):
        raise ValueError(
            f"{claim.claim_id} {label} absent from required bridge: {value}"
        )


class LiteralEndpointBindingDraft(StrictModel):
    claim_id: str = Field(min_length=1)
    relation_endpoint_anchors: list[str] = Field(default_factory=list)
    scope_qualifier_spans: list[str] = Field(default_factory=list)
    directional_qualifier_spans: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "LiteralEndpointBindingDraft":
        bound = bool(self.relation_endpoint_anchors)
        if bound:
            if len(self.relation_endpoint_anchors) < 2:
                raise ValueError(
                    "bound endpoint row requires at least two endpoints"
                )
            if self.abstention_reason is not None:
                raise ValueError(
                    "bound endpoint row cannot carry abstention_reason"
                )
        else:
            if not (self.abstention_reason or "").strip():
                raise ValueError(
                    "unbound endpoint row requires abstention_reason"
                )
            if self.scope_qualifier_spans or self.directional_qualifier_spans:
                raise ValueError(
                    "unbound endpoint row cannot carry qualifier spans"
                )

        for label, values in (
            ("relation_endpoint_anchors", self.relation_endpoint_anchors),
            ("scope_qualifier_spans", self.scope_qualifier_spans),
            ("directional_qualifier_spans", self.directional_qualifier_spans),
        ):
            normalized = [_normalize(value) for value in values]
            if any(not value for value in normalized):
                raise ValueError(f"{label} cannot contain empty values")
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{label} must be unique")
        return self


class LiteralEndpointBindingBatchDraft(StrictModel):
    schema_version: Literal[
        "relational-atomic-literal-endpoint-binding-batch-draft-v1"
    ] = "relational-atomic-literal-endpoint-binding-batch-draft-v1"

    bindings: list[LiteralEndpointBindingDraft] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_claim_ids(self) -> "LiteralEndpointBindingBatchDraft":
        ids = [row.claim_id for row in self.bindings]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate claim_id in endpoint binding batch")
        return self


BindingOutcome = Literal[
    "BOUND_LITERAL_ENDPOINTS",
    "ABSTAINED_UNBINDABLE",
]


class CompiledLiteralEndpointBinding(StrictModel):
    candidate_hypothesis_id: str
    final_hypothesis_id: str
    claim_id: str
    novelty_selection_role: str
    source_claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    relation_endpoint_anchors: list[str] = Field(default_factory=list)
    scope_qualifier_spans: list[str] = Field(default_factory=list)
    directional_qualifier_spans: list[str] = Field(default_factory=list)

    outcome: BindingOutcome
    abstention_reason: str | None = None

    claim_text_preserved: Literal[True] = True
    required_bridge_preserved: Literal[True] = True
    prediction_preserved: Literal[True] = True
    falsifier_preserved: Literal[True] = True
    identity_terms_preserved: Literal[True] = True
    novelty_selection_role_preserved: Literal[True] = True

    endpoint_spans_literal_in_claim: Literal[True] = True
    endpoint_spans_literal_in_required_bridge: Literal[True] = True
    endpoint_semantic_expansion_allowed: Literal[False] = False
    relation_nucleus_used_as_endpoint_authority: Literal[False] = False

    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_outcome(self) -> "CompiledLiteralEndpointBinding":
        if self.outcome == "BOUND_LITERAL_ENDPOINTS":
            if len(self.relation_endpoint_anchors) < 2:
                raise ValueError(
                    "bound endpoint result requires at least two endpoints"
                )
            if self.abstention_reason is not None:
                raise ValueError(
                    "bound endpoint result cannot carry abstention_reason"
                )
        else:
            if self.relation_endpoint_anchors:
                raise ValueError(
                    "abstained endpoint result cannot carry endpoints"
                )
            if not (self.abstention_reason or "").strip():
                raise ValueError(
                    "abstained endpoint result requires reason"
                )
        return self


class RelationalAtomicEndpointBindingReport(StrictModel):
    schema_version: Literal[
        "relational-atomic-endpoint-binding-report-v1"
    ] = "relational-atomic-endpoint-binding-report-v1"

    report_id: str
    source_binding_plan_id: str
    source_binding_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_name: str
    model_name: str

    bindings: list[CompiledLiteralEndpointBinding]
    selected_hypothesis_count: int = Field(ge=0)
    selected_claim_count: int = Field(ge=0)
    bound_claim_count: int = Field(ge=0)
    abstained_claim_count: int = Field(ge=0)
    novelty_bearing_bound_claim_count: int = Field(ge=0)

    llm_calls_performed: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    source_population_frozen_before_binding: Literal[True] = True
    only_binding_ready_claims_presented: Literal[True] = True
    only_binding_ready_hypotheses_presented: Literal[True] = True
    claim_content_mutated: Literal[False] = False
    scientific_content_added: Literal[False] = False
    endpoint_binding_only: Literal[True] = True
    verifier_result_observed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "RelationalAtomicEndpointBindingReport":
        if self.selected_claim_count != len(self.bindings):
            raise ValueError("selected_claim_count mismatch")
        bound = sum(
            row.outcome == "BOUND_LITERAL_ENDPOINTS"
            for row in self.bindings
        )
        if self.bound_claim_count != bound:
            raise ValueError("bound_claim_count mismatch")
        if self.abstained_claim_count != len(self.bindings) - bound:
            raise ValueError("abstained_claim_count mismatch")
        novelty_bound = sum(
            row.outcome == "BOUND_LITERAL_ENDPOINTS"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in self.bindings
        )
        if self.novelty_bearing_bound_claim_count != novelty_bound:
            raise ValueError(
                "novelty_bearing_bound_claim_count mismatch"
            )
        return self


@dataclass(frozen=True)
class EndpointBindingPrompt:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class EndpointBindingGeneration:
    draft: LiteralEndpointBindingBatchDraft
    input_tokens: int | None = None
    output_tokens: int | None = None


def selected_binding_claims(
    plan: RelationalAtomicBindingPlan,
) -> list[RelationalAtomicBindingClaimPlan]:
    rows: list[RelationalAtomicBindingClaimPlan] = []
    for hypothesis in plan.hypotheses:
        if (
            hypothesis.binding_status
            != "READY_FOR_LITERAL_ENDPOINT_BINDING"
        ):
            continue
        for claim in hypothesis.claims:
            if (
                claim.binding_status
                == "READY_FOR_LITERAL_ENDPOINT_BINDING"
            ):
                rows.append(claim)
    return rows


def build_endpoint_binding_prompt(
    plan: RelationalAtomicBindingPlan,
) -> EndpointBindingPrompt:
    claims = selected_binding_claims(plan)

    system = """You bind literal scientific relation endpoints for an already frozen atomic claim population.

This is annotation only. You MUST NOT create, rewrite, strengthen, weaken, combine, split, or reinterpret any scientific claim.

For every supplied claim_id, return exactly one binding row.

If the claim contains a clean atomic relation:
- select TWO OR MORE scientifically distinct endpoint phrases;
- every endpoint phrase must occur literally in BOTH claim_text and required_bridge;
- endpoints identify scientific entities, variables, constructs, or measured quantities at the ends of the relation;
- do not use relation operators such as "changes", "differs", "dependence", "effect", "association", "higher", or "lower" as endpoints;
- prior_art_identity_terms are branch/factor/context identity, not BASE-relation
  endpoints. Do NOT return a prior_art_identity_term itself as an endpoint, and
  do NOT hide a complete prior_art_identity_term inside a longer endpoint phrase.
  The lower-order BASE relation must remain meaningful after branch identity is projected away;
- do not select two nested phrases that name the same endpoint;
- scope_qualifier_spans and directional_qualifier_spans are optional and may be returned only when the exact phrase occurs literally in BOTH claim_text and required_bridge.

If two distinct literal scientific endpoints cannot be selected without paraphrase or inference:
- return no endpoints or qualifiers;
- give a concise abstention_reason.

Hard prohibitions:
- Do not use relation_nucleus_terms or search vocabulary as endpoint authority.
- Do not invent synonyms, abbreviations, normalized names, mechanisms, conditions, directions, or implied variables.
- Do not modify novelty roles or make novelty, truth, evidence, or production judgments.
"""

    payload = {
        "source_binding_plan_id": plan.plan_id,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "novelty_selection_role": claim.novelty_selection_role,
                "claim_text": claim.claim_text,
                "required_bridge": claim.required_bridge,
                "prior_art_identity_terms": list(
                    claim.prior_art_identity_terms
                ),
            }
            for claim in claims
        ],
    }
    user = (
        "Bind literal endpoints for every supplied claim. "
        "Return one row per claim_id and preserve exact source wording.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return EndpointBindingPrompt(
        system_prompt=system,
        user_prompt=user,
    )


class InstructorLiteralEndpointBindingBackend:
    backend_name = "instructor_openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 3,
        timeout: float | None = 180.0,
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key_env = api_key_env
        self.api_key = os.getenv(api_key_env)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(telemetry_context or {})
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env}."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Literal endpoint binding requires openai and instructor."
            ) from exc

        mode = getattr(instructor.Mode, self.instructor_mode, None)
        if mode is None:
            raise ValueError(f"Unknown Instructor mode: {self.instructor_mode}")
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        self._client = instructor.from_openai(
            OpenAI(**kwargs),
            mode=mode,
        )
        return self._client

    def generate(
        self,
        prompt: EndpointBindingPrompt,
    ) -> EndpointBindingGeneration:
        client = self._get_client()
        draft, event = run_instructor_structured_call(
            client.chat.completions,
            model=self.model_name,
            response_model=LiteralEndpointBindingBatchDraft,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "relational_atomic_endpoint_binding",
                "stage": "literal_endpoint_binding",
                "call_kind": "scientific_relation_annotation",
            },
            semantic_components={
                "binding_contract":
                    "literal_endpoint_binding_only_v1",
            },
        )
        if not isinstance(draft, LiteralEndpointBindingBatchDraft):
            draft = LiteralEndpointBindingBatchDraft.model_validate(draft)
        return EndpointBindingGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
        )


def _validate_endpoint_set(
    *,
    claim: RelationalAtomicBindingClaimPlan,
    endpoints: list[str],
) -> None:
    normalized = [_normalize(value) for value in endpoints]

    if len(normalized) < 2:
        raise ValueError(
            f"{claim.claim_id} requires at least two relation endpoints"
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"{claim.claim_id} relation endpoints must be unique"
        )

    for index, value in enumerate(endpoints):
        _validate_literal_span(
            claim=claim,
            label=f"relation_endpoint_anchors[{index}]",
            value=value,
        )
        if not _endpoint_content_tokens(value):
            raise ValueError(
                f"{claim.claim_id} endpoint lacks scientific content: {value}"
            )

        normalized_endpoint = _normalize(value)
        for identity in claim.prior_art_identity_terms:
            normalized_identity = _normalize(identity)
            if (
                normalized_identity
                and (
                    normalized_endpoint == normalized_identity
                    or normalized_identity in normalized_endpoint
                )
            ):
                raise ValueError(
                    f"{claim.claim_id} relation endpoint retains complete "
                    "branch identity instead of projecting it away: "
                    f"endpoint={value!r}; identity={identity!r}"
                )

    for left_index, left in enumerate(normalized):
        for right_index, right in enumerate(normalized):
            if left_index >= right_index:
                continue
            if left in right or right in left:
                raise ValueError(
                    f"{claim.claim_id} relation endpoints are nested/redundant: "
                    f"{endpoints[left_index]!r}, {endpoints[right_index]!r}"
                )


def compile_endpoint_bindings(
    *,
    plan: RelationalAtomicBindingPlan,
    draft: LiteralEndpointBindingBatchDraft,
    backend_name: str,
    model_name: str,
    llm_calls_performed: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> RelationalAtomicEndpointBindingReport:
    selected = selected_binding_claims(plan)
    selected_by_id = {row.claim_id: row for row in selected}

    if len(selected_by_id) != len(selected):
        raise ValueError("selected binding claim IDs must be unique")

    draft_by_id = {row.claim_id: row for row in draft.bindings}
    if set(draft_by_id) != set(selected_by_id):
        missing = sorted(set(selected_by_id) - set(draft_by_id))
        extra = sorted(set(draft_by_id) - set(selected_by_id))
        raise ValueError(
            "endpoint binding batch must cover exact frozen claim population; "
            f"missing={missing}; extra={extra}"
        )

    compiled: list[CompiledLiteralEndpointBinding] = []

    for claim in selected:
        row = draft_by_id[claim.claim_id]
        if not row.relation_endpoint_anchors:
            compiled.append(
                CompiledLiteralEndpointBinding(
                    candidate_hypothesis_id=claim.candidate_hypothesis_id,
                    final_hypothesis_id=claim.final_hypothesis_id,
                    claim_id=claim.claim_id,
                    novelty_selection_role=str(
                        claim.novelty_selection_role or ""
                    ),
                    source_claim_sha256=claim.source_claim_sha256,
                    outcome="ABSTAINED_UNBINDABLE",
                    abstention_reason=row.abstention_reason,
                )
            )
            continue

        _validate_endpoint_set(
            claim=claim,
            endpoints=row.relation_endpoint_anchors,
        )

        for label, values in (
            ("scope_qualifier_spans", row.scope_qualifier_spans),
            ("directional_qualifier_spans", row.directional_qualifier_spans),
        ):
            for index, value in enumerate(values):
                _validate_literal_span(
                    claim=claim,
                    label=f"{label}[{index}]",
                    value=value,
                )

        compiled.append(
            CompiledLiteralEndpointBinding(
                candidate_hypothesis_id=claim.candidate_hypothesis_id,
                final_hypothesis_id=claim.final_hypothesis_id,
                claim_id=claim.claim_id,
                novelty_selection_role=str(
                    claim.novelty_selection_role or ""
                ),
                source_claim_sha256=claim.source_claim_sha256,
                relation_endpoint_anchors=list(
                    row.relation_endpoint_anchors
                ),
                scope_qualifier_spans=list(row.scope_qualifier_spans),
                directional_qualifier_spans=list(
                    row.directional_qualifier_spans
                ),
                outcome="BOUND_LITERAL_ENDPOINTS",
            )
        )

    selected_hypotheses = {
        row.final_hypothesis_id
        for row in selected
    }
    report_id = (
        "relational_atomic_endpoint_binding:"
        + _sha256_json(
            {
                "source_binding_plan_id": plan.plan_id,
                "bindings": [
                    row.model_dump(mode="json")
                    for row in compiled
                ],
                "backend_name": backend_name,
                "model_name": model_name,
            }
        )[:20]
    )
    return RelationalAtomicEndpointBindingReport(
        report_id=report_id,
        source_binding_plan_id=plan.plan_id,
        source_binding_plan_sha256=plan.plan_sha256,
        backend_name=backend_name,
        model_name=model_name,
        bindings=compiled,
        selected_hypothesis_count=len(selected_hypotheses),
        selected_claim_count=len(compiled),
        bound_claim_count=sum(
            row.outcome == "BOUND_LITERAL_ENDPOINTS"
            for row in compiled
        ),
        abstained_claim_count=sum(
            row.outcome == "ABSTAINED_UNBINDABLE"
            for row in compiled
        ),
        novelty_bearing_bound_claim_count=sum(
            row.outcome == "BOUND_LITERAL_ENDPOINTS"
            and row.novelty_selection_role == "NOVELTY_BEARING"
            for row in compiled
        ),
        llm_calls_performed=llm_calls_performed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def run_endpoint_binding(
    *,
    plan: RelationalAtomicBindingPlan,
    backend: InstructorLiteralEndpointBindingBackend,
) -> tuple[
    RelationalAtomicEndpointBindingReport,
    EndpointBindingPrompt | None,
]:
    selected = selected_binding_claims(plan)
    if not selected:
        report = compile_endpoint_bindings(
            plan=plan,
            draft=LiteralEndpointBindingBatchDraft(bindings=[]),
            backend_name=backend.backend_name,
            model_name=backend.model_name,
            llm_calls_performed=0,
        )
        return report, None

    prompt = build_endpoint_binding_prompt(plan)
    generation = backend.generate(prompt)
    report = compile_endpoint_bindings(
        plan=plan,
        draft=generation.draft,
        backend_name=backend.backend_name,
        model_name=backend.model_name,
        llm_calls_performed=1,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
    )
    return report, prompt


__all__ = [
    "LiteralEndpointBindingDraft",
    "LiteralEndpointBindingBatchDraft",
    "CompiledLiteralEndpointBinding",
    "RelationalAtomicEndpointBindingReport",
    "EndpointBindingPrompt",
    "EndpointBindingGeneration",
    "InstructorLiteralEndpointBindingBackend",
    "selected_binding_claims",
    "build_endpoint_binding_prompt",
    "compile_endpoint_bindings",
    "run_endpoint_binding",
]
