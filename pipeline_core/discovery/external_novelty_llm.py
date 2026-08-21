from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReviewDraft,
    NoveltyClaim,
    NoveltyClaimDecompositionDraft,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call
from pipeline_core.discovery.prior_art_review_audit import (
    record_prior_art_review_call,
)


@dataclass(frozen=True)
class ExternalNoveltyPromptRecord:
    name: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


_DECOMPOSE_SYSTEM = """You decompose a generated scientific hypothesis into claim-level novelty assertions for prior-art search.

This is NOT a novelty judgment and NOT a truth judgment. Do not say that anything is novel, unprecedented, first, or unknown.

Identify the smallest scientifically meaningful claims that would make the hypothesis distinct if supported. Prefer claims about a mediator, moderator/interaction, context condition, pathway competition, descriptor interaction, mechanistic link, or a distinctive falsifiable prediction.

For each claim provide plain-text literature search concepts and 2-3 plain-text search queries. Search queries must not use Boolean operators or special search syntax, and should avoid hyphenated terms where possible. Keep them suitable for Semantic Scholar and Crossref relevance search.

Do not decompose generic background facts unless they are necessary to distinguish the generated hypothesis. Return only the structured NoveltyClaimDecompositionDraft requested by the caller."""


_REVIEW_SYSTEM = """You are a prior-art relationship reviewer in an external-novelty assessment pipeline.

You are given ONE claim and a bounded set of retrieved literature records. Use ONLY the supplied title/abstract metadata. Do not use outside knowledge. Do not claim literature-wide novelty.

For each truly relevant record, classify its relationship to the claim as one of:
- DIRECT_PRIOR_ART: the record directly states/tests essentially the same scientific relation or distinctive prediction.
- PARTIAL_PRIOR_ART: an ABSTRACT-BACKED record preserves the claim's RELATION NUCLEUS and establishes a substantial subset of that scientific relation, but not the full claim.
- TITLE_ONLY_NEIGHBOR: the title suggests a neighboring relation but the abstract is missing, so the substantive overlap cannot be confirmed.
- COMPONENT_ONLY: the record establishes one or more ingredients/components, variables, mechanisms, contexts, materials, or one arm of a comparison, but does not establish the claim's proposed interaction, dependence, mediation, conditionality, comparison, directional relation, or other relation nucleus.
- CONTEXTUAL_CONFLICT: the record challenges a broader descriptor/mechanistic assumption but differs materially in reaction domain, catalyst class, or site scope.
- CONFLICTING_PRIOR_ART: the record directly reports a materially opposing relation/result in a sufficiently overlapping scientific scope.
- UNRELATED: despite retrieval similarity, it does not materially bear on the claim.
- INSUFFICIENT_METADATA: the supplied metadata is too weak to judge.

RELATION-NUCLEUS RULES:
1. Shared entities, variables, mechanisms, contexts, materials, or thematic proximity alone are NOT sufficient for PARTIAL_PRIOR_ART. Use COMPONENT_ONLY unless the record actually establishes a substantial part of the asserted relation.
2. A thematically neighboring relation is not, by itself, PARTIAL_PRIOR_ART.
3. For mediator claims, PARTIAL_PRIOR_ART requires evidence linking the proposed mediator to the relevant relation or outcome. Evidence that merely discusses the mediator variable is COMPONENT_ONLY.
4. For moderator_interaction or descriptor_interaction claims, PARTIAL_PRIOR_ART requires an interaction, dependence, conditionality, or joint effect relevant to the asserted relation. Separate main effects or separate components are COMPONENT_ONLY.
5. For context_condition claims, PARTIAL_PRIOR_ART requires a comparison across contexts or an explicit context-dependent effect. Evidence from only one context is COMPONENT_ONLY.
6. For distinctive_prediction claims, PARTIAL_PRIOR_ART requires the same dependent relation or contrast even if direction, material scope, or a control condition is incomplete. Evidence for only one arm of the comparison or only the dependent variable is COMPONENT_ONLY.
7. For mechanistic_link claims, PARTIAL_PRIOR_ART requires substantially the same mechanistic link. Sharing mechanism ingredients without the link is COMPONENT_ONLY.
8. A scope mismatch can still permit PARTIAL_PRIOR_ART when the relation nucleus is preserved; scope similarity alone cannot create PARTIAL_PRIOR_ART.

SELF-CONSISTENCY CHECK:
If your rationale says that a record "does not compare X versus Y", "does not establish the claimed relationship", "only addresses one condition", "does not link X to Y", or an equivalent statement, PARTIAL_PRIOR_ART is usually inconsistent. Use COMPONENT_ONLY unless the same record still establishes another substantial part of the claim's relation nucleus.

WORK-ID COPY CONTRACT:
1. Every returned work_id MUST be copied byte-for-byte from the explicit ALLOWED_WORK_IDS block in the user message.
2. Never invent, reconstruct, shorten, normalize, index, or alias a work ID.
3. Never return candidate numbers, list indices, ordinal labels, or placeholders as work IDs. Examples of forbidden forms include "6", "work:6", "paper:6", and "prior_art_work:6" unless that exact string itself appears in ALLOWED_WORK_IDS.
4. Return at most one match per allowed work_id. Do not duplicate a work ID.
5. If you cannot copy the exact supplied ID for a record, OMIT that record. Do not create a match merely to mention that an ID is missing or invalid.
6. The set of returned work IDs must be a subset of ALLOWED_WORK_IDS.

Do not infer detailed results from a generic title alone. For CONFLICTING_PRIOR_ART, require not only an opposing result but also substantially matching reaction and catalyst/site scope; otherwise use CONTEXTUAL_CONFLICT. The deterministic compiler will independently enforce these constraints.

Return work IDs exactly as supplied. You may omit unrelated records. Your interpretation must describe only what the supplied bounded evidence shows."""


def _sha256(system: str, user: str) -> str:
    raw = (system + "\n---\n" + user).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class InstructorOpenAICompatibleExternalNoveltyBackend:
    backend_name = "instructor_openai_compatible_external_novelty"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 1,
        timeout: float | None = 180.0,
        capture_prompts: bool = False,
        max_abstract_chars: int = 1400,
        telemetry_path: str | os.PathLike[str] | None = None,
        telemetry_context: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env)
        self.api_key_env = api_key_env
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.capture_prompts = bool(capture_prompts)
        self.max_abstract_chars = int(max_abstract_chars)
        self.prompt_records: list[ExternalNoveltyPromptRecord] = []
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(telemetry_context or {})
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env} or pass api_key explicitly."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "External novelty LLM backend requires installed 'openai' and 'instructor'."
            ) from exc
        mode = getattr(instructor.Mode, self.instructor_mode, None)
        if mode is None:
            raise ValueError(f"Unknown Instructor mode: {self.instructor_mode}")
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        self._client = instructor.from_openai(OpenAI(**kwargs), mode=mode)
        return self._client

    def _record(self, name: str, system: str, user: str) -> None:
        if self.capture_prompts:
            self.prompt_records.append(
                ExternalNoveltyPromptRecord(
                    name=name,
                    system_prompt=system,
                    user_prompt=user,
                    prompt_sha256=_sha256(system, user),
                )
            )

    def decompose(
        self,
        hypothesis: HypothesisCard,
        *,
        max_claims: int,
    ) -> NoveltyClaimDecompositionDraft:
        prediction_lines = [
            f"- {row.observable} => {row.expected_direction}; rationale={row.rationale}"
            for row in hypothesis.predicted_observations
        ]
        user = "\n".join(
            [
                "HYPOTHESIS",
                "==========",
                f"hypothesis_id: {hypothesis.hypothesis_id}",
                f"title: {hypothesis.title}",
                f"statement: {hypothesis.hypothesis_statement}",
                f"inferential_bridge: {hypothesis.inferential_bridge}",
                "predictions:",
                *(prediction_lines or ["- NONE"]),
                "assumptions:",
                *([f"- {x}" for x in hypothesis.assumptions] or ["- NONE"]),
                "",
                f"Return at most {int(max_claims)} claim-level novelty assertions.",
                "At least one claim should capture the hypothesis's most distinctive interaction/condition/prediction rather than generic HER background.",
            ]
        )
        self._record(f"decompose_{hypothesis.hypothesis_id}", _DECOMPOSE_SYSTEM, user)
        result, _event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=NoveltyClaimDecompositionDraft,
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "external_novelty",
                "stage": "decompose",
                "call_kind": "structured",
                "hypothesis_id": hypothesis.hypothesis_id,
            },
        )
        if not isinstance(result, NoveltyClaimDecompositionDraft):
            result = NoveltyClaimDecompositionDraft.model_validate(result)
        return result

    def review_claim(
        self,
        claim: NoveltyClaim,
        works: list[dict[str, Any]],
    ) -> ClaimPriorArtReviewDraft:
        lines = [
            "CLAIM",
            "=====",
            f"claim_id: {claim.claim_id}",
            f"kind: {claim.kind}",
            f"importance: {claim.importance}",
            f"text: {claim.text}",
            "",
            "RETRIEVED PRIOR-ART CANDIDATES",
            "==============================",
        ]
        if not works:
            lines.append("- NONE")
        for index, work in enumerate(works, start=1):
            abstract = str(work.get("abstract") or "")
            if len(abstract) > self.max_abstract_chars:
                abstract = abstract[: self.max_abstract_chars - 1].rstrip() + "…"
            lines.extend(
                [
                    f"[{index}] work_id={work['work_id']}",
                    f"title: {work.get('title', '')}",
                    f"year: {work.get('year')}",
                    f"doi: {work.get('doi')}",
                    f"semantic_similarity: {float(work.get('semantic_similarity', 0.0)):.4f}",
                    f"lexical_coverage: {float(work.get('lexical_coverage', 0.0)):.4f}",
                    f"reaction_domain_relevance: {float(work.get('reaction_domain_relevance', 0.5)):.4f}",
                    f"catalyst_scope_relevance: {float(work.get('catalyst_scope_relevance', 0.5)):.4f}",
                    f"abstract: {abstract if abstract else '[NO ABSTRACT AVAILABLE]'}",
                    "",
                ]
            )
        allowed_work_ids = [
            str(work["work_id"])
            for work in works
        ]
        lines.extend(
            [
                "ALLOWED_WORK_IDS",
                "================",
                *allowed_work_ids,
                "",
                "WORK-ID OUTPUT REQUIREMENT",
                "==========================",
                "Every returned work_id must be copied byte-for-byte from ALLOWED_WORK_IDS above.",
                "Return at most one match per allowed work_id.",
                "Do not return candidate numbers, list indices, abbreviated IDs, reconstructed IDs, or placeholders.",
                "If you cannot copy the exact supplied work_id, omit that record.",
                "",
                "Only classify records that materially bear on the claim.",
                "Do not infer literature-wide absence from this bounded candidate set.",
            ]
        )
        user = "\n".join(lines)
        self._record(f"review_{claim.claim_id}", _REVIEW_SYSTEM, user)
        result, _event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=ClaimPriorArtReviewDraft,
            messages=[
                {"role": "system", "content": _REVIEW_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "external_novelty",
                "stage": "prior_art_review",
                "call_kind": "structured",
                "claim_id": claim.claim_id,
            },
        )
        if not isinstance(result, ClaimPriorArtReviewDraft):
            result = ClaimPriorArtReviewDraft.model_validate(result)
        record_prior_art_review_call(
            system_prompt=_REVIEW_SYSTEM,
            user_prompt=user,
            response_schema=ClaimPriorArtReviewDraft,
            result=result,
            model=self.model_name,
            instructor_mode=self.instructor_mode,
            temperature=self.temperature,
            claim_id=claim.claim_id,
            hypothesis_id=claim.hypothesis_id,
            claim_text=claim.text,
            works=works,
            telemetry_event=_event,
        )
        return result
