from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    PriorArtPacket,
)
from pipeline_core.discovery.novelty_closure_execution import (
    ClosureLiteratureQueryPlan,
    ExecutableClosureTarget,
)
from pipeline_core.discovery.novelty_closure_review import (
    ClosureSlotEvidenceReview,
    ClosureSlotReviewDraft,
    compile_closure_slot_review,
)
from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)


@dataclass(frozen=True)
class ClosureReviewPromptRecord:
    name: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


_CLOSURE_REVIEW_SYSTEM = """You are reviewing bounded prior-art metadata for ONE evidence-closure search target.

This is NOT a novelty judgment.
This is NOT a literature-wide absence judgment.
The supplied search target may be retrieval intent rather than a scientific proposition.

Use ONLY the supplied titles and abstracts.
Do not use outside knowledge.
Do not invent a missing scientific mechanism or relation.

RELATIONSHIP LABELS

ESTABLISHES_SLOT:
The abstract explicitly states, tests, compares, demonstrates, or otherwise establishes the scientific relation required by THIS closure slot.

PARTIAL_SLOT_RELATION:
The abstract establishes a substantial neighboring or incomplete part of the required relation, but does NOT establish the full slot.

COMPONENT_ONLY:
The paper contains relevant variables, mechanisms, materials, contexts, or separate effects, but does not establish the relation required by the slot.

TITLE_ONLY_NEIGHBOR:
The title appears relevant but no abstract is available to establish the relation.

UNRELATED:
The supplied metadata does not materially bear on the target.

INSUFFICIENT_METADATA:
The metadata is too weak or ambiguous to classify.

GLOBAL EPISTEMIC RULES

1. Co-mention is never enough to establish a relation.
2. Shared vocabulary is never enough to establish a relation.
3. Two separate effects must not be composed into a new relation.
4. Do not infer causality, moderation, mediation, threshold behavior, regime changes, or mechanism from thematic proximity.
5. A plausible scientific interpretation is not evidence unless the supplied abstract states or tests it.
6. PARTIAL_SLOT_RELATION is not equivalent to ESTABLISHES_SLOT.
7. Do not convert a retrieval query into a stronger scientific assertion.
8. Search-query wording is only retrieval provenance. It is not evidence.
9. If the abstract does not contain the relevant relation, prefer COMPONENT_ONLY or UNRELATED.
10. Copy every returned work_id byte-for-byte from ALLOWED_WORK_IDS.
11. Return at most one relationship per work_id.
12. Never claim that a relation is absent from the literature as a whole.

SLOT-SPECIFIC RULES

BASE_RELATION:
- The target describes the underlying relation nucleus.
- ESTABLISHES_SLOT requires an explicit relation among the relevant base variables/outcome.
- Mere discussion of the variables independently is COMPONENT_ONLY.
- Do not require the higher-order moderator, threshold, or novelty-bearing factor.

DISTINGUISHING_FACTOR_EFFECT:
- This target is deliberately a broad retrieval target, NOT a predefined scientific proposition.
- ESTABLISHES_SLOT requires the abstract to explicitly establish a nontrivial lower-order relation involving the distinguishing factor and at least one scientifically relevant variable, state, or outcome in the supplied relation context.
- Mere mention of the factor and contextual variables in the same paper is COMPONENT_ONLY.
- Separate independent statements must not be combined.
- Do NOT invent which mechanism the factor acts through.
- Do NOT rewrite the target into a specific factor-to-mechanism claim unless the abstract itself states that relation.

BRIDGE_RELATION:
- The target source_text is the already source-grounded bridge from the generated hypothesis.
- ESTABLISHES_SLOT requires substantially the same scientific connecting relation to be explicit in the abstract.
- Sharing only bridge ingredients is COMPONENT_ONLY.
- A materially incomplete version may be PARTIAL_SLOT_RELATION.

FULL_RELATION:
- The target source_text is the actual residual higher-order claim.
- ESTABLISHES_SLOT requires the full residual relation itself to be explicit or tested.
- For moderation, the moderator must explicitly alter/condition the base relation.
- For threshold/regime claims, the threshold/regime structure itself must be explicit.
- Base relations, factor effects, or mechanistic ingredients alone are COMPONENT_ONLY.
- A substantial but incomplete overlap is PARTIAL_SLOT_RELATION and must NOT be promoted to full establishment.

SELF-CONSISTENCY RULE

If your rationale says that the paper:
- does not establish the relation,
- only mentions the variables,
- only studies one component,
- does not test the interaction,
- does not show the threshold/regime,
- or requires combining separate statements,

then ESTABLISHES_SLOT is inconsistent.

Return only relationships supported by the supplied bounded metadata.
"""


def _sha256(
    system: str,
    user: str,
) -> str:
    raw = (
        system
        + "\n---\n"
        + user
    ).encode("utf-8")

    return hashlib.sha256(
        raw
    ).hexdigest()


def build_closure_review_user_prompt(
    *,
    target: ExecutableClosureTarget,
    candidates: ClaimPriorArtCandidateSet,
    packet: PriorArtPacket,
    max_abstract_chars: int = 1400,
) -> str:
    works = {
        row.work_id: row
        for row in packet.works
    }

    lines = [
        "CLOSURE TARGET",
        "==============",
        f"target_id: {target.target_id}",
        f"slot: {target.slot}",
        (
            "target_basis: "
            f"{target.target_basis}"
        ),
        (
            "source_claim_id: "
            f"{target.source_claim_id}"
        ),
        "",
        "SEARCH PROVENANCE",
        "=================",
        (
            "search_query: "
            f"{target.search_query}"
        ),
        (
            "search_terms: "
            + " | ".join(
                target.search_terms
            )
        ),
        "",
        "SOURCE MATERIAL",
        "===============",
        target.source_text,
        "",
        (
            "IMPORTANT: search_query is retrieval intent, "
            "not scientific evidence."
        ),
        "",
        "RANKED CANDIDATES",
        "=================",
    ]

    allowed: list[str] = []

    for index, ranked in enumerate(
        candidates.ranked_works,
        start=1,
    ):
        work = works.get(
            ranked.work_id
        )

        if work is None:
            continue

        allowed.append(
            work.work_id
        )

        abstract = str(
            work.abstract or ""
        )

        if len(abstract) > max_abstract_chars:
            abstract = (
                abstract[
                    : max_abstract_chars - 1
                ].rstrip()
                + "…"
            )

        lines.extend(
            [
                (
                    f"[{index}] "
                    f"work_id={work.work_id}"
                ),
                f"title: {work.title}",
                f"year: {work.year}",
                f"doi: {work.doi}",
                (
                    "relevance_score: "
                    f"{ranked.relevance_score:.4f}"
                ),
                (
                    "semantic_similarity: "
                    f"{ranked.semantic_similarity:.4f}"
                ),
                (
                    "lexical_coverage: "
                    f"{ranked.lexical_coverage:.4f}"
                ),
                (
                    "abstract: "
                    + (
                        abstract
                        if abstract
                        else (
                            "[NO ABSTRACT AVAILABLE]"
                        )
                    )
                ),
                "",
            ]
        )

    if not allowed:
        lines.append("- NONE")
        lines.append("")

    lines.extend(
        [
            "ALLOWED_WORK_IDS",
            "================",
            *(
                allowed
                if allowed
                else ["NONE"]
            ),
            "",
            "OUTPUT REQUIREMENTS",
            "===================",
            (
                "Every returned work_id must be copied "
                "exactly from ALLOWED_WORK_IDS."
            ),
            (
                "Do not infer literature-wide absence "
                "from this candidate set."
            ),
            (
                "Classify only the relationship between "
                "each record and THIS closure slot."
            ),
        ]
    )

    return "\n".join(
        lines
    )


class InstructorOpenAICompatibleClosureReviewBackend:
    backend_name = (
        "instructor_openai_compatible_"
        "closure_review"
    )

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
        telemetry_path: (
            str
            | os.PathLike[str]
            | None
        ) = None,
        telemetry_context: (
            dict[str, Any]
            | None
        ) = None,
    ) -> None:
        self.model_name = str(
            model
        )

        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(
                api_key_env
            )
        )

        self.api_key_env = (
            api_key_env
        )

        self.base_url = (
            base_url
            or os.getenv(
                "OPENAI_BASE_URL"
            )
            or None
        )

        self.instructor_mode = (
            str(
                instructor_mode
            ).upper()
        )

        self.temperature = float(
            temperature
        )

        self.parse_retries = int(
            parse_retries
        )

        self.timeout = timeout

        self.capture_prompts = bool(
            capture_prompts
        )

        self.max_abstract_chars = int(
            max_abstract_chars
        )

        self.prompt_records: list[
            ClosureReviewPromptRecord
        ] = []

        self.telemetry_path = (
            telemetry_path
        )

        self.telemetry_context = dict(
            telemetry_context or {}
        )

        self._client: Any | None = (
            None
        )

    def _get_client(
        self,
    ) -> Any:
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise RuntimeError(
                "No API key available. "
                f"Set {self.api_key_env} "
                "or pass api_key explicitly."
            )

        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Closure reviewer requires "
                "installed 'openai' and "
                "'instructor'."
            ) from exc

        mode = getattr(
            instructor.Mode,
            self.instructor_mode,
            None,
        )

        if mode is None:
            raise ValueError(
                "Unknown Instructor mode: "
                f"{self.instructor_mode}"
            )

        kwargs: dict[str, Any] = {
            "api_key":
                self.api_key,
        }

        if self.base_url:
            kwargs[
                "base_url"
            ] = self.base_url

        if self.timeout is not None:
            kwargs[
                "timeout"
            ] = self.timeout

        self._client = (
            instructor.from_openai(
                OpenAI(**kwargs),
                mode=mode,
            )
        )

        return self._client

    def review_target(
        self,
        *,
        target: ExecutableClosureTarget,
        candidates: ClaimPriorArtCandidateSet,
        packet: PriorArtPacket,
    ) -> ClosureSlotReviewDraft:
        user = (
            build_closure_review_user_prompt(
                target=target,
                candidates=candidates,
                packet=packet,
                max_abstract_chars=(
                    self.max_abstract_chars
                ),
            )
        )

        if self.capture_prompts:
            self.prompt_records.append(
                ClosureReviewPromptRecord(
                    name=(
                        "closure_review_"
                        + target.target_id
                    ),
                    system_prompt=(
                        _CLOSURE_REVIEW_SYSTEM
                    ),
                    user_prompt=user,
                    prompt_sha256=_sha256(
                        _CLOSURE_REVIEW_SYSTEM,
                        user,
                    ),
                )
            )

        result, _event = (
            run_instructor_structured_call(
                self._get_client()
                .chat.completions,
                model=self.model_name,
                response_model=(
                    ClosureSlotReviewDraft
                ),
                messages=[
                    {
                        "role": "system",
                        "content":
                            _CLOSURE_REVIEW_SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],
                temperature=(
                    self.temperature
                ),
                max_retries=(
                    self.parse_retries
                ),
                telemetry_path=(
                    self.telemetry_path
                ),
                telemetry_context={
                    **self.telemetry_context,
                    "pipeline":
                        "nonobviousness_closure",
                    "stage":
                        "closure_slot_review",
                    "call_kind":
                        "structured",
                    "target_id":
                        target.target_id,
                    "slot":
                        target.slot,
                    "source_claim_id":
                        target.source_claim_id,
                },
            )
        )

        if not isinstance(
            result,
            ClosureSlotReviewDraft,
        ):
            result = (
                ClosureSlotReviewDraft
                .model_validate(
                    result
                )
            )

        return result


def review_and_compile_closure_target(
    *,
    backend:
        InstructorOpenAICompatibleClosureReviewBackend,
    target: ExecutableClosureTarget,
    candidates: ClaimPriorArtCandidateSet,
    packet: PriorArtPacket,
    plan: ClosureLiteratureQueryPlan,
    min_positive_confidence: float = 0.65,
) -> ClosureSlotEvidenceReview:
    draft = backend.review_target(
        target=target,
        candidates=candidates,
        packet=packet,
    )

    return compile_closure_slot_review(
        target=target,
        draft=draft,
        candidates=candidates,
        packet=packet,
        plan=plan,
        min_positive_confidence=(
            min_positive_confidence
        ),
    )
