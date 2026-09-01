from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.external_novelty_contracts import (
    PriorArtPacket,
)
from pipeline_core.discovery.novelty_adjudication import (
    CompiledNonObviousnessAdjudication,
    NonObviousnessAdjudicationDraft,
    NonObviousnessEvidencePacket,
    NonObviousnessReviewGate,
    compile_nonobviousness_adjudication,
)
from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)


class _AdjudicationResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    proposed_verdict: Literal[
        "ROUTINE_FROM_PRIOR_ART",
        "INSUFFICIENT_FOR_JUDGMENT",
        "POTENTIALLY_NON_OBVIOUS",
    ]

    direct_reconstruction_from_known_relations: bool

    additional_scientific_assumptions: list[str] = Field(
        default_factory=list
    )

    prediction_distinguishes_from_routine_baseline: bool

    falsifier_is_specific: bool

    concise_basis: str = Field(
        min_length=1,
    )


_NONOBVIOUSNESS_SYSTEM = r"""
You are a scientific non-obviousness adjudicator operating AFTER
claim-level prior-art review.

This is NOT a literature search.
This is NOT a scientific truth judgment.
This is NOT a general novelty-scoring task.

You are given:
1. ONE residual scientific claim;
2. its structural characterization;
3. a bounded set of ESTABLISHED PRIOR-ART RELATIONS that have already
   been positively reviewed;
4. a proposed bridge, prediction, and falsification condition.

Use ONLY this supplied information.

Do not use outside scientific knowledge.
Do not invent literature.
Do not infer literature-wide absence.
Do not treat failure to find an exact paper as evidence of
non-obviousness.

CENTRAL QUESTION
================

Ask:

Can the residual claim be reconstructed from the supplied established
relations without adding a new scientific proposition?

If YES:
- direct_reconstruction_from_known_relations=true
- proposed_verdict=ROUTINE_FROM_PRIOR_ART

If NO, identify the additional scientific proposition that would be
required.

IMPORTANT:
An "additional scientific assumption" must be a proposition that is
actually required by the supplied claim/vector but is NOT supplied by
the established prior-art relations.

You may make an implicit requirement already encoded by the claim or
vector explicit.

You MUST NOT invent a new mechanism, material property, transition,
threshold, causal pathway, physical effect, biological effect, or
other scientific fact merely to make the claim sound interesting.

If the claim/vector itself does not specify the additional scientific
bridge with enough scientific content to state it clearly:
- do not invent one;
- return INSUFFICIENT_FOR_JUDGMENT.

LOGICAL NON-ENTAILMENT RULES
============================

Separate main effects do not establish an interaction.

For example:

X affects Y
M affects Y

does NOT establish:

M changes how X affects Y.

Likewise, a mediation chain:

M affects Z
Z affects Y

does not by itself establish that M moderates the X-to-Y relation.

Shared variables, shared mechanisms, semantic similarity, and
co-occurrence are not logical reconstruction.

ROUTINE_FROM_PRIOR_ART
======================

Use ROUTINE_FROM_PRIOR_ART only when the supplied established
relations reconstruct the scientific relation nucleus without a new
scientific bridge.

A routine conclusion may involve straightforward composition of
already established relations when that composition actually yields
the claimed logical relation.

INSUFFICIENT_FOR_JUDGMENT
=========================

Use this when:
- the supplied prior-art closure is not enough;
- the claim is under-specified;
- the necessary bridge cannot be stated without inventing scientific
  content;
- the prediction is merely generic;
- the distinction from a routine baseline is unclear.

This is the default when evidence or scientific specification is
insufficient.

POTENTIALLY_NON_OBVIOUS
=======================

Use this only when ALL are true:

1. the supplied established relations do not reconstruct the claim;
2. the claim/vector itself requires an identifiable additional
   scientific proposition;
3. you can state that proposition without importing outside
   knowledge;
4. the prediction distinguishes the claim from the routine baseline;
5. the supplied falsification condition specifically tests that
   distinction.

"POTENTIALLY_NON_OBVIOUS" means only that the search-bounded evidence
and explicit claim structure contain a genuine additional inferential
step.

It does NOT mean:
- proven novel;
- literature-wide novel;
- scientifically correct;
- surprising to all experts.

ADDITIONAL ASSUMPTION CONTRACT
==============================

additional_scientific_assumptions must contain only the minimum
scientific propositions required to get from the established
relations to the residual claim.

Do not list:
- generic statements such as "the hypothesis is true";
- absence-of-literature claims;
- restatements of established prior art;
- mechanisms invented from your own scientific knowledge.

PREDICTION CONTRACT
===================

prediction_distinguishes_from_routine_baseline=true only if the
supplied prediction would produce an observation meaningfully
different from what the established relations alone require.

FALSIFIER CONTRACT
==================

falsifier_is_specific=true only if the supplied falsification
condition directly tests the distinctive relation or regime, rather
than merely saying "the hypothesis is false."

Return only the requested structured output.
""".strip()


def build_nonobviousness_user_prompt(
    packet: NonObviousnessEvidencePacket,
) -> str:
    vector = packet.vector

    lines = [
        "RESIDUAL CLAIM",
        "==============",
        f"claim_id: {packet.claim_id}",
        f"claim: {packet.claim_text}",
        f"structural_status: {packet.structural_status}",
        "",
        "ADJUDICATION VECTOR",
        "===================",
        (
            "inferential_distance: "
            f"{vector.inferential_distance}"
        ),
        (
            "mechanistic_necessity: "
            f"{vector.mechanistic_necessity}"
        ),
        (
            "regime_specificity: "
            f"{vector.regime_specificity}"
        ),
        (
            "counterintuitiveness: "
            f"{vector.counterintuitiveness}"
        ),
        (
            "testable_distinctiveness: "
            f"{vector.testable_distinctiveness}"
        ),
        f"required_bridge: {vector.required_bridge}",
        (
            "predicted_observation: "
            f"{vector.predicted_observation}"
        ),
        (
            "falsification_condition: "
            f"{vector.falsification_condition}"
        ),
        "",
        "EVIDENCE STATE",
        "==============",
        (
            "direct_full_claim_prior_art: "
            f"{packet.direct_full_claim_prior_art}"
        ),
        (
            "evidence_closure_sufficient: "
            f"{packet.evidence_closure_sufficient}"
        ),
        "",
        "ESTABLISHED PRIOR-ART RELATIONS",
        "===============================",
    ]

    if not packet.established_relations:
        lines.append("- NONE")

    for index, relation in enumerate(
        packet.established_relations,
        1,
    ):
        lines.extend(
            [
                f"[{index}] {relation.relation_statement}",
                (
                    "relationship_status: "
                    f"{relation.relationship_status}"
                ),
                (
                    "scope_note: "
                    f"{relation.scope_note or '[NONE]'}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "DECISION REQUIREMENT",
            "====================",
            (
                "First decide whether the established relations "
                "reconstruct the residual claim without a new "
                "scientific proposition."
            ),
            (
                "If not, state only the minimum additional "
                "scientific assumption already required by the "
                "claim/vector."
            ),
            (
                "If that assumption cannot be stated without "
                "inventing scientific content, use "
                "INSUFFICIENT_FOR_JUDGMENT."
            ),
        ]
    )

    return "\n".join(lines)


class InstructorNonObviousnessAdjudicator:
    backend_name = (
        "instructor_openai_compatible_"
        "nonobviousness_adjudicator"
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
        telemetry_path: str | os.PathLike[str] | None = None,
        telemetry_context: dict[str, Any] | None = None,
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

        self.instructor_mode = str(
            instructor_mode
        ).upper()

        self.temperature = float(
            temperature
        )

        self.parse_retries = int(
            parse_retries
        )

        self.timeout = timeout

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
                "No API key available. Set "
                f"{self.api_key_env} or pass api_key."
            )

        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Non-obviousness LLM adjudicator requires "
                "'openai' and 'instructor'."
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
            "api_key": self.api_key,
        }

        if self.base_url:
            kwargs["base_url"] = self.base_url

        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        self._client = instructor.from_openai(
            OpenAI(**kwargs),
            mode=mode,
        )

        return self._client

    def adjudicate(
        self,
        packet: NonObviousnessEvidencePacket,
    ) -> NonObviousnessAdjudicationDraft:
        user = build_nonobviousness_user_prompt(
            packet
        )

        result, _event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=_AdjudicationResponse,
            messages=[
                {
                    "role": "system",
                    "content": _NONOBVIOUSNESS_SYSTEM,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "external_novelty",
                "stage": (
                    "nonobviousness_adjudication"
                ),
                "call_kind": "structured",
                "claim_id": packet.claim_id,
            },
        )

        if not isinstance(
            result,
            _AdjudicationResponse,
        ):
            result = (
                _AdjudicationResponse.model_validate(
                    result
                )
            )

        return NonObviousnessAdjudicationDraft(
            proposed_verdict=(
                result.proposed_verdict
            ),
            direct_reconstruction_from_known_relations=(
                result.direct_reconstruction_from_known_relations
            ),
            additional_scientific_assumptions=tuple(
                row.strip()
                for row
                in result.additional_scientific_assumptions
                if row.strip()
            ),
            prediction_distinguishes_from_routine_baseline=(
                result.prediction_distinguishes_from_routine_baseline
            ),
            falsifier_is_specific=(
                result.falsifier_is_specific
            ),
            concise_basis=(
                result.concise_basis.strip()
            ),
        )



# =====================================================================
# N10-C: independent evidence-constrained adjudication
# =====================================================================

@dataclass(frozen=True)
class NonObviousnessAdjudicationPromptRecord:
    name: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class SanitizedAdjudicationDraft:
    draft: NonObviousnessAdjudicationDraft
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class IndependentAdjudicationOutcome:
    review_performed: bool
    raw_draft: NonObviousnessAdjudicationDraft
    sanitized_draft: NonObviousnessAdjudicationDraft
    sanitizer_reason_codes: tuple[str, ...]
    compiled: CompiledNonObviousnessAdjudication


class NonObviousnessAdjudicationBackendProtocol(
    Protocol
):
    def adjudicate(
        self,
        *,
        packet: NonObviousnessEvidencePacket,
        prior_art: PriorArtPacket,
    ) -> NonObviousnessAdjudicationDraft: ...


_INDEPENDENT_ADJUDICATION_EXTENSION = r"""
INDEPENDENT N10 EVIDENCE CONTRACT
=================================

This review receives only positively ESTABLISHED prior-art relations
and the abstracts attached to those established work IDs.

Do not use retrieval-query wording as evidence.
Do not use outside scientific knowledge.
Missing prior art is never positive evidence of non-obviousness.

ADDITIONAL ASSUMPTION PROVENANCE
================================

Every item in additional_scientific_assumptions MUST be copied as an
exact contiguous span from one of the supplied SPECIFICATION_SOURCE_TEXTS:

- residual claim;
- required_bridge;
- predicted_observation;
- falsification_condition.

Do not paraphrase an assumption.
Do not invent a mechanism, threshold, regime, causal pathway, mediator,
moderator, material property, or scientific fact.

The adjudicator may decide that an already-specified bridge is additionally
required beyond established prior art. It may not create a new bridge.

POTENTIALLY_NON_OBVIOUS requires ALL of:
1. the established relations do not directly reconstruct the residual claim;
2. an explicit source-grounded additional scientific assumption is required;
3. the supplied prediction distinguishes the claim from routine composition;
4. the supplied falsifier specifically tests that distinction.

If these requirements are not supported, use INSUFFICIENT_FOR_JUDGMENT.
""".strip()


_INDEPENDENT_NONOBVIOUSNESS_SYSTEM = (
    _NONOBVIOUSNESS_SYSTEM
    + "\n\n"
    + _INDEPENDENT_ADJUDICATION_EXTENSION
)


def _independent_sha256(
    system: str,
    user: str,
) -> str:
    return hashlib.sha256(
        (
            system
            + "\n---\n"
            + user
        ).encode("utf-8")
    ).hexdigest()


def _specification_sources(
    packet: NonObviousnessEvidencePacket,
) -> tuple[str, ...]:
    values = (
        packet.claim_text,
        packet.vector.required_bridge,
        packet.vector.predicted_observation,
        packet.vector.falsification_condition,
    )

    result: list[str] = []

    for value in values:
        text = str(
            value or ""
        ).strip()

        if (
            text
            and text not in result
        ):
            result.append(text)

    return tuple(result)


def sanitize_adjudication_draft(
    *,
    packet: NonObviousnessEvidencePacket,
    draft: NonObviousnessAdjudicationDraft,
) -> SanitizedAdjudicationDraft:
    """Drop additional assumptions not extractively present in the claim."""

    sources = _specification_sources(
        packet
    )

    valid: list[str] = []
    reasons: list[str] = []

    for assumption in (
        draft.additional_scientific_assumptions
    ):
        candidate = str(
            assumption or ""
        ).strip()

        if not candidate:
            continue

        # Deliberately case-sensitive/extractive:
        # the adjudicator may select source text, not rewrite science.
        if any(
            candidate in source
            for source in sources
        ):
            if candidate not in valid:
                valid.append(candidate)
        else:
            reasons.append(
                "non_extractive_additional_assumption_dropped"
            )

    sanitized = NonObviousnessAdjudicationDraft(
        proposed_verdict=(
            draft.proposed_verdict
        ),
        direct_reconstruction_from_known_relations=(
            draft.direct_reconstruction_from_known_relations
        ),
        additional_scientific_assumptions=tuple(
            valid
        ),
        prediction_distinguishes_from_routine_baseline=(
            draft.prediction_distinguishes_from_routine_baseline
        ),
        falsifier_is_specific=(
            draft.falsifier_is_specific
        ),
        concise_basis=(
            draft.concise_basis
        ),
    )

    return SanitizedAdjudicationDraft(
        draft=sanitized,
        reason_codes=tuple(
            dict.fromkeys(reasons)
        ),
    )


def _allowed_positive_work_ids(
    packet: NonObviousnessEvidencePacket,
) -> tuple[str, ...]:
    result: list[str] = []

    for relation in (
        packet.established_relations
    ):
        for work_id in (
            relation.work_ids
        ):
            value = str(
                work_id or ""
            ).strip()

            if (
                value
                and value not in result
            ):
                result.append(value)

    return tuple(result)


def build_independent_nonobviousness_user_prompt(
    *,
    packet: NonObviousnessEvidencePacket,
    prior_art: PriorArtPacket,
    max_abstract_chars: int = 1800,
) -> str:
    """Extend the existing N9 prompt with positive provenance only."""

    lines = [
        build_nonobviousness_user_prompt(
            packet
        ),
        "",
        "SPECIFICATION_SOURCE_TEXTS",
        "==========================",
        f"claim_text: {packet.claim_text}",
        (
            "required_bridge: "
            f"{packet.vector.required_bridge}"
        ),
        (
            "predicted_observation: "
            f"{packet.vector.predicted_observation}"
        ),
        (
            "falsification_condition: "
            f"{packet.vector.falsification_condition}"
        ),
        "",
        "POSITIVE ESTABLISHED WORK ABSTRACTS",
        "===================================",
    ]

    allowed = (
        _allowed_positive_work_ids(
            packet
        )
    )

    works = {
        work.work_id: work
        for work in prior_art.works
        if work.work_id in set(allowed)
    }

    for work_id in allowed:
        work = works.get(
            work_id
        )

        if work is None:
            continue

        abstract = str(
            work.abstract or ""
        )

        if (
            len(abstract)
            > max_abstract_chars
        ):
            abstract = (
                abstract[
                    : max_abstract_chars - 1
                ].rstrip()
                + "…"
            )

        lines.extend(
            [
                f"work_id: {work.work_id}",
                f"title: {work.title}",
                f"year: {work.year}",
                f"doi: {work.doi}",
                (
                    "abstract: "
                    + (
                        abstract
                        if abstract
                        else "[NO ABSTRACT]"
                    )
                ),
                "",
            ]
        )

    if not allowed:
        lines.extend(
            [
                "- NONE",
                "",
            ]
        )

    lines.extend(
        [
            "ALLOWED_POSITIVE_WORK_IDS",
            "=========================",
            *(
                list(allowed)
                if allowed
                else ["NONE"]
            ),
            "",
            (
                "Only these established work IDs and their "
                "supplied abstracts may be used."
            ),
            (
                "Any additional_scientific_assumptions entry "
                "must be an exact span from "
                "SPECIFICATION_SOURCE_TEXTS."
            ),
        ]
    )

    return "\n".join(lines)


def build_nonobviousness_adjudication_user_prompt(
    *,
    packet: NonObviousnessEvidencePacket,
    prior_art: PriorArtPacket,
    max_abstract_chars: int = 1800,
) -> str:
    """N10 public compatibility wrapper.

    Keep the original N9 build_nonobviousness_user_prompt(packet)
    unchanged while exposing the evidence-enriched N10 prompt builder
    under the name used by the N10 integration tests/callers.
    """
    return build_independent_nonobviousness_user_prompt(
        packet=packet,
        prior_art=prior_art,
        max_abstract_chars=max_abstract_chars,
    )


class InstructorOpenAICompatibleNonObviousnessAdjudicationBackend(
    InstructorNonObviousnessAdjudicator
):
    """N10 extension preserving the original N9 adjudicator API."""

    backend_name = (
        "instructor_openai_compatible_"
        "independent_nonobviousness_adjudicator"
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
        max_abstract_chars: int = 1800,
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
        super().__init__(
            model=model,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            instructor_mode=instructor_mode,
            temperature=temperature,
            parse_retries=parse_retries,
            timeout=timeout,
            telemetry_path=telemetry_path,
            telemetry_context=telemetry_context,
        )

        self.capture_prompts = bool(
            capture_prompts
        )

        self.max_abstract_chars = int(
            max_abstract_chars
        )

        self.prompt_records: list[
            NonObviousnessAdjudicationPromptRecord
        ] = []

    def adjudicate(
        self,
        *,
        packet: NonObviousnessEvidencePacket,
        prior_art: PriorArtPacket,
    ) -> NonObviousnessAdjudicationDraft:
        user = (
            build_independent_nonobviousness_user_prompt(
                packet=packet,
                prior_art=prior_art,
                max_abstract_chars=(
                    self.max_abstract_chars
                ),
            )
        )

        if self.capture_prompts:
            self.prompt_records.append(
                NonObviousnessAdjudicationPromptRecord(
                    name=(
                        "nonobviousness_adjudication_"
                        + packet.claim_id
                    ),
                    system_prompt=(
                        _INDEPENDENT_NONOBVIOUSNESS_SYSTEM
                    ),
                    user_prompt=user,
                    prompt_sha256=(
                        _independent_sha256(
                            _INDEPENDENT_NONOBVIOUSNESS_SYSTEM,
                            user,
                        )
                    ),
                )
            )

        result, _event = (
            run_instructor_structured_call(
                self._get_client()
                .chat.completions,
                model=self.model_name,
                response_model=(
                    _AdjudicationResponse
                ),
                messages=[
                    {
                        "role":
                            "system",
                        "content":
                            _INDEPENDENT_NONOBVIOUSNESS_SYSTEM,
                    },
                    {
                        "role":
                            "user",
                        "content":
                            user,
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
                        "nonobviousness",
                    "stage":
                        "independent_adjudication",
                    "call_kind":
                        "structured",
                    "claim_id":
                        packet.claim_id,
                },
            )
        )

        if not isinstance(
            result,
            _AdjudicationResponse,
        ):
            result = (
                _AdjudicationResponse
                .model_validate(
                    result
                )
            )

        return NonObviousnessAdjudicationDraft(
            proposed_verdict=(
                result.proposed_verdict
            ),
            direct_reconstruction_from_known_relations=(
                result.direct_reconstruction_from_known_relations
            ),
            additional_scientific_assumptions=tuple(
                row.strip()
                for row
                in result.additional_scientific_assumptions
                if row.strip()
            ),
            prediction_distinguishes_from_routine_baseline=(
                result.prediction_distinguishes_from_routine_baseline
            ),
            falsifier_is_specific=(
                result.falsifier_is_specific
            ),
            concise_basis=(
                result.concise_basis.strip()
            ),
        )


def review_and_compile_nonobviousness_adjudication(
    *,
    backend: NonObviousnessAdjudicationBackendProtocol,
    readiness: NonObviousnessReviewGate,
    packet: NonObviousnessEvidencePacket,
    prior_art: PriorArtPacket,
) -> IndependentAdjudicationOutcome:
    """Independent review only after the deterministic readiness gate."""

    if (
        readiness.readiness
        != "READY_FOR_NONOBVIOUSNESS_REVIEW"
    ):
        raw = NonObviousnessAdjudicationDraft(
            proposed_verdict=(
                "INSUFFICIENT_FOR_JUDGMENT"
            ),
            direct_reconstruction_from_known_relations=False,
            additional_scientific_assumptions=(),
            prediction_distinguishes_from_routine_baseline=False,
            falsifier_is_specific=False,
            concise_basis="",
        )

        compiled = (
            compile_nonobviousness_adjudication(
                readiness=readiness,
                packet=packet,
                draft=raw,
            )
        )

        return IndependentAdjudicationOutcome(
            review_performed=False,
            raw_draft=raw,
            sanitized_draft=raw,
            sanitizer_reason_codes=(),
            compiled=compiled,
        )

    raw = backend.adjudicate(
        packet=packet,
        prior_art=prior_art,
    )

    sanitized = (
        sanitize_adjudication_draft(
            packet=packet,
            draft=raw,
        )
    )

    compiled = (
        compile_nonobviousness_adjudication(
            readiness=readiness,
            packet=packet,
            draft=sanitized.draft,
        )
    )

    return IndependentAdjudicationOutcome(
        review_performed=True,
        raw_draft=raw,
        sanitized_draft=(
            sanitized.draft
        ),
        sanitizer_reason_codes=(
            sanitized.reason_codes
        ),
        compiled=compiled,
    )
