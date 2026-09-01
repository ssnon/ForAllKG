from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.novelty_adjudication import (
    NonObviousnessAdjudicationDraft,
    NonObviousnessEvidencePacket,
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
