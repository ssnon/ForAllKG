from __future__ import annotations

from dataclasses import dataclass

from domains.sers.context_contracts import (
    SERSContextSignature,
)
from domains.sers.hypothesis_context_contracts import (
    HypothesisContextInterpretation,
    HypothesisContextInterpretationCompiler,
    HypothesisContextInterpretationDraft,
    HypothesisContextInterpretationValidationError,
)
from domains.sers.hypothesis_context_llm import (
    HypothesisContextBackend,
    HypothesisContextGeneration,
)
from domains.sers.hypothesis_context_prompt import (
    HypothesisContextPrompt,
    SERSHypothesisContextPromptAssembler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
)


@dataclass(frozen=True)
class HypothesisContextInterpreterOutcome:
    prompt: HypothesisContextPrompt
    generation: HypothesisContextGeneration
    canonical_draft: HypothesisContextInterpretationDraft
    interpretation: HypothesisContextInterpretation


class HypothesisContextInterpreterValidationError(
    ValueError
):
    """Preserve generation evidence when deterministic validation fails."""

    def __init__(
        self,
        *,
        prompt: HypothesisContextPrompt,
        generation: HypothesisContextGeneration,
        canonical_draft: HypothesisContextInterpretationDraft,
        validation_error:
            HypothesisContextInterpretationValidationError,
    ) -> None:
        self.prompt = prompt
        self.generation = generation
        self.canonical_draft = canonical_draft
        self.validation_error = validation_error
        self.issues = validation_error.issues

        super().__init__(
            "invalid SERS hypothesis-context "
            "interpretation after deterministic "
            "canonicalization: "
            + "; ".join(
                self.issues
            )
        )


def _canonicalize_interpretation_draft(
    *,
    draft: HypothesisContextInterpretationDraft,
    authoritative_hypothesis_id: str,
    source_signatures: list[
        SERSContextSignature
    ],
) -> HypothesisContextInterpretationDraft:
    """Canonicalize orchestration/citation noise without semantic repair.

    Three operations are permitted:

    1. hypothesis_id comes from the authoritative HypothesisCard
       identity rather than model reproduction.

    2. source_signature_ids comes from the authoritative supplied
       signature set rather than model reproduction.

    3. For treatments whose semantics require dimension preservation,
       remove *known* source facts from other dimensions only when at
       least one known source fact of the asserted dimension remains.

       If no correctly dimensioned source fact exists, leave the
       model output unchanged so the deterministic compiler rejects it.

    Unknown source fact IDs are never hidden by canonicalization.
    Reattach/combine/introduce/reference_only/uncertain are never
    dimension-projected here.
    """

    canonical_signature_ids = sorted({
        signature.signature_id
        for signature in source_signatures
    })

    fact_by_id = {
        fact.fact_id: fact
        for signature in source_signatures
        for fact in signature.facts
    }

    same_dimension_treatments = {
        "preserve",
        "generalize",
        "intentionally_vary",
    }

    canonical_assertions = []

    for assertion in draft.assertions:
        canonical_mentions = []

        for mention in assertion.mentions:
            canonical_mention = mention

            if (
                mention.treatment
                in same_dimension_treatments
                and mention.source_fact_ids
            ):
                matching_known = [
                    fact_id
                    for fact_id
                    in mention.source_fact_ids
                    if (
                        fact_id in fact_by_id
                        and fact_by_id[
                            fact_id
                        ].dimension
                        == mention.asserted_dimension
                    )
                ]

                if matching_known:
                    projected = []

                    for fact_id in (
                        mention.source_fact_ids
                    ):
                        fact = fact_by_id.get(
                            fact_id
                        )

                        # Unknown IDs stay visible so validation
                        # can still reject them.
                        if fact is None:
                            projected.append(
                                fact_id
                            )
                            continue

                        if (
                            fact.dimension
                            == mention.asserted_dimension
                        ):
                            projected.append(
                                fact_id
                            )

                    canonical_mention = (
                        mention.model_copy(
                            update={
                                "source_fact_ids":
                                    projected,
                            }
                        )
                    )

            canonical_mentions.append(
                canonical_mention
            )

        canonical_assertions.append(
            assertion.model_copy(
                update={
                    "mentions":
                        canonical_mentions,
                }
            )
        )

    return draft.model_copy(
        update={
            "hypothesis_id":
                authoritative_hypothesis_id,
            "source_signature_ids":
                canonical_signature_ids,
            "assertions":
                canonical_assertions,
        }
    )


class SERSHypothesisContextInterpreter:
    def __init__(
        self,
        backend: HypothesisContextBackend,
        *,
        prompt_assembler:
            SERSHypothesisContextPromptAssembler | None = None,
        compiler:
            HypothesisContextInterpretationCompiler | None = None,
    ) -> None:
        self.backend = backend

        self.prompt_assembler = (
            prompt_assembler
            or SERSHypothesisContextPromptAssembler()
        )

        self.compiler = (
            compiler
            or HypothesisContextInterpretationCompiler()
        )

    def interpret(
        self,
        *,
        card: HypothesisCard,
        source_signatures: list[
            SERSContextSignature
        ],
    ) -> HypothesisContextInterpreterOutcome:
        prompt = (
            self.prompt_assembler.build(
                card=card,
                source_signatures=
                    source_signatures,
            )
        )

        generation = (
            self.backend.interpret(
                prompt
            )
        )

        canonical_draft = (
            _canonicalize_interpretation_draft(
                draft=
                    generation.draft,
                authoritative_hypothesis_id=
                    card.hypothesis_id,
                source_signatures=
                    source_signatures,
            )
        )

        try:
            interpretation = (
                self.compiler.compile(
                    card=card,
                    source_signatures=
                        source_signatures,
                    draft=
                        canonical_draft,
                )
            )

        except (
            HypothesisContextInterpretationValidationError
        ) as exc:
            # Only source-reference hallucination is eligible for
            # automatic repair. Scientific/context-semantic validation
            # failures remain immediate fail-closed outcomes.
            reference_only = (
                bool(exc.issues)
                and all(
                    "unknown source fact "
                    in str(issue)
                    for issue in exc.issues
                )
            )

            if not reference_only:
                raise (
                    HypothesisContextInterpreterValidationError(
                        prompt=prompt,
                        generation=generation,
                        canonical_draft=
                            canonical_draft,
                        validation_error=exc,
                    )
                ) from exc

            repair_prompt = (
                self.prompt_assembler
                .repair_after_validation(
                    original_prompt=prompt,
                    previous_draft=
                        canonical_draft,
                    issues=list(
                        exc.issues
                    ),
                    source_signatures=
                        source_signatures,
                )
            )

            repair_generation = (
                self.backend.interpret(
                    repair_prompt
                )
            )

            repair_canonical_draft = (
                _canonicalize_interpretation_draft(
                    draft=
                        repair_generation.draft,
                    authoritative_hypothesis_id=
                        card.hypothesis_id,
                    source_signatures=
                        source_signatures,
                )
            )

            try:
                interpretation = (
                    self.compiler.compile(
                        card=card,
                        source_signatures=
                            source_signatures,
                        draft=
                            repair_canonical_draft,
                    )
                )

            except (
                HypothesisContextInterpretationValidationError
            ) as repair_exc:
                raise (
                    HypothesisContextInterpreterValidationError(
                        prompt=
                            repair_prompt,
                        generation=
                            repair_generation,
                        canonical_draft=
                            repair_canonical_draft,
                        validation_error=
                            repair_exc,
                    )
                ) from repair_exc

            # Final outcome corresponds to the accepted replacement
            # generation and its exact repair prompt.
            prompt = repair_prompt
            generation = repair_generation
            canonical_draft = (
                repair_canonical_draft
            )

        return (
            HypothesisContextInterpreterOutcome(
                prompt=prompt,
                generation=generation,
                canonical_draft=
                    canonical_draft,
                interpretation=
                    interpretation,
            )
        )
