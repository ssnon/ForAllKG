from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from domains.sers.context_contracts import (
    SERSContextSignature,
)
from domains.sers.hypothesis_context_contracts import (
    HypothesisContextInterpretationDraft,
    expected_hypothesis_context_assertions,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
)


HYPOTHESIS_CONTEXT_PROMPT_VERSION = (
    "sers-hypothesis-context-prompt-v1.1"
)


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _binding_payload(
    binding: object,
) -> dict[str, object] | None:
    if binding is None:
        return None

    return {
        "basis": binding.basis,
        "owner_ref_id": binding.owner_ref_id,
        "owner_label": binding.owner_label,
        "owner_type": binding.owner_type,
        "relation": binding.relation,
    }


def _source_fact_payload(
    signature: SERSContextSignature,
) -> dict[str, object]:
    return {
        "signature_id":
            signature.signature_id,
        "scope":
            signature.scope,
        "source_ref_id":
            signature.source_ref_id,
        "facts": [
            {
                "fact_id":
                    fact.fact_id,
                "dimension":
                    fact.dimension,
                "scientific_role":
                    fact.scientific_role,
                "knowledge_state":
                    fact.knowledge_state,
                "value":
                    fact.value,
                "normalized_value":
                    fact.normalized_value,
                "binding":
                    _binding_payload(
                        fact.binding
                    ),
                "tags":
                    sorted(
                        fact.tags
                    ),
            }
            for fact in sorted(
                signature.facts,
                key=lambda row:
                    row.fact_id,
            )
        ],
    }


@dataclass(frozen=True)
class HypothesisContextPrompt:
    prompt_version: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> "HypothesisContextPrompt":
        canonical = _compact_json({
            "prompt_version":
                HYPOTHESIS_CONTEXT_PROMPT_VERSION,
            "system_prompt":
                system_prompt,
            "user_prompt":
                user_prompt,
        })

        return cls(
            prompt_version=
                HYPOTHESIS_CONTEXT_PROMPT_VERSION,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_sha256=_sha256(
                canonical
            ),
        )


SYSTEM_PROMPT = """
You are the SERS hypothesis-context interpretation component of an
evidence-grounded scientific hypothesis system.

Your task is narrow and descriptive:
map scientific context mentions in the supplied hypothesis assertions
to the supplied source-context facts.

You do NOT decide whether the hypothesis passes, fails, is novel, or
requires repair. You do NOT assign compatibility labels. You do NOT
rewrite the hypothesis.

Use only the supplied hypothesis assertions and source-context facts.

Output requirements:
1. Return every expected assertion exactly once, with the supplied
   assertion_id, assertion_kind, and assertion_text unchanged.
2. An assertion may have zero or more context mentions.
3. mention_text must be an exact contiguous span of assertion_text
   after ordinary whitespace normalization.
4. source_fact_ids may reference only supplied source facts.
5. Preserve UNKNOWN source knowledge as unknown; never invent a
   material state, environment, optical condition, analyte, reporter,
   morphology, support role, or measurement context.
6. Use treatment precisely:
   - preserve: the source context is retained with the same scientific
     dimension and role.
   - generalize: one or more source facts of the same context dimension
     are abstracted to a broader description without changing their
     scientific attachment.
   - intentionally_vary: the same context dimension is explicitly used
     as an experimental variable, moderator, or comparison condition.
   - reattach: source context semantics are attached by the hypothesis
     to a different scientific owner, dimension, or role. Prefer this
     over combine whenever scientific attachment changes.
   - combine: two or more source facts are merged into one hypothesis
     mention while their scientific attachment is not transferred.
   - introduce: the hypothesis introduces a context claim for which no
     supplied source fact is an appropriate source.
   - reference_only: the hypothesis mentions a source context without
     using it as a variable, moderator, or mechanistic context.
   - uncertain: the mapping cannot be determined reliably from the
     supplied material.
7. asserted_owner_label describes the scientific object to which the
   hypothesis attaches the mentioned context. Use null when no owner is
   stated or reliably inferable, except reattach requires an owner.
8. Do not force intentional_variation merely because two experimental
   conditions are contrasted. If the hypothesis changes the source
   scientific role or owner while making that contrast, represent that
   attachment change explicitly.
9. Do not collapse support morphology, nanoparticle structural motif,
   nanogap regime, material identity, material state, and environment
   into one another.
10. Emit a context mention only when the phrase actually represents one
    of the supplied SERS context dimensions. Do not force response
    variables, causal mechanisms, enhancement magnitudes, or scientific
    outcomes into a context dimension merely because they are important.
11. In particular:
    - local electric-field intensity/distribution and SERS intensity are
      responses or outcomes, not optical_condition;
    - chemical/electromagnetic enhancement mechanisms are mechanisms,
      not material_state;
    - material_state means an actual physical/chemical state such as
      oxidation/state/phase where explicitly supported;
    - optical_condition concerns spectral/excitation/polarization or
      comparable optical setup context;
    - measurement_geometry may include explicit molecule-particle
      separation/proximity when that is genuinely asserted as geometry.
12. For preserve, generalize, or intentionally_vary, cite only source
    facts whose dimension equals asserted_dimension. Do not attach a
    material_identity fact to a substrate generalization, or vice versa.
13. If a scientifically relevant phrase is not itself a context variable,
    omit it from mentions rather than assigning the nearest available
    context dimension.
14. The treatment "introduce" has a narrow meaning here: no supplied
    typed source-context fact is an appropriate source for that context
    mention. It does NOT mean that the broader scientific evidence or
    grounded premises fail to support the scientific concept. Do not
    use this interpretation task to make broader evidence-support
    judgments.
""".strip()


class SERSHypothesisContextPromptAssembler:
    def build(
        self,
        *,
        card: HypothesisCard,
        source_signatures: list[
            SERSContextSignature
        ],
    ) -> HypothesisContextPrompt:
        if not source_signatures:
            raise ValueError(
                "source_signatures must not be empty"
            )

        if any(
            signature.domain_profile_id
            != card.domain_profile_id
            for signature in source_signatures
        ):
            raise ValueError(
                "source signature domain mismatch"
            )

        signatures = sorted(
            source_signatures,
            key=lambda row:
                row.signature_id,
        )

        payload = {
            "hypothesis": {
                "hypothesis_id":
                    card.hypothesis_id,
                "title":
                    card.title,
                "hypothesis_type":
                    card.hypothesis_type,
            },
            "expected_assertions":
                list(
                    expected_hypothesis_context_assertions(
                        card
                    )
                ),
            "source_context_signatures": [
                _source_fact_payload(
                    signature
                )
                for signature
                in signatures
            ],
        }

        user_prompt = (
            "SERS HYPOTHESIS-CONTEXT INTERPRETATION INPUT\n"
            "============================================\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n\n"
            "Return only the structured interpretation requested by "
            "the response schema."
        )

        return HypothesisContextPrompt.create(
            system_prompt=
                SYSTEM_PROMPT,
            user_prompt=
                user_prompt,
        )


    def repair_after_validation(
        self,
        *,
        original_prompt: HypothesisContextPrompt,
        previous_draft:
            HypothesisContextInterpretationDraft,
        issues: tuple[str, ...] | list[str],
        source_signatures: list[
            SERSContextSignature
        ],
    ) -> HypothesisContextPrompt:
        """Build one bounded reference-integrity repair prompt.

        This is not scientific hypothesis repair.  It only gives the
        interpreter one opportunity to correct deterministic reference
        errors while preserving the supplied assertions and context
        fact surface.
        """
        valid_fact_ids = sorted({
            fact.fact_id
            for signature
            in source_signatures
            for fact in signature.facts
        })

        repair_payload = {
            "validation_issues": [
                str(issue)
                for issue in issues
            ],
            "valid_source_fact_ids":
                valid_fact_ids,
            "previous_draft":
                previous_draft.model_dump(
                    mode="json"
                ),
        }

        repair_system = (
            SYSTEM_PROMPT
            + "\n\n"
            + """
VALIDATION REPAIR MODE
======================
The previous structured interpretation passed the response schema but
failed deterministic source-reference validation.

This is a bounded reference-integrity repair, not a new interpretation
task.

- Preserve every assertion_id, assertion_kind, and assertion_text.
- Do not add scientific claims or new hypothesis content.
- Do not invent or synthesize source fact IDs.
- Every source_fact_id in the replacement MUST be copied exactly from
  VALID_SOURCE_FACT_IDS.
- Repair only the context mentions necessary to resolve the listed
  deterministic issues.
- If no supplied source fact appropriately supports a context mention,
  do not fabricate one. Use the existing treatment semantics
  faithfully, including introduce with an empty source_fact_ids list
  when appropriate, or remove an unsupported mention when the phrase
  is not actually a context variable.
- Return one complete replacement draft.
""".strip()
        )

        repair_user = (
            original_prompt.user_prompt
            + "\n\n"
            + "VALIDATION REPAIR INPUT\n"
            + "=======================\n"
            + json.dumps(
                repair_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

        return HypothesisContextPrompt.create(
            system_prompt=
                repair_system,
            user_prompt=
                repair_user,
        )
