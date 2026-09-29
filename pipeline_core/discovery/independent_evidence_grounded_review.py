from __future__ import annotations

from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReviewDraft,
    NoveltyClaim,
)
from pipeline_core.discovery.prior_art_review_audit import (
    record_prior_art_review_call,
)
from pipeline_core.llm.llm_telemetry import (
    run_instructor_structured_call,
)


_SINGLE_WORK_SYSTEM = """You are a blinded scientific prior-art relationship adjudicator.

You receive ONE atomic scientific claim and ONE literature record. Use ONLY the supplied title and abstract. Do not use outside knowledge. You are not told any previous relationship labels.

Your job is to classify the record's relationship to the claim using this decision procedure.

DECISION PROCEDURE

1. DIRECT_PRIOR_ART
Use only when the title/abstract explicitly states, tests, compares, or demonstrates essentially the same scientific relation as the claim, including defining moderator/conditional/directional structure when that structure is part of the claim.

2. PARTIAL_PRIOR_ART
Use only when the abstract explicitly preserves the claim's RELATION NUCLEUS and establishes a substantial subset of the SAME scientific relation, but not the full formulation.
Shared variables, materials, mechanisms, or thematic proximity are not enough.

3. LOWER_ORDER_RELATION_PRIOR_ART
For a higher-order, moderator, conditional, mediated, or interaction claim, use this when the abstract explicitly establishes a nontrivial multivariable LOWER-ORDER SUBRELATION from the claim while omitting the higher-order moderator/conditional/interaction structure.
This is distinct from PARTIAL_PRIOR_ART:
- PARTIAL preserves the higher-level relation nucleus in incomplete form.
- LOWER_ORDER establishes a scientifically meaningful underlying relation after the higher-order extension is removed.

4. DIRECTIONAL_COUNTEREVIDENCE
Use when the claim has an ordered/directional prediction and the abstract materially challenges that direction in a scientifically relevant neighboring scope, such as an opposite trend, weak/absent correlation, regime dependence, or another determinant dominating.
Do not use this merely because the record fails to confirm the proposed direction.

5. COMPONENT_ONLY
Use when the record establishes relevant ingredients, variables, mechanisms, contexts, materials, separate main effects, or one comparison arm, but does not establish the claim's relation nucleus or a meaningful lower-order multivariable relation.

6. CONFLICTING_PRIOR_ART
Use only for a materially opposing relation/result in sufficiently overlapping scientific scope.

7. CONTEXTUAL_CONFLICT
Use when the record challenges a broader assumption or relation but materially differs in scientific scope.

8. UNRELATED
Use when the record does not materially bear on the claim.

9. INSUFFICIENT_METADATA
Use when the supplied title/abstract is inadequate to decide.

HIGHER-ORDER CONSISTENCY RULE
For a claim of logical form "M changes how X affects Y":
- M affects Y plus X affects Y separately => COMPONENT_ONLY.
- X affects Y explicitly, without M changing that relation => LOWER_ORDER_RELATION_PRIOR_ART.
- M changes/conditions the X-to-Y relation, but some scope/detail is incomplete => PARTIAL_PRIOR_ART.
- M changes/conditions the X-to-Y relation in essentially the claimed scope => DIRECT_PRIOR_ART.

EVIDENCE-SPAN CONTRACT
For every relationship other than UNRELATED or INSUFFICIENT_METADATA, provide 1-3 exact contiguous spans copied verbatim from the supplied abstract that justify the classification.
Do not paraphrase inside evidence_spans.
If the abstract does not contain a span supporting the asserted relationship, choose a weaker classification.

Return only the structured adjudication requested by the caller."""


def build_single_work_user_prompt(
    *,
    claim_text: str,
    work: dict[str, Any],
) -> str:
    abstract = str(work.get("abstract") or "")
    return "\n".join(
        [
            "ATOMIC CLAIM",
            "============",
            claim_text,
            "",
            "LITERATURE RECORD",
            "=================",
            f"work_id: {work['work_id']}",
            f"title: {work.get('title', '')}",
            f"doi: {work.get('doi')}",
            "abstract:",
            abstract if abstract else "[NO ABSTRACT AVAILABLE]",
            "",
            "Classify only the relationship between this record and this claim.",
        ]
    )


class IndependentEvidenceGroundedClaimReviewBackend:
    """S233/S234-equivalent one-claim x one-work review."""

    def __init__(self, base_backend: Any) -> None:
        self.base_backend = base_backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_backend, name)

    def _review_one(
        self,
        claim: NoveltyClaim,
        work: dict[str, Any],
    ) -> ClaimPriorArtReviewDraft:
        user = build_single_work_user_prompt(
            claim_text=claim.text,
            work=work,
        )

        record = getattr(self.base_backend, "_record", None)
        if callable(record):
            record(
                f"independent_review_{claim.claim_id}_{work['work_id']}",
                _SINGLE_WORK_SYSTEM,
                user,
            )

        result, event = run_instructor_structured_call(
            self.base_backend._get_client().chat.completions,
            model=self.base_backend.model_name,
            response_model=ClaimPriorArtReviewDraft,
            messages=[
                {"role": "system", "content": _SINGLE_WORK_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=self.base_backend.temperature,
            max_retries=self.base_backend.parse_retries,
            telemetry_path=self.base_backend.telemetry_path,
            telemetry_context={
                **getattr(self.base_backend, "telemetry_context", {}),
                "pipeline": "external_novelty",
                "stage": "independent_prior_art_review_s237b",
                "call_kind": "structured",
                "claim_id": claim.claim_id,
                "work_id": str(work["work_id"]),
            },
        )

        if not isinstance(result, ClaimPriorArtReviewDraft):
            result = ClaimPriorArtReviewDraft.model_validate(result)

        allowed = str(work["work_id"])
        result = result.model_copy(
            update={
                "matches": [
                    row
                    for row in result.matches
                    if row.work_id == allowed
                ][:1],
            }
        )

        record_prior_art_review_call(
            system_prompt=_SINGLE_WORK_SYSTEM,
            user_prompt=user,
            response_schema=ClaimPriorArtReviewDraft,
            result=result,
            model=self.base_backend.model_name,
            instructor_mode=self.base_backend.instructor_mode,
            temperature=self.base_backend.temperature,
            claim_id=claim.claim_id,
            hypothesis_id=claim.hypothesis_id,
            claim_text=claim.text,
            works=[work],
            telemetry_event=event,
        )

        return result

    def review_claim(
        self,
        claim: NoveltyClaim,
        works: list[dict[str, Any]],
    ) -> ClaimPriorArtReviewDraft:
        matches = []
        interpretations = []

        for work in works:
            draft = self._review_one(claim, work)
            matches.extend(draft.matches)
            if draft.interpretation:
                interpretations.append(draft.interpretation)

        return ClaimPriorArtReviewDraft(
            matches=matches,
            interpretation=(
                "Independent per-work review. "
                + " | ".join(interpretations)
            ),
        )

    def review_diagnostic_claim(
        self,
        claim: NoveltyClaim,
        works: list[dict[str, Any]],
    ):
        return self.base_backend.review_diagnostic_claim(claim, works)


__all__ = [
    "IndependentEvidenceGroundedClaimReviewBackend",
    "build_single_work_user_prompt",
]
