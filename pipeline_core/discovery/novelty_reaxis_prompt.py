from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.novelty_prior_art_boundary import render_higher_order_relational_gap_boundary

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyCard,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisContext,
)
from pipeline_core.discovery.hypothesis_prompt import (
    HypothesisPrompt,
    HypothesisPromptAssembler,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyGap,
)


def _prompt_sha(
    version: str,
    system: str,
    user: str,
) -> str:
    canonical = json.dumps(
        {
            "prompt_version": version,
            "system_prompt": system,
            "user_prompt": user,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


class FreshNoveltyReaxisPromptAssembler:
    """Fresh hypothesis generation after a strong known-axis signal.

    Unlike same-premise novelty refinement, this prompt may select a
    different subset of positive premises from the SAME grounded
    HypothesisContext.

    External prior art remains exclusion/boundary information only.
    """

    prompt_version = "novelty-fresh-context-reaxis-prompt-v2-relgap-boundary"

    def __init__(
        self,
        *,
        original: HypothesisCard,
        gap: NoveltyGap,
        targeted_card: ExternalNoveltyCard,
        allowed_premise_ids: list[str],
        required_unused_premise_ids: list[str],
    ) -> None:
        allowed = list(dict.fromkeys(map(str, allowed_premise_ids)))
        unused = list(
            dict.fromkeys(map(str, required_unused_premise_ids))
        )

        if not unused:
            raise ValueError(
                "fresh re-axis requires at least one unused eligible premise"
            )

        if not set(unused).issubset(set(allowed)):
            raise ValueError(
                "required unused premise IDs must be a subset of allowed IDs"
            )

        self.original = original
        self.gap = gap
        self.targeted_card = targeted_card
        self.allowed_premise_ids = allowed
        self.required_unused_premise_ids = unused

    def build(
        self,
        context: HypothesisContext,
    ) -> HypothesisPrompt:
        base = HypothesisPromptAssembler(
            max_hypotheses=1,
        ).build(context)

        system = (
            base.system_prompt
            + """

NOVELTY-DRIVEN FRESH CONTEXT RE-AXIS
====================================
The previous grounded hypothesis has a strong external known-axis signal.

This is NOT a same-premise wording refinement.

You may construct ONE fresh hypothesis from a DIFFERENT combination of
eligible positive premises in the SAME supplied HypothesisContext.

Rules specific to this re-axis:
1. Stay within the same scientific question, domain, and grounded context.
2. Use ONLY premise_statement_ids explicitly listed in ALLOWED FRESH PREMISE IDS.
3. If you generate a hypothesis, it MUST use at least one ID from REQUIRED UNUSED PREMISE IDS.
4. You may reuse some original premise IDs when scientifically useful.
5. You MAY change hypothesis_type when the new grounded relation requires it.
6. Prefer a genuinely different relation axis: moderator/interaction,
   context dependency, cross-evidence synthesis, mediator, boundary/failure
   regime, or another falsifiable dependency supported by the context.
7. Do NOT evade prior art by adding cosmetic qualifiers to the same known
   first-order relation.
8. External prior-art summaries below are EXCLUSION/BOUNDARY information only.
   They MUST NOT become positive scientific premises.
9. Do not claim novelty or literature-wide absence.
10. Return exactly ONE fresh hypothesis or abstain.

11. For every falsification criterion, its observable MUST copy verbatim one observable from predicted_observations. Put the contrary or falsifying result in falsifying_outcome; do NOT create, rename, paraphrase, or specialize the observable in the falsifier.

The new hypothesis must be grounded because its positive evidence comes from
the supplied context; the inferential relation itself remains a hypothesis.
"""
        )

        reviews: list[dict[str, object]] = []

        for review in self.targeted_card.claim_reviews:
            reviews.append(
                {
                    "claim_id": review.claim_id,
                    "importance": review.importance,
                    "status": review.status,
                    "claim_text": review.claim_text,
                    "interpretation": review.interpretation,
                    "matches": [
                        {
                            "relationship": match.relationship,
                            "title": match.title,
                            "rationale": match.rationale,
                        }
                        for match in review.matches[:3]
                    ],
                }
            )

        user = (
            base.user_prompt
            + "\n\n"
            + "\n".join(
                [
                    "NOVELTY RE-AXIS BOUNDARY",
                    "========================",
                    f"original_hypothesis_id: {self.original.hypothesis_id}",
                    f"original_title: {self.original.title}",
                    f"original_statement: {self.original.hypothesis_statement}",
                    (
                        "original_premise_statement_ids: "
                        + json.dumps(
                            self.original.premise_statement_ids,
                            ensure_ascii=False,
                        )
                    ),
                    (
                        "original_hypothesis_type: "
                        + self.original.hypothesis_type
                    ),
                    "",
                    f"gap_action: {self.gap.action}",
                    f"gap_differentiator: {self.gap.differentiator}",
                    "",
                    "ALREADY-KNOWN / ADJACENT BOUNDARY",
                    "=================================",
                    *(
                        [
                            f"- {value}"
                            for value in self.gap.already_known_boundary
                        ]
                        or ["- NONE"]
                    ),
                    "",
                    "UNRESOLVED BOUNDARY",
                    "===================",
                    *(
                        [
                            f"- {value}"
                            for value in self.gap.unresolved_boundary
                        ]
                        or ["- NONE"]
                    ),
                    "",
                    "TARGETED EXTERNAL PRIOR-ART STATUS",
                    "==================================",
                    f"status: {self.targeted_card.status}",
                    json.dumps(
                        reviews,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "",
                    "IMPORTANT: everything in the prior-art block above is",
                    "negative/exclusion boundary information only.",
                    "It is NOT a positive scientific premise.",
                    "",
                    "ALLOWED FRESH PREMISE IDS",
                    "=========================",
                    *[
                        f"- {statement_id}"
                        for statement_id in self.allowed_premise_ids
                    ],
                    "",
                    "REQUIRED UNUSED PREMISE IDS",
                    "===========================",
                    (
                        "A generated fresh hypothesis MUST select at least "
                        "one of these IDs:"
                    ),
                    *[
                        f"- {statement_id}"
                        for statement_id
                        in self.required_unused_premise_ids
                    ],
                    "",
                    "TASK",
                    "====",
                    (
                        "Generate exactly ONE fresh falsifiable hypothesis "
                        "from the same grounded context, or abstain."
                    ),
                    (
                        "Do not merely paraphrase or qualify the original "
                        "known/adjacent relation."
                    ),
                    (
                        "The selected premise_statement_ids must be a subset "
                        "of ALLOWED FRESH PREMISE IDS."
                    ),
                    (
                        "At least one selected premise must come from "
                        "REQUIRED UNUSED PREMISE IDS."
                    ),
                    (
                        "Predictions and falsifiers must test the new "
                        "relation axis itself."
                    ),
                    (
                        "External literature must not appear in "
                        "premise_statement_ids."
                    ),
                ]
            )
        )

        relational_gap_boundary = (
            render_higher_order_relational_gap_boundary(
                self.targeted_card
            )
        )

        if relational_gap_boundary:
            user = (
                user
                + "\n\n"
                + relational_gap_boundary
            )

        return HypothesisPrompt(
            prompt_version=self.prompt_version,
            system_prompt=system,
            user_prompt=user,
            prompt_sha256=_prompt_sha(
                self.prompt_version,
                system,
                user,
            ),
        )
