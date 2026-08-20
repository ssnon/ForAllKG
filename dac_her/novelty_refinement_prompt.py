from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.external_novelty_contracts import ExternalNoveltyCard
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard, HypothesisContext, HypothesisPortfolioDraft
from dac_her.hypothesis_prompt import HypothesisPrompt
from dac_her.novelty_refinement_contracts import NoveltyGap


def _sha(system: str, user: str) -> str:
    return hashlib.sha256((system + "\n---\n" + user).encode("utf-8")).hexdigest()


class NoveltyRefinementPromptAssembler:
    """Prompt adapter compatible with HypothesisDraftBackend.

    External prior art is an exclusion/boundary signal only. The model must keep
    the exact grounded premise IDs of the original hypothesis.
    """

    prompt_version = "novelty-refinement-prompt-v2.8.0-a6"

    def __init__(
        self,
        *,
        original: HypothesisCard,
        gap: NoveltyGap,
        targeted_card: ExternalNoveltyCard,
    ) -> None:
        self.original = original
        self.gap = gap
        self.targeted_card = targeted_card

    def build(self, context: HypothesisContext) -> HypothesisPrompt:
        system = """You refine ONE scientific hypothesis after a bounded external prior-art search.

Epistemic rules:
1. Grounded evidence is ONLY the supplied HypothesisContext premise statements.
2. External prior-art summaries are NOT positive scientific premises. They only tell you which formulations are already known, adjacent, conflicting, or insufficiently searched.
3. Preserve the exact premise_statement_ids and gap_statement_ids from the original hypothesis. Do not add, remove, or substitute evidence IDs.
4. Preserve the assigned discovery direction and scientific scope. Do not escape prior art by changing to a different reaction, catalyst class, or unrelated mechanism.
5. Make at most one bounded refinement. The new hypothesis must remain falsifiable and must sharpen the unresolved differentiator rather than merely adding qualifiers.
6. Never claim novelty, priority, first report, or literature-wide absence.
7. Return exactly ONE hypothesis, or abstain if a grounded refinement is not possible.

A useful refinement introduces a more precise moderator, mediator, conditional dependency, pathway competition, descriptor interaction, or distinctive prediction while staying supported by the same grounded premises.
"""
        reviews = []
        for review in self.targeted_card.claim_reviews:
            reviews.append(
                {
                    "claim_id": review.claim_id,
                    "status": review.status,
                    "claim_text": review.claim_text,
                    "interpretation": review.interpretation,
                    "matches": [
                        {
                            "relationship": x.relationship,
                            "title": x.title,
                            "rationale": x.rationale,
                            "reaction_domain_relevance": x.reaction_domain_relevance,
                            "catalyst_scope_relevance": x.catalyst_scope_relevance,
                        }
                        for x in review.matches[:3]
                    ],
                }
            )
        user = "\n".join(
            [
                "ORIGINAL HYPOTHESIS",
                "===================",
                f"title: {self.original.title}",
                f"statement: {self.original.hypothesis_statement}",
                f"type: {self.original.hypothesis_type}",
                f"premise_statement_ids: {json.dumps(self.original.premise_statement_ids)}",
                f"gap_statement_ids: {json.dumps(self.original.gap_statement_ids)}",
                f"inferential_bridge: {self.original.inferential_bridge}",
                "",
                "NOVELTY GAP",
                "===========",
                f"action: {self.gap.action}",
                f"differentiator: {self.gap.differentiator}",
                "already-known boundary:",
                *([f"- {x}" for x in self.gap.already_known_boundary] or ["- NONE"]),
                "unresolved boundary:",
                *([f"- {x}" for x in self.gap.unresolved_boundary] or ["- NONE"]),
                "",
                "TARGETED PRIOR-ART REASSESSMENT",
                "===============================",
                f"status: {self.targeted_card.status}",
                json.dumps(reviews, ensure_ascii=False, indent=2),
                "",
                "TASK",
                "====",
                "Return exactly ONE refined hypothesis using the SAME premise_statement_ids and gap_statement_ids.",
                "The refinement must focus on the unresolved differentiator above and avoid simply restating relations marked direct/partial prior art.",
                "Predictions and falsifiers must test the refined differentiator itself.",
                "If this cannot be done without treating external prior art as evidence or changing scientific scope, abstain.",
            ]
        )
        return HypothesisPrompt(
            prompt_version=self.prompt_version,
            system_prompt=system,
            user_prompt=user,
            prompt_sha256=_sha(system, user),
        )

    def repair_feedback(self, *, previous_draft: HypothesisPortfolioDraft, issues: list[object]) -> str:
        codes = [
            str(getattr(x, "code", getattr(x, "message", x)))
            for x in issues
        ]
        return "\n".join(
            [
                "REFINEMENT REPAIR",
                "=================",
                "Repair the single proposed hypothesis without changing premise_statement_ids or gap_statement_ids.",
                "Do not use external prior art as a positive premise.",
                "Issues:",
                *[f"- {x}" for x in codes],
                "Return exactly one corrected hypothesis or abstain.",
            ]
        )
