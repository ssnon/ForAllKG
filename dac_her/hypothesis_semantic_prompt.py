from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from dac_her.hypothesis_benchmark_contracts import HypothesisEvaluationReport
from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from dac_her.hypothesis_semantic_contracts import SEMANTIC_DIMENSIONS


CRITIC_PROMPT_VERSION = "hypothesis-semantic-critic-prompt-v2.6.2-b3"


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HypothesisSemanticPrompt:
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
    ) -> "HypothesisSemanticPrompt":
        canonical = _compact_json(
            {
                "prompt_version": CRITIC_PROMPT_VERSION,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return cls(
            prompt_version=CRITIC_PROMPT_VERSION,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_sha256=_sha256(canonical),
        )


SYSTEM_PROMPT = """You are the semantic critic for an evidence-grounded scientific hypothesis system.

Evaluate the hypothesis; do not repair or rewrite it.

A hypothesis is not evidence. The Hypothesis Maker is explicitly allowed to make a new inferential leap beyond what is directly reported, so do NOT penalize a hypothesis merely because the proposed mechanism is not already stated in the evidence.

Your task is to distinguish:
1. a supplied fact or evidence synthesis,
2. a reasonable, explicitly hypothetical inference built from the supplied evidence,
3. an unsupported assertion that silently upgrades, distorts, or overextends the evidence.

Use ONLY the supplied HypothesisContext, HypothesisPortfolio, and deterministic evaluation report. Do not introduce outside scientific facts, literature knowledge, novelty judgments, or your own preferred mechanism.

Do not modify the portfolio. Do not suggest a replacement hypothesis, revised wording, experiment protocol, or synthesis procedure.

Verdicts:
- pass: the dimension is semantically well controlled.
- warning: plausible/usable, but the inference is stronger, more specific, ambiguous, or less informative than the supplied evidence clearly warrants.
- fail: the dimension contains a substantive semantic violation, such as evidence distortion, unqualified causalization, candidate-to-fact promotion, cross-paper system conflation, or an effectively non-falsifiable claim.
- not_applicable: the dimension genuinely does not apply to this portfolio.

Evaluate EVERY required dimension exactly once.

Reference namespace discipline:
- hypothesis_ids refer only to hypothesis_portfolio.hypotheses[*].hypothesis_id.
- statement_ids refer only to HypothesisContext evidence_statements[*].statement_id.
- Prediction observation IDs, falsification criterion IDs, route IDs, and paper IDs are never statement_ids.
- Copy permitted IDs character-for-character from the supplied payload. Never synthesize, abbreviate, translate, renumber, or guess an ID. If no exact permitted ID is needed, leave that reference list empty.

Dimension definitions:
- premise_fidelity: whether selected positive premises are represented without
  distortion or silent strengthening. Evaluate the representation of the supplied
  premises themselves. Do not penalize an explicitly hypothetical inferential
  bridge merely because it goes beyond the premise; assess excessive inferential
  strength under inferential_proportionality or causal_strengthening.
- gap_discipline: whether unresolved/scope-limit content remains a research gap or motivation rather than being presented as established evidence.
- candidate_calibration: whether candidate/provisional evidence remains explicitly
  provisional when used. Pass when candidate status is clearly preserved and the
  dependency is transparently propagated into the hypothesis. Do not downgrade
  solely because the hypothesis assigns the candidate a proposed mechanistic role;
  assess excessive inferential strength separately.
- inferential_proportionality: whether the proposed bridge is a bounded scientific leap rather than an unsupported multi-step assertion.
- causal_strengthening: whether associations/observations are silently converted into established causation. An explicitly hypothetical causal mechanism is allowed.
- directional_specificity: whether increase/decrease/non-monotonic/shift claims are
  justified at the level of specificity stated. A generic qualitative_change alone
  is not an ordered directional claim. Use warning/fail for unsupported increase,
  decrease, monotonicity, non-monotonicity, magnitude, or other ordered trend/shape.
- prediction_linkage: whether predictions genuinely follow from the proposed hypothesis rather than being unrelated additions.
- falsifier_informativeness: whether the falsifying outcome would meaningfully
  count against the hypothesis, rather than merely restating it or being too vague.
  A falsifier need not logically eliminate every weaker interpretation of cautious
  wording such as "may contribute". Pass when the stated outcome would materially
  count against the central predicted relationship or mechanism. Warn when the
  criterion tests only a substantially stronger necessary/sufficient relation while
  leaving the stated central prediction essentially unchallenged.
- cross_paper_discipline: whether evidence from distinct papers/systems is kept distinct; cross-paper synthesis is allowed, but graph alignment or co-occurrence is not mechanistic evidence and distinct catalysts must not be silently merged.
- hypothesis_distinctness: whether multiple hypotheses in the portfolio are scientifically distinct rather than paraphrases. Use not_applicable for a single-hypothesis portfolio.
- abstention_appropriateness: whether abstention/non-abstention is appropriate for the supplied eligible evidence. For a nonempty portfolio, pass means there is enough eligible evidence to motivate at least a bounded hypothesis; it does not mean the hypothesis is proven.

Deterministic diagnostics are review hints, not truth labels. Independently assess the semantic dimension and explain your rationale using exact supplied hypothesis IDs and statement IDs when relevant.

Return only the requested structured semantic review draft."""


class HypothesisSemanticPromptAssembler:
    def build(
        self,
        context: HypothesisContext,
        portfolio: HypothesisPortfolio,
        evaluation: HypothesisEvaluationReport,
    ) -> HypothesisSemanticPrompt:
        if not evaluation.hard_gate_passed:
            raise ValueError(
                "Semantic critic may only review portfolios that pass deterministic hard gates."
            )
        if evaluation.context_id != context.context_id:
            raise ValueError("evaluation/context ID mismatch")
        if evaluation.context_sha256 != context.context_sha256:
            raise ValueError("evaluation/context SHA mismatch")
        if evaluation.portfolio_id != portfolio.portfolio_id:
            raise ValueError("evaluation/portfolio ID mismatch")

        eligible = [
            row.model_dump(mode="json")
            for row in context.evidence_statements
            if row.eligible_as_premise
        ]
        gaps = [
            row.model_dump(mode="json")
            for row in context.evidence_statements
            if row.eligible_as_gap
        ]
        restricted = [
            row.model_dump(mode="json")
            for row in context.evidence_statements
            if not row.eligible_as_premise and not row.eligible_as_gap
        ]
        diagnostics = [
            row.model_dump(mode="json") for row in evaluation.diagnostics
        ]

        payload = {
            "task": {
                "task_id": context.task_id,
                "question": context.question,
                "corpus_id": context.corpus_id,
            },
            "source_lineage": {
                "context_id": context.context_id,
                "context_sha256": context.context_sha256,
                "portfolio_id": portfolio.portfolio_id,
                "portfolio_sha256": evaluation.portfolio_sha256,
                "evaluator_version": evaluation.evaluator_version,
            },
            "eligible_positive_premises": eligible,
            "research_gaps": gaps,
            "restricted_nonpremise_statements": restricted,
            "mechanism_routes": [
                row.model_dump(mode="json") for row in context.mechanism_routes
            ],
            "partial_absence_blocked_paper_ids": context.partial_absence_blocked_paper_ids,
            "deterministic_diagnostics": diagnostics,
            "hypothesis_portfolio": portfolio.model_dump(mode="json"),
            "required_dimensions": list(SEMANTIC_DIMENSIONS),
        }

        user_prompt = (
            "SEMANTIC REVIEW INPUT\n"
            "=====================\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n\nOUTPUT REQUIREMENTS\n"
            "===================\n"
            "- Evaluate every required dimension exactly once.\n"
            "- Use only exact IDs from the permitted namespaces defined below.\n"
            "- hypothesis_ids may contain ONLY hypothesis_id values from hypothesis_portfolio.hypotheses.\n"
            "- statement_ids may contain ONLY statement_id values from the HypothesisContext evidence statements "
            "shown under eligible_positive_premises, research_gaps, or restricted_nonpremise_statements.\n"
            "- NEVER place observation_id, prediction IDs, criterion_id, falsifier IDs, route_id, paper IDs, "
            "or any other identifier in statement_ids.\n"
            "- Copy IDs exactly character-for-character. Do not invent an ID from a nearby label or object.\n"
            "- If you cannot identify an exact permitted reference, leave that list empty rather than guessing.\n"
            "- For prediction_linkage, directional_specificity, or falsifier_informativeness, use hypothesis_ids "
            "to identify the hypothesis being evaluated. Leave statement_ids empty unless a specific context "
            "evidence statement is directly relevant to the rationale.\n"
            "- hypothesis_ids may be empty for portfolio-level dimensions or abstention.\n"
            "- statement_ids may be empty when no specific context evidence statement is implicated.\n"
            "- Do not rewrite, repair, rank, or replace hypotheses.\n"
            "- Do not use external literature knowledge or assess novelty.\n"
        )
        return HypothesisSemanticPrompt.create(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
