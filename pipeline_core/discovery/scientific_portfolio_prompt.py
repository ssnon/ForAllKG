from __future__ import annotations

import json
from dataclasses import dataclass

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
    ScientificPortfolioEvaluationReport,
    ScientificPortfolioSelectionReport,
    evaluation_prompt_payload,
)


@dataclass(frozen=True)
class ScientificPortfolioPrompt:
    kind: str
    system_prompt: str
    user_prompt: str


_EVAL_SYSTEM = """You are evaluating a bounded scientific idea portfolio for research planning.

Critical epistemic rules:
- Evaluate only the supplied semantic idea content. Do not use outside knowledge.
- The input view is intentionally origin-blind. Do not infer or guess whether a candidate came from Frontier, Evolution, external literature, or another lane.
- Do NOT decide whether an idea is true, novel, publishable, or literature-wide unprecedented.
- Every candidate is INSPIRATION_ONLY and is not positive premise evidence.
- Evaluate EVERY supplied candidate exactly once and preserve candidate_id verbatim.
- Use only LOW, MODERATE, HIGH, or INDETERMINATE.

Common scientific-sketch normalization:
- Before assigning dimension levels, express EVERY candidate in the same bounded four-field normalized_sketch:
  1. hypothesis_frame
  2. predicted_observation
  3. falsification_condition
  4. discriminating_observation
- Derive this sketch only from the supplied scientific_intent, conceptual_change_summary, core_relations, and the stated research task.
- The normalized sketch is hypothetical planning language, not a claim that the mechanism is true.
- Do not reward a candidate merely because its source artifact happened to contain richer prediction/falsifier prose.
- Conversely, do not penalize a candidate merely because its source representation was sparse if a bounded test sketch follows from the supplied relation structure.
- If the semantic content itself is insufficient for a dimension, use INDETERMINATE for that dimension.

Dimension meanings:
- task_relevance: how directly the normalized idea addresses or productively extends the requested scientific task.
- mechanistic_coherence: whether the proposed relations/mechanism form a coherent explanatory structure, without asserting truth.
- falsifiability: whether the normalized sketch contains a concrete observation that could count against the idea.
- discriminating_power: whether the normalized sketch distinguishes the idea from a plausible alternative.
- operationalizability: whether the normalized sketch can be converted into a concrete hypothesis/validation design without inventing unsupported experimental details.
- information_gain: whether resolving the idea could meaningfully distinguish mechanisms, regimes, or interpretations.

This is advisory evaluation for bounded portfolio retention only. There is no production-selection, novelty, premise, or truth authority."""


_MATERIALIZE_SYSTEM = """You are materializing retained scientific ideas into the project's standard grounded hypothesis format.

Critical rules:
- Each selected idea is INSPIRATION_ONLY. It may guide the inferential bridge, prediction, or question framing, but it is NEVER itself positive premise evidence.
- premise_statement_ids may use ONLY evidence statements explicitly marked eligible_as_premise=true in the supplied context.
- gap_statement_ids may use ONLY statements explicitly marked eligible_as_gap=true.
- External literature work IDs and idea-lineage IDs are never premise IDs.
- Do not claim novelty, first report, absence from the literature, or established truth.
- Do not invent unsupported numeric predictions or experimental protocol details.
- Every materialized hypothesis requires at least one prediction and one falsification criterion.
- The falsifier observable must correspond to a predicted observable.
- If a selected idea cannot be responsibly grounded in the supplied positive premises, abstain for that candidate instead of fabricating support.
- Cover every selected candidate exactly once: either one materialization item or one abstention.

The output remains shadow-only and will be passed through the existing HypothesisCompiler and HypothesisValidator before any downstream verification."""


def build_evaluation_prompt(
    *,
    pool: ScientificPortfolioCandidatePool,
    task_source: str,
    task_target: str,
) -> ScientificPortfolioPrompt:
    payload = evaluation_prompt_payload(
        pool=pool,
        task_source=task_source,
        task_target=task_target,
    ).model_dump(mode="json")
    return ScientificPortfolioPrompt(
        kind="evaluation",
        system_prompt=_EVAL_SYSTEM,
        user_prompt=(
            "Evaluate the full projected candidate pool under the stated dimensions.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        ),
    )


def build_materialization_prompt(
    *,
    context: HypothesisContext,
    pool: ScientificPortfolioCandidatePool,
    evaluation: ScientificPortfolioEvaluationReport,
    selection: ScientificPortfolioSelectionReport,
) -> ScientificPortfolioPrompt:
    candidate_by_id = {x.candidate_id: x for x in pool.candidates}
    eval_by_id = {x.candidate_id: x for x in evaluation.evaluations}
    selected = []
    for entry in selection.entries:
        candidate = candidate_by_id[entry.candidate_id]
        assessment = eval_by_id[entry.candidate_id]
        selected.append(
            {
                "candidate_id": candidate.candidate_id,
                "assigned_profile": entry.assigned_profile,
                "eligible_profiles": entry.eligible_profiles,
                "verification_burden": entry.verification_burden,
                "origin": candidate.origin,
                "source_kind": candidate.source_kind,
                "operator_id": candidate.operator_id,
                "idea_form": candidate.idea_form,
                "scientific_intent": candidate.scientific_intent,
                "conceptual_change_summary": candidate.conceptual_change_summary,
                "core_relations": candidate.core_relations,
                "normalized_scientific_sketch": assessment.normalized_sketch.model_dump(
                    mode="json"
                ),
                "source_differential_prediction": candidate.differential_prediction,
                "source_falsification_condition": candidate.falsification_condition,
                "source_discriminating_observation": candidate.discriminating_observation,
                "task_relation_mode": candidate.task_relation_mode,
                "external_literature_lineage": candidate.external_literature_lineage,
                "candidate_or_unverified_lineage": candidate.candidate_or_unverified_lineage,
                "evaluation": assessment.model_dump(mode="json"),
            }
        )

    statements = [
        {
            "statement_id": row.statement_id,
            "text": row.text,
            "epistemic_role": row.epistemic_role,
            "claim_kind": row.claim_kind,
            "paper_ids": row.paper_ids,
            "requires_verification": row.requires_verification,
            "eligible_as_premise": row.eligible_as_premise,
            "eligible_as_gap": row.eligible_as_gap,
            "premise_restrictions": row.premise_restrictions,
        }
        for row in context.evidence_statements
        if row.eligible_as_premise or row.eligible_as_gap
    ]
    payload = {
        "research_question": context.question,
        "domain_profile_id": context.domain_profile_id,
        "selected_candidates": selected,
        "eligible_context_statements": statements,
    }
    return ScientificPortfolioPrompt(
        kind="materialization",
        system_prompt=_MATERIALIZE_SYSTEM,
        user_prompt=(
            "Materialize the retained candidates into grounded hypothesis drafts.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        ),
    )


__all__ = [
    "ScientificPortfolioPrompt",
    "build_evaluation_prompt",
    "build_materialization_prompt",
]
