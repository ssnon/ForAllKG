from __future__ import annotations

import json
from dataclasses import dataclass

from pipeline_core.discovery.reframing.critic_contracts import CRITIC_DIMENSIONS
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframeCandidate,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ScientificReframeEvidencePacket,
)


@dataclass(frozen=True)
class ScientificReframeCriticPrompt:
    candidate_id: str
    system_prompt: str
    user_prompt: str


_RUBRIC = """
Use one independent 0-3 rating for each dimension. Do not average or combine them.
0 = severe failure or unsupported move
1 = weak; material scientific concern
2 = adequate; defensible but limited or imperfect
3 = strong; no material concern in the supplied grounded scope

Dimension meanings:
- operator_validity: the candidate genuinely instantiates its declared reframing operator.
- premise_fidelity: the baseline/alternative models and rationale stay faithful to the cited grounded premises and do not convert speculation into reported fact.
- explanatory_span: the alternative explains multiple relevant observations/evidence families rather than one isolated statement.
- construct_burden: higher is better; proposed constructs are parsimonious relative to explanatory gain and do not multiply unsupported entities.
- task_obligation_coverage: the reframe addresses the scientific question represented by the supplied task rather than an adjacent topic.
- differential_prediction_quality: predictions actually separate baseline from alternative using discriminating observables.
- falsifiability: the falsifiers and test could genuinely disconfirm the alternative rather than merely produce ambiguous results.
- over_specificity: higher is better; the candidate avoids unsupported numerical, mechanistic, causal, or contextual detail beyond the premises.
- triviality: higher is better; the candidate is not merely a synonym, generic interaction effect, or restatement of an existing premise.
- reframe_depth: the candidate changes the explanatory model/representation, not just the magnitude of an already named factor.

Do not assess external novelty, priority, publication significance, or whether this candidate should be selected for production.
Do not compute an overall score, weighted average, winner, ranking, or accept/reject verdict.
Cite only statement IDs present in the supplied grounded evidence.
""".strip()


class ScientificReframeCriticPromptAssembler:
    def build(
        self,
        *,
        candidate: ScientificReframeCandidate,
        evidence: ScientificReframeEvidencePacket,
    ) -> ScientificReframeCriticPrompt:
        statement_rows = [
            {
                "statement_id": row.statement_id,
                "text": row.text,
                "epistemic_role": row.epistemic_role,
                "claim_kind": row.claim_kind,
                "paper_ids": row.paper_ids,
                "requires_verification": row.requires_verification,
            }
            for row in [*evidence.premise_statements, *evidence.gap_statements]
        ]
        system_prompt = (
            "You are a scientific model-reframing critic operating on a shadow-only "
            "hypothesis artifact. Evaluate the supplied candidate only against the "
            "supplied grounded evidence and task. Preserve epistemic boundaries.\n\n"
            + _RUBRIC
        )
        user_payload = {
            "task": {
                "task_id": evidence.task_id,
                "question": evidence.question,
            },
            "candidate": candidate.model_dump(mode="json"),
            "grounded_evidence_statements": statement_rows,
            "required_dimensions": list(CRITIC_DIMENSIONS),
            "output_instruction": (
                "Return candidate_id and one dimension review per requested dimension. "
                "Each review should contain dimension, rating, concise rationale, zero or "
                "more supporting_statement_ids, and zero or more concerns. You may also "
                "return cross_cutting_concerns."
            ),
        }
        return ScientificReframeCriticPrompt(
            candidate_id=candidate.reframe_id,
            system_prompt=system_prompt,
            user_prompt=json.dumps(
                user_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        )
