from __future__ import annotations

import json
from dataclasses import dataclass

from pipeline_core.discovery.reframing.reframe_contracts import (
    ImplementedReframeOperatorId,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ScientificReframeEvidencePacket,
)


@dataclass(frozen=True)
class ScientificReframePrompt:
    operator_id: ImplementedReframeOperatorId
    system_prompt: str
    user_prompt: str


_COMMON_SYSTEM = """You are a scientific reframing module operating in a shadow-only research lane.

Your task is NOT to summarize, rank, or restate the supplied evidence. Propose a distinct scientific model contrast only when the grounded evidence supports a falsifiable re-interpretation.

Hard rules:
1. Use only the supplied premise_statement_ids as positive scientific premises.
2. gap_statement_ids may motivate a question but are not positive evidence.
3. Do not treat retrieval similarity, missing extraction fields, or missing conditions as scientific negative evidence.
4. Do not claim external novelty, truth, confirmation, or causal establishment.
5. Every candidate must state a baseline model and a genuinely different alternative model.
6. Every candidate must include at least one differential prediction whose baseline and alternative expectations differ.
7. Every candidate must include falsifiers and a discriminating test with outcomes favoring each model.
8. Do not invent unsupported numerical thresholds. If a boundary location is unknown, keep it qualitative and testable.
9. Proposed constructs remain hypothesis-only and require verification.
10. Every statement ID used in baseline_model.explained_statement_ids or alternative_model.explained_statement_ids MUST also appear in that candidate's premise_statement_ids.
11. If no genuine reframe is supported, return zero candidates with a concise abstention_reason. Do not fill a quota.
"""


_OPERATOR_RULES = {
    "LATENT_VARIABLE": """Operator: LATENT_VARIABLE
- Ask whether one shared unmodeled construct could explain at least two grounded premise statements.
- The alternative model must explain at least two premise_statement_ids.
- Introduce no more than two speculative constructs.
- A renamed existing variable or a generic 'interaction effect' is not sufficient.
- Prefer explanatory leverage: one construct explaining multiple evidence families over many new variables.
- Populate latent_constructs and proposed_constructs. Set boundary_variables=[] and regime_change_kind=null; those fields belong only to REGIME_BOUNDARY.
""",
    "REGIME_BOUNDARY": """Operator: REGIME_BOUNDARY
- Ask whether the response model itself changes across experimental/computational conditions.
- A simple statement that an effect becomes stronger or weaker is NOT a regime reframe.
- Valid contrasts include threshold onset, saturation, slope/sign change, mechanism switch, non-monotonic transition, or another qualitative response-model change.
- Structured condition examples are clues, not exhaustive evidence. Missing conditions are not absences.
- The discriminating prediction should identify an observable pattern that separates one continuous model from a regime-changing model.
- Populate boundary_variables and regime_change_kind. Set latent_constructs=[]; that field belongs only to LATENT_VARIABLE.
""",
}


class ScientificReframePromptAssembler:
    def build(
        self,
        *,
        operator_id: ImplementedReframeOperatorId,
        evidence: ScientificReframeEvidencePacket,
    ) -> ScientificReframePrompt:
        premise_rows = [
            {
                "statement_id": row.statement_id,
                "text": row.text,
                "claim_kind": row.claim_kind,
                "paper_ids": row.paper_ids,
                "requires_verification": row.requires_verification,
            }
            for row in evidence.premise_statements
        ]
        gap_rows = [
            {
                "statement_id": row.statement_id,
                "text": row.text,
                "claim_kind": row.claim_kind,
                "paper_ids": row.paper_ids,
            }
            for row in evidence.gap_statements
        ]
        condition_rows = [
            row.model_dump(mode="json")
            for row in evidence.condition_examples
        ]

        user_payload = {
            "task_id": evidence.task_id,
            "question": evidence.question,
            "operator_id": operator_id,
            "grounded_positive_premises": premise_rows,
            "research_gaps": gap_rows,
            "structured_condition_examples": (
                condition_rows if operator_id == "REGIME_BOUNDARY" else []
            ),
            "scope_notes": {
                "grounded_source_chunk_count": evidence.grounded_source_chunk_count,
                "unresolved_grounded_node_count": evidence.unresolved_grounded_node_count,
                "missing_condition_is_negative_evidence": False,
                "external_novelty_evaluated": False,
            },
            "output_instruction": (
                "Return at most two candidates. Every candidate must use only listed "
                "statement IDs and satisfy the operator-specific constraints. Every "
                "explained_statement_id must also be repeated in that candidate's "
                "premise_statement_ids."
            ),
        }
        return ScientificReframePrompt(
            operator_id=operator_id,
            system_prompt=_COMMON_SYSTEM + "\n" + _OPERATOR_RULES[operator_id],
            user_prompt=json.dumps(
                user_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        )
