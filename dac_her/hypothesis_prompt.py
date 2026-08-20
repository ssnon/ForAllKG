from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolioDraft


PROMPT_VERSION = "hypothesis-maker-prompt-v2.6.1.1"


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate(text: str, limit: int) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class HypothesisPrompt:
    prompt_version: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str

    @classmethod
    def create(cls, *, system_prompt: str, user_prompt: str) -> "HypothesisPrompt":
        canonical = _compact_json(
            {
                "prompt_version": PROMPT_VERSION,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return cls(
            prompt_version=PROMPT_VERSION,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_sha256=_sha256(canonical),
        )


SYSTEM_PROMPT = """You are the Hypothesis Maker in an evidence-grounded scientific discovery system.

Unlike the Graph Explorer, you MAY propose scientific hypotheses that are not explicitly reported in the supplied evidence. Your task is to make the inferential leap explicit, bounded, and falsifiable.

A hypothesis is not evidence.

For every proposed hypothesis, clearly separate:
1. reported or evidence-synthesized premises already present in the supplied HypothesisContext,
2. the inferential bridge that you are proposing,
3. the new qualitative prediction that follows from that bridge, and
4. an observation that would falsify the hypothesis.

You MAY:
- propose a mechanistic extension of supplied evidence,
- connect multiple eligible evidence statements through an explicit inferential bridge,
- use unresolved statements only as research gaps or motivation,
- propose qualitative observables and falsification conditions,
- state assumptions needed for the proposed inference,
- abstain when the evidence is too weak for a useful falsifiable hypothesis.

You MUST NOT:
- present a generated hypothesis as reported evidence,
- use a statement marked ineligible_as_premise as a positive premise,
- use unresolved, navigation-only, scope-limit, or alignment-dependent content as established scientific evidence,
- treat registry/pattern alignment or graph navigability as causal/mechanistic support,
- claim that a hypothesis is novel, unprecedented, first, previously unknown, or otherwise novel in the external literature,
- invent quantitative values that are not already present in the selected positive premises,
- design an experimental protocol, synthesis recipe, electrolyte recipe, scan rate, temperature/time schedule, or instrument procedure,
- hide candidate dependence: candidate evidence remains provisional even when it motivates a hypothesis,
- fabricate statement IDs or any other IDs.

Candidate evidence may be selected only when the supplied context marks it eligible_as_premise=true. If you use such a premise, do not write as though the candidate relation were established fact; the deterministic compiler will propagate candidate dependency metadata.

An unresolved statement can motivate a gap_statement_id, but it cannot serve as a positive premise. A graph path or route may help organize context, but the positive premise IDs must come from eligible evidence statements.

If the evidence does not support a useful falsifiable hypothesis without violating these rules, return hypotheses=[] with a concise abstention_reason.

Return only the structured HypothesisPortfolioDraft requested by the caller. Local IDs are temporary labels used only inside the draft; do not invent final hypothesis, prediction, falsifier, portfolio, report, packet, or context IDs."""


class HypothesisPromptAssembler:
    def __init__(
        self,
        *,
        statement_text_limit: int = 1100,
        max_hypotheses: int = 3,
    ) -> None:
        if max_hypotheses < 1:
            raise ValueError("max_hypotheses must be >= 1")
        self.statement_text_limit = int(statement_text_limit)
        self.max_hypotheses = int(max_hypotheses)

    def build(self, context: HypothesisContext) -> HypothesisPrompt:
        lines: list[str] = [
            "TASK",
            "====",
            f"task_id: {context.task_id}",
            f"question: {context.question}",
            f"corpus_id: {context.corpus_id}",
            "",
            "SOURCE LINEAGE",
            "==============",
            f"context_id: {context.context_id}",
            f"context_sha256: {context.context_sha256}",
            f"source_packet_id: {context.source_packet_id}",
            f"source_packet_sha256: {context.source_packet_sha256}",
            f"source_report_id: {context.source_report_id}",
            f"source_report_sha256: {context.source_report_sha256}",
            "",
            "POLICY",
            "======",
            f"generated_hypotheses_allowed: {context.policy.generated_hypotheses_allowed}",
            f"external_novelty_claims_allowed: {context.policy.external_novelty_claims_allowed}",
            f"experiment_protocols_allowed: {context.policy.experiment_protocols_allowed}",
            f"unsupported_numeric_predictions_allowed: {context.policy.unsupported_numeric_predictions_allowed}",
            f"alignment_can_be_scientific_premise: {context.policy.alignment_can_be_scientific_premise}",
            f"unresolved_can_be_positive_premise: {context.policy.unresolved_can_be_positive_premise}",
            f"candidate_evidence_must_propagate: {context.policy.candidate_evidence_must_propagate}",
            f"falsifiable_prediction_required: {context.policy.falsifiable_prediction_required}",
            f"falsification_condition_required: {context.policy.falsification_condition_required}",
        ]

        eligible = [row for row in context.evidence_statements if row.eligible_as_premise]
        gaps = [row for row in context.evidence_statements if row.eligible_as_gap]
        restricted = [
            row
            for row in context.evidence_statements
            if not row.eligible_as_premise and not row.eligible_as_gap
        ]

        lines.extend(["", "ELIGIBLE POSITIVE PREMISES", "=========================="])
        if not eligible:
            lines.append("- NONE. Do not invent a positive premise; abstention is likely appropriate.")
        for row in eligible:
            candidate_note = "YES" if row.requires_verification else "NO"
            lines.append(
                f"- {row.statement_id}: role={row.epistemic_role}; claim_kind={row.claim_kind}; "
                f"papers={','.join(row.paper_ids) or '-'}; requires_verification={candidate_note}"
            )
            lines.append(f"  text: {_truncate(row.text, self.statement_text_limit)}")
            if row.premise_restrictions:
                lines.append(
                    "  restrictions/flags: " + ",".join(sorted(set(row.premise_restrictions)))
                )

        lines.extend(["", "RESEARCH GAPS", "============="])
        if not gaps:
            lines.append("- NONE")
        for row in gaps:
            lines.append(
                f"- {row.statement_id}: role={row.epistemic_role}; claim_kind={row.claim_kind}; "
                f"papers={','.join(row.paper_ids) or '-'}"
            )
            lines.append(f"  text: {_truncate(row.text, self.statement_text_limit)}")
            if row.premise_restrictions:
                lines.append(
                    "  NOT A POSITIVE PREMISE. restrictions: "
                    + ",".join(sorted(set(row.premise_restrictions)))
                )

        lines.extend(["", "MECHANISM ROUTES", "================"])
        if not context.mechanism_routes:
            lines.append("- NONE")
        for route in context.mechanism_routes:
            lines.append(
                f"- {route.route_id}: statements={','.join(route.statement_ids) or '-'}; "
                f"papers={','.join(route.paper_ids) or '-'}; type={route.structural_type}; "
                f"uses_alignment={route.uses_alignment}; reverse={route.uses_reverse_navigation}; "
                f"navigation_heavy={route.navigation_heavy}; requires_verification={route.requires_verification}"
            )
        lines.append(
            "Routes are organizational context only. A route is not itself a positive premise; select eligible statement IDs above."
        )

        lines.extend(["", "MECHANISTIC MOTIFS", "==================="])
        if not context.mechanistic_motifs:
            lines.append("- NONE")
        for motif in context.mechanistic_motifs:
            lines.append(
                f"- {motif.motif_id}: {motif.label}; statements={','.join(motif.statement_ids) or '-'}; "
                f"papers={','.join(motif.paper_ids) or '-'}; cross_paper={motif.cross_paper}"
            )

        lines.extend(["", "REPORTED DESIGN LEVERS", "======================"])
        if not context.reported_design_levers:
            lines.append("- NONE")
        for lever in context.reported_design_levers:
            lines.append(
                f"- {lever.lever_id}: {lever.label}; statements={','.join(lever.statement_ids) or '-'}; "
                f"papers={','.join(lever.paper_ids) or '-'}"
            )

        lines.extend(["", "RESTRICTED / NON-PREMISE STATEMENTS", "==================================="])
        if not restricted:
            lines.append("- NONE")
        for row in restricted:
            lines.append(
                f"- {row.statement_id}: role={row.epistemic_role}; claim_kind={row.claim_kind}; "
                f"papers={','.join(row.paper_ids) or '-'}; restrictions={','.join(row.premise_restrictions) or '-'}"
            )
            lines.append(f"  text: {_truncate(row.text, self.statement_text_limit)}")
        lines.append("These statements MUST NOT appear in premise_statement_ids.")

        lines.extend(["", "PARTIAL-PAPER ABSENCE SAFETY", "============================"])
        if context.partial_absence_blocked_paper_ids:
            lines.append(
                "Paper-level absence claims are unsafe for: "
                + ",".join(context.partial_absence_blocked_paper_ids)
            )
            lines.append(
                "You may say that the supplied context does not establish a relation; do not say that one of these papers/studies does not report or lacks it."
            )
        else:
            lines.append("- No partial-paper absence block is present in this context.")

        lines.extend(
            [
                "",
                "OUTPUT DISCIPLINE",
                "=================",
                f"- Return HypothesisPortfolioDraft only, with at most {self.max_hypotheses} focused hypotheses for this run.",
                "- Every hypothesis must select at least one exact eligible premise_statement_id from this prompt.",
                "- gap_statement_ids may contain only exact RESEARCH GAPS IDs above.",
                "- Write the inferential_bridge as the proposed inference, not as though it were reported evidence.",
                "- Provide at least one qualitative predicted observation and at least one falsification criterion for each hypothesis.",
                "- predicted_observations[].expected_direction MUST be exactly one of: increase, decrease, shift, non_monotonic, qualitative_change, unspecified.",
                "- Do not use conditional, context_dependent, mixed, stable, no_change, or free-form text as expected_direction. Put conditionality in observable/rationale; use unspecified when no allowed directional category is justified.",
                "- Make every falsification criterion refer to an observable that also appears among that hypothesis's predicted observations.",
                "- Do not introduce a numeric value unless that exact value occurs in one of the selected positive-premise statements.",
                "- Do not write an experimental protocol. State what should be observed, not how to synthesize, prepare, dose, heat, scan, or instrumentally measure it.",
                "- Do not claim external novelty. novelty_status is not assessed here.",
                "- If useful hypothesis generation would require treating a gap or restricted statement as established evidence, abstain instead.",
            ]
        )

        return HypothesisPrompt.create(
            system_prompt=SYSTEM_PROMPT,
            user_prompt="\n".join(lines),
        )

    def repair_feedback(
        self,
        *,
        previous_draft: HypothesisPortfolioDraft,
        issues: Iterable[object],
    ) -> str:
        issue_lines: list[str] = []
        for issue in issues:
            code = str(getattr(issue, "code", "UNKNOWN"))
            location = str(getattr(issue, "location", ""))
            message = str(getattr(issue, "message", issue))
            issue_lines.append(f"- {code} @ {location}: {message}")

        return "\n".join(
            [
                "REPAIR REQUEST",
                "==============",
                "The previous hypothesis draft failed deterministic compilation or validation.",
                "Revise only what is necessary to resolve the exact issues below.",
                "Preserve the scientific intent of each affected hypothesis whenever possible.",
                "Do not introduce new evidence IDs, new premises, external novelty claims, unsupported numerical values, experimental protocols, or unrelated replacement hypotheses.",
                "If the original hypothesis cannot be repaired without violating the supplied context, remove it; if no valid hypotheses remain, return an abstention with a concise reason.",
                "Return a complete replacement HypothesisPortfolioDraft.",
                "",
                "ISSUES",
                *issue_lines,
                "",
                "PREVIOUS DRAFT",
                previous_draft.model_dump_json(indent=2),
            ]
        )
