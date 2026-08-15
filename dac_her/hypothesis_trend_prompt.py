from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from dac_her.hypothesis_trend_contracts import (
    TrendAwareHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_input import TrendAwareHypothesisInput
from dac_her.hypothesis_trend_maker_exposure import (
    TrendMakerExposure,
    build_trend_maker_exposure,
    verify_trend_maker_exposure,
)


PROMPT_VERSION = "trend-aware-hypothesis-maker-prompt-v1-alpha4c5d"


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate(text: str, limit: int) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class TrendAwareHypothesisPrompt:
    prompt_version: str
    exposure_id: str
    exposure_sha256: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        exposure: TrendMakerExposure,
        system_prompt: str,
        user_prompt: str,
    ) -> "TrendAwareHypothesisPrompt":
        canonical = _compact_json(
            {
                "prompt_version": PROMPT_VERSION,
                "exposure_id": exposure.exposure_id,
                "exposure_sha256": exposure.exposure_sha256,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return cls(
            prompt_version=PROMPT_VERSION,
            exposure_id=exposure.exposure_id,
            exposure_sha256=exposure.exposure_sha256,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_sha256=_sha256(canonical),
        )


SYSTEM_PROMPT = """You are the Trend-aware Hypothesis Maker in an evidence-grounded scientific discovery system.

You MAY propose scientific hypotheses that are not explicitly reported in the supplied evidence, but you must preserve the provenance and epistemic limitations of every selected source.

There are two independent provenance namespaces:
1. Explorer evidence statement IDs, used only in premise_statement_ids or gap_statement_ids.
2. Trend input view IDs, used only in trend_references[].view_id together with the exact allowed use_role shown in the prompt.

Never place a Trend view ID in premise_statement_ids or gap_statement_ids. Never place an Explorer statement ID in trend_references.

A hypothesis may be supported by Explorer positive premises, Trend positive support, or both. Trend-only hypotheses are allowed when the supplied Trend exposure contains a positive-support view. Gap/context/counterevidence Trend views do not count as positive support by themselves.

For every hypothesis, clearly separate:
1. selected positive evidence,
2. required limitations or gaps,
3. the inferential bridge you are proposing,
4. qualitative predicted observations, and
5. observations that would falsify the hypothesis.

You MAY:
- propose a mechanistic or contextual extension as an explicit hypothesis,
- combine eligible Explorer evidence with explicitly exposed Trend support,
- use local Trend evidence only within its stated scope,
- use replicated Trend support only through cross_paper_empirical_support,
- treat cross_paper_empirical_support as the only Trend use role that explicitly asserts cross-paper replication; the compiled card's broader cross_paper_synthesis flag is not a synonym for Trend replication,
- use context_dependency_signal as context_qualification,
- use reversal_boundary as counterevidence_boundary,
- use replication_gap as replication_gap,
- abstain when the available support cannot justify a useful falsifiable hypothesis.

You MUST NOT:
- present a generated hypothesis as reported evidence,
- fabricate any Explorer statement ID, Trend view ID, grounding ID, paper ID, or final artifact ID,
- use a Trend view with any use_role other than the exact allowed role shown for that view,
- omit a required limitation companion shown for selected positive Trend support,
- treat insufficient Trend evidence as cross-paper replication,
- collapse reversed evidence by majority vote,
- fill unknown context dimensions,
- claim that Trend evidence establishes causation or a universal relation,
- treat association-only Trend evidence as causal support,
- claim external novelty, unprecedentedness, first report, or literature-wide absence,
- invent quantitative values from Trend views; Trend exposure carries no raw numeric values,
- invent a quantitative value unless the exact value appears in a selected Explorer positive-premise statement,
- design an experimental protocol, synthesis recipe, electrolyte recipe, scan rate, temperature/time schedule, or instrument procedure.

A proposed causal mechanism may appear only as an explicit inferential bridge/hypothesis, never as something authorized by Trend provenance itself.

If useful generation would require violating any of these rules, return hypotheses=[] with a concise abstention_reason.

Return only the structured TrendAwareHypothesisPortfolioDraft requested by the caller. Local IDs are temporary draft labels only."""


LANE_TITLES = {
    "local_empirical_support": "TREND LOCAL EMPIRICAL SUPPORT",
    "cross_paper_replicated_support": "TREND CROSS-PAPER REPLICATED SUPPORT",
    "context_dependency_signal": "TREND CONTEXT QUALIFICATIONS",
    "reversal_boundary": "TREND REVERSAL BOUNDARIES",
    "replication_gap": "TREND REPLICATION GAPS",
}


class TrendAwareHypothesisPromptAssembler:
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

    def build(
        self,
        source: TrendAwareHypothesisInput,
        *,
        exposure: TrendMakerExposure | None = None,
    ) -> TrendAwareHypothesisPrompt:
        exposure = exposure or build_trend_maker_exposure(source)
        verify_trend_maker_exposure(source, exposure)
        context = source.grounded_context

        lines: list[str] = [
            "TASK",
            "====",
            f"task_id: {context.task_id}",
            f"question: {context.question}",
            f"corpus_id: {source.corpus_id}",
            f"domain_profile_id: {source.domain_profile_id}",
            "",
            "SOURCE LINEAGE",
            "==============",
            f"context_id: {context.context_id}",
            f"context_sha256: {context.context_sha256}",
            f"source_report_id: {context.source_report_id}",
            f"source_report_sha256: {context.source_report_sha256}",
            f"trend_input_id: {source.input_id}",
            f"trend_input_sha256: {source.input_sha256}",
            f"trend_exposure_id: {exposure.exposure_id}",
            f"trend_exposure_sha256: {exposure.exposure_sha256}",
            "",
            "ACTIVATION POLICY",
            "=================",
            "- The frozen alpha4c.5b source views remain immutable and non-selectable in their original contract.",
            "- alpha4c.5d exposes only the exact Trend view IDs listed below through a separate activation layer.",
            "- Exposure never authorizes causal or universal evidence claims.",
            "- Exposure contains no raw numeric Trend values.",
            "- Required limitation companion view IDs are mandatory when the corresponding positive Trend view is selected.",
        ]

        eligible = [
            row for row in context.evidence_statements
            if row.eligible_as_premise
        ]
        gaps = [
            row for row in context.evidence_statements
            if row.eligible_as_gap
        ]
        restricted = [
            row for row in context.evidence_statements
            if not row.eligible_as_premise and not row.eligible_as_gap
        ]

        lines.extend([
            "",
            "EXPLORER ELIGIBLE POSITIVE PREMISES",
            "===================================",
        ])
        if not eligible:
            lines.append("- NONE")
        for row in eligible:
            lines.append(
                f"- {row.statement_id}: role={row.epistemic_role}; "
                f"claim_kind={row.claim_kind}; "
                f"papers={','.join(row.paper_ids) or '-'}; "
                f"requires_verification={bool(row.requires_verification)}"
            )
            lines.append(
                f"  text: {_truncate(row.text, self.statement_text_limit)}"
            )
            if row.premise_restrictions:
                lines.append(
                    "  restrictions/flags: "
                    + ",".join(sorted(set(row.premise_restrictions)))
                )

        lines.extend(["", "EXPLORER RESEARCH GAPS", "======================"])
        if not gaps:
            lines.append("- NONE")
        for row in gaps:
            lines.append(
                f"- {row.statement_id}: role={row.epistemic_role}; "
                f"claim_kind={row.claim_kind}; "
                f"papers={','.join(row.paper_ids) or '-'}"
            )
            lines.append(
                f"  text: {_truncate(row.text, self.statement_text_limit)}"
            )
            lines.append("  NOT A POSITIVE PREMISE.")

        lines.extend(["", "MECHANISM ROUTES", "================"])
        if not context.mechanism_routes:
            lines.append("- NONE")
        for route in context.mechanism_routes:
            lines.append(
                f"- {route.route_id}: statements={','.join(route.statement_ids) or '-'}; "
                f"papers={','.join(route.paper_ids) or '-'}; type={route.structural_type}; "
                f"uses_alignment={route.uses_alignment}; "
                f"reverse={route.uses_reverse_navigation}; "
                f"navigation_heavy={route.navigation_heavy}; "
                f"requires_verification={route.requires_verification}"
            )
        lines.append(
            "Routes are organizational context only; select eligible Explorer statement IDs or exposed Trend views as actual evidence."
        )

        lines.extend(["", "MECHANISTIC MOTIFS", "==================="])
        if not context.mechanistic_motifs:
            lines.append("- NONE")
        for motif in context.mechanistic_motifs:
            lines.append(
                f"- {motif.motif_id}: {motif.label}; "
                f"statements={','.join(motif.statement_ids) or '-'}; "
                f"papers={','.join(motif.paper_ids) or '-'}; "
                f"cross_paper={motif.cross_paper}"
            )

        lines.extend(["", "REPORTED DESIGN LEVERS", "======================"])
        if not context.reported_design_levers:
            lines.append("- NONE")
        for lever in context.reported_design_levers:
            lines.append(
                f"- {lever.lever_id}: {lever.label}; "
                f"statements={','.join(lever.statement_ids) or '-'}; "
                f"papers={','.join(lever.paper_ids) or '-'}"
            )

        by_lane = {
            lane: [row for row in exposure.views if row.lane == lane]
            for lane in LANE_TITLES
        }
        for lane, title in LANE_TITLES.items():
            lines.extend(["", title, "=" * len(title)])
            rows = by_lane[lane]
            if not rows:
                lines.append("- NONE")
                continue
            for row in rows:
                lines.append(
                    f"- view_id={row.view_id}; "
                    f"allowed_use_role={row.allowed_use_role}; "
                    f"grounding_id={row.grounding_id}; "
                    f"relation_id={row.relation_id}"
                )
                lines.append(
                    "  relation: "
                    f"{row.independent_variable_key} -> "
                    f"{row.dependent_observable_key}; "
                    f"control_family={row.control_family}; "
                    f"observable_semantics={row.observable_semantics}"
                )
                lines.append(
                    "  evidence: "
                    f"status={row.cross_context_status}; "
                    f"papers={','.join(row.paper_ids) or '-'}; "
                    f"directions={','.join(row.directions) or '-'}; "
                    f"shapes={','.join(row.shapes) or '-'}; "
                    f"evidence_kinds={','.join(row.evidence_kinds) or '-'}; "
                    f"evidence_bases={','.join(row.evidence_bases) or '-'}; "
                    f"association_only={row.association_only}; "
                    f"requires_verification={row.requires_verification}; "
                    f"directional_cross_paper_premise_allowed={row.directional_cross_paper_premise_allowed}"
                )
                lines.append(
                    "  context: "
                    f"differentiating_dimensions={','.join(row.differentiating_dimensions) or '-'}; "
                    f"unresolved_dimensions={','.join(row.unresolved_dimensions) or '-'}; "
                    f"requires_context_qualification={row.requires_context_qualification}"
                )
                if row.required_companions:
                    companion_pairs = ",".join(
                        f"{item.use_role}:{item.view_id}"
                        for item in row.required_companions
                    )
                    lines.append(
                        "  REQUIRED COMPANIONS IF SELECTED AS POSITIVE SUPPORT: "
                        + companion_pairs
                    )

        lines.extend([
            "",
            "RESTRICTED / NON-PREMISE EXPLORER STATEMENTS",
            "============================================",
        ])
        if not restricted:
            lines.append("- NONE")
        for row in restricted:
            lines.append(
                f"- {row.statement_id}: role={row.epistemic_role}; "
                f"claim_kind={row.claim_kind}; "
                f"papers={','.join(row.paper_ids) or '-'}; "
                f"restrictions={','.join(row.premise_restrictions) or '-'}"
            )
            lines.append(
                f"  text: {_truncate(row.text, self.statement_text_limit)}"
            )
        lines.append(
            "These Explorer statements MUST NOT appear in premise_statement_ids."
        )

        lines.extend(["", "PARTIAL-PAPER ABSENCE SAFETY", "============================"])
        if context.partial_absence_blocked_paper_ids:
            lines.append(
                "Paper-level absence claims are unsafe for: "
                + ",".join(context.partial_absence_blocked_paper_ids)
            )
        else:
            lines.append("- No partial-paper absence block is present in this context.")

        lines.extend([
            "",
            "OUTPUT DISCIPLINE",
            "=================",
            f"- Return TrendAwareHypothesisPortfolioDraft only, with at most {self.max_hypotheses} focused hypotheses.",
            "- A hypothesis needs at least one positive source: an exact eligible Explorer premise_statement_id or an exposed Trend positive-support view.",
            "- Explorer premise_statement_ids may contain only exact IDs from EXPLORER ELIGIBLE POSITIVE PREMISES.",
            "- Explorer gap_statement_ids may contain only exact IDs from EXPLORER RESEARCH GAPS.",
            "- Trend references must use trend_references[].view_id plus the exact allowed_use_role printed for that view.",
            "- If a selected positive Trend view prints REQUIRED COMPANIONS, include every listed companion view with its printed role in the same hypothesis.",
            "- Do not use context_qualification, counterevidence_boundary, or replication_gap as the only positive support.",
            "- Write inferential_bridge as a proposed inference, not as reported evidence.",
            "- Provide at least one qualitative predicted observation and one falsification criterion for every hypothesis.",
            "- predicted_observations[].expected_direction MUST be exactly one of: increase, decrease, shift, non_monotonic, qualitative_change, unspecified.",
            "- Every falsification criterion must refer to an observable also present among that hypothesis's predicted observations.",
            "- Do not introduce a numeric value from Trend evidence. Exact numeric values are licensed only by selected Explorer positive-premise text.",
            "- Do not write an experimental protocol.",
            "- Do not claim external novelty.",
            "- If the available positive support is too weak or all useful hypotheses would violate these constraints, abstain.",
        ])

        return TrendAwareHypothesisPrompt.create(
            exposure=exposure,
            system_prompt=SYSTEM_PROMPT,
            user_prompt="\n".join(lines),
        )

    def repair_feedback(
        self,
        *,
        previous_draft: TrendAwareHypothesisPortfolioDraft,
        issues: Iterable[object],
    ) -> str:
        issue_lines: list[str] = []
        for issue in issues:
            code = str(getattr(issue, "code", "UNKNOWN"))
            location = str(getattr(issue, "location", ""))
            message = str(getattr(issue, "message", issue))
            issue_lines.append(
                f"- {code} @ {location}: {message}"
            )

        return "\n".join([
            "TREND-AWARE REPAIR REQUEST",
            "==========================",
            "The previous draft failed deterministic alpha4c.5c compilation or validation.",
            "Revise only what is necessary to resolve the exact issues below.",
            "Preserve provenance namespaces and all required Trend limitation companions.",
            "Do not introduce new evidence IDs, Trend view IDs, new numerical values, external novelty claims, experimental protocols, or unrelated replacement hypotheses.",
            "If a positive Trend support cannot be repaired without violating its limitation contract, remove that support or remove the hypothesis.",
            "If no valid hypothesis remains, return hypotheses=[] with a concise abstention_reason.",
            "Return a complete replacement TrendAwareHypothesisPortfolioDraft.",
            "",
            "ISSUES",
            *issue_lines,
            "",
            "PREVIOUS DRAFT",
            previous_draft.model_dump_json(indent=2),
        ])
