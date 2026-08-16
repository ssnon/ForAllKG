from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from dac_her.hypothesis_trend_contract_hardened_contracts import (
    ContractHardenedTrendHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_contract_hardened_exposure import (
    ContractHardenedTrendMakerExposure,
    build_contract_hardened_trend_maker_exposure,
    verify_contract_hardened_trend_maker_exposure,
)
from dac_her.hypothesis_trend_input import TrendAwareHypothesisInput


PROMPT_VERSION = (
    "contract-hardened-trend-hypothesis-maker-prompt-v1-alpha4c5i"
)


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
class ContractHardenedTrendHypothesisPrompt:
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
        exposure: ContractHardenedTrendMakerExposure,
        system_prompt: str,
        user_prompt: str,
    ) -> "ContractHardenedTrendHypothesisPrompt":
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


SYSTEM_PROMPT = """You are the alpha4c.5i contract-hardened Trend Hypothesis Maker.

Your scientific role is synthesis, not epistemic bookkeeping.

You MAY:
- select exact eligible Explorer premise IDs,
- select exact eligible Explorer gap IDs,
- select exact exposed Trend view IDs with their printed allowed use_role,
- propose a mechanistic hypothesis, inferential bridge, assumptions, conceptual falsifier, and exploratory prediction,
- abstain.

You MUST NOT author or copy into your free-form prose:
- the direction of a Trend-bound relation,
- a decreasing/smaller/lower OR increasing/larger/higher restatement of the bound independent variable,
- a new paper/study/article/literature-level absence claim,
- external novelty, protocols, unsupported numbers, causal evidence authority, or universal evidence authority.

For a Trend-bound prediction:
- set prediction_kind="trend_bound",
- provide only exact trend_view_ids plus a mechanistic_rationale,
- do NOT provide an observable or expected direction; the compiler owns both.

For an exploratory prediction:
- set prediction_kind="exploratory",
- provide exploratory_observable and exploratory_expected_direction,
- trend_view_ids must be empty.

Each falsification criterion references prediction_local_id. The compiler copies the prediction observable into the final criterion, so you do not author that observable twice.

Every selected positive Trend support view must be bound to at least one trend_bound prediction. Do not average or majority-vote incompatible Trend directions.

Paper-level absence facts are code-owned. If a grounded gap is useful, select its exact gap_statement_id; do not paraphrase it into 'the paper did not report...', 'the study lacks...', or literature-wide absence language.

Return only ContractHardenedTrendHypothesisPortfolioDraft. If useful generation would require violating these constraints, abstain."""


class ContractHardenedTrendHypothesisPromptAssembler:
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
        exposure: ContractHardenedTrendMakerExposure | None = None,
    ) -> ContractHardenedTrendHypothesisPrompt:
        exposure = (
            exposure
            or build_contract_hardened_trend_maker_exposure(source)
        )
        verify_contract_hardened_trend_maker_exposure(
            source,
            exposure,
        )
        context = source.grounded_context

        lines: list[str] = [
            "TASK",
            "====",
            f"task_id: {context.task_id}",
            f"question: {context.question}",
            f"corpus_id: {source.corpus_id}",
            f"domain_profile_id: {source.domain_profile_id}",
            "",
            "5I AUTHORITY POLICY",
            "===================",
            "- LLM scientific synthesis: enabled.",
            "- LLM-authored Trend direction: FORBIDDEN.",
            "- LLM-authored Trend-bound observable: FORBIDDEN.",
            "- New LLM-authored paper/literature absence claim: FORBIDDEN.",
            "- Trend direction and Trend-bound observable: compiler-owned.",
            "- Trend directional prose: deterministic renderer-owned.",
            "- Falsifier observable: compiler-bound to prediction_local_id.",
            "- Causal/universal evidence authorization: FALSE.",
            "",
            "ELIGIBLE EXPLORER PREMISES",
            "==========================",
        ]

        premises = [
            row
            for row in context.evidence_statements
            if row.eligible_as_premise
        ]
        if not premises:
            lines.append("- NONE")
        for row in premises:
            lines.append(
                f"- {row.statement_id}: role={row.epistemic_role}; "
                f"claim_kind={row.claim_kind}; "
                f"papers={','.join(row.paper_ids) or '-'}; "
                f"requires_verification={bool(row.requires_verification)}"
            )
            lines.append(
                "  text: "
                + _truncate(row.text, self.statement_text_limit)
            )

        lines.extend(
            [
                "",
                "ELIGIBLE EXPLORER GAPS",
                "======================",
            ]
        )
        gaps = [
            row
            for row in context.evidence_statements
            if row.eligible_as_gap
        ]
        if not gaps:
            lines.append("- NONE")
        blocked = set(exposure.partial_blocked_gap_statement_ids)
        for row in gaps:
            suffix = (
                " [PARTIAL-PAPER ABSENCE AUTHORITY BLOCKED]"
                if row.statement_id in blocked
                else ""
            )
            lines.append(
                f"- {row.statement_id}: role={row.epistemic_role}; "
                f"claim_kind={row.claim_kind}; "
                f"papers={','.join(row.paper_ids) or '-'}{suffix}"
            )
            lines.append(
                "  text: "
                + _truncate(row.text, self.statement_text_limit)
            )
            lines.append(
                "  Selection as a gap ID does not authorize a new "
                "paper-level absence paraphrase."
            )

        lines.extend(
            [
                "",
                "EXPOSED TREND VIEWS",
                "===================",
            ]
        )
        if not exposure.views:
            lines.append("- NONE")
        for row in exposure.views:
            view = row.source_view
            lines.append(
                f"- view_id={view.view_id}; "
                f"allowed_use_role={view.allowed_use_role}; "
                f"lane={view.lane}; "
                f"relation={view.independent_variable_key}->"
                f"{view.dependent_observable_key}; "
                f"directions={','.join(view.directions) or '-'}; "
                f"shapes={','.join(view.shapes) or '-'}; "
                f"status={view.cross_context_status}"
            )
            if view.required_companions:
                lines.append(
                    "  REQUIRED COMPANIONS: "
                    + ",".join(
                        f"{item.use_role}:{item.view_id}"
                        for item in view.required_companions
                    )
                )

        if exposure.partial_absence_blocked_paper_ids:
            lines.extend(
                [
                    "",
                    "PARTIAL-PAPER ABSENCE BLOCK",
                    "===========================",
                    "- "
                    + ",".join(
                        exposure.partial_absence_blocked_paper_ids
                    ),
                ]
            )

        lines.extend(
            [
                "",
                "OUTPUT",
                "======",
                f"- Return at most {self.max_hypotheses} hypotheses.",
                "- Use exact IDs only.",
                "- Positive Trend support requires a trend_bound prediction.",
                "- Do not write the bound IV direction in title, "
                "mechanistic_proposal, inferential_bridge, assumptions, "
                "mechanistic_rationale, or falsifying_outcome.",
                "- Do not author paper/study/article/literature absence prose.",
                "- If constraints prevent a useful hypothesis, abstain.",
            ]
        )

        return ContractHardenedTrendHypothesisPrompt.create(
            exposure=exposure,
            system_prompt=SYSTEM_PROMPT,
            user_prompt="\n".join(lines),
        )

    def repair_feedback(
        self,
        *,
        previous_draft: ContractHardenedTrendHypothesisPortfolioDraft,
        issues: Iterable[object],
    ) -> str:
        issue_lines = [
            "- "
            + str(getattr(issue, "code", "UNKNOWN"))
            + " @ "
            + str(getattr(issue, "location", ""))
            + ": "
            + str(getattr(issue, "message", issue))
            for issue in issues
        ]
        return "\n".join(
            [
                "ALPHA4C.5I CONTRACT REPAIR REQUEST",
                "=================================",
                "Repair only the listed deterministic contract failures.",
                "Do not weaken, paraphrase around, or bypass the authority "
                "rules.",
                "For direction failures: remove LLM-authored bound-IV "
                "directional wording; leave direction to the compiler.",
                "For absence failures: remove the paper/literature absence "
                "assertion; selecting an eligible gap ID is sufficient.",
                "Do not introduce new evidence IDs, numbers, protocols, "
                "novelty claims, causal authority, or universal authority.",
                "If repair is not possible, remove the hypothesis or abstain.",
                "",
                "ISSUES",
                *issue_lines,
                "",
                "PREVIOUS DRAFT",
                previous_draft.model_dump_json(indent=2),
            ]
        )
