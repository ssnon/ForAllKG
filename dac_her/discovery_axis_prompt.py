from __future__ import annotations

import hashlib
import json
from typing import Iterable

from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxis
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolioDraft,
)
from pipeline_core.discovery.hypothesis_prompt import HypothesisPrompt, HypothesisPromptAssembler
from pipeline_core.discovery.evidence_family_selection import (
    EvidenceFamilyHierarchy,
    render_family_hierarchy_guidance,
)


PROMPT_VERSION = "hypothesis-maker-discovery-axis-prompt-v2.8.0-a4"
FAMILY_AWARE_PROMPT_VERSION = (
    "hypothesis-maker-discovery-axis-prompt-v2.9.1-ec2c"
)


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SYSTEM_APPENDIX = r"""
DISCOVERY-AXIS SYNTHESIS POLICY
===============================
This run is assigned exactly ONE discovery axis. The axis is an unverified,
non-evidentiary creativity constraint. It is NOT a positive premise.

Your task is narrower than ordinary hypothesis generation:
- return exactly ONE hypothesis that substantively depends on the assigned
  discovery axis, OR abstain if grounded premises cannot support a bounded
  extension using that axis;
- the assigned axis must create a meaningful mediator, moderator, conditional
  dependence, pathway competition, descriptor interaction, or other additional
  scientific dependency;
- the inferential_bridge MUST explain how the grounded premises are extended
  through the assigned axis;
- at least one predicted observation and its matching falsification criterion
  MUST probe the axis-specific dependency, not merely the already-grounded
  canonical relationship;
- positive premise_statement_ids still come ONLY from eligible statements in
  the grounded HypothesisContext;
- discovery inspiration IDs, candidate-unit IDs, node IDs, path IDs, and edge
  IDs MUST NEVER appear in premise_statement_ids;
- do not assert the discovery axis as reported fact. Phrase its role as the
  proposed inference being tested;
- do not claim external novelty, precedence, first discovery, or absence from
  literature.

A hypothesis fails this task if removing the assigned discovery axis leaves
its central hypothesis and predictions essentially unchanged. Merely adding
axis vocabulary to an otherwise canonical coordination→adsorption→HER chain is
not sufficient.
""".strip()


class DiscoveryAxisHypothesisPromptAssembler(HypothesisPromptAssembler):
    def __init__(
        self,
        axis: DiscoveryAxis,
        *,
        statement_text_limit: int = 1100,
        family_hierarchy: EvidenceFamilyHierarchy | None = None,
    ) -> None:
        super().__init__(
            statement_text_limit=statement_text_limit,
            max_hypotheses=1,
        )
        self.axis = axis
        self.family_hierarchy = family_hierarchy

    def build(self, context: HypothesisContext) -> HypothesisPrompt:
        base = super().build(context)
        axis = self.axis
        lines = [
            "",
            "ASSIGNED DISCOVERY AXIS (INSPIRATION ONLY; NOT EVIDENCE)",
            "======================================================",
            f"axis_id: {axis.axis_id}",
            f"axis_rank: {axis.axis_rank}",
            f"inspiration_id: {axis.inspiration_id}",
            f"candidate_unit_id: {axis.candidate_unit_id}",
            f"label: {axis.label}",
            f"entry_anchor: {axis.entry_anchor_label}",
            f"exit_anchor: {axis.exit_anchor_label}",
            (
                "proposed_semantics: "
                f"{axis.proposed_subject} | {axis.proposed_relation} | {axis.proposed_object}"
            ),
            f"candidate_unit_score: {axis.candidate_unit_score:.3f}",
            f"planner_score: {axis.planner_score:.3f}",
            f"mechanistic_continuity: {axis.mechanistic_continuity_band}",
            f"reaction_domain_switch_penalty: {axis.reaction_domain_switch_penalty:.2f}",
            f"route_context: {axis.rendered_path}",
            "STATUS: unverified inspiration; eligible_as_positive_premise=false",
            "",
            "AXIS-SPECIFIC OUTPUT DISCIPLINE",
            "===============================",
            "- Generate exactly ONE focused hypothesis for this axis, or abstain.",
            "- The hypothesis must still be scientifically anchored by at least one eligible positive premise.",
            "- The assigned axis must be essential to the new inferential bridge; do not merely mention it.",
            "- Make at least one prediction directly distinguish the axis-mediated/axis-conditioned account from the simpler grounded account.",
            "- Make at least one falsifier test the same axis-specific observable.",
            "- Do not simply restate a fully exposed corpus chain even if it is well grounded.",
            "- If the axis cannot be integrated without overclaiming, return hypotheses=[] and a concise abstention_reason.",
        ]
        system_prompt = base.system_prompt.rstrip() + "\n\n" + _SYSTEM_APPENDIX + "\n"

        user_sections = [base.user_prompt.rstrip()]
        prompt_version = PROMPT_VERSION
        if self.family_hierarchy is not None:
            user_sections.append(
                render_family_hierarchy_guidance(
                    self.family_hierarchy
                )
            )
            prompt_version = FAMILY_AWARE_PROMPT_VERSION
        user_sections.append("\n".join(lines))
        user_prompt = "\n\n".join(user_sections).rstrip() + "\n"

        canonical = _compact_json(
            {
                "prompt_version": prompt_version,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return HypothesisPrompt(
            prompt_version=prompt_version,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_sha256=_sha256(canonical),
        )

    def repair_feedback(
        self,
        *,
        previous_draft: HypothesisPortfolioDraft,
        issues: Iterable[object],
    ) -> str:
        base = super().repair_feedback(previous_draft=previous_draft, issues=issues)
        return (
            base
            + "\n\nDISCOVERY-AXIS REMINDER\n"
            + "=======================\n"
            + f"Assigned axis: {self.axis.label}\n"
            + "Repair the draft without turning the assigned axis into evidence. "
            + "The axis must remain an explicit proposed dependency in the inferential bridge and prediction.\n"
        )

    def fidelity_repair_feedback(
        self,
        *,
        previous_draft: HypothesisPortfolioDraft,
        reason: str,
    ) -> str:
        return "\n".join(
            [
                "DISCOVERY-AXIS FIDELITY REPAIR",
                "==============================",
                "The previous draft passed grounded evidence compilation but did not substantively use the assigned discovery axis.",
                f"Assigned axis: {self.axis.label}",
                (
                    "Proposed axis semantics: "
                    f"{self.axis.proposed_subject} | {self.axis.proposed_relation} | {self.axis.proposed_object}"
                ),
                f"Reason: {reason}",
                "Revise the hypothesis so that this axis creates an essential mediator, moderator, pathway competition, descriptor interaction, or conditional dependency.",
                "At least one prediction and matching falsifier must probe that axis-specific dependency.",
                "Do not change discovery content into a positive premise and do not claim external novelty.",
                "If this cannot be done from the grounded premises, abstain.",
                "Return a complete replacement HypothesisPortfolioDraft.",
                "",
                "PREVIOUS DRAFT",
                previous_draft.model_dump_json(indent=2),
            ]
        )

    def novelty_repair_feedback(
        self,
        *,
        previous_draft: HypothesisPortfolioDraft,
        novelty_status: str,
        interpretation: str,
        route_summary: str = "",
    ) -> str:
        lines = [
            "CORPUS-INTERNAL NOVELTY REPAIR",
            "==============================",
            "The previous proposal is acceptable as a grounded hypothesis but is too close to an existing corpus claim/route for this discovery-axis run.",
            f"Internal novelty status: {novelty_status}",
            f"Assessment: {interpretation}",
        ]
        if route_summary:
            lines.append(f"Existing-route signal: {route_summary}")
        lines.extend(
            [
                f"Assigned discovery axis: {self.axis.label}",
                "Do NOT merely paraphrase or lengthen the existing corpus chain.",
                "Use the assigned axis to introduce a materially additional dependency whose prediction would differ from the simpler existing account.",
                "Keep all positive premises grounded and keep the axis explicitly hypothetical.",
                "Do not claim external novelty.",
                "If the grounded premises cannot support such a revision, abstain.",
                "Return a complete replacement HypothesisPortfolioDraft.",
                "",
                "PREVIOUS DRAFT",
                previous_draft.model_dump_json(indent=2),
            ]
        )
        return "\n".join(lines)
