from __future__ import annotations

import hashlib
import json
from pipeline_core.discovery.hypothesis_specification_prompt import (
    SELF_CONTAINED_SPECIFICATION_RULES,
)
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


PROMPT_VERSION = "hypothesis-maker-discovery-axis-prompt-v2.8.0-a4-s17-intent-v1"
FAMILY_AWARE_PROMPT_VERSION = (
    "hypothesis-maker-discovery-axis-prompt-v2.9.1-ec2c-s17-intent-v1"
)


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()



_KNOWN_RELATION_COMPONENT_SYSTEM_APPENDIX = r'''
S25d KNOWN-RELATION-COMPONENT SPECIAL CASE
==========================================
For this assigned axis, the external source provenance indicates that the
axis subject-relation-object is source-reported rather than an explicit
bounded synthesis. This changes ONLY its novelty-generation role.

The axis remains inspiration-only, unverified in the grounded context, and is
NOT a positive premise. Do not treat the retrieved literature as evidence for
the hypothesis.

However, do NOT use the source-reported axis relation itself as the central
novelty-bearing claim. Preserve it only as the baseline inspiration relation
that the hypothesis is trying to go beyond.

The central proposed dependency must instead be ONE grounded, testable
second-order relation selected from these domain-neutral forms:
- MODERATOR
- INTERACTION
- RESIDUAL
- BOUNDARY
- PROXY_DECOUPLING
- COMPENSATION_LIMIT

Every moderator, factor, correction, observable, regime, proxy, or contextual
state needed to instantiate that second-order relation must already be
supported by eligible grounded HypothesisContext statements. Do not invent
scientific content merely to instantiate an operator.

The prediction and matching falsifier must discriminate the second-order
dependency from the simpler source-reported baseline relation.

If the grounded context cannot support such a second-order dependency, abstain.

This special case overrides any generic instruction above that would otherwise
make the assigned source-reported S-R-O relation itself the central exploratory
claim. It does NOT promote external literature into evidence and does NOT
create novelty authority.
'''.strip()


def _s25d_known_relation_component_guidance(
    axis: DiscoveryAxis,
) -> str:
    if not axis.second_order_gap_required:
        return ""

    return "\\n".join(
        [
            "S25d KNOWN RELATION COMPONENT MODE",
            "==================================",
            (
                "axis_novelty_role: "
                + axis.inspiration_role
            ),
            (
                "external_relation_source_mode: "
                + axis.external_relation_source_mode
            ),
            (
                "baseline_external_relation: "
                + axis.proposed_subject
                + " | "
                + axis.proposed_relation
                + " | "
                + axis.proposed_object
            ),
            "",
            (
                "The baseline external relation is NOT the novelty target and "
                "is NOT a positive premise."
            ),
            (
                "Generate a testable second-order dependency only when the "
                "needed scientific content is already present in eligible "
                "grounded statements."
            ),
            (
                "Allowed reasoning operators: MODERATOR, INTERACTION, "
                "RESIDUAL, BOUNDARY, PROXY_DECOUPLING, COMPENSATION_LIMIT."
            ),
            (
                "Do not invent variables, mechanisms, regimes, corrections, "
                "proxies, materials, or conditions to force an operator."
            ),
            (
                "At least one prediction and matching falsifier must "
                "distinguish the second-order dependency from the baseline "
                "relation."
            ),
            (
                "If no grounded second-order dependency is supportable, abstain."
            ),
        ]
    )


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
axis vocabulary to an otherwise familiar downstream relation chain is not
sufficient.

DISCOVERY-INTENT RETENTION
==========================
The assigned discovery axis remains inspiration-only and MUST NOT become a
positive premise. Preserve its proposed semantics as the central exploratory
dependency rather than translating it back into a familiar downstream claim.

- The central dependency in hypothesis_statement must preserve the scientific
  roles of the assigned proposed_subject, proposed_relation, and
  proposed_object. Paraphrase is allowed, but role substitution is not.
- inferential_bridge must make that same assigned dependency the primary
  extension being proposed, rather than using it as decorative vocabulary.
- At least one predicted observation and one matching falsification criterion
  must directly test the assigned dependency.
- Do not replace the assigned proposed_object with a familiar downstream
  endpoint merely because that endpoint is well represented in the grounded
  context.
- Grounded canonical relations may explain plausibility, but they may not
  replace the assigned axis as the central hypothesis relation.
- Do not add unsupported sign, ordering, optimum, threshold, or quantitative
  specificity.
- If the assigned relation cannot be preserved without unsupported content,
  abstain.

This policy strengthens discovery-intent transfer only. It does not authorize
an external novelty claim.
""".strip()

def _s26a_task_generation_guidance(
    *,
    task_source: str | None,
    task_target: str | None,
    task_question: str | None,
) -> str:
    """
    Generation-time task anchoring.

    These strings are orchestration constraints, not scientific evidence.
    They never become premise IDs or novelty authority.
    """

    source = str(task_source or "").strip()
    target = str(task_target or "").strip()
    question = str(task_question or "").strip()

    if bool(source) != bool(target):
        raise ValueError(
            "S26a task anchoring requires both task_source and task_target"
        )

    if not source:
        return ""

    lines = [
        "S26a ORIGINAL-TASK PRESERVATION",
        "================================",
        (
            "The orchestration layer supplied the original scientific task "
            "endpoints below. They are generation constraints, NOT evidence."
        ),
        f"task_source: {source}",
        f"task_target: {target}",
    ]

    if question:
        lines.append(f"original_question: {question}")

    lines.extend(
        [
            "",
            (
                "The central hypothesis MUST remain about how the task_source "
                "relates to the task_target. Preserve both scientific roles; "
                "paraphrase is allowed, endpoint replacement is not."
            ),
            (
                "The assigned discovery axis may contribute only as a "
                "mediator, moderator, interaction, boundary condition, "
                "residual structure, proxy-decoupling condition, "
                "compensation-limit condition, or other subordinate "
                "dependency that changes or explains the task_source -> "
                "task_target relationship."
            ),
            (
                "Do NOT turn the discovery-axis subject/object into a new "
                "task that replaces either original endpoint."
            ),
            (
                "A hypothesis about axis_source -> axis_object is invalid "
                "when the original task_source/task_target relation has "
                "disappeared or become merely decorative."
            ),
            (
                "At least one predicted observation and matching falsifier "
                "must test the task-preserving axis-conditioned relation."
            ),
            (
                "The task endpoints are not positive premises. Scientific "
                "support must still come only from eligible grounded "
                "HypothesisContext statements."
            ),
            (
                "If the assigned axis cannot be connected to the original "
                "task without inventing unsupported scientific content, "
                "ABSTAIN instead of replacing the task."
            ),
        ]
    )

    return "\n".join(lines)


class DiscoveryAxisHypothesisPromptAssembler(HypothesisPromptAssembler):
    def __init__(
        self,
        axis: DiscoveryAxis,
        *,
        task_source: str | None = None,
        task_target: str | None = None,
        task_question: str | None = None,
        statement_text_limit: int = 1100,
        family_hierarchy: EvidenceFamilyHierarchy | None = None,
    ) -> None:
        super().__init__(
            statement_text_limit=statement_text_limit,
            max_hypotheses=1,
        )
        self.axis = axis
        self.family_hierarchy = family_hierarchy
        self.task_source = (
            str(task_source).strip()
            if task_source is not None
            else None
        )
        self.task_target = (
            str(task_target).strip()
            if task_target is not None
            else None
        )
        self.task_question = (
            str(task_question).strip()
            if task_question is not None
            else None
        )

        _s26a_task_generation_guidance(
            task_source=self.task_source,
            task_target=self.task_target,
            task_question=self.task_question,
        )

    def build(self, context: HypothesisContext) -> HypothesisPrompt:
        base = super().build(context)
        axis = self.axis
        task_guidance = (
            _s26a_task_generation_guidance(
                task_source=self.task_source,
                task_target=self.task_target,
                task_question=self.task_question,
            )
        )
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
            f"axis_novelty_role: {axis.inspiration_role}",
            (
                "external_relation_source_mode: "
                + axis.external_relation_source_mode
            ),
            (
                "second_order_gap_required: "
                + str(axis.second_order_gap_required).lower()
            ),
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
        if task_guidance:
            lines.extend(
                [
                    "",
                    task_guidance,
                ]
            )

        system_prompt = (
            base.system_prompt.rstrip()
            + "\n\n"
            + _SYSTEM_APPENDIX
            + "\n"
        )

        s25d_guidance = (
            _s25d_known_relation_component_guidance(
                axis
            )
        )
        if s25d_guidance:
            system_prompt = (
                system_prompt.rstrip()
                + "\n\n"
                + _KNOWN_RELATION_COMPONENT_SYSTEM_APPENDIX
                + "\n"
            )

        user_sections = [base.user_prompt.rstrip()]
        prompt_version = PROMPT_VERSION
        if self.family_hierarchy is not None:
            user_sections.append(
                render_family_hierarchy_guidance(
                    self.family_hierarchy
                )
            )
            prompt_version = FAMILY_AWARE_PROMPT_VERSION

        if s25d_guidance:
            user_sections.append(
                s25d_guidance
            )
            prompt_version = (
                prompt_version
                + "-s25d-known-relation-component"
            )

        user_sections.append("\n".join(lines))
        user_prompt = "\n\n".join(user_sections).rstrip() + "\n"

        if task_guidance:
            prompt_version = (
                prompt_version
                + "-s26a-task-anchored"
            )

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

    def inference_repair_feedback(
        self,
        *,
        previous_draft: HypothesisPortfolioDraft,
        review: object,
    ) -> str:
        review_json = (
            review.model_dump_json(indent=2)
            if hasattr(review, "model_dump_json")
            else str(review)
        )

        return "\n".join(
            [
                "DISCOVERY-AXIS INFERENCE-STRENGTH REPAIR",
                "========================================",
                SELF_CONTAINED_SPECIFICATION_RULES,
                (
                    "The previous hypothesis is scientifically usable in core form, "
                    "but the inference-strength critic found one or more assertions "
                    "that exceed the support supplied by the selected positive premises "
                    "and the inspiration-only discovery axis."
                ),
                "",
                f"Assigned axis: {self.axis.label}",
                (
                    "Proposed axis semantics: "
                    f"{self.axis.proposed_subject} | "
                    f"{self.axis.proposed_relation} | "
                    f"{self.axis.proposed_object}"
                ),
                "",
                "REPAIR RULES",
                "------------",
                "- Preserve assertions marked KEEP.",
                "- Preserve A_AXIS content as explicitly hypothetical.",
                (
                    "- For OPEN_DIRECTION, remove unsupported sign, ordering, "
                    "response shape, optimum, threshold, or directional specificity."
                ),
                (
                    "- For REFRAME, retain only the minimum open-direction relation "
                    "needed to test the central grounded-plus-axis synthesis."
                ),
                (
                    "- For REMOVE, remove that unsupported prediction/consequence "
                    "unless a weaker directly testable replacement is necessary to "
                    "keep the hypothesis falsifiable."
                ),
                (
                    "- If a prediction is removed or replaced, update its matching "
                    "falsification criterion so the final draft remains internally "
                    "consistent."
                ),
                "- Do not add new positive premises.",
                "- Do not promote the discovery axis into evidence.",
                (
                    "- Do not introduce new increase/decrease/shift/non-monotonic/"
                    "optimum/threshold claims unless they are explicitly supported "
                    "by the selected premises or axis semantics."
                ),
                "- Keep exactly one hypothesis for this axis, or abstain.",
                "- Return a complete replacement HypothesisPortfolioDraft.",
                "",
                "INFERENCE REVIEW",
                "----------------",
                review_json,
                "",
                "PREVIOUS DRAFT",
                "--------------",
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
