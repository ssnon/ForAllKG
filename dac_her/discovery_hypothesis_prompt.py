from __future__ import annotations

import hashlib
import json
from typing import Iterable

from dac_her.discovery_contracts import DiscoveryBundle
from dac_her.hypothesis_contracts import HypothesisContext, HypothesisPortfolioDraft
from dac_her.hypothesis_prompt import HypothesisPrompt, HypothesisPromptAssembler


PROMPT_VERSION = "hypothesis-maker-discovery-prompt-v2.8.0-a3"


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_DISCOVERY_SYSTEM_APPENDIX = """

DISCOVERY-INSPIRATION POLICY
============================
You may also receive a DiscoveryBundle containing graph routes selected for exploration potential.
These routes are NOT evidence and are NOT positive premises. They may suggest an inferential direction,
moderator, competing mechanism, descriptor failure, or cross-paper combination worth hypothesizing.

You MUST:
- cite positive premises only through exact premise_statement_ids from the grounded HypothesisContext;
- never place discovery inspiration IDs, path IDs, node IDs, or edge IDs in premise_statement_ids;
- never write an inspiration-only relation as though it were established by the literature;
- treat candidate/requires-verification inspirations as especially provisional;
- keep any scientific leap inspired by discovery routes explicit in inferential_bridge;
- prefer bounded, falsifiable extensions that combine or condition grounded premises rather than merely
  restating a well-established single-premise chain;
- prefer inspirations with high mechanistic continuity and low generic-entity/registry-hop burden when multiple routes suggest similar ideas;
- for candidate-unit routes, treat the candidate unit as ONE unverified semantic bridge grounded by distinct entry/exit anchors; do not interpret the navigation direction candidate->exit as a causal direction;
- prefer candidate units with high candidate_unit_score and low reaction_domain_switch_penalty; unrelated reaction-domain detours are weak discovery support;
- treat one-sided/low-continuity cross-paper routes as search hints, not as a complete mechanistic bridge;
- remain forbidden from claiming external novelty, precedence, first discovery, or absence from literature.

Discovery routes are a creativity surface, not an epistemic upgrade.
""".strip()


class DiscoveryAwareHypothesisPromptAssembler(HypothesisPromptAssembler):
    def __init__(
        self,
        discovery_bundle: DiscoveryBundle,
        *,
        statement_text_limit: int = 1100,
        max_hypotheses: int = 3,
    ) -> None:
        super().__init__(
            statement_text_limit=statement_text_limit,
            max_hypotheses=max_hypotheses,
        )
        self.discovery_bundle = discovery_bundle

    def build(self, context: HypothesisContext) -> HypothesisPrompt:
        if context.corpus_id != self.discovery_bundle.corpus_id:
            raise ValueError("DiscoveryBundle corpus_id does not match HypothesisContext")
        base = super().build(context)

        lines = [
            "",
            "DISCOVERY INSPIRATIONS (NOT POSITIVE PREMISES)",
            "=============================================",
            f"bundle_id: {self.discovery_bundle.bundle_id}",
            f"bundle_sha256: {self.discovery_bundle.bundle_sha256}",
            "These items may inspire the inferential bridge only. They are not evidence and have no premise_statement_id.",
        ]
        if not self.discovery_bundle.inspirations:
            lines.append("- NONE")
        for item in self.discovery_bundle.inspirations:
            flags = ",".join(item.reason_codes) or "-"
            lines.extend(
                [
                    (
                        f"- {item.inspiration_id}: score={item.exploration_score:.3f}; "
                        f"type={item.path_type}; mode={item.source_mode}; "
                        f"papers={','.join(item.paper_ids) or '-'}; "
                        f"requires_verification={item.requires_verification}; "
                        f"continuity={item.mechanistic_continuity_band}; "
                        f"generic_entity_fraction={item.generic_entity_fraction:.2f}; "
                        f"registry_hop_fraction={item.registry_hop_fraction:.2f}; "
                        f"grounding_semantic_overlap={item.semantic_similarity_to_grounding:.2f}; "
                        f"selected_semantic_overlap={item.max_semantic_similarity_to_selected:.2f}"
                    ),
                    f"  route: {item.rendered_path}",
                    *(
                        [
                            (
                                f"  candidate_unit: id={item.candidate_unit_id}; "
                                f"label={item.candidate_unit_label}; score={item.candidate_unit_score:.3f}; "
                                f"reaction_switch_penalty={item.reaction_domain_switch_penalty:.2f}"
                            ),
                            f"  candidate_entry_anchor: {item.candidate_entry_anchor_label} ({item.candidate_entry_anchor_id})",
                            f"  candidate_exit_anchor: {item.candidate_exit_anchor_label} ({item.candidate_exit_anchor_id})",
                            (
                                "  candidate_proposal: "
                                f"{item.candidate_proposed_subject} | "
                                f"{item.candidate_proposed_relation} | "
                                f"{item.candidate_proposed_object}"
                            ),
                            "  candidate_semantics: one unverified unit; entry/exit are grounding anchors, not a causal arrow",
                        ]
                        if item.candidate_unit_id
                        else []
                    ),
                    f"  exploration_reasons: {flags}",
                    "  STATUS: inspiration_only; eligible_as_positive_premise=false",
                ]
            )

        lines.extend(
            [
                "",
                "DISCOVERY USE DISCIPLINE",
                "========================",
                "- Prefer a discovery-inspired hypothesis only when at least one grounded positive premise makes the proposed leap scientifically anchored.",
                "- A route that merely shares an entity, support, metal, registry hub, or navigation scaffold is not itself a mechanism.",
                "- Prefer high-continuity routes where mechanism-bearing content exists on both sides of a cross-paper alignment.",
                "- Discount routes flagged generic_entity_hopping, registry_hop_heavy, or mechanistic_continuity_low unless they only motivate a targeted question.",
                "- Avoid generating two hypotheses from semantically redundant inspiration routes when a distinct mechanistic axis is available.",
                "- If a discovery route suggests a moderator or competing mechanism, formulate it as a conditional/falsifiable inference rather than an asserted fact.",
                "- Do not claim that a high exploration_score means the hypothesis is novel. Internal/external novelty are assessed later.",
            ]
        )

        system_prompt = base.system_prompt.rstrip() + "\n\n" + _DISCOVERY_SYSTEM_APPENDIX + "\n"
        user_prompt = base.user_prompt.rstrip() + "\n" + "\n".join(lines) + "\n"
        canonical = _compact_json(
            {
                "prompt_version": PROMPT_VERSION,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return HypothesisPrompt(
            prompt_version=PROMPT_VERSION,
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
            + "\n\nDISCOVERY REMINDER\n"
            + "==================\n"
            + "Discovery inspiration IDs/path IDs/node IDs/edge IDs are never valid positive premise_statement_ids. "
            + "Preserve discovery content only as an explicitly hypothetical inferential bridge when grounded premises support doing so.\n"
        )
