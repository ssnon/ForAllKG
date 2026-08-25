from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from pipeline_core.discovery.explorer_contracts import GraphExplorerPacket
from pipeline_core.discovery.explorer_draft import ExplorationDraft


PROMPT_VERSION = "graph-explorer-prompt-v2.5.1.3"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _truncate(text: str, limit: int = 900) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class ExplorerPrompt:
    prompt_version: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str

    @classmethod
    def create(cls, *, system_prompt: str, user_prompt: str) -> "ExplorerPrompt":
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


SYSTEM_PROMPT = """You are the Graph Explorer in an evidence-bounded scientific discovery system.

Your task is to organize and synthesize ONLY the evidence contained in the supplied GraphExplorerPacket view. You are not the Hypothesis Maker.

You MAY:
- summarize reported observations and reported mechanism claims,
- describe evidence-supported graph routes,
- compare evidence contained in the packet,
- identify recurring reported mechanistic motifs,
- identify packet-scoped tensions or unresolved connections,
- identify design levers that are explicitly supported by supplied evidence.

You MUST NOT:
- propose a new catalyst or material,
- propose a new mechanism,
- predict catalyst performance,
- recommend an experiment,
- invent a scientific relation,
- treat embedding/retrieval similarity as scientific confidence,
- treat registry/pattern alignment as mechanistic or causal evidence,
- reverse the scientific direction of an edge because graph navigation traversed it in reverse,
- infer paper-specific absence unless the packet explicitly allows absence claims for that paper.

Paper-specific absence is a strict extraction-completeness claim. If a paper has absence_claims_allowed=false, do NOT say that the paper "did not report", contains "no evidence", provides "no support", "did not observe", or otherwise lacks a relation/result. Incomplete extraction cannot establish a negative fact. If the missing relation matters, describe only the packet-level limitation (for example, that the supplied packet is insufficient to determine whether the relation was reported) without asserting absence from that paper.

A graph path proves navigability, not causality.
Cross-paper navigation through an alignment hub establishes a graph connection, not a reported cross-paper causal mechanism.
Use epistemic_role='reported' only when the statement itself is directly source-grounded by cited node/edge evidence. Use epistemic_role='evidence_synthesis' when you organize evidence across paths/papers. Use 'navigation_note' for graph-structural observations and 'unresolved' for packet-scoped limitations.

Every substantive statement must cite exact packet IDs in its support_* fields. Never fabricate an ID. Prefer the smallest sufficient support set. If a scientific statement cites a path containing reverse navigation, also cite the original scientific node or edge that supports the assertion.

Use claim_kind='mechanism' only when the cited support contains mechanism-bearing evidence under the packet's selected scientific domain. If the support establishes only an observation or association, use the weaker claim kind rather than labeling it mechanism. A recurring_mechanistic_motif must be backed by mechanism-bearing scientific node/edge evidence; navigation or alignment alone is insufficient.

Do not use strong causal wording in motif/design-lever labels unless the cited scientific evidence itself supports that causal wording. When uncertain, preserve a weaker association/observation rather than strengthening it.

Return only the structured ExplorationDraft requested by the caller. Local IDs are temporary labels for references inside the draft; use simple unique values such as s1, route1, motif1. Do not create final report IDs."""


class ExplorerPromptAssembler:
    def __init__(self, *, node_text_limit: int = 900, pointer_limit: int = 700) -> None:
        self.node_text_limit = int(node_text_limit)
        self.pointer_limit = int(pointer_limit)

    def build(self, packet: GraphExplorerPacket) -> ExplorerPrompt:
        lines: list[str] = []
        task = packet.task
        lines.extend(
            [
                "TASK",
                "====",
                f"task_id: {task.task_id}",
                f"domain_profile_id: {packet.domain_profile_id}",
                f"question: {task.question}",
                f"objective: {task.objective}",
                f"traversal_mode: {task.traversal_mode}",
                f"source_query: {task.source_query or '-'}",
                f"target_query: {task.target_query or '-'}",
                f"waypoint_query: {task.waypoint_query or '-'}",
                "",
                "PACKET SAFETY STATE",
                "===================",
                f"packet_sha256: {packet.packet_sha256}",
                f"provenance_missing_edges: {packet.provenance_summary.missing_pointer_edge_count}",
                f"novel_hypotheses_allowed: {packet.policy.novel_hypotheses_allowed}",
                f"scientific_direction_must_be_preserved: {packet.policy.scientific_direction_must_be_preserved}",
                f"retrieval_similarity_is_scientific_confidence: {packet.policy.retrieval_similarity_is_scientific_confidence}",
            ]
        )

        limited_papers = [
            paper
            for paper in packet.corpus.papers
            if paper.quality_status != "complete" or not paper.absence_claims_allowed
        ]
        lines.extend(["", "PAPER COMPLETENESS", "=================="])
        if limited_papers:
            for paper in limited_papers:
                lines.append(
                    f"- {paper.paper_id}: quality={paper.quality_status}; "
                    f"coverage={paper.source_token_coverage}; "
                    f"absence_claims_allowed={paper.absence_claims_allowed}"
                )
        else:
            lines.append("- All papers in scope are complete under the supplied manifest.")

        if packet.direct_concept_hits:
            lines.extend(["", "DIRECT CONCEPT HITS", "==================="])
            for hit in packet.direct_concept_hits:
                node = packet.evidence_catalog.nodes.get(hit.node_evidence_ref)
                lines.append(
                    f"- {hit.hit_id}: node={hit.node_id}; tier={hit.hit_tier}; "
                    f"basis={hit.quality_basis}; retrieval_similarity=({hit.source_similarity}, {hit.target_similarity}); "
                    f"requires_verification={hit.requires_verification}"
                )
                if node is not None:
                    lines.append(f"  evidence: [{node.node_type}] {_truncate(node.node_text, self.node_text_limit)}")

        node_catalog = packet.evidence_catalog.nodes
        edge_catalog = packet.evidence_catalog.edges
        lines.extend(["", "SELECTED PATHS", "=============="])
        for path in packet.paths:
            q = path.quality
            lines.extend(
                [
                    "",
                    f"PATH {path.path_id} (bundle_rank={path.bundle_rank})",
                    f"type={q.path_type}; mechanistic_content={q.mechanistic_content}; "
                    f"navigation_burden={q.navigation_burden}; reverse_fraction={q.reverse_fraction:.3f}; "
                    f"candidate_fraction={q.candidate_fraction:.3f}",
                    f"visited_papers={','.join(path.visited_paper_ids) or '-'}",
                    f"endpoint_source={path.endpoint.source_node_id} | {path.endpoint.source_label or ''} "
                    f"[retrieval_similarity={path.endpoint.source_similarity}; exact={path.endpoint.source_exact}]",
                    f"endpoint_target={path.endpoint.target_node_id} | {path.endpoint.target_label or ''} "
                    f"[retrieval_similarity={path.endpoint.target_similarity}; exact={path.endpoint.target_exact}]",
                ]
            )
            if path.waypoint is not None:
                lines.append(
                    f"waypoint={path.waypoint.node_id} | {path.waypoint.label or ''} "
                    f"[retrieval_similarity={path.waypoint.semantic_similarity}; tier={path.waypoint.semantic_tier}]"
                )
            lines.append("scientific assertions / navigation steps:")
            for index, step in enumerate(path.steps, start=1):
                reverse_note = " [NAVIGATED IN REVERSE]" if step.traversal_direction == "reverse" else ""
                edge = edge_catalog.get(step.edge_evidence_ref)
                provenance = edge.provenance_status if edge is not None else "UNKNOWN"
                lines.append(
                    f"  E{index}: {step.scientific_source} --{step.relation}--> {step.scientific_target}; "
                    f"edge_id={step.edge_evidence_ref}; edge_class={step.edge_class}; provenance={provenance}{reverse_note}"
                )
            lines.append("path node labels:")
            for node_id in path.node_ids:
                node = node_catalog.get(node_id)
                if node is None:
                    continue
                lines.append(f"  - {node_id}: [{node.node_type}] {_truncate(node.label, 320)}")

        if packet.alignment_contexts:
            lines.extend(["", "ALIGNMENT CONTEXTS", "=================="])
            lines.append("These contexts are navigation/entity-alignment metadata, NOT causal/mechanistic assertions.")
            for context in packet.alignment_contexts:
                lines.append(
                    f"- {context.context_id}: path={context.path_id}; hub={context.hub_node_id}; "
                    f"label={context.hub_label or '-'}; member_papers={','.join(context.member_paper_ids)}; "
                    f"entry={','.join(context.traversed_entry_node_ids) or '-'}; "
                    f"exit={','.join(context.traversed_exit_node_ids) or '-'}"
                )

        lines.extend(["", "NODE EVIDENCE CATALOG", "====================="])
        for node_id in sorted(node_catalog):
            node = node_catalog[node_id]
            lines.append(
                f"NODE {node_id}\n"
                f"  type={node.node_type}; evidence_status={node.evidence_status}; "
                f"requires_verification={node.requires_verification}; source_paper={node.source_paper_id or '-'}\n"
                f"  text={_truncate(node.node_text, self.node_text_limit)}"
            )

        lines.extend(["", "EDGE EVIDENCE CATALOG", "====================="])
        for edge_id in sorted(edge_catalog):
            edge = edge_catalog[edge_id]
            pointer_text = _truncate(_compact_json(edge.evidence_pointers), self.pointer_limit)
            lines.append(
                f"EDGE {edge_id}\n"
                f"  {edge.scientific_source} --{edge.relation}--> {edge.scientific_target}\n"
                f"  evidence_status={edge.evidence_status}; graph_layer={edge.graph_layer}; "
                f"requires_verification={edge.requires_verification}; provenance={edge.provenance_status}; "
                f"source_papers={','.join(edge.source_paper_ids) or '-'}\n"
                f"  evidence_pointers={pointer_text}"
            )

        lines.extend(
            [
                "",
                "OUTPUT DISCIPLINE",
                "=================",
                "- Return ExplorationDraft only.",
                "- Every statement needs at least one exact support ID from this prompt.",
                "- Do not use a path alone as the only support for a scientific claim when that path contains reverse navigation; also cite original node/edge evidence.",
                "- Do not describe alignment edges as mechanism evidence.",
                "- Use claim_kind='mechanism' only with mechanism-bearing node/edge/path support under the selected domain semantics.",
                "- Do not create recurring_mechanistic_motifs without mechanism-bearing scientific node/edge support.",
                "- Before writing any paper-specific absence statement, check PAPER COMPLETENESS. If absence_claims_allowed=false for any implicated paper, do not assert absence; express only a packet-scoped uncertainty/limitation without negative factual wording.",
                "- Do not turn a multi-paper navigation route into a causal chain.",
                "- If the packet only connects two mechanisms through registry/alignment navigation, say that explicitly and add an unresolved scope-limit statement rather than filling the missing causal link.",
                "- For unresolved_connections.reason use exactly one of: alignment_only, navigation_heavy, candidate_only, missing_direct_relation_in_packet, insufficient_provenance, partial_source_scope, insufficient_context.",
                "- Keep reported design levers to relationships explicitly supported in this packet; do not recommend a new composition, catalyst, or experiment.",
            ]
        )
        return ExplorerPrompt.create(system_prompt=SYSTEM_PROMPT, user_prompt="\n".join(lines))

    def repair_feedback(
        self,
        *,
        previous_draft: ExplorationDraft,
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
                "The previous draft failed deterministic compilation/validation.",
                "Revise only what is necessary to address the exact issues below.",
                "Do not add new scientific content, new IDs, new numbers, hypotheses, predictions, or experiments.",
                "For PAPER_ABSENCE_CLAIM_NOT_ALLOWED, do not preserve or paraphrase the unsupported paper-specific negative claim. Remove that claim or replace it only with a non-negative packet-scoped limitation already justified by the supplied packet.",
                "Return a complete replacement ExplorationDraft.",
                "",
                "ISSUES",
                *issue_lines,
                "",
                "PREVIOUS DRAFT",
                previous_draft.model_dump_json(indent=2),
            ]
        )
