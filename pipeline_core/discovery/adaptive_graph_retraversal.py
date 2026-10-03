from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Iterable, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(canonical_json(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _strings(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        str(item).strip()
        for item in value
        if str(item).strip()
    }


def _step_edge_id(step: Mapping[str, Any]) -> str:
    return str(
        step.get("selected_original_edge_id")
        or step.get("navigation_edge_id")
        or step.get("edge_id")
        or ""
    ).strip()


def path_footprint(path: Mapping[str, Any]) -> dict[str, set[str]]:
    edges = {
        edge_id
        for step in path.get("steps", [])
        if isinstance(step, Mapping)
        for edge_id in [_step_edge_id(step)]
        if edge_id
    }
    nodes = _strings(path.get("nodes"))
    papers = set()
    for key in ("visited_paper_ids", "supporting_paper_ids", "source_paper_ids"):
        papers |= _strings(path.get(key))
    for step in path.get("steps", []):
        if isinstance(step, Mapping):
            papers |= _strings(step.get("source_paper_ids"))
    return {
        "path_ids": {str(path.get("path_id") or "").strip()} - {""},
        "edge_ids": edges,
        "node_ids": nodes,
        "paper_ids": papers,
    }


def traversal_footprint(traversal: Mapping[str, Any]) -> dict[str, set[str]]:
    result = {
        "path_ids": set(),
        "edge_ids": set(),
        "node_ids": set(),
        "paper_ids": set(),
    }
    for row in traversal.get("paths", []):
        if not isinstance(row, Mapping):
            continue
        footprint = path_footprint(row)
        for key in result:
            result[key] |= footprint[key]
    return result


def context_footprint(context: Mapping[str, Any]) -> dict[str, set[str]]:
    result = {
        "path_ids": set(),
        "edge_ids": set(),
        "node_ids": set(),
        "paper_ids": set(),
    }
    for row in context.get("evidence_statements", []):
        if not isinstance(row, Mapping):
            continue
        result["path_ids"] |= _strings(row.get("support_path_ids"))
        result["path_ids"] |= _strings(row.get("alignment_path_ids"))
        result["edge_ids"] |= _strings(row.get("scientific_support_edge_ids"))
        result["node_ids"] |= _strings(row.get("scientific_support_node_ids"))
        result["paper_ids"] |= _strings(row.get("paper_ids"))
    for row in context.get("mechanism_routes", []):
        if isinstance(row, Mapping):
            rid = str(row.get("route_id") or "").strip()
            if rid:
                result["path_ids"].add(rid)
            result["paper_ids"] |= _strings(row.get("paper_ids"))
    return result


def merge_footprints(
    footprints: Iterable[Mapping[str, set[str]]],
) -> dict[str, set[str]]:
    result = {
        "path_ids": set(),
        "edge_ids": set(),
        "node_ids": set(),
        "paper_ids": set(),
    }
    for footprint in footprints:
        for key in result:
            result[key] |= set(footprint.get(key, set()))
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _new_fraction(current: set[str], prior: set[str]) -> float:
    if not current:
        return 0.0
    return len(current - prior) / len(current)


def score_retraversal_candidate(
    path: Mapping[str, Any],
    *,
    prior: Mapping[str, set[str]],
    min_new_edge_fraction: float,
    min_new_paper_fraction: float,
    max_prior_edge_jaccard: float,
) -> dict[str, Any]:
    footprint = path_footprint(path)
    path_id = str(path.get("path_id") or "")
    exact_reuse = path_id in prior.get("path_ids", set())
    new_edge_fraction = _new_fraction(
        footprint["edge_ids"], prior.get("edge_ids", set())
    )
    new_paper_fraction = _new_fraction(
        footprint["paper_ids"], prior.get("paper_ids", set())
    )
    new_node_fraction = _new_fraction(
        footprint["node_ids"], prior.get("node_ids", set())
    )
    edge_jaccard = _jaccard(
        footprint["edge_ids"], prior.get("edge_ids", set())
    )

    quality = path.get("path_quality", {})
    if not isinstance(quality, Mapping):
        quality = {}
    try:
        mechanism_score = float(quality.get("mechanistic_content_score", 0.0))
    except (TypeError, ValueError):
        mechanism_score = 0.0
    mechanism_score = max(0.0, min(1.0, mechanism_score))

    admissible = (
        not exact_reuse
        and edge_jaccard <= max_prior_edge_jaccard
        and (
            new_edge_fraction >= min_new_edge_fraction
            or new_paper_fraction >= min_new_paper_fraction
        )
    )
    score = (
        0.50 * new_edge_fraction
        + 0.30 * new_paper_fraction
        + 0.10 * new_node_fraction
        + 0.10 * mechanism_score
    )
    return {
        "path_id": path_id,
        "admissible": bool(admissible),
        "score": score,
        "exact_path_reuse": exact_reuse,
        "new_edge_fraction": new_edge_fraction,
        "new_paper_fraction": new_paper_fraction,
        "new_node_fraction": new_node_fraction,
        "prior_edge_jaccard": edge_jaccard,
        "mechanistic_content_score": mechanism_score,
        "edge_ids": sorted(footprint["edge_ids"]),
        "paper_ids": sorted(footprint["paper_ids"]),
        "node_ids": sorted(footprint["node_ids"]),
    }


def select_retraversal_paths(
    candidate_paths: list[dict[str, Any]],
    *,
    prior: Mapping[str, set[str]],
    top_k: int = 6,
    min_new_edge_fraction: float = 0.35,
    min_new_paper_fraction: float = 0.25,
    max_prior_edge_jaccard: float = 0.85,
    max_selected_edge_jaccard: float = 0.85,
    paper_expansion_reserve: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    scored = [
        (
            row,
            score_retraversal_candidate(
                row,
                prior=prior,
                min_new_edge_fraction=min_new_edge_fraction,
                min_new_paper_fraction=min_new_paper_fraction,
                max_prior_edge_jaccard=max_prior_edge_jaccard,
            ),
        )
        for row in candidate_paths
        if isinstance(row, dict)
    ]
    scored.sort(
        key=lambda item: (
            not item[1]["admissible"],
            -float(item[1]["score"]),
            float(item[0].get("total_cost", 0.0)),
            str(item[0].get("path_id") or ""),
        )
    )

    if paper_expansion_reserve < 0:
        raise ValueError("paper_expansion_reserve must be nonnegative")

    selected: list[dict[str, Any]] = []
    selected_edges: list[set[str]] = []
    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()

    def try_select(
        path: dict[str, Any],
        diagnostic: dict[str, Any],
        *,
        reason: str,
    ) -> bool:
        pid = str(diagnostic["path_id"])
        if pid in seen or not diagnostic["admissible"]:
            return False
        edges = set(diagnostic["edge_ids"])
        if any(
            _jaccard(edges, incumbent) > max_selected_edge_jaccard
            for incumbent in selected_edges
        ):
            diagnostics.append(
                {
                    **diagnostic,
                    "selected": False,
                    "selection_reason": "selected_path_edge_overlap",
                }
            )
            seen.add(pid)
            return False
        selected.append(path)
        selected_edges.append(edges)
        diagnostics.append(
            {
                **diagnostic,
                "selected": True,
                "selection_reason": reason,
            }
        )
        seen.add(pid)
        return True

    # v2B: when the persistent KG exposes paths supported by papers that were
    # not present in the prior grounded footprint, reserve a bounded number of
    # slots for those paths. This is a preference, not a hard requirement:
    # same-paper structural expansion remains legal when no new-paper path is
    # available.
    paper_candidates = [
        item
        for item in scored
        if (
            item[1]["admissible"]
            and float(item[1]["new_paper_fraction"]) > 0.0
        )
    ]
    paper_candidates.sort(
        key=lambda item: (
            -float(item[1]["new_paper_fraction"]),
            -float(item[1]["score"]),
            float(item[0].get("total_cost", 0.0)),
            str(item[0].get("path_id") or ""),
        )
    )

    reserve_target = min(
        top_k,
        paper_expansion_reserve,
        len(paper_candidates),
    )
    reserved = 0
    for path, diagnostic in paper_candidates:
        if reserved >= reserve_target:
            break
        if try_select(
            path,
            diagnostic,
            reason="paper_expansion_reserve",
        ):
            reserved += 1

    for path, diagnostic in scored:
        if len(selected) >= top_k:
            break
        try_select(
            path,
            diagnostic,
            reason="novel_grounded_neighborhood",
        )

    selected_ids = {str(row.get("path_id") or "") for row in selected}
    for _, diagnostic in scored:
        if diagnostic["path_id"] in seen:
            continue
        diagnostics.append(
            {
                **diagnostic,
                "selected": diagnostic["path_id"] in selected_ids,
                "selection_reason": (
                    "novel_grounded_neighborhood"
                    if diagnostic["path_id"] in selected_ids
                    else (
                        "inadmissible_against_prior_footprint"
                        if not diagnostic["admissible"]
                        else "selection_budget_exhausted"
                    )
                ),
            }
        )

    report = {
        "schema_version": "adaptive-graph-retraversal-selection-v1",
        "candidate_count": len(scored),
        "admissible_count": sum(bool(item[1]["admissible"]) for item in scored),
        "selected_count": len(selected),
        "selected_path_ids": sorted(selected_ids),
        "paper_expansion_candidate_count": len(paper_candidates),
        "selected_paper_expansion_count": sum(
            1
            for row in diagnostics
            if row.get("selected")
            and float(row.get("new_paper_fraction", 0.0)) > 0.0
        ),
        "selected_structural_only_count": sum(
            1
            for row in diagnostics
            if row.get("selected")
            and float(row.get("new_paper_fraction", 0.0)) <= 0.0
        ),
        "policy": {
            "top_k": top_k,
            "min_new_edge_fraction": min_new_edge_fraction,
            "min_new_paper_fraction": min_new_paper_fraction,
            "max_prior_edge_jaccard": max_prior_edge_jaccard,
            "max_selected_edge_jaccard": max_selected_edge_jaccard,
            "paper_expansion_reserve": paper_expansion_reserve,
            "paper_expansion_is_preference_not_requirement": True,
        },
        "prior_footprint_counts": {
            key: len(prior.get(key, set()))
            for key in ("path_ids", "edge_ids", "node_ids", "paper_ids")
        },
        "diagnostics": diagnostics,
        "external_prior_art_used_as_positive_premise": False,
        "external_prior_art_used_for_path_selection": False,
        "canonical_graph_mutated": False,
    }
    report["selection_id"] = stable_id(
        "adaptive_graph_retraversal_selection", report
    )
    return selected, report


def _path_type_groups(paths: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for row in paths:
        quality = row.get("path_quality", {})
        if not isinstance(quality, Mapping):
            quality = {}
        path_type = str(quality.get("path_type") or "UNKNOWN")
        groups.setdefault(path_type, []).append(str(row.get("path_id") or ""))
    return {key: value for key, value in sorted(groups.items())}


def _paper_ids(paths: list[dict[str, Any]]) -> list[str]:
    papers: set[str] = set()
    for row in paths:
        papers |= path_footprint(row)["paper_ids"]
    return sorted(papers)


def materialize_selected_traversal(
    broad_traversal: Mapping[str, Any],
    *,
    selected_paths: list[dict[str, Any]],
    selection_report: Mapping[str, Any],
    source_context_id: str,
) -> dict[str, Any]:
    if not selected_paths:
        raise ValueError("cannot materialize retraversal with zero selected paths")

    body = copy.deepcopy(dict(broad_traversal))
    groups = _path_type_groups(selected_paths)
    paper_ids = _paper_ids(selected_paths)

    body["paths"] = copy.deepcopy(selected_paths)
    body["path_count"] = len(selected_paths)
    body["returned_path_count"] = len(selected_paths)
    body["candidate_paths"] = copy.deepcopy(selected_paths)
    body["candidate_paths_included"] = True
    body["candidate_path_count"] = len(selected_paths)
    body["candidate_path_count_before_top_k"] = len(selected_paths)
    body["waypoint_relevance_pool_count"] = len(selected_paths)
    body["path_groups"] = groups
    body["returned_path_groups"] = groups
    body["candidate_path_groups"] = groups
    body["path_type_counts"] = {key: len(value) for key, value in groups.items()}
    body["returned_path_type_counts"] = dict(body["path_type_counts"])
    body["candidate_path_type_counts"] = dict(body["path_type_counts"])
    body["direct_concept_hits"] = []
    body["direct_concept_hit_count"] = 0
    body["direct_concept_hits_enabled"] = False
    body["bundle_selection"] = {
        "enabled": True,
        "selector": "adaptive_graph_retraversal_v2",
        "selected_path_ids": [str(row.get("path_id") or "") for row in selected_paths],
    }
    body["path_paper_coverage"] = {
        "candidate_distinct_paper_count": len(paper_ids),
        "candidate_paper_ids": paper_ids,
        "returned_distinct_paper_count": len(paper_ids),
        "returned_paper_ids": paper_ids,
    }
    body["adaptive_graph_retraversal"] = {
        "schema_version": "adaptive-graph-retraversal-materialization-v1",
        "source_context_id": source_context_id,
        "selection_id": selection_report.get("selection_id"),
        "selected_path_ids": selection_report.get("selected_path_ids", []),
        "direct_concept_hits_suppressed": True,
        "external_prior_art_used_as_positive_premise": False,
        "canonical_graph_mutated": False,
    }
    return body


def eligible_premise_ids(context: Mapping[str, Any]) -> set[str]:
    return {
        str(row.get("statement_id") or "")
        for row in context.get("evidence_statements", [])
        if (
            isinstance(row, Mapping)
            and row.get("eligible_as_premise")
            and not row.get("requires_verification")
            and not row.get("premise_restrictions")
            and str(row.get("epistemic_role") or "")
            in {"reported", "evidence_synthesis"}
            and str(row.get("statement_id") or "")
        )
    }


def _norm_text(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def eligible_premise_rows(
    context: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("statement_id") or ""): dict(row)
        for row in context.get("evidence_statements", [])
        if (
            isinstance(row, Mapping)
            and row.get("eligible_as_premise")
            and not row.get("requires_verification")
            and not row.get("premise_restrictions")
            and str(row.get("epistemic_role") or "")
            in {"reported", "evidence_synthesis"}
            and str(row.get("statement_id") or "")
        )
    }


def audit_context_delta(
    *,
    source_context: Mapping[str, Any],
    output_context: Mapping[str, Any],
) -> dict[str, Any]:
    source_rows = eligible_premise_rows(source_context)
    output_rows = eligible_premise_rows(output_context)

    source_texts = {
        _norm_text(row.get("text"))
        for row in source_rows.values()
        if _norm_text(row.get("text"))
    }
    source_edges = {
        item
        for row in source_rows.values()
        for item in _strings(row.get("scientific_support_edge_ids"))
    }
    source_nodes = {
        item
        for row in source_rows.values()
        for item in _strings(row.get("scientific_support_node_ids"))
    }
    source_papers = {
        item
        for row in source_rows.values()
        for item in _strings(row.get("paper_ids"))
    }
    source_paths = {
        item
        for row in source_rows.values()
        for item in (
            _strings(row.get("support_path_ids"))
            | _strings(row.get("alignment_path_ids"))
        )
    }

    rows: list[dict[str, Any]] = []
    structurally_new: list[str] = []
    id_new: list[str] = []

    for statement_id, row in sorted(output_rows.items()):
        if statement_id in source_rows:
            continue
        id_new.append(statement_id)
        edges = _strings(row.get("scientific_support_edge_ids"))
        nodes = _strings(row.get("scientific_support_node_ids"))
        papers = _strings(row.get("paper_ids"))
        paths = (
            _strings(row.get("support_path_ids"))
            | _strings(row.get("alignment_path_ids"))
        )
        new_edges = edges - source_edges
        new_nodes = nodes - source_nodes
        new_papers = papers - source_papers
        new_paths = paths - source_paths
        exact_text_reuse = _norm_text(row.get("text")) in source_texts
        # Path IDs are diagnostic lineage, not sufficient scientific novelty:
        # the same scientific route can receive a different path ID under a
        # different traversal algorithm. Require a new grounded scientific
        # edge or a new supporting paper.
        support_novel = bool(new_edges or new_papers)
        structurally_new_row = bool(
            support_novel and not exact_text_reuse
        )
        if structurally_new_row:
            structurally_new.append(statement_id)
        rows.append(
            {
                "statement_id": statement_id,
                "text": str(row.get("text") or ""),
                "exact_normalized_text_reuse": exact_text_reuse,
                "new_support_edge_ids": sorted(new_edges),
                "new_support_node_ids": sorted(new_nodes),
                "new_paper_ids": sorted(new_papers),
                "new_support_path_ids": sorted(new_paths),
                "support_novel": support_novel,
                "structurally_new_eligible_premise": structurally_new_row,
                "novelty_class": (
                    "NEW_PAPER_AND_GRAPH_SUPPORT"
                    if new_papers and (new_edges or new_paths)
                    else (
                        "NEW_PAPER_SUPPORT"
                        if new_papers
                        else (
                            "NEW_GRAPH_SUPPORT"
                            if new_edges or new_paths
                            else (
                                "NEW_NODE_ONLY"
                                if new_nodes
                                else "ID_ONLY_OR_TEXT_REUSE"
                            )
                        )
                    )
                ),
            }
        )

    report = {
        "schema_version": "adaptive-context-delta-audit-v1",
        "source_context_id": source_context.get("context_id"),
        "output_context_id": output_context.get("context_id"),
        "source_eligible_premise_count": len(source_rows),
        "output_eligible_premise_count": len(output_rows),
        "id_new_eligible_premise_ids": id_new,
        "id_new_eligible_premise_count": len(id_new),
        "structurally_new_eligible_premise_ids": structurally_new,
        "structurally_new_eligible_premise_count": len(structurally_new),
        "statement_deltas": rows,
        "id_novelty_alone_is_sufficient": False,
        "external_prior_art_as_positive_premise": False,
    }
    report["audit_id"] = stable_id(
        "adaptive_context_delta_audit",
        report,
    )
    return report


def build_retraversal_request(
    *,
    decision: Mapping[str, Any],
    source_context: Mapping[str, Any],
    attempt_history: list[dict[str, Any]],
    context_epoch: int,
    retraversal_index: int,
    reason: str,
) -> dict[str, Any]:
    body = {
        "schema_version": "adaptive-graph-retraversal-request-v2",
        "root_hypothesis_id": decision.get("root_hypothesis_id"),
        "current_hypothesis_id": decision.get("current_hypothesis_id"),
        "current_epistemic_state": decision.get("current_epistemic_state"),
        "current_external_status": decision.get("current_external_status"),
        "source_context_id": source_context.get("context_id"),
        "source_context_sha256": source_context.get("context_sha256"),
        "task_id": source_context.get("task_id"),
        "question": source_context.get("question"),
        "corpus_id": source_context.get("corpus_id"),
        "domain_profile_id": source_context.get("domain_profile_id"),
        "context_epoch": int(context_epoch),
        "retraversal_index": int(retraversal_index),
        "request_reason": str(reason),
        "already_known_boundary": list(
            decision.get("already_known_boundary", []) or []
        ),
        "unresolved_boundary": list(
            decision.get("unresolved_boundary", []) or []
        ),
        "target_claim_ids": list(decision.get("target_claim_ids", []) or []),
        "attempt_history": list(attempt_history),
        "external_prior_art_as_positive_premise": False,
        "external_boundary_used_for_positive_path_selection": False,
        "canonical_graph_mutation_requested": False,
    }
    body["request_id"] = stable_id("adaptive_graph_retraversal_request", body)
    return body
