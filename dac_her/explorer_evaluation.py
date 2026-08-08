from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .explorer_benchmark_contracts import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkIssue,
)

_ABSENCE_RE = re.compile(
    r"\b(absent|absence|not\s+reported|does\s+not\s+report|did\s+not\s+report|"
    r"no\s+evidence\s+of|lacks?|without)\b",
    re.IGNORECASE,
)

_CAUSAL_RE = re.compile(
    r"\b(caus(?:e|es|ed|ing)|driv(?:e|es|en|ing)|promot(?:e|es|ed|ing)|"
    r"enhanc(?:e|es|ed|ing)|facilitat(?:e|es|ed|ing)|lead(?:s)?\s+to|"
    r"result(?:s)?\s+in|mediate(?:s|d|ing)?|regulat(?:e|es|ed|ing)|"
    r"control(?:s|led|ling)?|lower(?:s|ed|ing)?|stabiliz(?:e|es|ed|ing)?)\b",
    re.IGNORECASE,
)

_NEGATION_WINDOW_RE = re.compile(
    r"\b(?:does\s+not|do\s+not|did\s+not|not|cannot|can't|fails?\s+to|"
    r"doesn't|is\s+not|are\s+not)\b.{0,45}\b"
    r"(?:cause|causes|drive|drives|promote|promotes|enhance|enhances|"
    r"facilitate|facilitates|lead\s+to|leads\s+to|mediate|mediates|"
    r"regulate|regulates|control|controls|lower|lowers|stabilize|stabilizes)\b",
    re.IGNORECASE,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _norm(text: str) -> str:
    # Canonicalize Unicode punctuation so lexical benchmark terms such as
    # "charge transfer" also match scientifically equivalent spellings like
    # "charge-transfer" / "charge–transfer" / "charge—transfer".
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = re.sub(r"[\u2010-\u2015\u2212-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _statements(report: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    items = report.get("statements", [])
    return [x for x in items if isinstance(x, Mapping)]


def _path_index(packet: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(p.get("path_id")): p
        for p in packet.get("paths", [])
        if isinstance(p, Mapping) and p.get("path_id")
    }


def _candidate_ids(packet: Mapping[str, Any]) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    nodes: Set[str] = set()
    edges: Set[str] = set()
    hits: Set[str] = set()
    paths: Set[str] = set()

    catalog = packet.get("evidence_catalog", {})
    for node_id, node in (catalog.get("nodes", {}) or {}).items():
        if isinstance(node, Mapping) and bool(node.get("requires_verification")):
            nodes.add(str(node_id))
    for edge_id, edge in (catalog.get("edges", {}) or {}).items():
        if isinstance(edge, Mapping) and bool(edge.get("requires_verification")):
            edges.add(str(edge_id))
    for hit in packet.get("direct_concept_hits", []) or []:
        if isinstance(hit, Mapping) and bool(hit.get("requires_verification")):
            if hit.get("hit_id"):
                hits.add(str(hit["hit_id"]))
            if hit.get("node_id"):
                nodes.add(str(hit["node_id"]))
    for path in packet.get("paths", []) or []:
        if not isinstance(path, Mapping) or not path.get("path_id"):
            continue
        is_candidate = bool(path.get("requires_verification"))
        quality = path.get("quality", {}) or {}
        if isinstance(quality, Mapping) and float(quality.get("candidate_fraction", 0.0) or 0.0) > 0:
            is_candidate = True
        for step in path.get("steps", []) or []:
            if isinstance(step, Mapping) and bool(step.get("requires_verification")):
                is_candidate = True
                if step.get("edge_evidence_ref"):
                    edges.add(str(step["edge_evidence_ref"]))
        if is_candidate:
            paths.add(str(path["path_id"]))
    return nodes, edges, hits, paths


def _alignment_paths(packet: Mapping[str, Any]) -> Set[str]:
    result: Set[str] = set()
    for path in packet.get("paths", []) or []:
        if not isinstance(path, Mapping) or not path.get("path_id"):
            continue
        quality = path.get("quality", {}) or {}
        if bool(quality.get("uses_alignment")) or int(quality.get("alignment_edge_count", 0) or 0) > 0:
            result.add(str(path["path_id"]))
            continue
        for step in path.get("steps", []) or []:
            if not isinstance(step, Mapping):
                continue
            edge_class = _norm(step.get("edge_class", ""))
            relation = _norm(step.get("relation", ""))
            if "alignment" in edge_class or "registry" in edge_class or "pattern_alignment" in edge_class:
                result.add(str(path["path_id"]))
                break
            if relation in {"registry_alignment", "pattern_alignment"}:
                result.add(str(path["path_id"]))
                break
    return result


def _partial_absence_blocked_papers(packet: Mapping[str, Any]) -> Set[str]:
    blocked: Set[str] = set()
    corpus = packet.get("corpus", {}) or {}
    for paper in corpus.get("papers", []) or []:
        if not isinstance(paper, Mapping):
            continue
        if paper.get("paper_id") and paper.get("absence_claims_allowed") is False:
            blocked.add(str(paper["paper_id"]))
    return blocked


def _role_counts(report: Mapping[str, Any]) -> Dict[str, int]:
    counts = {"reported": 0, "evidence_synthesis": 0, "navigation_note": 0, "unresolved": 0}
    for stmt in _statements(report):
        role = str(stmt.get("epistemic_role", ""))
        if role in counts:
            counts[role] += 1
    return counts


def _repair_metrics(
    draft: Optional[Mapping[str, Any]],
    repair: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not draft or not repair:
        return {
            "repair_artifact_available": False,
            "repair_text_change_count": 0,
            "repair_new_statement_count": 0,
            "repair_deleted_statement_count": 0,
            "support_only_repair": None,
        }

    dmap = {str(s.get("local_id")): s for s in draft.get("statements", []) if isinstance(s, Mapping)}
    rmap = {str(s.get("local_id")): s for s in repair.get("statements", []) if isinstance(s, Mapping)}
    common = set(dmap) & set(rmap)
    text_changes = sum(_norm(dmap[k].get("text", "")) != _norm(rmap[k].get("text", "")) for k in common)
    new_statements = len(set(rmap) - set(dmap))
    deleted_statements = len(set(dmap) - set(rmap))

    # Treat statement text / epistemic role / claim kind and section labels/references as semantic.
    def semantic_projection(doc: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["statements"] = sorted(
            [
                (
                    str(s.get("local_id")),
                    str(s.get("text", "")),
                    str(s.get("epistemic_role", "")),
                    str(s.get("claim_kind", "")),
                )
                for s in doc.get("statements", [])
                if isinstance(s, Mapping)
            ]
        )
        for key in (
            "mechanism_routes",
            "recurring_mechanistic_motifs",
            "cross_paper_connections",
            "evidence_tensions",
            "unresolved_connections",
            "reported_design_levers",
        ):
            vals = []
            for item in doc.get(key, []) or []:
                if not isinstance(item, Mapping):
                    continue
                vals.append(
                    tuple(
                        (k, json.dumps(v, sort_keys=True, ensure_ascii=False))
                        for k, v in sorted(item.items())
                        if not (
                            k.startswith("support_")
                            or k in {"mechanism_node_ids", "mechanism_edge_ids", "outcome_node_ids"}
                        )
                    )
                )
            out[key] = sorted(vals, key=repr)
        return out

    support_only = semantic_projection(draft) == semantic_projection(repair)
    return {
        "repair_artifact_available": True,
        "repair_text_change_count": text_changes,
        "repair_new_statement_count": new_statements,
        "repair_deleted_statement_count": deleted_statements,
        "support_only_repair": support_only,
    }


def evaluate_case(
    case: BenchmarkCase,
    *,
    repo_root: Path,
) -> BenchmarkCaseResult:
    packet_path = (repo_root / case.packet).resolve()
    prefix = (repo_root / case.output_prefix).resolve()
    report_path = Path(str(prefix) + ".report.json")
    run_path = Path(str(prefix) + ".run.json")
    validation_path = Path(str(prefix) + ".validation.json")
    draft_path = Path(str(prefix) + ".draft.json")
    repair_path = Path(str(prefix) + ".repair1.draft.json")

    issues: List[BenchmarkIssue] = []

    required = {
        "packet": packet_path,
        "report": report_path,
        "run": run_path,
        "validation": validation_path,
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        return BenchmarkCaseResult(
            case_id=case.case_id,
            passes=False,
            issues=[
                BenchmarkIssue(
                    severity="error",
                    code="MISSING_ARTIFACTS",
                    message="; ".join(missing),
                )
            ],
            metrics={"missing_artifacts": missing},
        )

    packet = load_json(packet_path)
    report = load_json(report_path)
    run = load_json(run_path)
    validation = load_json(validation_path)
    draft = load_json(draft_path) if draft_path.exists() else None
    repair = load_json(repair_path) if repair_path.exists() else None

    exp = case.expectations
    roles = _role_counts(report)
    unresolved_count = len(report.get("unresolved_connections", []) or [])
    route_count = len(report.get("mechanism_routes", []) or [])
    repairs = int(run.get("repair_attempts", 0) or 0)
    compile_issues = int(run.get("compile_issue_count", 0) or 0)
    validation_pass = bool(validation.get("passes", False)) and bool(run.get("final_validation_passed", False))

    def error(code: str, message: str) -> None:
        issues.append(BenchmarkIssue(severity="error", code=code, message=message))

    def warning(code: str, message: str) -> None:
        issues.append(BenchmarkIssue(severity="warning", code=code, message=message))

    if exp.must_validate and not validation_pass:
        error("FINAL_VALIDATION_FAILED", "validation/run record did not both report PASS")
    if repairs > exp.max_repairs:
        error("REPAIR_BUDGET_EXCEEDED", f"repair_attempts={repairs} > {exp.max_repairs}")
    if compile_issues > exp.max_compile_issues:
        error("COMPILE_ISSUE_BUDGET_EXCEEDED", f"compile_issue_count={compile_issues} > {exp.max_compile_issues}")
    if roles["reported"] < exp.min_reported_statements:
        error("TOO_FEW_REPORTED", f"reported={roles['reported']} < {exp.min_reported_statements}")
    if roles["evidence_synthesis"] < exp.min_synthesis_statements:
        error("TOO_FEW_SYNTHESIS", f"evidence_synthesis={roles['evidence_synthesis']} < {exp.min_synthesis_statements}")
    if unresolved_count < exp.min_unresolved_connections:
        error("TOO_FEW_UNRESOLVED", f"unresolved_connections={unresolved_count} < {exp.min_unresolved_connections}")
    if route_count < exp.min_mechanism_routes:
        error("TOO_FEW_MECHANISM_ROUTES", f"mechanism_routes={route_count} < {exp.min_mechanism_routes}")

    if exp.require_report_packet_sha_match:
        packet_sha = str(packet.get("packet_sha256", ""))
        report_sha = str(report.get("source_packet_sha256", ""))
        run_sha = str(run.get("packet_sha256", ""))
        if not packet_sha or report_sha != packet_sha or run_sha != packet_sha:
            error(
                "PACKET_SHA_MISMATCH",
                f"packet={packet_sha!r} report={report_sha!r} run={run_sha!r}",
            )

    all_text = "\n".join(str(s.get("text", "")) for s in _statements(report))
    all_text_norm = _norm(all_text)
    for term in exp.required_terms_anywhere:
        if _norm(term) not in all_text_norm:
            error("REQUIRED_TERM_MISSING", f"required term not found: {term!r}")

    by_role: Dict[str, List[str]] = {}
    for s in _statements(report):
        by_role.setdefault(str(s.get("epistemic_role", "")), []).append(str(s.get("text", "")))
    for role, terms in exp.required_terms_by_role.items():
        role_text = _norm("\n".join(by_role.get(role, [])))
        for term in terms:
            if _norm(term) not in role_text:
                error("ROLE_REQUIRED_TERM_MISSING", f"role={role}: required term {term!r} missing")
    for role, terms in exp.forbidden_terms_by_role.items():
        role_text = _norm("\n".join(by_role.get(role, [])))
        for term in terms:
            if _norm(term) in role_text:
                error("ROLE_FORBIDDEN_TERM_PRESENT", f"role={role}: forbidden term {term!r} present")
    for pattern in exp.forbidden_regexes:
        if re.search(pattern, all_text, re.IGNORECASE | re.MULTILINE):
            error("FORBIDDEN_REGEX_MATCH", f"forbidden regex matched: {pattern!r}")

    candidate_nodes, candidate_edges, candidate_hits, candidate_paths = _candidate_ids(packet)
    candidate_propagation_violations = 0
    if exp.require_candidate_verification_propagation:
        for s in _statements(report):
            uses_candidate = (
                bool(set(map(str, s.get("support_node_ids", []) or [])) & candidate_nodes)
                or bool(set(map(str, s.get("support_edge_ids", []) or [])) & candidate_edges)
                or bool(set(map(str, s.get("support_direct_hit_ids", []) or [])) & candidate_hits)
                or bool(set(map(str, s.get("support_path_ids", []) or [])) & candidate_paths)
            )
            if uses_candidate and not bool(s.get("requires_verification")):
                candidate_propagation_violations += 1
                error(
                    "CANDIDATE_VERIFICATION_NOT_PROPAGATED",
                    f"statement {s.get('statement_id')} uses candidate evidence but requires_verification=false",
                )
        for route in report.get("mechanism_routes", []) or []:
            if not isinstance(route, Mapping):
                continue
            uses_candidate = bool(set(map(str, route.get("path_ids", []) or [])) & candidate_paths)
            if uses_candidate and not bool(route.get("requires_verification")):
                candidate_propagation_violations += 1
                error(
                    "CANDIDATE_ROUTE_VERIFICATION_NOT_PROPAGATED",
                    f"route {route.get('route_id')} uses candidate path but requires_verification=false",
                )

    partial_blocked = _partial_absence_blocked_papers(packet)
    partial_absence_violations = 0
    if exp.forbid_partial_paper_absence_claims and partial_blocked:
        for s in _statements(report):
            papers = set(map(str, s.get("paper_ids", []) or []))
            text = str(s.get("text", ""))
            if papers & partial_blocked and _ABSENCE_RE.search(text):
                partial_absence_violations += 1
                error(
                    "PARTIAL_PAPER_ABSENCE_CLAIM",
                    f"statement {s.get('statement_id')} makes absence-like claim for blocked paper(s) "
                    f"{sorted(papers & partial_blocked)}",
                )

    alignment_paths = _alignment_paths(packet)
    alignment_causal_risks = 0
    if exp.forbid_alignment_causal_claims and alignment_paths:
        for s in _statements(report):
            if str(s.get("epistemic_role")) in {"navigation_note", "unresolved"}:
                continue
            used_paths = set(map(str, s.get("support_path_ids", []) or []))
            if not (used_paths & alignment_paths):
                continue
            text = str(s.get("text", ""))
            papers = set(map(str, s.get("paper_ids", []) or []))
            if len(papers) >= 2 and _CAUSAL_RE.search(text) and not _NEGATION_WINDOW_RE.search(text):
                alignment_causal_risks += 1
                error(
                    "ALIGNMENT_CAUSAL_CLAIM",
                    f"statement {s.get('statement_id')} uses alignment path(s) and causal language across papers",
                )

    rm = _repair_metrics(draft, repair)
    if exp.repair_expectation == "none" and repairs != 0:
        error("UNEXPECTED_REPAIR", f"repair_attempts={repairs}, expected none")
    elif exp.repair_expectation == "required" and repairs == 0:
        error("EXPECTED_REPAIR_MISSING", "repair was required by benchmark but repair_attempts=0")

    if repairs > 0 and not rm["repair_artifact_available"]:
        warning("REPAIR_ARTIFACT_MISSING", "repair_attempts>0 but .repair1.draft.json was not found")
    if exp.max_repair_text_changes is not None and rm["repair_text_change_count"] > exp.max_repair_text_changes:
        error(
            "REPAIR_TEXT_CHANGE_BUDGET_EXCEEDED",
            f"repair_text_change_count={rm['repair_text_change_count']} > {exp.max_repair_text_changes}",
        )
    if exp.require_support_only_repair and repairs > 0 and rm["support_only_repair"] is not True:
        error("REPAIR_NOT_SUPPORT_ONLY", "repair changed semantic text/labels/references, not only support bookkeeping")

    motif_local_single_paper = 0
    for motif in report.get("recurring_mechanistic_motifs", []) or []:
        if isinstance(motif, Mapping) and len(set(map(str, motif.get("paper_ids", []) or []))) <= 1:
            motif_local_single_paper += 1
    if motif_local_single_paper:
        warning(
            "LOCAL_MOTIF_IN_RECURRING_SECTION",
            f"{motif_local_single_paper} motif(s) are supported by <=1 paper; track recurrence precision in v2.5.2",
        )

    metrics: Dict[str, Any] = {
        "final_validation_passed": validation_pass,
        "validation_errors": int(validation.get("errors", 0) or 0),
        "validation_warnings": int(validation.get("warnings", 0) or 0),
        "generation_attempts": int(run.get("generation_attempts", 0) or 0),
        "repair_attempts": repairs,
        "compile_issue_count": compile_issues,
        "reported_statement_count": roles["reported"],
        "synthesis_statement_count": roles["evidence_synthesis"],
        "navigation_note_count": roles["navigation_note"],
        "unresolved_statement_count": roles["unresolved"],
        "unresolved_connection_count": unresolved_count,
        "mechanism_route_count": route_count,
        "candidate_node_count": len(candidate_nodes),
        "candidate_edge_count": len(candidate_edges),
        "candidate_hit_count": len(candidate_hits),
        "candidate_path_count": len(candidate_paths),
        "candidate_verification_violations": candidate_propagation_violations,
        "partial_absence_blocked_paper_count": len(partial_blocked),
        "partial_absence_violations": partial_absence_violations,
        "alignment_path_count": len(alignment_paths),
        "alignment_causal_risk_count": alignment_causal_risks,
        "local_single_paper_motif_count": motif_local_single_paper,
        **rm,
    }
    passes = not any(i.severity == "error" for i in issues)
    return BenchmarkCaseResult(
        case_id=case.case_id,
        passes=passes,
        issues=issues,
        metrics=metrics,
    )
