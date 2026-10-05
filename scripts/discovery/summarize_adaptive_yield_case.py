from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.adaptive_yield_evaluation import AdaptiveYieldCaseAudit


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def unique_candidates(audit: AdaptiveYieldCaseAudit):
    unique = {}
    for row in audit.candidates:
        unique.setdefault(row.scientific_fingerprint, row)
    return list(unique.items())


def validity_index(audit: AdaptiveYieldCaseAudit, report: dict[str, Any]):
    alias_to_fp = {
        f"V{i:02d}": fp
        for i, (fp, _row) in enumerate(unique_candidates(audit), start=1)
    }
    out = {}
    for row in report.get("candidates", []) or []:
        if not isinstance(row, dict):
            continue
        alias = str(row.get("candidate_alias") or "")
        fp = alias_to_fp.get(alias)
        if not fp:
            continue
        dims = {
            str(d.get("dimension") or ""): {
                "verdict": str(d.get("verdict") or ""),
                "rationale": str(d.get("rationale") or ""),
            }
            for d in row.get("dimensions", []) or []
            if isinstance(d, dict)
        }
        out[fp] = {
            "candidate_alias": alias,
            "summary": str(row.get("summary") or ""),
            "dimensions": dims,
            "fail_dimensions": sorted(k for k,v in dims.items() if v["verdict"] == "FAIL"),
            "warn_dimensions": sorted(k for k,v in dims.items() if v["verdict"] == "WARN"),
            "unknown_dimensions": sorted(k for k,v in dims.items() if v["verdict"] == "UNKNOWN"),
        }
    return out


def novelty_index(report: dict[str, Any]):
    out = {}
    for row in report.get("candidates", []) or []:
        if not isinstance(row, dict):
            continue
        fp = str(row.get("scientific_fingerprint") or "")
        if fp:
            out[fp] = {
                "status": str(row.get("status") or "UNKNOWN"),
                "report": str(row.get("report") or ""),
                "fresh_query_decomposition": bool(row.get("fresh_query_decomposition")),
                "prior_art_memory_used": bool(row.get("prior_art_memory_used")),
                "independent_evidence_review": bool(row.get("independent_evidence_review")),
            }
    return out


def retraversal_lineage(run_dir: Path):
    path = run_dir / "scientific_portfolio_shadow" / "adaptive_graph_retraversal_shadow" / "context_retraversal.lineage.json"
    if not path.is_file():
        return {}
    payload = load(path)
    out = {}
    for event in payload.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        hid = str(event.get("generated_hypothesis_id") or "")
        if hid:
            out[hid] = {
                "retraversal_index": event.get("retraversal_index"),
                "source_hypothesis_id": event.get("source_hypothesis_id"),
                "source_context_id": event.get("source_context_id"),
                "output_context_id": event.get("output_context_id"),
                "selected_path_ids": event.get("selected_path_ids", []) or [],
                "structurally_new_eligible_premise_ids": event.get("structurally_new_eligible_premise_ids", []) or [],
                "generation_decision": event.get("generation_decision"),
            }
    return out


def classify_candidate(novelty_status: str, fail_dimensions: list[str], warn_dimensions: list[str]) -> str:
    fail = set(fail_dimensions)
    if fail & {
        "COMPARISON_CONTEXT_COMPATIBILITY",
        "CONTROL_VARIABLE_ADEQUACY",
        "MEASUREMENT_SCALE",
        "VARIANCE_SEMANTICS",
        "VARIABLE_IDENTITY",
    }:
        return "SCIENTIFIC_IDENTIFICATION_RISK"
    if novelty_status in {"WELL_ESTABLISHED", "LITERATURE_SUPPORTED_EXTENSION"}:
        return "PRIOR_ART_SATURATED_OR_EXTENSION"
    if novelty_status in {"NEW_COMBINATION_OF_KNOWN_EFFECTS", "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"}:
        return "RELATIONAL_GAP_OR_NEW_COMBINATION"
    if novelty_status == "PLAUSIBLY_NOVEL":
        return "NOVELTY_SURVIVOR_REQUIRES_VALIDITY_REVIEW"
    if novelty_status == "INSUFFICIENT_SEARCH_EVIDENCE":
        return "NOVELTY_UNRESOLVED"
    return "UNCLASSIFIED"


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Adaptive Scientific Yield Case Synthesis v1", "",
        f"- Case: `{payload['case_id']}`",
        f"- Baseline: `{payload['baseline_commit']}`",
        f"- Domain: `{payload['domain_profile_id']}`",
        f"- Question: {payload['question']}", "",
        "## Stage summary", "",
        "| Condition | Effective | Marginal | Search attempts | Unresolved | Retraversals | New contexts | New structural premises |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["stages"]:
        lines.append(
            f"| {row['condition']} | {row['cumulative_effective_count']} | {row['marginal_effective_count']} | "
            f"{row.get('search_attempt_unique_count', 0)} | {row['unresolved_count']} | {row['graph_retraversal_count']} | "
            f"{row['new_context_count']} | {row['structurally_new_premise_count']} |"
        )
    lines += ["", "## Candidate synthesis", "",
              "| Alias | Source | Title | Fresh-search novelty | FAIL | WARN | Interpretation |",
              "|---|---|---|---|---|---|---|"]
    for row in payload["candidates"]:
        fail = ", ".join(row["validity"]["fail_dimensions"]) or "—"
        warn = ", ".join(row["validity"]["warn_dimensions"]) or "—"
        title = row["title"].replace("|", "\\|")
        lines.append(
            f"| {row['validity']['candidate_alias'] or '—'} | {row['condition']} | {title} | "
            f"{row['fresh_search_novelty']['status']} | {fail} | {warn} | {row['synthesis_class']} |"
        )
    lines += ["", "## Evaluation boundaries", "",
              "- Evaluation-only; no production authority.",
              "- Candidate count is not a scientific quality score.",
              "- Fresh-search novelty stress is not independent scientific ground truth.",
              "- Relation-validity findings have no rejection authority.",
              "- External prior art is not positive premise evidence.",
              "- Production selection and Stage 8 inputs are unchanged.", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case-audit", required=True, type=Path)
    p.add_argument("--relation-validity", required=True, type=Path)
    p.add_argument("--novelty-stress", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-md", required=True, type=Path)
    args = p.parse_args()

    audit = AdaptiveYieldCaseAudit.model_validate_json(args.case_audit.read_text(encoding="utf-8"))
    validity = validity_index(audit, load(args.relation_validity))
    novelty = novelty_index(load(args.novelty_stress))
    lineage = retraversal_lineage(Path(audit.run_dir))

    rows = []
    for fp, candidate in unique_candidates(audit):
        valid = validity.get(fp, {
            "candidate_alias": None, "summary": "", "dimensions": {},
            "fail_dimensions": [], "warn_dimensions": [], "unknown_dimensions": [],
        })
        novel = novelty.get(fp, {
            "status": "NOT_EVALUATED", "report": "", "fresh_query_decomposition": False,
            "prior_art_memory_used": False, "independent_evidence_review": False,
        })
        rows.append({
            "scientific_fingerprint": fp,
            "record_id": candidate.record_id,
            "record_kind": getattr(candidate, "record_kind", "UNKNOWN"),
            "condition": candidate.condition,
            "hypothesis_id": candidate.hypothesis_id,
            "epoch_index": candidate.epoch_index,
            "title": candidate.title,
            "hypothesis_statement": candidate.hypothesis_statement,
            "premise_statement_ids": candidate.premise_statement_ids,
            "marginal_from_previous_condition": candidate.marginal_from_previous_condition,
            "fresh_search_novelty": novel,
            "validity": valid,
            "graph_lineage": lineage.get(candidate.hypothesis_id, {}),
            "synthesis_class": classify_candidate(
                novel.get("status", "UNKNOWN"),
                valid.get("fail_dimensions", []),
                valid.get("warn_dimensions", []),
            ),
        })

    payload = {
        "schema_version": "adaptive-scientific-yield-case-synthesis-v1",
        "case_id": audit.case_id,
        "baseline_commit": audit.baseline_commit,
        "run_dir": audit.run_dir,
        "domain_profile_id": audit.domain_profile_id,
        "question": audit.question,
        "stages": [x.model_dump(mode="json") for x in audit.stages],
        "candidate_count": len(rows),
        "candidates": rows,
        "fresh_search_novelty_status_counts": dict(Counter(x["fresh_search_novelty"]["status"] for x in rows)),
        "validity_fail_dimension_counts": dict(Counter(dim for x in rows for dim in x["validity"]["fail_dimensions"])),
        "validity_warn_dimension_counts": dict(Counter(dim for x in rows for dim in x["validity"]["warn_dimensions"])),
        "synthesis_class_counts": dict(Counter(x["synthesis_class"] for x in rows)),
        "evaluation_only": True,
        "candidate_count_is_quality_signal": False,
        "fresh_search_novelty_is_ground_truth": False,
        "relation_validity_has_rejection_authority": False,
        "external_prior_art_as_positive_premise": False,
        "production_selection_changed": False,
        "stage8_input_changed": False,
    }
    write(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(payload), encoding="utf-8")

    print("Adaptive Scientific Yield case synthesis complete")
    print("case:", payload["case_id"])
    print("candidates:", payload["candidate_count"])
    print("fresh-search novelty:", payload["fresh_search_novelty_status_counts"])
    print("validity FAIL dimensions:", payload["validity_fail_dimension_counts"])
    print("synthesis classes:", payload["synthesis_class_counts"])
    print("production selection changed: false")
    print("json:", args.output_json)
    print("markdown:", args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
