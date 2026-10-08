"""M2-E: deterministic frozen Q-A capability comparison; no network or LLM.

Important: This is an evidence inventory and blinded-review handoff, NOT
an automated scientific quality or novelty evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ARM_PATHS = {
    "V31": {
        "portfolio": "scientific_portfolio_shadow/sis_v3_2_e2e/arms/v3_1/final.materialized.portfolio.json",
        "summary": "scientific_portfolio_shadow/sis_v3_2_e2e/arms/v3_1/arm.summary.json",
        "freeze_key": "v3_1_final",
    },
    "V34": {
        "portfolio": "scientific_portfolio_shadow/sis_v3_4_e2e/final.materialized.portfolio.json",
        "summary": "scientific_portfolio_shadow/sis_v3_4_e2e/arm.summary.json",
        "freeze_key": "v3_4_final",
    },
}


def _load(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _text(value: object) -> str:
    return str(value or "").strip()


def _observation(value: Any) -> str:
    if isinstance(value, dict):
        pieces = [
            value.get("observable"), value.get("expected_direction"),
            value.get("rationale"),
        ]
        return "; ".join(_text(p) for p in pieces if _text(p))
    return _text(value)


def _falsification(value: Any) -> str:
    if isinstance(value, dict):
        pieces = [value.get("falsifying_outcome"), value.get("rationale")]
        return "; ".join(_text(p) for p in pieces if _text(p))
    return _text(value)


def _unique_field(h: dict[str, Any], key: str) -> list[str]:
    arr = h.get(key) or []
    if not isinstance(arr, list):
        raise ValueError(f"{key} must be an array")
    return list(dict.fromkeys(_text(v) for v in arr if _text(v)))


def _inventory(label: str, p: dict[str, Any]) -> list[dict[str, Any]]:
    hs = p.get("hypotheses")
    if not isinstance(hs, list):
        raise ValueError(f"{label}: missing hypotheses list")
    ids = [_text(h.get("hypothesis_id")) for h in hs]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{label}: missing or duplicate hypothesis IDs")
    rows: list[dict[str, Any]] = []
    for h in hs:
        if not isinstance(h, dict):
            raise ValueError(f"{label}: invalid hypothesis object")
        pred = h.get("predicted_observations") or []
        fals = h.get("falsification_criteria") or []
        if not isinstance(pred, list) or not isinstance(fals, list):
            raise ValueError(f"{label}: malformed prediction/falsifier list")
        ep = h.get("evidence_profile") or {}
        if not isinstance(ep, dict):
            ep = {}
        rows.append({
            "arm": label,
            "hypothesis_id": h["hypothesis_id"],
            "title": _text(h.get("title")),
            "statement": _text(h.get("hypothesis_statement")),
            "bridge": _text(h.get("inferential_bridge")),
            "assumptions": list(h.get("assumptions") or []),
            "predictions": [_observation(v) for v in pred],
            "falsifiers": [_falsification(v) for v in fals],
            "premise_statement_ids": _unique_field(h, "premise_statement_ids"),
            "source_paper_ids": _unique_field(h, "source_paper_ids"),
            "gap_statement_ids": _unique_field(h, "gap_statement_ids"),
            "evidence_profile": ep,
            "novelty_status": _text(h.get("novelty_status")),
            "status": _text(h.get("status")),
            "candidate_dependency": _text(h.get("candidate_dependency")),
            "cross_paper_synthesis": bool(h.get("cross_paper_synthesis", False)),
            "hypothesis_type": _text(h.get("hypothesis_type")),
        })
    return rows


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "hypothesis_count": len(rows),
        "with_nonempty_prediction": sum(bool(r["predictions"]) for r in rows),
        "with_nonempty_falsifier": sum(bool(r["falsifiers"]) for r in rows),
        "with_nonempty_inferential_bridge": sum(bool(r["bridge"]) for r in rows),
        "distinct_premise_statement_ids": len(set(v for r in rows for v in r["premise_statement_ids"])),
        "distinct_source_paper_ids": len(set(v for r in rows for v in r["source_paper_ids"])),
        "distinct_gap_statement_ids": len(set(v for r in rows for v in r["gap_statement_ids"])),
        "hypothesis_type_counts": {
            t: sum(r["hypothesis_type"] == t for r in rows)
            for t in sorted(set(r["hypothesis_type"] for r in rows))
        },
        "novelty_status_counts": {
            t: sum(r["novelty_status"] == t for r in rows)
            for t in sorted(set(r["novelty_status"] for r in rows))
        },
    }


def _validate_m2c(summary: dict[str, Any], manifest: dict[str, Any], m2c_p0: Path) -> dict[str, Any]:
    if not summary.get("old_p0_exact_parity_checked"):
        raise ValueError("M2-C old P0 exact parity was not checked")
    if not summary.get("old_selection_parity_checked"):
        raise ValueError("M2-C old selection parity was not checked")
    if summary.get("early_hypothesis_materialization_executed") is not False:
        raise ValueError("M2-C early materialization must not have executed")
    if int(summary.get("early_materialization_llm_calls", -1)) != 0:
        raise ValueError("M2-C early materialization LLM calls must equal zero")
    if int(summary.get("selected_research_idea_count", -1)) != 8:
        raise ValueError("M2-C expected frozen Q-A cohort of eight P0 ideas")
    expected_seed = manifest.get("comparison", {}).get("v3_1", {}).get("source_p0_seed_sha256")
    if expected_seed and summary.get("seed_sha256") != expected_seed:
        raise ValueError("M2-C P0 seed SHA does not match M1 manifest")
    if not m2c_p0.is_file():
        raise FileNotFoundError(f"M2-C direct P0 missing: {m2c_p0}")
    execution = _load(m2c_p0)
    expected_report_id = manifest.get("comparison", {}).get("v3_4", {}).get("source_p0_report_id")
    if expected_report_id and execution.get("report_id") != expected_report_id:
        raise ValueError("M2-C P0 report id differs from frozen Q-A v3.4 input")
    if summary.get("p0_execution_report_id") != execution.get("report_id"):
        raise ValueError("M2-C summary P0 report ID mismatch")
    return {"passed": True, "p0_execution_report_id": execution["report_id"], "p0_sha256": _sha(m2c_p0)}


def _validation(case_dir: Path, manifest: dict[str, Any], m2c_summary: dict[str, Any], m2c_p0: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    expected_artifacts = {v["key"]: v for v in manifest.get("artifacts", []) if v.get("present")}
    raw: dict[str, Any] = {}
    inv: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    context = None
    for arm, paths in ARM_PATHS.items():
        p = case_dir / paths["portfolio"]
        s = case_dir / paths["summary"]
        if not p.is_file() or not s.is_file():
            raise FileNotFoundError(f"Missing frozen {arm} portfolio/summary: {p}, {s}")
        expected = expected_artifacts.get(paths["freeze_key"])
        if not expected:
            raise ValueError(f"M1 manifest does not have {paths['freeze_key']}")
        hash_now = _sha(p)
        if hash_now != expected["source_sha256"]:
            raise ValueError(f"M1 frozen SHA mismatch for {arm}: {p}")
        portfolio, summary = _load(p), _load(s)
        if portfolio.get("schema_version") != "hypothesis-portfolio-v1":
            raise ValueError(f"{arm}: unsupported hypothesis-portfolio schema")
        ctx = (portfolio.get("source_context_id"), portfolio.get("source_context_sha256"))
        if not all(ctx):
            raise ValueError(f"{arm}: context provenance missing")
        if context is None:
            context = ctx
        elif ctx != context:
            raise ValueError("Cannot compare portfolios with different source contexts")
        rows = _inventory(arm, portfolio)
        declared = summary.get("final_hypothesis_count")
        if declared is not None and int(declared) != len(rows):
            raise ValueError(f"{arm} arm summary final count differs from portfolio")
        raw[arm] = summary
        inv[arm] = rows
        meta[arm] = {"sha256": hash_now, "path": str(p), "metrics": _metrics(rows),
            "generation_llm_calls": summary.get("continuation_generation_llm_calls"),
            "realization_llm_calls": summary.get("continuation_realization_llm_calls")}
    p0 = _validate_m2c(m2c_summary, manifest, m2c_p0)
    result = {
        "status": "OFFLINE_EVIDENCE_INVENTORY_COMPLETE",
        "context": {"id": context[0], "sha256": context[1]},
        "m2c_exact_p0_parity": p0,
        "arms": meta,
        "limitations": {
            "m2d_cache_integration_or_savings_measured": False,
            "v3_4_common_external_verifier_executed_here": False,
            "equal_compute_comparison": False,
            "scientific_novelty_certified": False,
            "semantic_quality_or_causal_effect_proven": False,
            "cost_counts_exclude_prior_art_family_and_other_llm_calls": True,
        },
    }
    return result, inv, raw


def _safe_line(text: Any) -> str:
    return re.sub(r"\s+", " ", _text(text)).replace("|", "\\|")


def _review_packet(rows: list[dict[str, Any]], blind_key: str) -> tuple[str, list[dict[str, str]]]:
    # Deterministic pseudonyms; all real arm labels are held in a separate key file.
    ranked = sorted(rows, key=lambda row: hashlib.sha256((blind_key + "|" + str(row["hypothesis_id"]) + "|" + row["arm"]).encode()).hexdigest())
    lines = ["# M2-E — Blinded scientific review packet", "", "Contains hypotheses from two pipeline configurations in mixed order. Do not infer arm from text style.", "", "**Scientific novelty must be checked against the literature separately; this packet does not certify it.**", ""]
    mapping: list[dict[str, str]] = []
    for i, row in enumerate(ranked, 1):
        code = f"H{i:02d}"
        mapping.append({"blind_id": code, "arm": row["arm"], "hypothesis_id": row["hypothesis_id"]})
        lines += [f"## {code} — {_safe_line(row['title'])}", "", "**Claim:** " + _safe_line(row["statement"]), "", "**Inferential bridge:** " + _safe_line(row["bridge"]), "", "**Predictions:**"]
        lines += [f"- {_safe_line(t)}" for t in row["predictions"]] or ["- Not specified"]
        lines += ["", "**Falsifiers:**"]
        lines += [f"- {_safe_line(t)}" for t in row["falsifiers"]] or ["- Not specified"]
        lines += ["", "**Assumptions:**"]
        lines += [f"- {_safe_line(t)}" for t in row["assumptions"]] or ["- Not specified"]
        lines += ["", "**Provenance (IDs only; not verification):**"]
        lines += [f"- Premises: {', '.join(row['premise_statement_ids']) or 'None'}", f"- Paper IDs: {', '.join(row['source_paper_ids']) or 'None'}", f"- Evidence gaps: {', '.join(row['gap_statement_ids']) or 'None'}", "", "---", ""]
    return "\n".join(lines), mapping


def _write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})


def analyze(case_dir: Path, manifest_path: Path, m2c_summary_path: Path, output_dir: Path, blind_key: str) -> dict[str, Any]:
    # Output must not overwrite an existing evidence package or source folder.
    case_dir, output_dir = case_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing M2-E output: {output_dir}")
    if output_dir == case_dir or case_dir in output_dir.parents:
        raise ValueError("M2-E output must be outside frozen case source")
    manifest, m2c = _load(manifest_path), _load(m2c_summary_path)
    m2c_p0 = Path(_text(m2c.get("p0_execution"))).expanduser()
    report, inventories, summaries = _validation(case_dir, manifest, m2c, m2c_p0)
    all_rows = inventories["V31"] + inventories["V34"]
    packet, key = _review_packet(all_rows, blind_key)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write(output_dir / "m2e.metrics.json", report)
    columns = ["arm", "hypothesis_id", "title", "statement", "bridge", "novelty_status", "status", "hypothesis_type", "candidate_dependency", "cross_paper_synthesis"]
    flat = [{**r, "prediction_count": len(r["predictions"]), "falsifier_count": len(r["falsifiers"]), "premise_count": len(r["premise_statement_ids"]), "source_paper_count": len(r["source_paper_ids"]), "gap_count": len(r["gap_statement_ids"])} for r in all_rows]
    _write_csv(output_dir / "hypothesis_inventory.csv", flat, columns + ["prediction_count", "falsifier_count", "premise_count", "source_paper_count", "gap_count"])
    (output_dir / "review_packet_blinded.md").write_text(packet, encoding="utf-8")
    _write(output_dir / "review_key_KEEP_PRIVATE.json", {"warning": "Do not show to blinded reviewers", "mapping": key})
    _write_csv(output_dir / "review_scores_template.csv", [{"blind_id": row["blind_id"]} for row in key], ["blind_id", "reviewer", "mechanistic_specificity_0_4", "discriminating_test_0_4", "causal_identifiability_0_4", "quantitative_commitment_0_4", "experimental_feasibility_0_4", "literature_novelty_requires_external_review", "notes"])
    a, b = report["arms"]["V31"], report["arms"]["V34"]
    lines = ["# M2-E — Minimal AI Scientist Frozen Capability Gate", "", "**Status:** OFFLINE_EVIDENCE_INVENTORY_COMPLETE (not scientific quality certification)", "", "## Input integrity", "", "- M1 frozen final portfolio SHA checked for both arms", "- Both arms share the same source context and have valid, unique hypothesis identifiers", "- M2-C exact P0 parity and zero early materialization verified", "", "## Structural inventory", "", "| Observation | v3.1 | v3.4 |", "|---|---:|---:|", f"| Final hypotheses | {a['metrics']['hypothesis_count']} | {b['metrics']['hypothesis_count']} |", f"| With predictions | {a['metrics']['with_nonempty_prediction']} | {b['metrics']['with_nonempty_prediction']} |", f"| With falsifier | {a['metrics']['with_nonempty_falsifier']} | {b['metrics']['with_nonempty_falsifier']} |", f"| With inferential bridge | {a['metrics']['with_nonempty_inferential_bridge']} | {b['metrics']['with_nonempty_inferential_bridge']} |", f"| Distinct premise statement IDs | {a['metrics']['distinct_premise_statement_ids']} | {b['metrics']['distinct_premise_statement_ids']} |", f"| Distinct paper IDs | {a['metrics']['distinct_source_paper_ids']} | {b['metrics']['distinct_source_paper_ids']} |", f"| Continuation generation calls (only) | {a['generation_llm_calls']} | {b['generation_llm_calls']} |", f"| Continuation realization calls (only) | {a['realization_llm_calls']} | {b['realization_llm_calls']} |", "", "## Scientific interpretation guardrails", "", "- Counts, references and formatted prompts are not novelty, reproducibility or scientific quality scores.", "- v3.4 did not undergo the same common downstream novelty/feasibility verifier in this audit.", "- M2-D cache savings and its equivalence to the baseline are unmeasured: cache unit tests alone do not establish either.", "- This is NOT an equal-compute comparison, and the continuation counts exclude literature, family and other calls.", "- Review `review_packet_blinded.md` without opening `review_key_KEEP_PRIVATE.json`; complete `review_scores_template.csv` and evaluate literature novelty separately.", "- Do not delete Stage7, HO, search semantics or verifier modules based on these descriptive statistics.", "", "## Next decision", "", "If reviewers find v3.4 no worse than v3.1 on important dimensions, compare opt-in M2-D feedback against unmodified v3.4 using identical frozen inputs and fixed budget. Only then consider production-path deprecation.", ""]
    (output_dir / "M2E_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--m1-freeze-manifest", type=Path, required=True)
    p.add_argument("--m2c-summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--blind-key", default="m2e-frozen-qa-v1")
    a = p.parse_args()
    result = analyze(a.case_dir, a.m1_freeze_manifest, a.m2c_summary, a.output_dir, a.blind_key)
    print("M2-E offline audit ready:", a.output_dir)
    print("M2-C exact P0 parity:", result["m2c_exact_p0_parity"]["passed"])
    print("v3.1 final hypotheses:", result["arms"]["V31"]["metrics"]["hypothesis_count"])
    print("v3.4 final hypotheses:", result["arms"]["V34"]["metrics"]["hypothesis_count"])
    print("LLM/network calls: 0")
    print("Scientific quality/novelty not certified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
