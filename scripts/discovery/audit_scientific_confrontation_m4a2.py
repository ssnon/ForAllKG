"""M4-A2: offline, fail-closed audit of M4-A1 confrontation drafts.

Never certifies scientific truth, physical feasibility, measurement independence,
mechanism exclusion, falsification, or novelty.  Never edits source artifacts.
Runs without LLM, network, or scientific provider calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "m4a2-confrontation-readiness-audit-v1"
SOURCE_SCHEMA = "m4a1-scientific-confrontation-v1"
SOURCE_STATUS = "DRAFT_CONFRONTATIONS_NEED_REVIEW"
CASE_STATUS = "REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED"
MEASUREMENT_ROLES = {"TARGET_OUTCOME", "INDEPENDENT_DISCRIMINATOR", "CONTROL"}
PAIR_KINDS = {"COMPETING_PARTIAL", "MUTUALLY_EXCLUSIVE", "ADJACENT_NOT_EXCLUSIVE"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError("M4A2_INPUT_REJECTED: " + msg)


def nonblank(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def load(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    required(isinstance(obj, dict), "root must be JSON object")
    return obj


def index_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    required(report.get("schema_version") == SOURCE_SCHEMA, "M4A1 schema")
    required(report.get("status") == SOURCE_STATUS, "M4A1 overall status")
    for k in ("external_evidence_validated", "human_or_expert_science_certification",
              "hypothesis_cards_modified", "production_selection_changed", "source_files_mutated"):
        required(report.get(k) is False, f"non-authorizing M4A1 property {k}")
    required(report.get("llm_or_network_calls") == 0, "M4A1 call count")
    source = report.get("source")
    required(isinstance(source, dict) and source.get("source_count", 0) > 0, "source count")
    required(nonblank(report.get("input_case_spec_sha256")), "case spec digest")
    records = report.get("cases")
    required(isinstance(records, list) and records, "empty cases")
    required(report.get("case_count") == len(records), "case_count mismatch")
    counts = Counter(c.get("status") for c in records if isinstance(c, dict))
    required(dict(counts) == report.get("case_status_counts"), "case status counts mismatch")
    seen: set[str] = set()
    refs_seen: set[str] = set()
    for case in records:
        required(isinstance(case, dict), "non-object case")
        cid = case.get("case_id")
        required(nonblank(cid) and cid not in seen, "duplicate/empty case ID")
        seen.add(cid)
        required(case.get("status") == CASE_STATUS, f"{cid}: source is not an unreviewed draft")
        for field in ("confrontation_validated", "experiment_physically_feasible_certified",
                      "falsification_logic_certified", "novelty_or_truth_authority"):
            required(case.get(field) is False, f"{cid}: nonauthoritative field {field}")
        refs = case.get("hypothesis_refs")
        inputs = case.get("input_hypotheses")
        required(isinstance(refs, list) and bool(refs) and len(refs) == len(set(refs)), f"{cid}: references")
        required(isinstance(inputs, list) and {i.get("ref") for i in inputs} == set(refs), f"{cid}: hypothesis mismatch")
        for item in inputs:
            required(nonblank(item.get("hypothesis_id")) and nonblank(item.get("scientific_intent")), f"{cid}: missing hypothesis identity/intent")
            required(nonblank(item.get("arm")), f"{cid}: missing arm")
        refs_seen.update(refs)
        e = case.get("proposed_experiment_UNREVIEWED")
        ms = case.get("proposed_mechanisms_UNREVIEWED")
        required(isinstance(e, dict) and isinstance(ms, list) and len(ms) == 2, f"{cid}: missing draft")
        required({m.get("role") for m in ms} == {"A", "B"}, f"{cid}: roles")
        required(all(nonblank(m.get("explanation")) and nonblank(m.get("predicted_pattern")) for m in ms), f"{cid}: mechanism text")
        required(e.get("pair_relationship") in PAIR_KINDS, f"{cid}: relationship")
        required(all(nonblank(e.get(k)) for k in ("intervention", "decision_rule", "inconclusive_rule")), f"{cid}: incomplete intervention/rules")
        required(isinstance(e.get("matched_controls"), list) and bool(e["matched_controls"]), f"{cid}: controls")
        obs = e.get("observables")
        required(isinstance(obs, list) and len(obs) >= 2, f"{cid}: observables")
        required(any(x.get("role") == "TARGET_OUTCOME" for x in obs), f"{cid}: no target")
        required(any(x.get("role") == "INDEPENDENT_DISCRIMINATOR" for x in obs), f"{cid}: no discriminator")
        names: set[str] = set()
        for x in obs:
            required(isinstance(x, dict), f"{cid}: bad observable")
            name = x.get("name")
            required(nonblank(name) and name.casefold().strip() not in names, f"{cid}: duplicated observable")
            names.add(name.casefold().strip())
            required(x.get("role") in MEASUREMENT_ROLES, f"{cid}: bad measurement role")
            required(x.get("independent_of_sers_outcome") in {"YES", "NO", "UNKNOWN"}, f"{cid}: bad independence enum")
            required(nonblank(x.get("measurement_method")), f"{cid}: missing measurement method")
            if x["independent_of_sers_outcome"] == "YES":
                required(nonblank(x.get("independence_basis")), f"{cid}: unjustified independence assertion")
        falsifier = e.get("falsifier")
        required(isinstance(falsifier, dict), f"{cid}: falsifier")
        required(falsifier.get("scope") in {"MECHANISM_IN_SPECIFIED_CONDITIONS", "WHOLE_RESEARCH_QUESTION"}, f"{cid}: scope")
        required(falsifier.get("target_role") in {"A", "B", "BOTH"}, f"{cid}: target role")
        required(nonblank(falsifier.get("outcome")), f"{cid}: empty falsifier")
        required(falsifier.get("necessity_review") in {"UNREVIEWED", "SUPPORTED", "UNSUPPORTED"}, f"{cid}: necessity status")
        required(falsifier.get("alternatives_review") in {"UNREVIEWED", "CONTROLLED", "UNCONTROLLED"}, f"{cid}: alternatives status")
        required(isinstance(case.get("issues"), list) and isinstance(case.get("existing_experiment_critic_diagnostic_codes"), list), f"{cid}: diagnostics")
    required(len(refs_seen) <= source["source_count"], "more selected refs than source hypotheses")
    return records


def evaluate(case: dict[str, Any]) -> dict[str, Any]:
    cid = case["case_id"]
    e = case["proposed_experiment_UNREVIEWED"]
    f = e["falsifier"]
    obs = e["observables"]
    issues = set(case["issues"])
    checks: list[dict[str, str]] = []

    def check(key: str, title: str, state: str, reason: str) -> None:
        checks.append({"check_id": key, "criterion": title, "state": state, "reason": reason})

    preds = {m["role"]: m["predicted_pattern"].strip().casefold() for m in case["proposed_mechanisms_UNREVIEWED"]}
    if preds["A"] == preds["B"]:
        check("PREDICTION_SEPARATION", "A/B predictions differ in observable outcomes", "FAILED", "Identical predicted patterns")
    else:
        check("PREDICTION_SEPARATION", "Quantified A/B predictions distinguishable above uncertainty", "UNREVIEWED", "Distinct wording does not establish statistical or physical identifiability")
    for idx, o in enumerate(obs, 1):
        if o["role"] != "INDEPENDENT_DISCRIMINATOR":
            continue
        state = "FAILED" if o["independent_of_sers_outcome"] == "NO" else "UNREVIEWED"
        check(f"MEASUREMENT_INDEPENDENCE_{idx}", f"Independent readout: {o['name']}", state,
              "SERS-outcome-derived discriminator is circular" if state == "FAILED" else "Declared method/basis is not independently verified; require calibration and shared-error audit")
    check("CONFOUNDING_AND_MATCHING", "Intervention shifts targeted mechanism without uncontrolled co-changes", "UNREVIEWED", "Specify nuisance variables, manipulation check, and matching uncertainty")
    check("REPEATABILITY_POWER", "Effect size, repeatability, temporal/spatial resolution and measurement sensitivity", "UNREVIEWED", "No externally checked statistical power or sensitivity")
    check("FALSIFIER_NECESSITY", "Observed outcome is necessary consequence under stated scope", "FAILED" if f["necessity_review"] == "UNSUPPORTED" else "UNREVIEWED", "An absence of signal does not refute a mechanism without detectability and exclusion assumptions")
    check("ALTERNATIVE_CAUSES", "Competing explanations, mixed causes, and omitted drivers addressed", "UNREVIEWED", "Case draft does not independently establish alternative control")
    if e["pair_relationship"] == "ADJACENT_NOT_EXCLUSIVE":
        check("JOINT_INCREMENTAL_MODEL", "Joint or nested models tested, rather than winner-takes-all", "UNREVIEWED", "Overlapping mechanisms require conditional incremental predictive testing and a shared-variable audit")
    else:
        check("MIXTURE_OR_INTERACTION", "Both mechanisms can coexist or interact", "UNREVIEWED", "COMPETING_PARTIAL is not mutual exclusivity")
    if f["scope"] == "WHOLE_RESEARCH_QUESTION":
        check("FALSIFIER_SCOPE", "Falsifier cannot invalidate a research question wholesale", "FAILED", "Falsifying a conditional mechanism is not falsifying all possible explanations")
    else:
        check("FALSIFIER_SCOPE", "Falsifier scoped to mechanism in tested conditions", "DRAFT_ONLY", "Scope is well-formed; scientific validity still unreviewed")
    if "INDEPENDENT_EXPERT_REVIEW_ABSENT" in issues:
        check("EXTERNAL_SCIENCE_REVIEW", "Independent scientific review of measurement and logic", "UNREVIEWED", "No expert or independent empirical confirmation")
    failed = any(x["state"] == "FAILED" for x in checks)
    return {
        "case_id": cid, "hypothesis_refs": case["hypothesis_refs"],
        "proposed_mechanisms_UNREVIEWED": case["proposed_mechanisms_UNREVIEWED"],
        "proposed_experiment_UNREVIEWED": e,
        "m4a1_issues": sorted(issues),
        "structural_review_checks": checks,
        "check_state_counts": dict(sorted(Counter(x["state"] for x in checks).items())),
        "readiness": "BLOCKED_DRAFT_INVALID" if failed else "NOT_READY_EMPIRICAL_ADJUDICATION",
        "scientific_truth_or_falsification_certified": False,
        "physical_feasibility_certified": False,
        "independent_measurement_certified": False,
        "requires_independent_review": True,
    }


def reviewer_template(rows: list[dict[str, Any]], report_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "m4a2-independent-science-review-template-v1",
        "source_m4a1_report_sha256": report_sha,
        "status": "UNREVIEWED_NO_AUTHORITY",
        "reviews": [{
            "case_id": row["case_id"],
            "prediction_separation": "UNREVIEWED",
            "target_estimand_and_uncertainty": "",
            "measurement_reviews": [{
                "observable": o["name"], "role": o["role"],
                "independence_verified": "UNREVIEWED", "independent_witness_or_protocol": "",
                "resolution_sensitivity_and_shared_error_notes": ""
            } for o in row["proposed_experiment_UNREVIEWED"]["observables"]],
            "intervention_manipulation_check": "",
            "confounder_control_review": "UNREVIEWED",
            "confounder_evidence_or_plan": "",
            "falsifier_necessary_under_conditions": "UNREVIEWED",
            "falsifier_target_scope_review": "UNREVIEWED",
            "alternative_mechanisms_review": "UNREVIEWED",
            "additive_or_interacting_mechanisms_plan": "",
            "inconclusive_outcome_rules_reviewed": "UNREVIEWED",
            "data_or_method_references": [],
            "reviewer": "", "scientific_notes": "",
        } for row in rows],
        "review_is_not_automatic_scientific_certification": True,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = ["# M4-A2 — Confrontation Readiness Audit (PRIVATE)", "",
             f"Status: `{result['status']}`", "",
             f"Input M4-A1 SHA-256: `{result['input_m4a1_sha256']}`", "",
             "No experiment has been performed and no independent measurement, physical feasibility, falsifier or scientific truth is certified.", "",
             "| Case | Readiness | Failed checks | Unreviewed checks |", "|---|---|---:|---:|"]
    for r in result["cases"]:
        c = r["check_state_counts"]
        lines.append(f"| {r['case_id']} | {r['readiness']} | {c.get('FAILED',0)} | {c.get('UNREVIEWED',0)} |")
    lines.extend(["", "## Required next step", "", "Complete the independent science-review template with verifiable measurement methods, confound controls, discriminating estimands, and mechanism-scoped falsifier logic. Unknown or non-exclusive causes must remain inconclusive. No automatic promotion from this audit.", ""])
    return "\n".join(lines)


def run(*, input_report: Path, expected_sha: str, output_dir: Path) -> dict[str, Any]:
    source = input_report.expanduser().resolve()
    out = output_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    required(source.is_file(), "M4A1 report not found")
    required(nonblank(expected_sha) and len(expected_sha) == 64, "explicit M4A1 SHA-256 required")
    required(sha256(source) == expected_sha.lower(), "M4A1 SHA mismatch")
    required(not out.exists(), "output directory already exists")
    required(repo != out and repo not in out.parents, "private output cannot be written inside repository")
    required(source != out and out not in source.parents, "source cannot be within output")
    inputs = index_cases(load(source))
    cases = [evaluate(x) for x in inputs]
    status = "STRUCTURAL_FAILURE_NO_SCIENCE_CERTIFICATION" if any(x["readiness"] == "BLOCKED_DRAFT_INVALID" for x in cases) else "REVIEW_REQUIRED_NO_SCIENCE_CERTIFICATION"
    result = {
        "schema_version": SCHEMA, "status": status,
        "input_m4a1_sha256": sha256(source), "source_case_count": len(inputs),
        "case_count": len(cases), "cases": cases,
        "fresh_llm_or_network_calls": 0, "source_files_mutated": False,
        "scientific_truth_or_novelty_certified": False, "physical_feasibility_certified": False,
        "deletion_or_production_authority": False,
        "reviewer_template_completed": False,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "M4A2_READINESS_PRIVATE.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (out / "M4A2_EXPERT_REVIEW_TEMPLATE_PRIVATE.json").write_text(json.dumps(reviewer_template(cases, expected_sha), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (out / "M4A2_REPORT_PRIVATE.md").write_text(markdown(result), encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m4a1-report", type=Path, required=True)
    p.add_argument("--expected-m4a1-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    try:
        result = run(input_report=a.m4a1_report, expected_sha=a.expected_m4a1_sha256, output_dir=a.output_dir)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        p.exit(2, f"M4A2 STOPPED (no source modified): {e}\n")
    print("M4A2:", result["status"])
    print("cases:", result["case_count"])
    print("science certified: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
