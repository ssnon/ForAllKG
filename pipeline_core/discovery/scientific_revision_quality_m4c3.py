"""M4-C3 read-only, conservative scientific revision quality audit.

This gate tests provenance preservation and *presence* of a concrete scientific
revision, not scientific truth, scientific novelty or experiment feasibility.
Text/template differences cannot certify an improvement.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Mapping

SCENARIOS = frozenset({
    "A_ONLY_IN_SCOPE", "B_ONLY_IN_SCOPE", "BOTH_COMPATIBLE",
    "NEITHER_COMPATIBLE", "NOT_IDENTIFIABLE",
})
FALSE_FLAGS = (
    "empirical_scientific_learning_demonstrated", "hypothesis_cards_modified",
    "novelty_certification_authority", "original_research_ideas_modified",
    "production_selection_changed", "real_measurements_consumed",
    "research_idea_nodes_created", "scientific_truth_or_falsification_authority",
)


def check(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("M4C3_INTEGRITY_FAILURE: " + why)


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def kernel_payload(value: Any, label: str) -> dict[str, Any]:
    check(isinstance(value, dict), label + " must be object")
    required = ("canonical_intent", "core_scientific_commitments", "scope_commitments",
                "contrastive_commitments", "question_commitment")
    check(all(k in value for k in required), label + " missing fields")
    check(isinstance(value["core_scientific_commitments"], list) and
          all(isinstance(x, str) and x.strip() for x in value["core_scientific_commitments"]),
          label + " invalid core")
    return value


def evaluate_quality(*, c2: dict[str, Any], a1: dict[str, Any],
                     trajectories: dict[str, Any], sha_a1: str,
                     sha_traces: str, sha_c2: str) -> dict[str, Any]:
    check(c2.get("schema_version") == "m4c2-synthetic-kernel-revision-replay-v1", "C2 schema")
    check(c2.get("status") == "SYNTHETIC_KERNEL_DRAFTS_ONLY_NOT_EMPIRICAL_LEARNING", "C2 status")
    check(c2.get("input_sha256", {}).get("m4a1") == sha_a1, "C2/A1 SHA mismatch")
    check(c2.get("input_sha256", {}).get("m3c1_trajectories") == sha_traces, "C2/M3C1 SHA mismatch")
    check(c2.get("new_llm_or_network_calls") == 0, "unexpected model calls")
    report_without_id = {k: v for k, v in c2.items() if k != "report_id"}
    check(c2.get("report_id") == "m4c2_kernel_revision_replay:" + sha(report_without_id)[:20], "C2 report checksum mismatch")
    for field in FALSE_FLAGS:
        check(c2.get(field) is False, "C2 authority flag: " + field)
    check(a1.get("schema_version") == "m4a1-scientific-confrontation-v1", "A1 schema")
    check(a1.get("status") == "DRAFT_CONFRONTATIONS_NEED_REVIEW", "A1 status")
    check(trajectories.get("status") == "SHA_VERIFIED_SCIENTIFIC_DELTAS_READY_FOR_HUMAN_REVIEW", "M3C1 status")
    check(trajectories.get("original_data_mutated") is False, "M3C1 original mutated")
    check(trajectories.get("authoritative_science_judgment") is False, "M3C1 authoritative")
    case_list = a1.get("cases")
    check(isinstance(case_list, list) and bool(case_list), "A1 cases invalid")
    cases = {c.get("case_id"): c for c in case_list if isinstance(c, dict)}
    check(len(cases) == len(case_list) == a1.get("case_count"), "A1 duplicate/invalid cases")
    trace_list = trajectories.get("rows")
    check(isinstance(trace_list, list), "M3C1 rows invalid")
    traces = {r.get("blind_id"): r for r in trace_list if isinstance(r, dict)}
    check(len(traces) == len(trace_list), "M3C1 duplicate blind id")
    rows = c2.get("rows")
    check(isinstance(rows, list) and len(rows) == c2.get("scenario_count"), "C2 row count")
    check(c2.get("source_case_count") == len(cases), "case count mismatch")
    seen = set()
    out = []
    question_cases = defaultdict(set)
    for row in rows:
        check(isinstance(row, dict), "C2 row not object")
        cid, scenario = row.get("case_id"), row.get("synthetic_scenario")
        check(cid in cases and scenario in SCENARIOS, "case/scenario unsupported")
        key = (cid, scenario)
        check(key not in seen, "duplicate case/scenario")
        seen.add(key)
        case = cases[cid]
        hyps = case.get("input_hypotheses")
        check(isinstance(hyps, list) and bool(hyps), "A1 case hypotheses missing")
        ids = [h["terminal_idea_id"] for h in hyps]
        check(row.get("source_hypothesis_refs") == case["hypothesis_refs"], "source refs mismatch")
        check(row.get("original_terminal_idea_ids_PRESERVED") == ids, "original parent ID mismatch")
        check(row.get("seed_parent_id") == ids[0], "seed parent mismatch")
        check(row.get("comparison_parent_ids_PRESERVED_SEPARATELY") == ids[1:], "comparison parent mismatch")
        check(row.get("multi_parent_composition_NOT_CLAIMED") is True, "implicit program merger")
        check(row.get("pair_relationship") == case["proposed_experiment_UNREVIEWED"]["pair_relationship"], "relationship mismatch")
        check(row.get("scenario_kind") == "SYNTHETIC_POLICY_FIXTURE_NOT_OBSERVED", "scenario not synthetic")
        check(row.get("requires_human_science_review") is True, "missing human-review requirement")
        for flag in ("research_idea_node_created", "original_research_ideas_modified", "scientific_claim_adjudicated", "real_measurement_consumed"):
            check(row.get(flag) is False, "row improperly authoritative: " + flag)
        parent_kernels = {}
        for h in hyps:
            trace = traces.get(h["ref"])
            check(isinstance(trace, dict) and trace.get("hypothesis_id") == h["hypothesis_id"] and
                  trace.get("terminal_idea_id") == h["terminal_idea_id"] and
                  trace.get("lineage_validation") == "PASS_SOURCE_HASH_AND_KERNEL_TRAJECTORY", "broken M3C1 lineage")
            steps = trace.get("steps_earliest_to_latest", [])
            check(isinstance(steps, list) and bool(steps) and steps[-1].get("idea_id") == h["terminal_idea_id"], "M3C1 terminal mismatch")
            parent = kernel_payload(steps[-1].get("kernel"), "M3C1 parent")
            parent_kernels[h["terminal_idea_id"]] = parent
            check(row.get("parent_kernel_sha256_by_id", {}).get(h["terminal_idea_id"]) == sha(parent), "parent kernel hash mismatch")
            check(row.get("original_parent_scientific_core_by_id", {}).get(h["terminal_idea_id"]) == parent["core_scientific_commitments"], "parent core mismatch")
        if row["pair_relationship"] == "ADJACENT_NOT_EXCLUSIVE":
            check(len(ids) >= 2, "nonexclusive pair needs separate parents")
            check("NONEXCLUSIVE_MECHANISMS_NO_AUTOMATIC_EXCLUSION" in row.get("scientific_cautions", []), "nonexclusive caution missing")
        draft = row.get("draft_scientific_kernel_NOT_CREATED")
        if scenario == "NOT_IDENTIFIABLE":
            check(draft is None and row.get("draft_kernel_sha256") is None and
                  row.get("draft_id_NOT_RESEARCH_IDEA_ID") is None and
                  row.get("revision_disposition") == "NO_REVISION_UNIDENTIFIABLE", "unidentifiable scenario revised")
            out.append({"case_id": cid, "scenario": scenario, "draft_id": None,
                        "structural_integrity": "PASS", "scientific_delta_state": "NO_REVISION_REQUIRED",
                        "science_review_required": False, "failure_reasons": [],
                        "independent_science_confirmed": False})
            continue
        check(row.get("revision_disposition") == "DRAFT_KERNEL_FOR_EXPERT_REVIEW_ONLY", "unexpected draft disposition")
        check(isinstance(row.get("draft_id_NOT_RESEARCH_IDEA_ID"), str) and
              row["draft_id_NOT_RESEARCH_IDEA_ID"].startswith("m4c2_kernel_draft:"), "invalid draft identity")
        d = kernel_payload(draft, "draft")
        expected_id = "m4c2_kernel_draft:" + sha([sha_traces, c2["input_sha256"]["m4c0"], cid, scenario, ids, d])[:24]
        check(row["draft_id_NOT_RESEARCH_IDEA_ID"] == expected_id, "draft ID mismatch")
        check(d.get("schema_version") == "research-idea-kernel-v1", "draft kernel schema")
        check(sha(d) == row.get("draft_kernel_sha256"), "draft hash mismatch")
        seed = parent_kernels[ids[0]]
        check(d["core_scientific_commitments"] == seed["core_scientific_commitments"], "seed core modified")
        check(d["canonical_intent"].startswith(seed["canonical_intent"]), "original scientific intent removed")
        check(d["scope_commitments"][:len(seed["scope_commitments"])] == seed["scope_commitments"], "original scopes removed")
        check(d["contrastive_commitments"][:len(seed["contrastive_commitments"])] == seed["contrastive_commitments"], "original contrasts removed")
        # A new sentence alone is not a new mechanistic commitment, measured
        # estimand or autonomous scientific insight. Not a semantic equivalence test.
        reasons = ["CORE_COMMITMENTS_UNCHANGED", "SYNTHETIC_SCENARIO_NOT_EVIDENCE",
                   "OPERATIONAL_DISCRIMINATING_ESTIMAND_NOT_SPECIFIED_IN_DRAFT",
                   "INDEPENDENT_MEASUREMENT_PLAN_NOT_VALIDATED",
                   "CONTRASTIVE_CLAIM_NOT_SCIENTIFICALLY_ADJUDICATED"]
        if row["pair_relationship"] == "ADJACENT_NOT_EXCLUSIVE":
            reasons.append("NONEXCLUSIVE_PARENT_PROGRAMS_MUST_REMAIN_SEPARATE")
        question_cases[d["question_commitment"]].add(cid)
        out.append({"case_id": cid, "scenario": scenario,
                    "draft_id": row["draft_id_NOT_RESEARCH_IDEA_ID"],
                    "structural_integrity": "PASS",
                    "scientific_delta_state": "CONDITIONAL_REFRAME_ONLY_SUBSTANTIVE_DELTA_UNPROVEN",
                    "core_commitments_changed": False,
                    "new_conditional_question_present": d["question_commitment"] != seed["question_commitment"],
                    "science_review_required": True, "failure_reasons": reasons,
                    "independent_science_confirmed": False,
                    "review_requirements": ["mechanism_specific_delta", "discriminating_estimand",
                                            "independent_measurement_path", "confound_controls",
                                            "alternative_explanations", "scope_of_possible_falsifier"]})
    check(seen == {(c, s) for c in cases for s in SCENARIOS}, "incomplete scenario grid")
    check(sum(x["draft_id"] is not None for x in out) == c2.get("draft_kernel_count"), "draft count mismatch")
    check(sum(x["draft_id"] is None for x in out) == c2.get("no_revision_count"), "no revision mismatch")
    for row in out:
        if row["draft_id"] is not None and len(question_cases[draft_question(c2, row["case_id"], row["scenario"])]) > 1:
            row["failure_reasons"].append("GENERIC_QUESTION_SHARED_ACROSS_CASES")
    result = {
        "schema_version": "m4c3-scientific-revision-quality-v1",
        "status": "INTEGRITY_PASS_SCIENTIFIC_DELTA_UNPROVEN",
        "input_sha256": {"m4c2": sha_c2, "m4a1": sha_a1, "m3c1": sha_traces},
        "case_count": len(cases), "scenario_count": len(rows),
        "draft_count": c2["draft_kernel_count"], "no_revision_count": c2["no_revision_count"],
        "substantive_revisions_scientifically_confirmed": 0,
        "review_required_count": c2["draft_kernel_count"],
        "rows": out,
        "source_files_mutated": False, "research_idea_nodes_created": False,
        "production_selection_changed": False, "empirical_learning_certified": False,
        "novelty_or_truth_authority": False, "llm_or_network_calls": 0,
    }
    return result


def draft_question(c2: Mapping[str, Any], cid: str, scenario: str) -> str:
    for r in c2["rows"]:
        if r["case_id"] == cid and r["synthetic_scenario"] == scenario:
            return r["draft_scientific_kernel_NOT_CREATED"]["question_commitment"]
    raise ValueError("M4C3_INTEGRITY_FAILURE: missing draft question")


def render_report(result: dict[str, Any]) -> str:
    lines = ["# M4-C3 — Scientific Revision Quality Gate (PRIVATE)", "",
             f"Status: `{result['status']}`", "",
             "This audit validates lineage and detects unmet scientific-review prerequisites. It DOES NOT establish scientific semantic equivalence, novelty, empirical success, or idea improvement.", "",
             f"Cases: {result['case_count']}; hypothetical scenarios: {result['scenario_count']}; draft kernels: {result['draft_count']}; no-revision: {result['no_revision_count']}.",
             f"Substantive scientific improvements demonstrated: **{result['substantive_revisions_scientifically_confirmed']}** (not assessed as genuine discoveries).", "",
             "| Case | Drafts to review | Scientific improvements confirmed |", "|---|---:|---:|"]
    by_case = Counter(row["case_id"] for row in result["rows"] if row["draft_id"])
    for cid in sorted({x["case_id"] for x in result["rows"]}):
        lines.append(f"| {cid} | {by_case[cid]} | 0 |")
    lines += ["", "## Minimum expert adjudication", "",
              "A genuine revision needs a new mechanism-specific conditional prediction or estimand, a test that can distinguish it from the parent/alternative under independent observation, a scoped failure condition, and explicit preservation of independent parent programs.", "",
              "The 16 current drafts add hypothetical scenario questions and repeat source patterns. Without an operational estimand or independently verifiable discriminating measurement, they must not be promoted to ResearchIdeaNodes or SIS offspring.", "",
              "The four NOT_IDENTIFIABLE fixtures correctly produce no revision. No empirical data was consumed.", ""]
    return "\n".join(lines)
