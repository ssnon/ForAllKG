"""M4-C2 synthetic-only ResearchIdea kernel revision *draft* replay.

Produces reviewed-needed proposals, never a ResearchIdeaNode or empirical
revision. No models, retrieval, data reading, premise promotion, or mutation.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Mapping

from pydantic import ValidationError

from pipeline_core.discovery.empirical_revision_policy_m4c0 import (
    SCENARIOS, validated_cases,
)
from pipeline_core.discovery.research_idea_contracts import ResearchIdeaKernel


def _check(predicate: bool, message: str) -> None:
    if not predicate:
        raise ValueError("M4C2_INTEGRITY_FAILURE: " + message)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _is_false(obj: Mapping[str, Any], fields: tuple[str, ...], label: str) -> None:
    for field in fields:
        _check(obj.get(field) is False, f"{label}.{field} must be false")


def _text(value: Any, label: str) -> str:
    _check(isinstance(value, str) and bool(value.strip()), label + " must be nonblank")
    return value.strip()


def _validated_kernel(payload: Any, label: str) -> ResearchIdeaKernel:
    try:
        return ResearchIdeaKernel.model_validate(payload)
    except (ValidationError, TypeError, ValueError) as exc:
        raise ValueError("M4C2_INTEGRITY_FAILURE: invalid ResearchIdeaKernel " + label) from exc


def _check_previous_runs(
    *, a1: dict[str, Any], c0: dict[str, Any], c1: dict[str, Any],
    trajectories: dict[str, Any], sha_a1: str, sha_c0: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    cases = validated_cases(a1)
    _check(c0.get("schema_version") == "m4c0-synthetic-revision-policy-v1", "C0 schema")
    _check(c0.get("status") == "SYNTHETIC_POLICY_REPLAY_ONLY_NO_EMPIRICAL_LEARNING", "C0 status")
    _check(c0.get("input_m4a1_sha256") == sha_a1, "C0/A1 SHA linkage")
    _check(c0.get("case_count") == len(cases), "C0 case count")
    _check(c0.get("scenario_count_per_case") == len(SCENARIOS), "C0 scenarios per case")
    _check(c0.get("new_llm_or_network_calls") == 0, "C0 model calls")
    _is_false(c0, ("scientific_learning_demonstrated", "real_measurements_consumed",
                   "original_research_ideas_modified", "hypothesis_cards_modified",
                   "scientific_truth_or_falsification_authority", "production_selection_changed"), "C0")
    all_rows = c0.get("cases")
    _check(isinstance(all_rows, list) and c0.get("policy_row_count") == len(all_rows), "C0 row count")
    row_map: dict[tuple[str, str], dict[str, Any]] = {}
    case_by_id = {case["case_id"]: case for case in cases}
    for row in all_rows:
        _check(isinstance(row, dict), "C0 row must be object")
        cid, scenario = row.get("case_id"), row.get("synthetic_scenario")
        _check(cid in case_by_id and scenario in SCENARIOS, "C0 unknown case/scenario")
        key = (cid, scenario)
        _check(key not in row_map, "C0 duplicate scenario " + str(key))
        source = case_by_id[cid]
        ids = [h["terminal_idea_id"] for h in source["input_hypotheses"]]
        _check(row.get("original_terminal_idea_ids_PRESERVED") == ids, "C0 parent ID mismatch")
        _check(row.get("hypothesis_refs") == source["hypothesis_refs"], "C0 hypothesis reference mismatch")
        _check(row.get("pair_relationship") == source["proposed_experiment_UNREVIEWED"]["pair_relationship"], "C0 relationship mismatch")
        _check(row.get("scenario_kind") == "SYNTHETIC_POLICY_FIXTURE_NOT_OBSERVED", "C0 scenario not synthetic")
        _check(row.get("requires_human_science_review") is True, "C0 must require review")
        _is_false(row, ("hypothesis_falsified", "new_research_idea_created", "observed_evidence_ingested"), "C0 row")
        _check(isinstance(row.get("suggested_review_actions_NOT_EXECUTED"), list) and row["suggested_review_actions_NOT_EXECUTED"], "C0 missing actions")
        _check(isinstance(row.get("scientific_cautions"), list) and row["scientific_cautions"], "C0 missing cautions")
        if row["pair_relationship"] == "ADJACENT_NOT_EXCLUSIVE":
            _check("NONEXCLUSIVE_MECHANISMS_NO_AUTOMATIC_EXCLUSION" in row["scientific_cautions"], "nonexclusive caution missing")
        row_map[key] = row
    _check(len(row_map) == len(cases) * len(SCENARIOS), "C0 incomplete scenario matrix")

    _check(c1.get("schema_version") == "m4c1-evidence-event-audit-v1", "C1 schema")
    _check(c1.get("status") == "EVIDENCE_EVENT_INTAKE_INCOMPLETE", "C1 cannot be populated evidence")
    _check(c1.get("input_m4a1_sha256") == sha_a1, "C1/A1 SHA linkage")
    _check(c1.get("input_m4c0_sha256") == sha_c0, "C1/C0 SHA linkage")
    _check(c1.get("case_count") == len(cases), "C1 case count")
    _check(c1.get("file_sha256_match_count") == 0, "C1 evidence files found: synthetic replay prohibited")
    _check(c1.get("real_measurement_values_read") is False, "C1 measured values consumed")
    _check(c1.get("new_llm_or_network_calls") == 0, "C1 model calls")
    _is_false(c1, ("hypothesis_cards_modified", "independence_certified",
                   "new_idea_generated", "original_research_ideas_modified",
                   "production_selection_changed", "scientific_truth_or_falsification_authority",
                   "source_files_mutated"), "C1")
    indexed_c1 = {}
    for row in c1.get("cases", []):
        _check(isinstance(row, dict) and row.get("case_id") not in indexed_c1, "invalid/duplicate C1 case")
        indexed_c1[row["case_id"]] = row
    _check(set(indexed_c1) == set(case_by_id), "C1 case set mismatch")
    for cid, case in case_by_id.items():
        meta = indexed_c1[cid]
        _check(meta.get("original_terminal_idea_ids_PRESERVED") == [h["terminal_idea_id"] for h in case["input_hypotheses"]], "C1 original IDs mismatch")
        _check(meta.get("independent_measurements_certified") is False and meta.get("claim_adjudication_authorized") is False, "C1 cannot certify science")

    _check(trajectories.get("status") == "SHA_VERIFIED_SCIENTIFIC_DELTAS_READY_FOR_HUMAN_REVIEW", "M3C1 trajectory status")
    _check(trajectories.get("root_set_conserved_across_arms") is True, "M3C1 root preservation")
    _check(trajectories.get("original_data_mutated") is False, "M3C1 mutation")
    _check(trajectories.get("new_llm_calls") == 0, "M3C1 calls")
    _check(trajectories.get("authoritative_science_judgment") is False, "M3C1 authority")
    trajectory_index = {}
    for item in trajectories.get("rows", []):
        _check(isinstance(item, dict) and item.get("blind_id") not in trajectory_index, "duplicate/invalid blind id")
        trajectory_index[item["blind_id"]] = item
    for case in cases:
        for h in case["input_hypotheses"]:
            ref = h["ref"]
            _check(ref in trajectory_index, "missing M3C1 trace " + ref)
            record = trajectory_index[ref]
            _check(record.get("lineage_validation") == "PASS_SOURCE_HASH_AND_KERNEL_TRAJECTORY", "M3C1 lineage")
            _check(record.get("hypothesis_id") == h["hypothesis_id"], "M3C1 hypothesis ID")
            _check(record.get("terminal_idea_id") == h["terminal_idea_id"], "M3C1 terminal idea ID")
            steps = record.get("steps_earliest_to_latest")
            _check(isinstance(steps, list) and bool(steps) and steps[-1].get("idea_id") == h["terminal_idea_id"], "M3C1 terminal step")
            _validated_kernel(steps[-1].get("kernel"), ref)
    return cases, row_map, trajectory_index


def _proposal_kernel(
    base: ResearchIdeaKernel, *, scenario: str, relationship: str,
    mechanism_a: Mapping[str, str], mechanism_b: Mapping[str, str],
) -> ResearchIdeaKernel | None:
    if scenario == "NOT_IDENTIFIABLE":
        return None
    a = _text(mechanism_a.get("explanation"), "mechanism A")
    b = _text(mechanism_b.get("explanation"), "mechanism B")
    pa = _text(mechanism_a.get("predicted_pattern"), "prediction A")
    pb = _text(mechanism_b.get("predicted_pattern"), "prediction B")
    if scenario == "A_ONLY_IN_SCOPE":
        question = "Under a hypothetical independently verified A-like outcome, where does explanation B fail to account for the measured pattern, and what additional controls could restore B?"
        contrast = f"Synthetic A-like pattern to interrogate: {pa}"
        target = f"Challenge B only within matched measurement conditions: {b}"
        mode = "SCOPE_B_CHALLENGE"
    elif scenario == "B_ONLY_IN_SCOPE":
        question = "Under a hypothetical independently verified B-like outcome, which A-specific prediction requires revision, and what alternative controls could explain the discrepancy?"
        contrast = f"Synthetic B-like pattern to interrogate: {pb}"
        target = f"Challenge A only within matched measurement conditions: {a}"
        mode = "SCOPE_A_CHALLENGE"
    elif scenario == "BOTH_COMPATIBLE":
        question = ("Under an ambiguous hypothetical outcome, what additional orthogonal measurement separates the explanations or estimates their joint contribution?"
                    if relationship == "ADJACENT_NOT_EXCLUSIVE" else
                    "Under an ambiguous hypothetical outcome, can a joint or regime-dependent model explain both patterns without forcing a false exclusive choice?")
        contrast = f"Joint/explanation A target prediction: {pa}"
        target = f"Joint/explanation B target prediction: {pb}"
        mode = "JOINT_OR_REGIME_MODEL"
    elif scenario == "NEITHER_COMPATIBLE":
        question = "If independently reproduced observations contradicted both proposed patterns, which third mediator or regime boundary would provide a falsifiable alternative?"
        contrast = f"Assumed unresolved pattern A, NOT actually falsified: {pa}"
        target = f"Assumed unresolved pattern B, NOT actually falsified: {pb}"
        mode = "THIRD_MECHANISM_SEARCH"
    else:
        raise ValueError("M4C2_INTEGRITY_FAILURE: unsupported scenario")
    conditional = "Hypothetical synthetic replay only; no observation occurred and neither mechanism has been verified."
    # Preserve core commitments exactly. Update explicit research question, scope,
    # and contrastive tests so this is a content-bearing *proposal*, not renaming.
    return ResearchIdeaKernel(
        canonical_intent=(f"{base.canonical_intent} Conditional revision objective ({mode}): {question}"),
        core_scientific_commitments=list(base.core_scientific_commitments),
        scope_commitments=list(base.scope_commitments) + [
            conditional,
            "Restrict any potential inference to matched sample, time, measurement and confound-controlled conditions; require independent replication.",
            f"Revision target: {mode}; all original ResearchIdea parent programs remain available.",
        ],
        contrastive_commitments=list(base.contrastive_commitments) + [contrast, target],
        question_commitment=question,
    )


def build_revision_replay(
    *, a1: dict[str, Any], c0: dict[str, Any], c1: dict[str, Any],
    trajectories: dict[str, Any], sha_a1: str, sha_c0: str, sha_c1: str,
    sha_trajectories: str,
) -> dict[str, Any]:
    cases, policies, traces = _check_previous_runs(a1=a1, c0=c0, c1=c1,
        trajectories=trajectories, sha_a1=sha_a1, sha_c0=sha_c0)
    rows = []
    for case in cases:
        cid = case["case_id"]
        hyps = case["input_hypotheses"]
        parent_ids = [hyp["terminal_idea_id"] for hyp in hyps]
        parent_kernels = [_validated_kernel(
            traces[hyp["ref"]]["steps_earliest_to_latest"][-1]["kernel"], hyp["ref"]
        ) for hyp in hyps]
        pair_rel = case["proposed_experiment_UNREVIEWED"]["pair_relationship"]
        mechanisms = {m["role"]: m for m in case["proposed_mechanisms_UNREVIEWED"]}
        for scenario in SCENARIOS:
            policy = policies[cid, scenario]
            kernel = _proposal_kernel(parent_kernels[0], scenario=scenario,
                relationship=pair_rel, mechanism_a=mechanisms["A"], mechanism_b=mechanisms["B"])
            if kernel is not None:
                _check(kernel.core_scientific_commitments == parent_kernels[0].core_scientific_commitments, "core kernel altered")
                _check(kernel.canonical_intent != parent_kernels[0].canonical_intent, "no substantive question delta")
                _check(len(kernel.scope_commitments) > len(parent_kernels[0].scope_commitments), "scope not changed")
                _check(len(kernel.contrastive_commitments) > len(parent_kernels[0].contrastive_commitments), "contrast not changed")
            d = kernel.model_dump(mode="json") if kernel else None
            row = {
                "case_id": cid,
                "scenario_kind": "SYNTHETIC_POLICY_FIXTURE_NOT_OBSERVED",
                "synthetic_scenario": scenario,
                "pair_relationship": pair_rel,
                "source_hypothesis_refs": case["hypothesis_refs"],
                "original_terminal_idea_ids_PRESERVED": parent_ids,
                "seed_parent_id": parent_ids[0],
                "comparison_parent_ids_PRESERVED_SEPARATELY": parent_ids[1:],
                "multi_parent_composition_NOT_CLAIMED": True,
                "parent_kernel_sha256_by_id": {
                    hyp["terminal_idea_id"]: _sha(p.model_dump(mode="json"))
                    for hyp, p in zip(hyps, parent_kernels)
                },
                "original_parent_scientific_core_by_id": {
                    hyp["terminal_idea_id"]: list(p.core_scientific_commitments)
                    for hyp, p in zip(hyps, parent_kernels)
                },
                "draft_id_NOT_RESEARCH_IDEA_ID": None,
                "draft_scientific_kernel_NOT_CREATED": d,
                "draft_kernel_sha256": _sha(d) if d else None,
                "revision_disposition": "DRAFT_KERNEL_FOR_EXPERT_REVIEW_ONLY" if d else "NO_REVISION_UNIDENTIFIABLE",
                "preferred_policy_actions_NOT_EXECUTED": policy["suggested_review_actions_NOT_EXECUTED"],
                "scientific_cautions": policy["scientific_cautions"],
                "requires_human_science_review": True,
                "research_idea_node_created": False,
                "original_research_ideas_modified": False,
                "scientific_claim_adjudicated": False,
                "real_measurement_consumed": False,
            }
            if d is not None:
                row["draft_id_NOT_RESEARCH_IDEA_ID"] = "m4c2_kernel_draft:" + _sha([
                    sha_trajectories, sha_c0, cid, scenario, parent_ids, d
                ])[:24]
            rows.append(row)
    result = {
        "schema_version": "m4c2-synthetic-kernel-revision-replay-v1",
        "status": "SYNTHETIC_KERNEL_DRAFTS_ONLY_NOT_EMPIRICAL_LEARNING",
        "input_sha256": {"m4a1":sha_a1,"m4c0":sha_c0,"m4c1":sha_c1,"m3c1_trajectories":sha_trajectories},
        "source_case_count": len(cases),
        "scenario_count": len(rows),
        "draft_kernel_count": sum(x["draft_scientific_kernel_NOT_CREATED"] is not None for x in rows),
        "no_revision_count": sum(x["draft_scientific_kernel_NOT_CREATED"] is None for x in rows),
        "rows": rows,
        "real_measurements_consumed": False,
        "research_idea_nodes_created": False,
        "original_research_ideas_modified": False,
        "hypothesis_cards_modified": False,
        "scientific_truth_or_falsification_authority": False,
        "novelty_certification_authority": False,
        "production_selection_changed": False,
        "new_llm_or_network_calls": 0,
        "empirical_scientific_learning_demonstrated": False,
    }
    result["report_id"] = "m4c2_kernel_revision_replay:" + _sha(result)[:20]
    return result


def render_report(data: dict[str, Any]) -> str:
    counts = Counter(x["revision_disposition"] for x in data["rows"])
    lines = ["# M4-C2 — Conditional ResearchIdea Kernel Drafts (PRIVATE)", "",
        f"Status: `{data['status']}`", "",
        "All scenarios are SYNTHETIC, NOT OBSERVED. Drafts are not ResearchIdeaNodes, do not have ResearchIdea IDs, and have no scientific/reproduction authority.",
        "Original scientific core commitments are copied unchanged. Scenario-conditioned research questions, contrastive tests, and scope are drafted for expert review only.", "",
        f"Cases: {data['source_case_count']}; scenarios: {data['scenario_count']}; proposed kernels: {data['draft_kernel_count']}; no revision: {data['no_revision_count']}.", "",
        "| Case | Draft proposals | No revision | Nonexclusive caution |", "|---|---:|---:|---|",]
    case_ids=list(dict.fromkeys(x['case_id'] for x in data['rows']))
    for cid in case_ids:
        rr=[x for x in data['rows'] if x['case_id']==cid]
        lines.append(f"| {cid} | {sum(x['draft_scientific_kernel_NOT_CREATED'] is not None for x in rr)} | {sum(x['draft_scientific_kernel_NOT_CREATED'] is None for x in rr)} | {'YES' if rr[0]['pair_relationship']=='ADJACENT_NOT_EXCLUSIVE' else 'NO'} |")
    lines.extend(["", "**Boundary:** no measured values, no generation, no scientific falsification, no production changes. An unidentifiable scenario yields no kernel draft.",
        "", "**Next:** Expert-review draft scientific meaning and then test accepted revision semantics in an isolated shadow runtime; obtain independent empirical observations before any empirical-learning claim."])
    return '\n'.join(lines)+'\n'
