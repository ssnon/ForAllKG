"""M4-C0 synthetic, *non-empirical* research-idea revision policy replay.

This intentionally does not accept or interpret measured values, generate a new
ResearchIdea, verify a mechanism, or mutate anything. It exercises only the
policy that will eventually consume independently validated evidence events.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SCHEMA_VERSION = "m4c0-synthetic-revision-policy-v1"
SCENARIOS = (
    "A_ONLY_IN_SCOPE",
    "B_ONLY_IN_SCOPE",
    "BOTH_COMPATIBLE",
    "NEITHER_COMPATIBLE",
    "NOT_IDENTIFIABLE",
)
EXPECTED_SOURCE_STATUS = "DRAFT_CONFRONTATIONS_NEED_REVIEW"
ALLOWED_RELATIONSHIPS = {"COMPETING_PARTIAL", "ADJACENT_NOT_EXCLUSIVE"}


def _sha_obj(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("M4C0_INTEGRITY_FAILURE: " + message)


def _unique_text(values: Any, label: str) -> list[str]:
    _check(isinstance(values, list) and bool(values), label + " must be a nonempty array")
    _check(all(isinstance(x, str) and x.strip() for x in values), label + " must contain nonblank strings")
    _check(len(values) == len(set(values)), label + " contains duplicates")
    return list(values)


def validated_cases(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    _check(isinstance(report, dict), "source root must be object")
    _check(report.get("schema_version") == "m4a1-scientific-confrontation-v1", "unexpected M4-A1 schema")
    _check(report.get("status") == EXPECTED_SOURCE_STATUS, "unexpected M4-A1 status")
    for field, expected in (
        ("external_evidence_validated", False),
        ("human_or_expert_science_certification", False),
        ("hypothesis_cards_modified", False),
        ("source_files_mutated", False),
        ("production_selection_changed", False),
        ("llm_or_network_calls", 0),
    ):
        _check(type(report.get(field)) is type(expected) and report[field] == expected, field + " violates non-authority boundary")
    cases = report.get("cases")
    _check(isinstance(cases, list) and len(cases) > 0, "missing cases")
    _check(report.get("case_count") == len(cases), "case count mismatch")
    ids: set[str] = set()
    for case in cases:
        _check(isinstance(case, dict), "case must be object")
        cid = case.get("case_id")
        _check(isinstance(cid, str) and cid and cid not in ids, "missing/duplicate case id")
        ids.add(cid)
        _check(case.get("status") == "REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED", f"{cid}: case improperly certified")
        for field in ("confrontation_validated", "experiment_physically_feasible_certified", "falsification_logic_certified", "novelty_or_truth_authority"):
            _check(case.get(field) is False, f"{cid}: {field} is not false")
        refs = _unique_text(case.get("hypothesis_refs"), f"{cid}: hypothesis_refs")
        hyps = case.get("input_hypotheses")
        _check(isinstance(hyps, list) and hyps, f"{cid}: input_hypotheses missing")
        _check(len(hyps) == len(refs), f"{cid}: hypothesis reference count mismatch")
        _check(set(h.get("ref") for h in hyps if isinstance(h, dict)) == set(refs), f"{cid}: hypothesis refs mismatch")
        for hyp in hyps:
            _check(isinstance(hyp, dict), f"{cid}: invalid hypothesis")
            for field in ("hypothesis_id", "terminal_idea_id", "scientific_intent"):
                _check(isinstance(hyp.get(field), str) and bool(hyp[field].strip()), f"{cid}: missing {field}")
        experiment = case.get("proposed_experiment_UNREVIEWED")
        _check(isinstance(experiment, dict), f"{cid}: experiment missing")
        _check(experiment.get("pair_relationship") in ALLOWED_RELATIONSHIPS, f"{cid}: unknown pair relationship")
        mechanisms = case.get("proposed_mechanisms_UNREVIEWED")
        _check(isinstance(mechanisms, list) and len(mechanisms) == 2, f"{cid}: requires two mechanisms")
        _check({m.get("role") for m in mechanisms if isinstance(m, dict)} == {"A", "B"}, f"{cid}: requires A/B mechanisms")
    return cases


def _actions(scenario: str, relationship: str) -> tuple[list[str], list[str]]:
    if scenario == "A_ONLY_IN_SCOPE":
        actions = ["REVIEW_RELATIVE_SUPPORT_FOR_A_IN_SCOPE", "CHALLENGE_B_REALIZATION_IN_SCOPE", "REQUEST_INDEPENDENT_REPLICATION"]
        cautions = ["A_NOT_PROVEN", "B_NOT_GLOBALLY_REJECTED"]
    elif scenario == "B_ONLY_IN_SCOPE":
        actions = ["REVIEW_RELATIVE_SUPPORT_FOR_B_IN_SCOPE", "CHALLENGE_A_REALIZATION_IN_SCOPE", "REQUEST_INDEPENDENT_REPLICATION"]
        cautions = ["B_NOT_PROVEN", "A_NOT_GLOBALLY_REJECTED"]
    elif scenario == "BOTH_COMPATIBLE":
        actions = ["RETAIN_BOTH_MECHANISMS", "REQUEST_ADDITIONAL_DISCRIMINATOR"]
        cautions = ["NO_MECHANISM_SELECTION"]
    elif scenario == "NEITHER_COMPATIBLE":
        actions = ["RETAIN_BOTH_ORIGINAL_PROGRAMS", "SEARCH_MISSING_MECHANISM", "REFRAME_IN_TESTED_SCOPE"]
        cautions = ["FAILURE_OF_TWO_MODELS_NOT_EXHAUSTIVE"]
    elif scenario == "NOT_IDENTIFIABLE":
        actions = ["RESOLVE_MEASUREMENT_INDEPENDENCE", "REPAIR_CONFOUND_CONTROLS", "REQUEST_DISCRIMINATING_TEST"]
        cautions = ["NO_ADJUDICATION_FROM_UNIDENTIFIABLE_OBSERVATION"]
    else:
        raise ValueError("M4C0_INTEGRITY_FAILURE: unsupported scenario " + str(scenario))
    if relationship == "ADJACENT_NOT_EXCLUSIVE":
        actions.append("EVALUATE_JOINT_OR_ADDITIVE_EXPLANATORY_POWER")
        cautions.append("NONEXCLUSIVE_MECHANISMS_NO_AUTOMATIC_EXCLUSION")
    return actions, cautions


def build_policy_replay(report: dict[str, Any], *, source_sha256: str) -> dict[str, Any]:
    _check(isinstance(source_sha256, str) and len(source_sha256) == 64 and all(c in "0123456789abcdef" for c in source_sha256), "invalid source SHA-256")
    cases = validated_cases(report)
    rows = []
    for case in cases:
        relationship = case["proposed_experiment_UNREVIEWED"]["pair_relationship"]
        idea_ids = [x["terminal_idea_id"] for x in case["input_hypotheses"]]
        for scenario in SCENARIOS:
            actions, cautions = _actions(scenario, relationship)
            rows.append({
                "case_id": case["case_id"],
                "hypothesis_refs": list(case["hypothesis_refs"]),
                "original_terminal_idea_ids_PRESERVED": list(idea_ids),
                "pair_relationship": relationship,
                "scenario_kind": "SYNTHETIC_POLICY_FIXTURE_NOT_OBSERVED",
                "synthetic_scenario": scenario,
                "suggested_review_actions_NOT_EXECUTED": actions,
                "scientific_cautions": cautions,
                "new_research_idea_created": False,
                "hypothesis_falsified": False,
                "observed_evidence_ingested": False,
                "requires_human_science_review": True,
            })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "SYNTHETIC_POLICY_REPLAY_ONLY_NO_EMPIRICAL_LEARNING",
        "input_m4a1_sha256": source_sha256,
        "case_count": len(cases),
        "scenario_count_per_case": len(SCENARIOS),
        "policy_row_count": len(rows),
        "cases": rows,
        "original_research_ideas_modified": False,
        "hypothesis_cards_modified": False,
        "scientific_truth_or_falsification_authority": False,
        "production_selection_changed": False,
        "new_llm_or_network_calls": 0,
        "scientific_learning_demonstrated": False,
        "real_measurements_consumed": False,
    }
    payload["report_id"] = "m4c0_synthetic_policy:" + _sha_obj(payload)[:20]
    return payload


def render_report(replay: Mapping[str, Any]) -> str:
    lines = [
        "# M4-C0 — Synthetic Epistemic Revision Policy Replay (PRIVATE)", "",
        f"Status: `{replay['status']}`", "",
        "This is a policy test, not empirical science or actual ResearchIdea evolution.",
        "Synthetic scenario labels do not represent observed outcomes. No ResearchIdea was created or modified.", "",
        f"Cases: {replay['case_count']}; synthetic scenarios per case: {replay['scenario_count_per_case']}; total: {replay['policy_row_count']}.", "",
        "| Case | Pair relationship | Synthetic scenarios |", "|---|---|---:|",
    ]
    case_counts = {}
    for row in replay["cases"]:
        case_counts[row["case_id"]] = (row["pair_relationship"], case_counts.get(row["case_id"], (None, 0))[1] + 1)
    for case_id, (relationship, count) in case_counts.items():
        lines.append(f"| {case_id} | {relationship} | {count} |")
    lines.extend(["", "**Next scientific gate:** verified observational measurements, sample/time linkage, measurement independence, uncertainty, confound controls, and scope-specific falsifier review.", "", "No deletion, claim promotion, or reproduction is authorized."])
    return "\n".join(lines) + "\n"
