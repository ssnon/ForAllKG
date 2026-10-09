"""M4-C1 source-byte anchored evidence-event intake, NEVER scientific certification.

Scopes candidate files to frozen M4-A1 observable requests. The only file content
operation on evidence is SHA-256 hashing; it cannot determine physical independence,
correct numeric extraction, sample registration, or support for a hypothesis.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from pipeline_core.discovery.empirical_revision_policy_m4c0 import SCENARIOS, validated_cases

SCHEMA = "m4c1-evidence-event-template-v1"
EVENT_CLASSES = frozenset({
    "PRIMARY_INSTRUMENT", "DERIVED_FROM_PRIMARY", "LITERATURE_TEXT",
    "SIMULATION", "SYNTHETIC_FIXTURE", "KG_DERIVED", "UNCLASSIFIED",
})
NOT_FILLED = frozenset({"", "UNKNOWN", "UNREVIEWED", "TBD", "N/A", "NONE", "NULL"})
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
EVENT_FIELDS = (
    "event_id", "case_id", "observable_name", "observable_role",
    "source_class", "source_relative_path", "source_sha256", "value_locator",
    "sample_id", "condition_id", "time_window", "measurement_method",
    "units", "uncertainty_description", "confound_controls",
    "independence_basis", "origin_citation", "reviewer_notes",
)


def _check(condition: bool, msg: str) -> None:
    if not condition:
        raise ValueError("M4C1_INTEGRITY_FAILURE: " + msg)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _filled(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() not in NOT_FILLED


def validate_source(m4a1: dict[str, Any], m4c0: dict[str, Any], m4a1_sha: str) -> list[dict[str, Any]]:
    cases = validated_cases(m4a1)
    _check(m4c0.get("schema_version") == "m4c0-synthetic-revision-policy-v1", "M4-C0 schema")
    _check(m4c0.get("status") == "SYNTHETIC_POLICY_REPLAY_ONLY_NO_EMPIRICAL_LEARNING", "M4-C0 status")
    _check(m4c0.get("input_m4a1_sha256") == m4a1_sha, "M4-C0 source SHA mismatch")
    _check(m4c0.get("case_count") == len(cases), "M4-C0 case count mismatch")
    _check(m4c0.get("policy_row_count") == len(cases) * 5, "M4-C0 policy coverage mismatch")
    for flag in ("scientific_learning_demonstrated", "real_measurements_consumed",
                 "original_research_ideas_modified", "hypothesis_cards_modified",
                 "scientific_truth_or_falsification_authority", "production_selection_changed"):
        _check(m4c0.get(flag) is False, f"M4-C0 forbidden authority: {flag}")
    original_cases = {x["case_id"]: x for x in cases}
    seen_policy: set[tuple[str, str]] = set()
    for row in m4c0.get("cases", []):
        cid = row.get("case_id")
        _check(cid in original_cases, "M4-C0 unknown case")
        _check(row.get("scenario_kind") == "SYNTHETIC_POLICY_FIXTURE_NOT_OBSERVED", "M4-C0 scenario origin")
        scenario = row.get("synthetic_scenario")
        _check(scenario in SCENARIOS and (cid, scenario) not in seen_policy, "M4-C0 duplicate or invalid scenario")
        seen_policy.add((cid, scenario))
        for key in ("new_research_idea_created", "hypothesis_falsified"):
            _check(row.get(key) is False, "M4-C0 row forbidden authority: " + key)
        _check(row.get("observed_evidence_ingested") is False, "M4-C0 observed flag")
        _check(row.get("original_terminal_idea_ids_PRESERVED") == [x["terminal_idea_id"] for x in original_cases[cid]["input_hypotheses"]], "M4-C0 lineage mismatch")
    _check(seen_policy == {(cid, scenario) for cid in original_cases for scenario in SCENARIOS}, "M4-C0 incomplete scenario coverage")
    return cases


def _requirements(cases: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for case in cases:
        obs = case["proposed_experiment_UNREVIEWED"].get("observables")
        _check(isinstance(obs, list) and obs, "missing source observables")
        for o in obs:
            name, role = o.get("name"), o.get("role")
            _check(_filled(name) and role in {"TARGET_OUTCOME", "INDEPENDENT_DISCRIMINATOR"}, "invalid source observable")
            key = (case["case_id"], name)
            _check(key not in expected, "duplicate observable within case")
            expected[key] = {"case_id": case["case_id"], "observable_name": name,
                             "observable_role": role, "hypothesis_refs": case["hypothesis_refs"],
                             "original_terminal_idea_ids": [x["terminal_idea_id"] for x in case["input_hypotheses"]]}
    return expected


def make_event_template(cases: list[dict[str, Any]], *, m4a1_sha: str, m4c0_sha: str) -> dict[str, Any]:
    requirements = _requirements(cases)
    rows = []
    for (case_id, name), req in requirements.items():
        event_id = "event_template:" + _canonical_sha([case_id, name])[:20]
        row = {k: "" for k in EVENT_FIELDS}
        row.update({"event_id": event_id, "case_id": case_id,
                    "observable_name": name, "observable_role": req["observable_role"],
                    "source_class": "UNCLASSIFIED"})
        rows.append(row)
    return {
        "schema_version": SCHEMA, "status": "EMPTY_TEMPLATE_NOT_OBSERVED",
        "source_m4a1_sha256": m4a1_sha, "source_m4c0_sha256": m4c0_sha,
        "events": rows, "is_verified_scientific_evidence": False,
        "notes": "Every template row is UNREVIEWED. Fill only from independently checked sources; never use generated claims as observed evidence.",
    }


def _safe_source(root: Path, rel: str) -> tuple[Path | None, str | None]:
    # Do not allow paths outside evidence root; do not follow symlinks.
    if not _filled(rel):
        return None, "SOURCE_PATH_NOT_PROVIDED"
    normalized = rel.replace("\\", "/")
    if re.match(r"^[a-zA-Z]:/", normalized) or normalized.startswith("//"):
        return None, "SOURCE_PATH_OUTSIDE_ALLOWED_ROOT"
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(p in {"..", "."} for p in pure.parts) or normalized.startswith("~"):
        return None, "SOURCE_PATH_OUTSIDE_ALLOWED_ROOT"
    candidate = root / Path(*pure.parts)
    if not candidate.resolve(strict=False).is_relative_to(root.resolve()):
        return None, "SOURCE_PATH_OUTSIDE_ALLOWED_ROOT"
    parent = root
    for part in pure.parts:
        parent = parent / part
        if parent.is_symlink():
            return None, "SYMLINK_SOURCE_DISALLOWED"
    if not candidate.is_file():
        return None, "SOURCE_FILE_NOT_FOUND"
    return candidate, None


def _inspect(row: Mapping[str, Any], root: Path) -> dict[str, Any]:
    issues: list[str] = []
    klass = row["source_class"]
    if klass == "UNCLASSIFIED":
        issues.append("SOURCE_CLASS_UNCLASSIFIED")
    if klass in {"SYNTHETIC_FIXTURE", "SIMULATION", "LITERATURE_TEXT", "KG_DERIVED"}:
        issues.append("NOT_PRIMARY_EMPIRICAL_MEASUREMENT")
    path, path_issue = _safe_source(root, row["source_relative_path"])
    file_sha_match = False
    observed_sha = None
    if path_issue:
        issues.append(path_issue)
    else:
        recorded = row["source_sha256"]
        if not isinstance(recorded, str) or not HEX64.fullmatch(recorded):
            issues.append("SOURCE_SHA256_MISSING_OR_INVALID")
        else:
            observed_sha = _digest(path)
            file_sha_match = observed_sha == recorded
            if not file_sha_match:
                issues.append("SOURCE_SHA256_MISMATCH")
    for field in ("value_locator", "sample_id", "condition_id", "time_window", "measurement_method",
                  "units", "uncertainty_description", "confound_controls", "origin_citation"):
        if not _filled(row[field]):
            issues.append("MISSING_" + field.upper())
    if row["observable_role"] == "INDEPENDENT_DISCRIMINATOR" and not _filled(row["independence_basis"]):
        issues.append("INDEPENDENCE_BASIS_NOT_DESCRIBED")
    if not _filled(row["reviewer_notes"]):
        issues.append("REVIEWER_NOTE_UNRECORDED")
    return {
        "event_id": row["event_id"], "case_id": row["case_id"],
        "observable_name": row["observable_name"], "observable_role": row["observable_role"],
        "source_class": klass, "source_relative_path_PRIVATE": row["source_relative_path"],
        "file_byte_sha256_matched": file_sha_match,
        "observed_file_sha256_PRIVATE": observed_sha,
        "metadata_complete_as_declared": not issues,
        "issues": issues,
        "physical_measurement_independence_certified": False,
        "data_values_inspected": False, "scientific_claim_authority": False,
    }


def audit_event_template(template: dict[str, Any], cases: list[dict[str, Any]], *,
                         root: Path, m4a1_sha: str, m4c0_sha: str) -> dict[str, Any]:
    _check(isinstance(template, dict) and template.get("schema_version") == SCHEMA, "input template schema")
    _check(template.get("source_m4a1_sha256") == m4a1_sha, "event source M4-A1 SHA mismatch")
    _check(template.get("source_m4c0_sha256") == m4c0_sha, "event source M4-C0 SHA mismatch")
    _check(template.get("is_verified_scientific_evidence") is False, "template must not claim evidence certification")
    expected = _requirements(cases)
    rows = template.get("events")
    _check(isinstance(rows, list), "events must be list")
    seen_eids, seen_keys = set(), set()
    results = []
    for row in rows:
        _check(isinstance(row, dict) and set(row) == set(EVENT_FIELDS), "event field set differs from pinned contract")
        for f in EVENT_FIELDS:
            _check(isinstance(row[f], str), f"event.{f} must be string")
        _check(_filled(row["event_id"]) and row["event_id"] not in seen_eids, "missing/duplicate event ID")
        seen_eids.add(row["event_id"])
        key = row["case_id"], row["observable_name"]
        _check(key in expected and key not in seen_keys, "duplicate/foreign observable")
        seen_keys.add(key)
        _check(row["observable_role"] == expected[key]["observable_role"], "observable role mismatch")
        _check(row["source_class"] in EVENT_CLASSES, "invalid source class")
        results.append(_inspect(row, root))
    missing = sorted(set(expected) - seen_keys)
    complete = bool(results) and not missing and all(x["metadata_complete_as_declared"] for x in results)
    cases_out = []
    for case in cases:
        cid = case["case_id"]
        sub = [x for x in results if x["case_id"] == cid]
        requirements = [x for x in expected if x[0] == cid]
        # Matching labels is NOT a sample/time linkage certification. Results do not expose values.
        cases_out.append({
            "case_id": cid, "pair_relationship": case["proposed_experiment_UNREVIEWED"]["pair_relationship"], "declared_event_count": len(sub),
            "expected_observable_count": len(requirements),
            "all_required_observables_declared": len(sub) == len(requirements),
            "matching_sample_time_certified": False,
            "independent_measurements_certified": False,
            "claim_adjudication_authorized": False,
            "original_terminal_idea_ids_PRESERVED": [x["terminal_idea_id"] for x in case["input_hypotheses"]],
        })
    report = {
        "schema_version": "m4c1-evidence-event-audit-v1",
        "status": "SOURCE_METADATA_ONLY_EXPERT_REVIEW_REQUIRED" if complete else "EVIDENCE_EVENT_INTAKE_INCOMPLETE",
        "input_m4a1_sha256": m4a1_sha, "input_m4c0_sha256": m4c0_sha,
        "case_count": len(cases), "expected_event_count": len(expected),
        "submitted_event_count": len(results), "missing_observable_keys": [list(x) for x in missing],
        "file_sha256_match_count": sum(x["file_byte_sha256_matched"] for x in results),
        "issue_counts": dict(sorted(Counter(code for x in results for code in x["issues"]).items())),
        "events": results, "cases": cases_out,
        "source_files_mutated": False, "hypothesis_cards_modified": False,
        "original_research_ideas_modified": False, "real_measurement_values_read": False,
        "independence_certified": False, "scientific_truth_or_falsification_authority": False,
        "new_idea_generated": False, "production_selection_changed": False, "new_llm_or_network_calls": 0,
        "safe_next_step": "EXPERT_DATA_PROVENANCE_AND_SCIENTIFIC_DESIGN_REVIEW_NOT_AUTOMATED_ADJUDICATION",
    }
    report["report_id"] = "m4c1_evidence_event_audit:" + _canonical_sha(report)[:20]
    return report


def render_report(obj: Mapping[str, Any]) -> str:
    lines = ["# M4-C1 — Evidence Event Intake (PRIVATE)", "",
             f"Status: `{obj['status']}`", "",
             f"Observables: {obj['expected_event_count']}; submitted events: {obj['submitted_event_count']}; files with matching SHA-256: {obj['file_sha256_match_count']}.",
             "", "A matching file hash proves only local byte identity, not measurement truth, extraction accuracy, or experimental independence.",
             "No values were read, no ResearchIdea was revised, and no hypothesis was adjudicated.", "",
             "| Case | Events / observables | Scientific adjudication |", "|---|---:|---|"]
    for c in obj["cases"]:
        lines.append(f"| {c['case_id']} | {c['declared_event_count']} / {c['expected_observable_count']} | NOT_AUTHORIZED |")
    lines.extend(["", "## Remaining issues", ""])
    for issue, count in obj["issue_counts"].items():
        lines.append(f"- {issue}: {count}")
    if not obj["issue_counts"]:
        lines.append("- No syntactic metadata issues; independent scientific review still required")
    lines += ["", "**Next:** Review actual original data, source methods, synchronized sample/time, measurement independence, confounds and uncertainties. The presence of a source file is not an evidence-of-mechanism decision."]
    return "\n".join(lines) + "\n"
