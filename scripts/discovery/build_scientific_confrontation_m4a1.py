"""M4-A1: offline, non-authoritative scientific confrontation review adapter.

Input is either SHA-verified M3-C1 *trajectory exports* or source HypothesisPortfolio
JSON files. A case spec is an explicit AI/human draft, NOT empirical evidence.
This script does not choose explanations, run models, or mutate input data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "m4a1-scientific-confrontation-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path}")
    return data


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("M4A1_INTEGRITY_FAILURE: " + message)


def nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def source_index(*, trajectories: Path | None, portfolios: list[str], expected_sha: str | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    require(bool(trajectories) != bool(portfolios), "select exactly one input mode")
    paths: dict[str, Path] = {}
    rows: dict[str, dict[str, Any]] = {}
    if trajectories:
        src = trajectories.expanduser().resolve()
        require(src.is_file(), "trajectory file not found")
        paths["trajectories"] = src
        if expected_sha:
            require(digest(src) == expected_sha, "trajectory SHA-256 mismatch")
        report = read_json(src)
        require(report.get("status") == "SHA_VERIFIED_SCIENTIFIC_DELTAS_READY_FOR_HUMAN_REVIEW", "M3C1 status invalid")
        require(report.get("authoritative_science_judgment") is False, "M3C1 must be non-authoritative")
        require(report.get("original_data_mutated") is False, "M3C1 reports source mutation")
        for item in report.get("rows", []):
            key = item.get("blind_id")
            require(nonblank(key) and key not in rows, "invalid/duplicate blind ID")
            require(nonblank(item.get("hypothesis_id")) and nonblank(item.get("arm")), f"invalid identity for {key}")
            require(item.get("lineage_validation") == "PASS_SOURCE_HASH_AND_KERNEL_TRAJECTORY", f"unverified trajectory {key}")
            steps = item.get("steps_earliest_to_latest", [])
            require(isinstance(steps, list) and bool(steps), f"empty trajectory {key}")
            require(steps[-1].get("idea_id") == item.get("terminal_idea_id"), f"terminal mismatch {key}")
            rows[key] = {
                "ref": key, "arm": item["arm"], "hypothesis_id": item["hypothesis_id"],
                "terminal_idea_id": item["terminal_idea_id"], "p0_root_idea_id": item.get("p0_root_idea_id"),
                "scientific_intent": steps[-1]["kernel"].get("canonical_intent", ""),
                "predictions": item.get("card_predictions_UNREVIEWED", []),
                "falsifiers": item.get("card_falsification_criteria_UNREVIEWED", []),
                "premise_ids": item.get("premise_statement_ids", []),
                "context_id": None,
            }
        require(len(rows) == 21, "expected all 21 M3-C1 trajectory rows")
        kind = "M3C1_TRAJECTORIES"
    else:
        for argument in portfolios:
            require("=" in argument, "portfolio input must use LABEL=/path")
            label, filename = argument.split("=", 1)
            require(nonblank(label) and label not in paths, "duplicate/empty portfolio label")
            src = Path(filename).expanduser().resolve()
            require(src.is_file(), f"portfolio not found: {src}")
            paths[label] = src
            # Reuse ForAllKG's strict hypothesis-card validation when available.
            from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
            portfolio = HypothesisPortfolio.model_validate(read_json(src))
            for card in portfolio.hypotheses:
                ref = f"{label}:{card.hypothesis_id}"
                require(ref not in rows, "duplicate hypothesis reference")
                rows[ref] = {
                    "ref": ref, "arm": label, "hypothesis_id": card.hypothesis_id,
                    "terminal_idea_id": None, "p0_root_idea_id": None,
                    "scientific_intent": card.hypothesis_statement,
                    "predictions": [p.model_dump(mode="json") for p in card.predicted_observations],
                    "falsifiers": [p.model_dump(mode="json") for p in card.falsification_criteria],
                    "premise_ids": list(card.premise_statement_ids), "context_id": card.source_context_id,
                }
        require(bool(rows), "empty portfolios")
        kind = "HYPOTHESIS_PORTFOLIOS"
    return rows, {"source_kind": kind, "source_sha256": {label: digest(path) for label, path in paths.items()},
                  "source_paths_PRIVATE": {label: str(path) for label, path in paths.items()},
                  "source_count": len(rows)}


def make_template(rows: dict[str, dict[str, Any]], refs: list[str]) -> dict[str, Any]:
    selected = refs or sorted(rows)
    require(len(set(selected)) == len(selected), "duplicate hypothesis refs")
    require(all(ref in rows for ref in selected), "unknown hypothesis ref in template")
    return {"schema_version": "m4a1-case-spec-v1", "authority": "AI_OR_HUMAN_DRAFT_NOT_CERTIFIED", "cases": [
        {"case_id": f"case:{ref}", "hypothesis_refs": [ref],
         "mechanisms": [{"role": "A", "explanation": "", "predicted_pattern": ""},
                        {"role": "B", "explanation": "", "predicted_pattern": ""}],
         "intervention": "", "matched_controls": [],
         "pair_relationship": "COMPETING_PARTIAL",
         "observables": [{"name": "", "role": "INDEPENDENT_DISCRIMINATOR", "measurement_method": "", "independent_of_sers_outcome": "UNKNOWN", "independence_basis": ""}],
         "decision_rule": "", "inconclusive_rule": "",
         "falsifier": {"target_role": "A", "outcome": "", "scope": "MECHANISM_IN_SPECIFIED_CONDITIONS",
                       "necessity_review": "UNREVIEWED", "alternatives_review": "UNREVIEWED"},
         "reviewer": "", "review_source": ""}
        for ref in selected]}


def _critique(case: dict[str, Any], records: list[dict[str, Any]], source_context_id: str | None) -> list[str]:
    """Run the existing lexical/structural critic as advisory only.

    Its MEASUREMENT_INDEPENDENCE_UNRESOLVED flag remains an instrumentation
    warning, even if the input spec claims an independent measurement.
    """
    ctxs = {r["context_id"] for r in records if r["context_id"]}
    if source_context_id:
        require(not ctxs or ctxs == {source_context_id}, "source context assertion disagrees with portfolio")
        ctxs.add(source_context_id)
    # For trajectories, the context ID is not exported. Do not fabricate it.
    if not ctxs:
        return ["LEGACY_CRITIC_SKIPPED_MISSING_CONTEXT_ID"]
    if len(ctxs) > 1:
        return ["LEGACY_CRITIC_SKIPPED_MULTIPLE_CONTEXTS"]
    from pipeline_core.discovery.higher_order_discriminating_experiments import (
        DiscriminatingExperimentCandidate, DiscriminatingPrediction,
    )
    from pipeline_core.discovery.higher_order_experiment_critic import critique_discriminating_experiment
    p = case["mechanisms"]
    source_arms = {"V31_FROZEN": 1, "V34_FROZEN": 2, "M3A_NEW": 3}
    if any(rec["arm"] not in source_arms for rec in records):
        return ["LEGACY_CRITIC_SKIPPED_UNMAPPED_SOURCE_ARM"]
    row = DiscriminatingExperimentCandidate(
        experiment_id=case["case_id"], tension_id="M4A1_EXPLICIT_CASE",
        pair_id=case["case_id"], tension_type="explicit_competing_mechanisms",
        experiment_mode="matched_modifier_intervention",
        requested_source=case["intervention"],
        requested_target=next(x["name"] for x in case["observables"] if x["role"] == "TARGET_OUTCOME"),
        intervention_or_sweep=case["intervention"],
        held_constant_conditions=case["matched_controls"],
        observables=[x["name"] for x in case["observables"]],
        predictions=[DiscriminatingPrediction(explanation_role=x["role"], explanation_id=f"{case['case_id']}:{x['role']}", predicted_pattern=x["predicted_pattern"]) for x in p],
        decision_rule=case["decision_rule"], inconclusive_rule=case["inconclusive_rule"],
        source_arm_indices=sorted({source_arms[rec["arm"]] for rec in records}),
        source_context_ids=sorted(ctxs),
        basis_premise_ids=sorted({p for rec in records for p in rec["premise_ids"]}),
    )
    codes = [issue.code for issue in critique_discriminating_experiment(row).issues]
    # The existing critic's BASIS_ALIGNMENT check is defined for graph relation
    # endpoints. The M4-A1 case is grounded in hypothesis text; it contains no
    # such basis relation, so a zero lexical alignment score is not evidence.
    if not row.basis_relation_texts and "BASIS_ALIGNMENT_UNRESOLVED" in codes:
        codes.remove("BASIS_ALIGNMENT_UNRESOLVED")
        codes.append("LEGACY_BASIS_ALIGNMENT_NOT_APPLICABLE_NO_GRAPH_RELATION")
    return sorted(set(codes))


def evaluate_case(case: dict[str, Any], rows: dict[str, dict[str, Any]], *, source_context_id: str | None) -> dict[str, Any]:
    require(isinstance(case, dict), "case must be an object")
    cid = case.get("case_id")
    require(nonblank(cid), "case_id required")
    refs = case.get("hypothesis_refs")
    require(isinstance(refs, list) and bool(refs) and len(refs) <= 4 and len(set(refs)) == len(refs), f"{cid}: 1-4 unique refs required")
    require(all(ref in rows for ref in refs), f"{cid}: reference not present in source")
    mechanisms = case.get("mechanisms")
    require(isinstance(mechanisms, list) and len(mechanisms) == 2 and {x.get("role") for x in mechanisms} == {"A", "B"}, f"{cid}: exactly A and B needed")
    for m in mechanisms:
        require(nonblank(m.get("explanation")) and nonblank(m.get("predicted_pattern")), f"{cid}: explanation/prediction missing")
    by_role = {x["role"]: x for x in mechanisms}
    require(by_role["A"]["predicted_pattern"].strip().casefold() != by_role["B"]["predicted_pattern"].strip().casefold(), f"{cid}: identical predictions")
    require(case.get("pair_relationship") in {"MUTUALLY_EXCLUSIVE", "COMPETING_PARTIAL", "ADJACENT_NOT_EXCLUSIVE"}, f"{cid}: unknown mechanism relationship")
    controls, observables = case.get("matched_controls"), case.get("observables")
    require(nonblank(case.get("intervention")) and nonblank(case.get("decision_rule")) and nonblank(case.get("inconclusive_rule")), f"{cid}: missing comparison plan")
    require(isinstance(controls, list) and controls and all(nonblank(x) for x in controls), f"{cid}: controls missing")
    require(isinstance(observables, list) and len(observables) >= 2, f"{cid}: >=2 observables required")
    names = [o.get("name", "") for o in observables]
    require(all(nonblank(name) for name in names) and len(set(x.strip().casefold() for x in names)) == len(names), f"{cid}: missing/duplicate observable")
    role_counts = Counter(o.get("role") for o in observables)
    require(role_counts["TARGET_OUTCOME"] >= 1 and role_counts["INDEPENDENT_DISCRIMINATOR"] >= 1, f"{cid}: requires target and independent discriminator")
    require(set(role_counts) <= {"TARGET_OUTCOME", "INDEPENDENT_DISCRIMINATOR", "CONTROL"}, f"{cid}: invalid observable role")
    for o in observables:
        require(nonblank(o.get("measurement_method")), f"{cid}: measurement method absent")
        require(o.get("independent_of_sers_outcome") in {"YES", "NO", "UNKNOWN"}, f"{cid}: independence enum invalid")
        if o["independent_of_sers_outcome"] == "YES":
            require(nonblank(o.get("independence_basis")), f"{cid}: claimed independent without witness")
    falsifier = case.get("falsifier")
    require(isinstance(falsifier, dict), f"{cid}: falsifier object missing")
    require(falsifier.get("target_role") in {"A", "B", "BOTH"}, f"{cid}: falsifier target invalid")
    require(nonblank(falsifier.get("outcome")), f"{cid}: falsifier outcome missing")
    require(falsifier.get("scope") in {"MECHANISM_IN_SPECIFIED_CONDITIONS", "WHOLE_RESEARCH_QUESTION"}, f"{cid}: invalid falsifier scope")
    require(falsifier.get("necessity_review") in {"UNREVIEWED", "SUPPORTED", "UNSUPPORTED"}, f"{cid}: necessity assessment invalid")
    require(falsifier.get("alternatives_review") in {"UNREVIEWED", "CONTROLLED", "UNCONTROLLED"}, f"{cid}: alternatives assessment invalid")
    records = [rows[ref] for ref in refs]
    context_ids = {rec["context_id"] for rec in records if rec["context_id"]}
    require(len(context_ids) <= 1, f"{cid}: mixed scientific contexts")
    if source_context_id:
        require(not context_ids or source_context_id in context_ids, f"{cid}: context mismatch")
    issues: list[str] = []
    if falsifier["scope"] == "WHOLE_RESEARCH_QUESTION":
        issues.append("INVALID_WHOLE_QUESTION_FALSIFIER_SCOPE")
    if falsifier["necessity_review"] == "UNSUPPORTED":
        issues.append("FALSIFIER_NOT_NECESSARY_FOR_TARGET")
    elif falsifier["necessity_review"] == "UNREVIEWED":
        issues.append("FALSIFIER_NECESSITY_NOT_REVIEWED")
    if falsifier["alternatives_review"] != "CONTROLLED":
        issues.append("ALTERNATIVE_MECHANISMS_NOT_ESTABLISHED_CONTROLLED")
    discriminators = [o for o in observables if o["role"] == "INDEPENDENT_DISCRIMINATOR"]
    if case["pair_relationship"] == "ADJACENT_NOT_EXCLUSIVE":
        issues.append("NONEXCLUSIVE_MECHANISMS_REQUIRE_JOINT_OR_ADDITIVE_TEST")
    if any(o["independent_of_sers_outcome"] == "NO" for o in discriminators):
        issues.append("CIRCULAR_OUTCOME_DERIVED_MEASUREMENT")
    if any(o["independent_of_sers_outcome"] == "UNKNOWN" for o in discriminators):
        issues.append("MEASUREMENT_INDEPENDENCE_NOT_ESTABLISHED")
    # A common, scientifically meaningful but non-fatal uncertainty.
    if not nonblank(case.get("reviewer")) or not nonblank(case.get("review_source")):
        issues.append("INDEPENDENT_EXPERT_REVIEW_ABSENT")
    critic_issues = _critique(case, records, source_context_id)
    hard = {"INVALID_WHOLE_QUESTION_FALSIFIER_SCOPE", "FALSIFIER_NOT_NECESSARY_FOR_TARGET", "CIRCULAR_OUTCOME_DERIVED_MEASUREMENT"}
    return {
        "case_id": cid, "hypothesis_refs": refs,
        "input_hypotheses": [{"ref": r["ref"], "arm": r["arm"], "hypothesis_id": r["hypothesis_id"], "terminal_idea_id": r["terminal_idea_id"], "scientific_intent": r["scientific_intent"], "source_predictions_UNREVIEWED": r["predictions"], "source_falsifiers_UNREVIEWED": r["falsifiers"]} for r in records],
        "proposed_mechanisms_UNREVIEWED": mechanisms, "proposed_experiment_UNREVIEWED": {k: case[k] for k in ("pair_relationship", "intervention", "matched_controls", "observables", "decision_rule", "inconclusive_rule", "falsifier")},
        "issues": sorted(set(issues)), "existing_experiment_critic_diagnostic_codes": sorted(set(critic_issues)),
        "status": "BLOCKED_LOGIC_OR_MEASUREMENT" if hard.intersection(issues) else "REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED",
        "confrontation_validated": False, "experiment_physically_feasible_certified": False,
        "falsification_logic_certified": False, "novelty_or_truth_authority": False,
    }


def execute(*, trajectories: Path | None, portfolios: list[str], cases: Path | None, output_dir: Path,
            expected_sha: str | None = None, source_context_id: str | None = None,
            select_refs: list[str] | None = None) -> dict[str, Any]:
    rows, sources = source_index(trajectories=trajectories, portfolios=portfolios, expected_sha=expected_sha)
    if cases:
        spec_path = cases.expanduser().resolve()
        require(spec_path.is_file(), "case spec not found")
        spec = read_json(spec_path)
        require(spec.get("schema_version") == "m4a1-case-spec-v1", "case spec schema invalid")
        require(spec.get("authority") == "AI_OR_HUMAN_DRAFT_NOT_CERTIFIED", "case spec must have no science authority")
        require(isinstance(spec.get("cases"), list) and bool(spec["cases"]), "no cases")
        names = [x.get("case_id") for x in spec["cases"]]
        require(len(set(names)) == len(names), "duplicate case ID")
        evaluated = [evaluate_case(x, rows, source_context_id=source_context_id) for x in spec["cases"]]
        template = None
    else:
        evaluated = []
        template = make_template(rows, select_refs or [])
    report = {"schema_version": SCHEMA, "status": "DRAFT_CONFRONTATIONS_NEED_REVIEW" if cases else "TEMPLATE_READY_UNREVIEWED",
              "source": sources, "source_context_id_ASSERTED": source_context_id,
              "input_case_spec_sha256": digest(cases.expanduser().resolve()) if cases else None,
              "cases": evaluated, "case_count": len(evaluated), "case_status_counts": dict(Counter(c["status"] for c in evaluated)),
              "source_files_mutated": False, "llm_or_network_calls": 0,
              "human_or_expert_science_certification": False, "production_selection_changed": False,
              "external_evidence_validated": False, "hypothesis_cards_modified": False}
    out = output_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    require(out != repo and repo not in out.parents, "output directory must stay outside Git repository")
    require(not out.exists(), "output directory already exists")
    for path in sources["source_paths_PRIVATE"].values():
        require(not (out == Path(path) or out in Path(path).parents), "output overlaps source")
    out.mkdir(parents=True, exist_ok=False)
    (out / "M4A1_CONFRONTATION_REPORT_PRIVATE.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if template is not None:
        (out / "M4A1_CASE_SPEC_TEMPLATE_PRIVATE.json").write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# M4-A1 — Scientific Confrontation (PRIVATE, NON-AUTHORITATIVE)", "", f"Status: {report['status']}",
             f"Source items: {sources['source_count']}", f"Cases evaluated: {len(evaluated)}", "",
             "No mechanism is validated by lexical similarity or a completed draft.\n"]
    for c in evaluated:
        lines.extend([f"## {c['case_id']}", f"Status: {c['status']}",
                      "Issues: " + (", ".join(c["issues"]) or "none from explicit contract"),
                      "Legacy critic diagnostics: " + (", ".join(c["existing_experiment_critic_diagnostic_codes"]) or "none"), ""])
    (out / "M4A1_REPORT_PRIVATE.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--trajectories", type=Path)
    inputs.add_argument("--portfolio", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--cases", type=Path, help="Explicit draft confrontation cases (omit to build template)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-trajectories-sha256")
    p.add_argument("--source-context-id", help="Optional external assertion; MUST match portfolio when present")
    p.add_argument("--select-ref", action="append", default=[], help="Template-only filter, e.g., B12")
    a = p.parse_args()
    try:
        result = execute(trajectories=a.trajectories, portfolios=a.portfolio, cases=a.cases,
                         output_dir=a.output_dir, expected_sha=a.expected_trajectories_sha256,
                         source_context_id=a.source_context_id, select_refs=a.select_ref)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        p.exit(2, f"M4A1 STOPPED (no source modified): {e}\n")
    print(f"M4A1: {result['status']} — {result['case_count']} cases; science certification=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
