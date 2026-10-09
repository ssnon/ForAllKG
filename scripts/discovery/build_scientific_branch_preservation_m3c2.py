"""M3-C2 read-only, human-review-required scientific branch preservation gate.

This tool accepts PRIVATE post-unblinding audit inputs. It never mutates source
files, executes models/retrieval, materializes hypotheses, or grants scientific
claim/deletion authority. Outputs MUST be kept outside the Git repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ARMS = frozenset({"V31_FROZEN", "V34_FROZEN", "M3A_NEW"})
DECISIONS = frozenset({
    "PRESERVE_CONTRASTIVE_PROGRAM", "GROUP_FOR_CONTRASTIVE_REVIEW",
    "KEEP_BASELINE_REFERENCE", "PRESERVE_WITH_FALSIFIER_REPAIR",
    "KEEP_PARENT_AND_CHILD", "PRESERVE_PARENT_PROGRAM",
})
PROTECTED = frozenset({
    "PRESERVE_CONTRASTIVE_PROGRAM", "GROUP_FOR_CONTRASTIVE_REVIEW",
    "PRESERVE_WITH_FALSIFIER_REPAIR", "KEEP_PARENT_AND_CHILD",
    "PRESERVE_PARENT_PROGRAM",
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return obj


def checked(condition: bool, msg: str) -> None:
    if not condition:
        raise ValueError("M3C2_INTEGRITY_FAILURE: " + msg)


def _unique_by(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        checked(isinstance(row, dict), f"invalid row in {label}")
        val = row.get(key)
        checked(isinstance(val, str) and bool(val), f"missing {key} in {label}")
        checked(val not in indexed, f"duplicate {key}={val} in {label}")
        indexed[val] = row
    return indexed


def validate_inputs(
    *,
    lineage: dict[str, Any], trajectory: dict[str, Any], review_rows: list[dict[str, str]],
    integrity: dict[str, Any], source_hashes: dict[str, Any],
    lineage_path: Path, source_hashes_path: Path, expected_commit: str | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, str]], str]:
    checked(lineage.get("status") == "M3C0_STRUCTURAL_LINEAGE_AUDIT_PASS_NO_SCIENCE_CERTIFICATION", "M3C0 status")
    checked(trajectory.get("status") == "SHA_VERIFIED_SCIENTIFIC_DELTAS_READY_FOR_HUMAN_REVIEW", "M3C1 status")
    checked(trajectory.get("authoritative_science_judgment") is False, "trajectory must have no science authority")
    checked(trajectory.get("original_data_mutated") is False, "source data reported mutated")
    checked(trajectory.get("root_set_conserved_across_arms") is True, "P0 roots not conserved")
    checked(integrity.get("original_data_mutated") is False, "M3C1 integrity reports mutation")
    checked(integrity.get("input_m3c0_audit_sha256") == sha256(lineage_path), "M3C0 audit SHA mismatch")
    checked(integrity.get("input_m3c0_manifest_sha256") == sha256(source_hashes_path), "M3C0 input manifest SHA mismatch")
    commit = trajectory.get("commit")
    checked(isinstance(commit, str) and commit == lineage.get("code_commit"), "commit mismatch")
    if expected_commit is not None:
        checked(commit == expected_commit, "unexpected HEAD")
    arm_rows = _unique_by(lineage.get("arms", []), "arm", "M3C0 arms")
    checked(set(arm_rows) == ARMS, "expected exactly three arms")
    base_p0 = None
    base_roots = None
    for name, row in arm_rows.items():
        checked(row.get("terminal_hypothesis_count") == 7, f"{name} terminal count != 7")
        checked(row.get("p0_research_idea_ids") and len(row["p0_research_idea_ids"]) == 8, f"{name} P0 count")
        p0_ids = set(row["p0_research_idea_ids"])
        checked(len(p0_ids) == 8, f"{name} duplicate P0 IDs")
        checked(base_p0 is None or base_p0 == p0_ids, f"{name} nonidentical P0")
        base_p0 = p0_ids
        checked(source_hashes.get(f"{name}.final_portfolio") == row.get("final_sha256"), f"{name} portfolio SHA mismatch")
        items = _unique_by(row.get("items", []), "hypothesis_id", f"{name} M3C0 hypotheses")
        checked(len(items) == 7, f"{name} item count != 7")
        roots = set()
        for item in items.values():
            r = item.get("p0_root_research_idea_ids", [])
            checked(len(r) == 1 and r[0] in p0_ids, f"{name} invalid P0 root")
            checked(item.get("lineage_validation") == "PASS_STRUCTURAL_ONLY", f"{name} invalid M3C0 item")
            roots.add(r[0])
        checked(len(roots) == 7, f"{name} duplicate terminal roots")
        checked(base_roots is None or roots == base_roots, f"{name} divergent terminal root sets")
        base_roots = roots
    trajectories = _unique_by(trajectory.get("rows", []), "blind_id", "M3C1 trajectories")
    reviews = _unique_by(review_rows, "blind_id", "M3C1 review")
    checked(len(trajectories) == 21 and set(reviews) == set(trajectories), "must match all 21 review/trajectory IDs")
    seen_terminals: set[tuple[str, str]] = set()
    for blind_id, tr in trajectories.items():
        arm = tr.get("arm")
        checked(arm in ARMS, f"{blind_id} arm invalid")
        row = reviews[blind_id]
        checked(row.get("arm") == arm, f"{blind_id} review arm mismatch")
        checked(row.get("p0_root_id") == tr.get("p0_root_idea_id"), f"{blind_id} review root mismatch")
        checked(row.get("terminal_idea_id") == tr.get("terminal_idea_id"), f"{blind_id} review idea mismatch")
        checked(str(tr.get("terminal_birth")) == str(row.get("terminal_birth")), f"{blind_id} birth mismatch")
        decision = row.get("capability_preservation_decision", "")
        checked(decision in DECISIONS, f"{blind_id} review decision unknown")
        checked(tr.get("lineage_validation") == "PASS_SOURCE_HASH_AND_KERNEL_TRAJECTORY", f"{blind_id} trajectory unverified")
        steps = tr.get("steps_earliest_to_latest", [])
        checked(bool(steps) and steps[0].get("idea_id") == tr.get("p0_root_idea_id") and steps[-1].get("idea_id") == tr.get("terminal_idea_id"), f"{blind_id} broken steps")
        checked(len({x.get("idea_id") for x in steps}) == len(steps), f"{blind_id} duplicated step ID")
        arm_item = next((x for x in arm_rows[arm]["items"] if x.get("blind_id") == blind_id), None)
        checked(arm_item is not None, f"{blind_id} not in M3C0 arm")
        checked(arm_item.get("hypothesis_id") == tr.get("hypothesis_id") and arm_item.get("terminal_research_idea_id") == tr.get("terminal_idea_id"), f"{blind_id} M3C0 / M3C1 terminal discrepancy")
        checked(arm_item.get("p0_root_research_idea_ids") == [tr.get("p0_root_idea_id")], f"{blind_id} M3C0 root discrepancy")
        key = (arm, tr["terminal_idea_id"])
        checked(key not in seen_terminals, f"{blind_id} duplicate terminal in arm")
        seen_terminals.add(key)
    return arm_rows, trajectories, reviews, commit


def create_plan(
    *, arm_rows: dict[str, dict[str, Any]], trajectories: dict[str, dict[str, Any]],
    reviews: dict[str, dict[str, str]], target_arm: str, commit: str,
    file_hashes: dict[str, str],
) -> dict[str, Any]:
    checked(target_arm in ARMS, "invalid target arm")
    target = arm_rows[target_arm]
    target_items = [t for t in trajectories.values() if t["arm"] == target_arm]
    target_by_root = {r["p0_root_idea_id"]: r for r in target_items}
    realized = {r["terminal_idea_id"] for r in target_items}
    active = set(target["final_active_research_idea_ids"])
    historic = set(target["historical_only_idea_ids"])
    checked(not (active & historic), "active and historical ResearchIdea IDs overlap")
    proposals = []
    for key in sorted(trajectories):
        tr = trajectories[key]
        review = reviews[key]
        terminal = tr["terminal_idea_id"]
        root = tr["p0_root_idea_id"]
        if terminal in realized:
            reachability = "EXACT_IDEA_TERMINAL_REALIZED"
        elif terminal in active:
            reachability = "ACTIVE_IDEA_NOT_TERMINAL_REALIZED"
        elif terminal in historic:
            reachability = "HISTORICAL_IDEA_NOT_TERMINAL_REALIZED"
        else:
            reachability = "FROZEN_SOURCE_ARM_ONLY"
        decision = review["capability_preservation_decision"]
        if decision in PROTECTED and reachability == "HISTORICAL_IDEA_NOT_TERMINAL_REALIZED":
            action = "REVIEW_ARCHIVED_IDEA_THEN_OPTIONAL_STRICT_RE_REALIZATION"
        elif decision in PROTECTED and reachability == "ACTIVE_IDEA_NOT_TERMINAL_REALIZED":
            action = "REVIEW_ACTIVE_IDEA_THEN_OPTIONAL_STRICT_REALIZATION"
        elif decision in PROTECTED and reachability == "FROZEN_SOURCE_ARM_ONLY":
            action = "PRESERVE_FROZEN_ARM_NO_AUTOMATIC_IMPORT"
        elif decision in PROTECTED:
            action = "RETAIN_TERMINAL_OPTION_AND_REVIEW_SCIENCE"
        else:
            action = "RETAIN_FROZEN_BASELINE_REFERENCE"
        paired = target_by_root.get(root)
        checked(paired is not None, f"missing target root {root}")
        proposals.append({
            "blind_id": key, "source_arm": tr["arm"], "target_arm": target_arm,
            "p0_root_idea_id": root, "source_terminal_idea_id": terminal,
            "target_terminal_idea_id_from_same_p0": paired["terminal_idea_id"],
            "same_p0_does_not_imply_program_equivalence": True,
            "decision_from_AI_draft_NOT_HUMAN_APPROVED": decision,
            "target_reachability": reachability, "proposed_preservation_action": action,
            "scientific_review_REQUIRED": decision in PROTECTED,
            "falsifier_logic_from_AI_draft_UNVALIDATED": review["falsifier_logical_validity"],
            "independent_observables_from_AI_draft_UNVALIDATED": review["decisive_experiment_and_independent_observables"],
            "human_review_status": "UNREVIEWED",
            "hypothesis_id": tr["hypothesis_id"],
        })
    counts = dict(sorted(Counter(x["target_reachability"] for x in proposals).items()))
    actions = dict(sorted(Counter(x["proposed_preservation_action"] for x in proposals).items()))
    unresolved = [x for x in proposals if x["scientific_review_REQUIRED"] and x["target_reachability"] != "EXACT_IDEA_TERMINAL_REALIZED"]
    return {
        "schema_version": "m3c2-scientific-branch-preservation-shadow-v1",
        "status": "M3C2_STRUCTURAL_PLAN_READY_HUMAN_SCIENCE_REVIEW_REQUIRED",
        "source_commit": commit, "target_arm": target_arm,
        "lineage_and_review_rows": len(proposals),
        "frozen_p0_ids": list(target["p0_research_idea_ids"]),
        "target_active_idea_count": len(active), "target_historical_only_idea_count": len(historic),
        "source_input_sha256": file_hashes,
        "reachability_counts": counts, "proposed_action_counts": actions,
        "unresolved_protected_count": len(unresolved),
        "unresolved_protected_blind_ids_PRIVATE": [x["blind_id"] for x in unresolved],
        "proposals_PRIVATE": proposals,
        "scientific_novelty_certified": False,
        "scientific_truth_certified": False,
        "strict_grounding_or_falsifier_validation_executed": False,
        "new_llm_calls": 0, "new_network_or_provider_calls": 0,
        "historical_archive_is_not_runnable_hypothesis": True,
        "archive_rescue_needs_fresh_strict_realization_and_common_verification": True,
        "cross_arm_hypothesis_import_executed": False,
        "production_path_switched": False,
        "source_modules_deleted_or_modified": False,
        "simplification_authorized": False,
        "source_lane_causal_necessity_proven": False,
        "all_preservation_decisions_AI_DRAFT_NOT_EXPERT_CERTIFIED": True,
    }


def _render(plan: dict[str, Any]) -> str:
    p = [
        "# M3-C2 Scientific Branch Preservation — PRIVATE / REVIEW REQUIRED", "",
        f"- Status: `{plan['status']}`",
        f"- Commit: `{plan['source_commit']}`",
        f"- Candidate opt-in arm: `{plan['target_arm']}` (legacy defaults unchanged)",
        f"- Protected missing/nonterminal scientific directions: **{plan['unresolved_protected_count']}**",
        "- AI postlock decisions are proposals, **not human adjudications**.",
        "- No scientific novelty/truth/falsifier certification, no new API calls, no source modification.",
        "- Historical ideas are **not** runnable HypothesisCards. M2-F rescue is a separate, fresh strict-realization+verification path.",
        "- Do not delete Stage7/HO/SIS, overwrite frozen arms, or promote prior-art to positive evidence.", "",
        "| Blind ID | Original arm | Source branch location in candidate | Action |", "|---|---|---|---|",
    ]
    for x in plan["proposals_PRIVATE"]:
        p.append(f"| {x['blind_id']} | {x['source_arm']} | {x['target_reachability']} | {x['proposed_preservation_action']} |")
    p += ["", "## Executable boundary", "",
          "Opt-in entrypoint remains `scripts/discovery/run_minimal_ai_scientist_m3a.py`.",
          "The v3.1/v3.2/v3.4 comparison wrappers remain frozen and callable; default production routing is unchanged.",
          "This script only **reports** scientific branches to protect; it never injects a ResearchIdea into a population.",
          "No deletion/refactor approval until reviewed branch coverage and matched multi-case ablations.", ""]
    return "\n".join(p)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "m3c0_lineage": args.lineage,
        "m3c0_source_hashes": args.source_hashes,
        "m3c1_trajectories": args.trajectories,
        "m3c1_integrity": args.integrity,
        "m3c1_ai_postlock_csv": args.review_csv,
        "m3c1_ai_postlock_manifest": args.review_manifest,
    }
    dest = args.output_dir.expanduser().resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if dest == repository_root or repository_root in dest.parents:
        raise ValueError("private audit outputs must stay OUTSIDE the repository")
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite output: {dest}")
    resolved = {name: path.expanduser().resolve() for name, path in inputs.items()}
    for name, path in resolved.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
        if path == dest or dest in path.parents:
            raise ValueError("output must not contain source inputs")
    postlock_manifest = load_json(resolved["m3c1_ai_postlock_manifest"])
    checked(postlock_manifest.get("status") == "POSTLOCK_AI_DRAFT_NOT_EXPERT_CERTIFIED", "postlock review manifest status")
    checked(postlock_manifest.get("row_count") == 21, "postlock review manifest row count")
    checked(postlock_manifest.get("output_files_sha256", {}).get("M3C1_REVIEW_SHEET_AI_POSTLOCK_DRAFT_PRIVATE.csv") == sha256(resolved["m3c1_ai_postlock_csv"]), "postlock review CSV SHA mismatch")
    checked(postlock_manifest.get("input_files_sha256", {}).get("M3C1_KERNEL_TRAJECTORIES_PRIVATE.json") == sha256(resolved["m3c1_trajectories"]), "trajectory SHA does not match postlock review")
    checked(postlock_manifest.get("input_files_sha256", {}).get("M3C1_INPUT_INTEGRITY_PRIVATE.json") == sha256(resolved["m3c1_integrity"]), "integrity SHA does not match postlock review")
    lineage, trajectory, integrity, source_hashes = (
        load_json(resolved["m3c0_lineage"]), load_json(resolved["m3c1_trajectories"]),
        load_json(resolved["m3c1_integrity"]), load_json(resolved["m3c0_source_hashes"]),
    )
    with resolved["m3c1_ai_postlock_csv"].open(newline="", encoding="utf-8-sig") as f:
        review_rows = list(csv.DictReader(f))
    arm_rows, trajectories, reviews, commit = validate_inputs(
        lineage=lineage, trajectory=trajectory, review_rows=review_rows,
        integrity=integrity, source_hashes=source_hashes,
        lineage_path=resolved["m3c0_lineage"], source_hashes_path=resolved["m3c0_source_hashes"],
        expected_commit=args.expected_commit,
    )
    checked(postlock_manifest.get("commit") == commit, "review commit mismatch")
    plan = create_plan(
        arm_rows=arm_rows, trajectories=trajectories, reviews=reviews,
        target_arm=args.target_arm, commit=commit,
        file_hashes={name: sha256(path) for name, path in resolved.items()},
    )
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "M3C2_BRANCH_PLAN_PRIVATE.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dest / "M3C2_REPORT_PRIVATE.md").write_text(_render(plan), encoding="utf-8")
    reviewer_template = {
        "schema_version": "m3c2-human-branch-review-template-v1",
        "source_plan_sha256": sha256(dest / "M3C2_BRANCH_PLAN_PRIVATE.json"),
        "status": "UNREVIEWED_NO_AUTHORITY",
        "decisions": [
            {"blind_id": x["blind_id"], "keep_as_separate_research_option": None,
             "scientific_reason": "", "independent_discriminator_validated": None,
             "falsifier_logic_validated": None, "reviewer": ""}
            for x in plan["proposals_PRIVATE"] if x["scientific_review_REQUIRED"]
        ],
        "no_automatic_approval_from_template": True,
    }
    (dest / "M3C2_HUMAN_REVIEW_TEMPLATE_PRIVATE.json").write_text(json.dumps(reviewer_template, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lineage", type=Path, required=True)
    ap.add_argument("--source-hashes", type=Path, required=True)
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--integrity", type=Path, required=True)
    ap.add_argument("--review-csv", type=Path, required=True)
    ap.add_argument("--review-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--target-arm", choices=sorted(ARMS), default="M3A_NEW")
    ap.add_argument("--expected-commit", default=None)
    args = ap.parse_args()
    try:
        p = execute(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        ap.exit(2, f"M3C2 AUDIT STOPPED (source unmodified): {exc}\n")
    print("M3C2:", p["status"])
    print("unresolved protected directions:", p["unresolved_protected_count"])
    print("private plan:", args.output_dir / "M3C2_BRANCH_PLAN_PRIVATE.json")
    print("deletion/simplification authorized: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
