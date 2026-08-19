from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

SPEC_PATH = ROOT / "dac_her/sers_i0_integrated_orchestration_spec_v1.json"

R2_RUN_ROOT = ROOT / "evaluation/sers_novelty_gap/r2_final_reassessment_run_v1"
R2_REPORT_PATH = R2_RUN_ROOT / "r2_report.json"
R2_COMPLETE_PATH = R2_RUN_ROOT / "R2_COMPLETE.json"
R2_FREEZE_ROOT = ROOT / "evaluation/sers_novelty_gap/r2_final_reassessment_freeze_v1"
R2_FREEZE_MANIFEST = R2_FREEZE_ROOT / "freeze_manifest.json"
R2_FREEZE_READY = R2_FREEZE_ROOT / "FREEZE_READY.json"

R0_ADJ_PATH = ROOT / "evaluation/sers_novelty_gap/r0_manual_scientific_adjudication_v1/adjudication.json"
R0_FREEZE_ROOT = ROOT / "evaluation/sers_novelty_gap/r0_manual_scientific_adjudication_freeze_v1"
R0_FREEZE_MANIFEST = R0_FREEZE_ROOT / "freeze_manifest.json"
R0_FREEZE_READY = R0_FREEZE_ROOT / "FREEZE_READY.json"

T1_FREEZE_MANIFEST = ROOT / "evaluation/sers_novelty_gap/t1_live_targeted_retrieval_freeze_v2/freeze_manifest.json"
T0_FREEZE_MANIFEST = ROOT / "evaluation/sers_novelty_gap/t0_targeted_retrieval_canonicalization_freeze_v2/freeze_manifest.json"
GAP_PLAN_PATH = ROOT / "evaluation/sers_novelty_gap/t1_frozen_input_bundle_v1/novelty_gap_plan.json"

OUTPUT_ROOT = ROOT / "evaluation/sers_novelty_gap/i0_integrated_orchestration_run_v1"
HANDOFF_PATH = OUTPUT_ROOT / "i0_handoff.json"
COMPLETE_PATH = OUTPUT_ROOT / "I0_COMPLETE.json"

RUN_PATH = ROOT / "campaigns/sers_alpha4_epoch/post_t1/cli/run_sers_i0_integrated_orchestration_v1.py"
VERIFY_PATH = ROOT / "campaigns/sers_alpha4_epoch/post_t1/cli/verify_sers_i0_integrated_orchestration_v1.py"
FREEZE_CREATE_PATH = ROOT / "campaigns/sers_alpha4_epoch/post_t1/cli/freeze_sers_i0_integrated_orchestration_v1.py"
FREEZE_VERIFY_PATH = ROOT / "campaigns/sers_alpha4_epoch/post_t1/cli/verify_sers_i0_integrated_orchestration_freeze_v1.py"
TEST_PATH = ROOT / "tests/test_sers_i0_integrated_orchestration_v1.py"

ALLOWED_DISPOSITIONS = {
    "KEEP_BOUNDED_EXTENSION",
    "REJECT_AS_FORMULATED",
    "KEEP_RELATIONAL_GAP_CANDIDATE",
}


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_bytes_at(ref: str, relpath: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{relpath}"], cwd=ROOT)


def tracked_at(ref: str, relpath: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{relpath}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def is_ancestor(ancestor: str, descendant: str = "HEAD") -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_spec() -> dict[str, Any]:
    spec = load_json(SPEC_PATH)
    payload = dict(spec)
    spec_id = payload.pop("spec_id", None)
    spec_sha = payload.pop("spec_sha256", None)
    recomputed = sha256_bytes(canonical(payload).encode("utf-8"))
    expected_id = "sers_i0_integrated_orchestration_spec_v1:" + recomputed[:20]
    if spec_sha != recomputed or spec_id != expected_id:
        raise ValueError("I0 spec hash/ID mismatch")
    return spec


def _validate_hashed_object(
    value: dict[str, Any],
    *,
    id_key: str,
    sha_key: str,
    id_prefix: str,
    label: str,
) -> None:
    payload = dict(value)
    object_id = payload.pop(id_key, None)
    object_sha = payload.pop(sha_key, None)
    recomputed = sha256_bytes(canonical(payload).encode("utf-8"))
    if object_sha != recomputed:
        raise ValueError(f"{label} SHA mismatch")
    if object_id != id_prefix + recomputed[:20]:
        raise ValueError(f"{label} ID mismatch")


def validate_r2_portfolio(report: dict[str, Any]) -> None:
    decisions = report.get("hypothesis_decisions")
    portfolio = report.get("portfolio_decision")
    guards = report.get("epistemic_guards")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("R2 hypothesis decisions missing")
    if not isinstance(portfolio, dict):
        raise ValueError("R2 portfolio decision missing")
    if not isinstance(guards, dict):
        raise ValueError("R2 epistemic guards missing")

    rows: dict[str, dict[str, Any]] = {}
    for row in decisions:
        hypothesis_id = row.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            raise ValueError("R2 decision missing hypothesis_id")
        if hypothesis_id in rows:
            raise ValueError("duplicate R2 hypothesis decision")
        if row.get("candidate_disposition") not in ALLOWED_DISPOSITIONS:
            raise ValueError("unknown R2 candidate disposition")
        if row.get("hypothesis_rewrite_performed") is not False:
            raise ValueError("R2 decision records hypothesis rewrite")
        rows[hypothesis_id] = row

    primary = portfolio.get("primary_remaining_candidate_hypothesis_id")
    secondary = portfolio.get("secondary_bounded_extension_hypothesis_id")
    rejected = portfolio.get("rejected_as_formulated_hypothesis_ids")
    if primary not in rows:
        raise ValueError("R2 primary candidate absent from decisions")
    if secondary not in rows:
        raise ValueError("R2 secondary bounded extension absent from decisions")
    if not isinstance(rejected, list):
        raise ValueError("R2 rejected set malformed")
    if primary == secondary or primary in rejected or secondary in rejected:
        raise ValueError("R2 portfolio roles overlap")
    if rows[primary].get("candidate_disposition") != "KEEP_RELATIONAL_GAP_CANDIDATE":
        raise ValueError("R2 primary candidate disposition mismatch")
    if rows[secondary].get("candidate_disposition") != "KEEP_BOUNDED_EXTENSION":
        raise ValueError("R2 secondary candidate disposition mismatch")
    rejected_from_rows = sorted(
        hid
        for hid, row in rows.items()
        if row.get("candidate_disposition") == "REJECT_AS_FORMULATED"
    )
    if sorted(rejected) != rejected_from_rows:
        raise ValueError("R2 rejected set does not match decision rows")

    required_portfolio_false = [
        "r1_executed",
        "fresh_reserve_c_consumed",
        "fresh_reserve_c_authorized",
        "automatic_next_stage_authorized",
        "i0_started",
    ]
    for key in required_portfolio_false:
        if portfolio.get(key) is not False:
            raise ValueError(f"R2 portfolio guard changed:{key}")
    if portfolio.get("hypothesis_rewrites") != 0:
        raise ValueError("R2 portfolio records hypothesis rewrite")
    if portfolio.get("stop_after_r2_freeze") is not True:
        raise ValueError("R2 portfolio STOP boundary missing")
    if portfolio.get("ranking_is_scientific_review_not_empirical_validation") is not True:
        raise ValueError("R2 empirical-validation disclaimer missing")

    required_guard_false = [
        "r1_executed",
        "hypothesis_rewrite_called",
        "external_prior_art_used_as_positive_premise",
        "literature_absence_claimed",
        "i0_started",
        "fresh_reserve_c_consumed",
        "fresh_reserve_c_authorized",
        "automatic_next_stage_authorized",
    ]
    for key in required_guard_false:
        if guards.get(key) is not False:
            raise ValueError(f"R2 epistemic guard changed:{key}")
    if guards.get("runtime_llm_calls") != 0:
        raise ValueError("R2 runtime LLM calls changed")
    if guards.get("runtime_network_calls") != 0:
        raise ValueError("R2 runtime network calls changed")
    if guards.get("stop_after_r2") is not True:
        raise ValueError("R2 runtime STOP boundary missing")


def _validate_exact_tracked_file(path: Path, ref: str, label: str) -> None:
    rp = rel(path)
    if not tracked_at(ref, rp):
        raise ValueError(f"{label} is not tracked at {ref}:{rp}")
    if git_bytes_at(ref, rp) != path.read_bytes():
        raise ValueError(f"{label} working file differs from {ref}:{rp}")


def _validate_r2_lineage(spec: dict[str, Any], head: str) -> dict[str, Any]:
    contract = spec["source_contract"]
    required_commit = contract["required_r2_freeze_commit"]
    if not is_ancestor(required_commit, head):
        raise ValueError("required R2 freeze commit is not an ancestor of HEAD")
    for path in [R2_FREEZE_MANIFEST, R2_FREEZE_READY]:
        _validate_exact_tracked_file(path, required_commit, "R2 frozen artifact")

    manifest = load_json(R2_FREEZE_MANIFEST)
    ready = load_json(R2_FREEZE_READY)
    _validate_hashed_object(
        manifest,
        id_key="freeze_id",
        sha_key="manifest_sha256",
        id_prefix="sers_r2_final_reassessment_freeze_v1:",
        label="R2 freeze manifest",
    )
    if manifest.get("freeze_id") != contract["required_r2_freeze_id"]:
        raise ValueError("unexpected R2 freeze ID")
    if manifest.get("manifest_sha256") != contract["required_r2_manifest_sha256"]:
        raise ValueError("unexpected R2 freeze manifest SHA")
    if manifest.get("source_r2_report_commit") != contract["required_r2_report_commit"]:
        raise ValueError("unexpected R2 source report commit")
    if ready.get("ready") is not True or ready.get("stop") is not True:
        raise ValueError("R2 FREEZE_READY boundary invalid")
    if ready.get("freeze_id") != manifest.get("freeze_id"):
        raise ValueError("R2 FREEZE_READY ID mismatch")
    if ready.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("R2 FREEZE_READY SHA mismatch")
    if ready.get("source_r2_report_commit") != manifest.get("source_r2_report_commit"):
        raise ValueError("R2 FREEZE_READY source commit mismatch")

    for key in [
        "r1_executed",
        "hypothesis_rewrite_called",
        "i0_started",
        "fresh_reserve_c_consumed",
        "fresh_reserve_c_authorized",
        "automatic_next_stage_authorized",
    ]:
        if manifest.get(key) is not False:
            raise ValueError(f"R2 freeze guard changed:{key}")
    if manifest.get("r2_complete") is not True:
        raise ValueError("R2 freeze completion flag missing")
    if manifest.get("runtime_llm_calls") != 0 or manifest.get("runtime_network_calls") != 0:
        raise ValueError("R2 runtime call counts changed")
    if manifest.get("stop_after_freeze") is not True:
        raise ValueError("R2 freeze STOP boundary missing")

    report_commit = contract["required_r2_report_commit"]
    if not is_ancestor(report_commit, head):
        raise ValueError("required R2 report commit is not an ancestor of HEAD")
    for path in [R2_REPORT_PATH, R2_COMPLETE_PATH]:
        _validate_exact_tracked_file(path, report_commit, "R2 report artifact")

    report = load_json(R2_REPORT_PATH)
    complete = load_json(R2_COMPLETE_PATH)
    _validate_hashed_object(
        report,
        id_key="report_id",
        sha_key="report_sha256",
        id_prefix="sers_r2_final_reassessment_report_v1:",
        label="R2 report",
    )
    if report.get("report_id") != contract["required_r2_report_id"]:
        raise ValueError("unexpected R2 report ID")
    if report.get("report_sha256") != contract["required_r2_report_sha256"]:
        raise ValueError("unexpected R2 report SHA")
    if complete.get("complete") is not True or complete.get("stop") is not True:
        raise ValueError("R2 completion marker invalid")
    if complete.get("report_id") != report.get("report_id"):
        raise ValueError("R2 completion marker report ID mismatch")
    if complete.get("report_sha256") != report.get("report_sha256"):
        raise ValueError("R2 completion marker report SHA mismatch")
    validate_r2_portfolio(report)

    lineage = report.get("source_lineage", {})
    if lineage.get("source_r0_freeze_id") != contract["required_r0_freeze_id"]:
        raise ValueError("R2 report R0 freeze ID mismatch")
    if lineage.get("source_r0_manifest_sha256") != contract["required_r0_manifest_sha256"]:
        raise ValueError("R2 report R0 freeze SHA mismatch")
    return {"manifest": manifest, "ready": ready, "report": report, "complete": complete}


def _validate_upstream_frozen_lineage(spec: dict[str, Any], head: str) -> dict[str, Any]:
    contract = spec["source_contract"]

    r0_manifest = load_json(R0_FREEZE_MANIFEST)
    r0_ready = load_json(R0_FREEZE_READY)
    _validate_hashed_object(
        r0_manifest,
        id_key="freeze_id",
        sha_key="manifest_sha256",
        id_prefix="sers_r0_manual_scientific_adjudication_freeze_v1:",
        label="R0 freeze manifest",
    )
    if r0_manifest.get("freeze_id") != contract["required_r0_freeze_id"]:
        raise ValueError("unexpected R0 freeze ID")
    if r0_manifest.get("manifest_sha256") != contract["required_r0_manifest_sha256"]:
        raise ValueError("unexpected R0 freeze manifest SHA")
    if r0_manifest.get("source_adjudication_commit") != contract["required_r0_source_adjudication_commit"]:
        raise ValueError("unexpected R0 source adjudication commit")
    if r0_ready.get("ready") is not True or r0_ready.get("stop") is not True:
        raise ValueError("R0 FREEZE_READY boundary invalid")
    if not is_ancestor(contract["required_r0_source_adjudication_commit"], head):
        raise ValueError("R0 source adjudication commit is not an ancestor of HEAD")
    _validate_exact_tracked_file(
        R0_ADJ_PATH,
        contract["required_r0_source_adjudication_commit"],
        "R0 adjudication payload",
    )
    adjudication = load_json(R0_ADJ_PATH)
    source_lineage = adjudication.get("source_lineage", {})
    if source_lineage.get("gap_plan_id") != contract["required_gap_plan_id"]:
        raise ValueError("R0 gap-plan ID lineage mismatch")
    if source_lineage.get("gap_plan_sha256") != contract["required_gap_plan_sha256"]:
        raise ValueError("R0 gap-plan SHA lineage mismatch")
    if source_lineage.get("t1_run_id") != contract["required_t1_run_id"]:
        raise ValueError("R0 T1 run lineage mismatch")
    if source_lineage.get("t1_freeze_id") != contract["required_t1_freeze_id"]:
        raise ValueError("R0 T1 freeze lineage mismatch")
    if source_lineage.get("t1_manifest_sha256") != contract["required_t1_manifest_sha256"]:
        raise ValueError("R0 T1 manifest lineage mismatch")
    stage = adjudication.get("stage_boundary", {})
    if stage.get("r1_authorized_for_any_hypothesis") is not False:
        raise ValueError("R0 authorizes R1")
    if stage.get("fresh_reserve_c_authorized") is not False:
        raise ValueError("R0 authorizes Reserve C")

    t1_manifest = load_json(T1_FREEZE_MANIFEST)
    _validate_hashed_object(
        t1_manifest,
        id_key="freeze_id",
        sha_key="manifest_sha256",
        id_prefix="sers_targeted_retrieval_t1_final_freeze_v2:",
        label="T1 freeze manifest",
    )
    if t1_manifest.get("freeze_id") != contract["required_t1_freeze_id"]:
        raise ValueError("unexpected T1 freeze ID")
    if t1_manifest.get("manifest_sha256") != contract["required_t1_manifest_sha256"]:
        raise ValueError("unexpected T1 manifest SHA")
    if t1_manifest.get("v2_run_id") != contract["required_t1_run_id"]:
        raise ValueError("unexpected T1 run ID")
    if t1_manifest.get("v2_outcome") != "SERS_T1_LIVE_TARGETED_RETRIEVAL_V2_MECHANICAL_PASS":
        raise ValueError("T1 mechanical outcome changed")
    if t1_manifest.get("all_structural_checks_pass") is not True:
        raise ValueError("T1 structural checks no longer pass")
    if t1_manifest.get("failed_provider_execution_count") != 0:
        raise ValueError("T1 contains failed provider executions")
    for key in [
        "ranker_called",
        "claim_reviewer_called",
        "scientific_novelty_reassessed",
        "hypothesis_rewrite_called",
        "fresh_reserve_c_consumed",
        "automatic_next_stage_authorized",
    ]:
        if t1_manifest.get(key) is not False:
            raise ValueError(f"T1 frozen guard changed:{key}")
    if t1_manifest.get("llm_calls") != 0:
        raise ValueError("T1 LLM call count changed")

    if sha256_file(T0_FREEZE_MANIFEST) != contract["required_t0_manifest_file_sha256"]:
        raise ValueError("T0 freeze manifest file SHA mismatch")
    t0_manifest = load_json(T0_FREEZE_MANIFEST)
    if t0_manifest.get("freeze_id") != contract["required_t0_freeze_id"]:
        raise ValueError("unexpected T0 freeze ID")
    if t1_manifest.get("t0_freeze_id") != contract["required_t0_freeze_id"]:
        raise ValueError("T1/T0 freeze lineage mismatch")

    gap_plan = load_json(GAP_PLAN_PATH)
    if gap_plan.get("plan_id") != contract["required_gap_plan_id"]:
        raise ValueError("unexpected gap-plan ID")
    if gap_plan.get("plan_sha256") != contract["required_gap_plan_sha256"]:
        raise ValueError("unexpected gap-plan SHA")

    for path in [
        R0_FREEZE_MANIFEST,
        R0_FREEZE_READY,
        T1_FREEZE_MANIFEST,
        T0_FREEZE_MANIFEST,
        GAP_PLAN_PATH,
    ]:
        _validate_exact_tracked_file(path, "HEAD", "upstream frozen input")
    return {
        "r0_manifest": r0_manifest,
        "r0_ready": r0_ready,
        "r0_adjudication": adjudication,
        "t1_manifest": t1_manifest,
        "t0_manifest": t0_manifest,
        "gap_plan": gap_plan,
    }


def _validate_i0_rules(spec: dict[str, Any]) -> None:
    rules = spec.get("orchestration_rules", {})
    if rules.get("consume_frozen_r2_decisions_without_reassessment") is not True:
        raise ValueError("I0 frozen R2 consumption rule missing")
    for key in [
        "new_scientific_judgment_allowed",
        "new_retrieval_allowed",
        "network_allowed",
        "llm_allowed",
        "ranker_allowed",
        "claim_reviewer_allowed",
        "hypothesis_rewrite_allowed",
        "r1_execution_allowed",
        "external_prior_art_can_be_positive_premise",
        "literature_absence_claimed",
        "fresh_reserve_c_consumption_allowed",
        "fresh_reserve_c_authorization_allowed",
        "automatic_next_stage_authorized",
    ]:
        if rules.get(key) is not False:
            raise ValueError(f"I0 forbidden capability enabled:{key}")

    policy = spec.get("handoff_policy", {})
    for key in [
        "preserve_r2_hypothesis_decisions_exactly",
        "preserve_r2_portfolio_decision_exactly",
        "primary_candidate_is_not_empirically_validated",
        "i0_is_orchestration_not_scientific_reassessment",
        "reserve_c_readiness_is_not_assessed_in_i0",
        "separate_explicit_reserve_c_authorization_required",
        "stop_after_i0_freeze",
    ]:
        if policy.get(key) is not True:
            raise ValueError(f"I0 handoff policy requirement missing:{key}")


def _validate_i0_implementation_tracked(head: str) -> str:
    implementation = [
        SPEC_PATH,
        RUN_PATH,
        VERIFY_PATH,
        FREEZE_CREATE_PATH,
        FREEZE_VERIFY_PATH,
        TEST_PATH,
    ]
    for path in implementation:
        _validate_exact_tracked_file(path, "HEAD", "I0 implementation")
    code_commit = git_text("log", "-1", "--format=%H", "--", rel(SPEC_PATH))
    if not code_commit:
        raise ValueError("cannot resolve I0 implementation commit")
    for path in implementation:
        rp = rel(path)
        if not tracked_at(code_commit, rp):
            raise ValueError(f"I0 code commit missing implementation:{rp}")
    if not is_ancestor(code_commit, head):
        raise ValueError("I0 implementation commit is not an ancestor of HEAD")
    return code_commit


def validate_inputs(
    *,
    require_output_absent: bool,
    enforce_execution_workspace: bool = True,
) -> dict[str, Any]:
    issues: list[str] = []
    spec: dict[str, Any] | None = None
    try:
        spec = load_spec()
    except Exception as exc:
        issues.append(str(exc))

    branch = git_text("branch", "--show-current")

    if enforce_execution_workspace:
        if spec is not None:
            expected_branch = spec["source_contract"]["required_branch"]
            if branch != expected_branch:
                issues.append(f"unexpected branch:{branch}")

        tracked_dirty = (
            subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
            or subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=ROOT,
            ).returncode != 0
        )
        if tracked_dirty:
            issues.append("tracked working tree/index is not clean")

    required_paths = [
        SPEC_PATH,
        R2_REPORT_PATH,
        R2_COMPLETE_PATH,
        R2_FREEZE_MANIFEST,
        R2_FREEZE_READY,
        R0_ADJ_PATH,
        R0_FREEZE_MANIFEST,
        R0_FREEZE_READY,
        T1_FREEZE_MANIFEST,
        T0_FREEZE_MANIFEST,
        GAP_PLAN_PATH,
    ]
    for path in required_paths:
        if not path.is_file():
            issues.append(f"required file missing:{rel(path)}")
    if require_output_absent and OUTPUT_ROOT.exists():
        issues.append("I0 output root already exists")
    if issues:
        raise ValueError("; ".join(issues))

    base = subprocess.run(
        [sys.executable, "-m", "campaigns.sers_alpha4_epoch.post_t1.cli.verify_sers_r2_final_reassessment_freeze_v1"],
        cwd=ROOT,
        text=True,
    )
    if base.returncode != 0:
        raise ValueError("R2 final reassessment freeze verifier failed")

    head = git_text("rev-parse", "HEAD")
    _validate_i0_rules(spec)
    r2 = _validate_r2_lineage(spec, head)
    upstream = _validate_upstream_frozen_lineage(spec, head)
    i0_code_commit = _validate_i0_implementation_tracked(head)

    return {
        "head": head,
        "branch": branch,
        "spec": spec,
        "i0_code_commit": i0_code_commit,
        "r2_manifest": r2["manifest"],
        "r2_ready": r2["ready"],
        "r2_report": r2["report"],
        "r2_complete": r2["complete"],
        **upstream,
    }


def build_handoff(ctx: dict[str, Any]) -> dict[str, Any]:
    spec = ctx["spec"]
    contract = spec["source_contract"]
    report = ctx["r2_report"]
    decisions = json.loads(canonical(report["hypothesis_decisions"]))
    portfolio = json.loads(canonical(report["portfolio_decision"]))

    payload = {
        "schema_version": "sers-i0-integrated-orchestration-handoff-v1",
        "source_lineage": {
            "source_i0_spec_id": spec["spec_id"],
            "source_i0_spec_sha256": spec["spec_sha256"],
            "source_i0_code_commit": ctx["i0_code_commit"],
            "source_r2_freeze_commit": contract["required_r2_freeze_commit"],
            "source_r2_freeze_id": ctx["r2_manifest"]["freeze_id"],
            "source_r2_manifest_sha256": ctx["r2_manifest"]["manifest_sha256"],
            "source_r2_report_commit": ctx["r2_manifest"]["source_r2_report_commit"],
            "source_r2_report_id": report["report_id"],
            "source_r2_report_sha256": report["report_sha256"],
            "source_r0_freeze_id": ctx["r0_manifest"]["freeze_id"],
            "source_r0_manifest_sha256": ctx["r0_manifest"]["manifest_sha256"],
            "source_r0_adjudication_commit": ctx["r0_manifest"]["source_adjudication_commit"],
            "source_t1_run_id": ctx["t1_manifest"]["v2_run_id"],
            "source_t1_freeze_id": ctx["t1_manifest"]["freeze_id"],
            "source_t1_manifest_sha256": ctx["t1_manifest"]["manifest_sha256"],
            "source_t0_freeze_id": ctx["t0_manifest"]["freeze_id"],
            "source_gap_plan_id": ctx["gap_plan"]["plan_id"],
            "source_gap_plan_sha256": ctx["gap_plan"]["plan_sha256"],
        },
        "stage_ledger": {
            "g0_g2": "FROZEN_INPUT_BOUND",
            "t0": "FROZEN_INPUT_BOUND",
            "t1": "FROZEN_VERIFIED",
            "r0": "FROZEN_VERIFIED",
            "r1": "NOT_EXECUTED_NOT_AUTHORIZED",
            "r2": "FROZEN_VERIFIED",
            "i0": "COMPLETE",
            "fresh_reserve_c": "UNTOUCHED_UNAUTHORIZED",
        },
        "frozen_r2_hypothesis_decisions": decisions,
        "frozen_r2_portfolio_decision": portfolio,
        "orchestration_guards": {
            "scientific_reassessment_performed": False,
            "new_scientific_judgment_performed": False,
            "new_retrieval_performed": False,
            "ranker_called": False,
            "claim_reviewer_called": False,
            "hypothesis_rewrite_called": False,
            "r1_executed": False,
            "runtime_network_calls": 0,
            "runtime_llm_calls": 0,
            "external_prior_art_used_as_positive_premise": False,
            "literature_absence_claimed": False,
            "automatic_next_stage_authorized": False,
        },
        "reserve_c_boundary": {
            "readiness_assessed": False,
            "authorization_required": "explicit_separate_guarded_one_shot_authorization",
            "authorized": False,
            "consumed": False,
            "marker_write_allowed": False,
            "holdout_execution_allowed": False,
        },
        "epistemic_usage": "frozen_decision_handoff_only_not_new_scientific_evidence",
        "stop_after_i0": True,
    }
    handoff_sha = sha256_bytes(canonical(payload).encode("utf-8"))
    handoff = dict(payload)
    handoff["orchestration_id"] = "sers_i0_integrated_orchestration_v1:" + handoff_sha[:20]
    handoff["orchestration_sha256"] = handoff_sha
    return handoff


def _atomic_write_output(handoff: dict[str, Any], marker: dict[str, Any]) -> None:
    parent = OUTPUT_ROOT.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_root = parent / f".{OUTPUT_ROOT.name}.tmp"
    if tmp_root.exists():
        raise ValueError("I0 temporary output root already exists")
    tmp_root.mkdir()
    try:
        (tmp_root / HANDOFF_PATH.name).write_text(
            json.dumps(handoff, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (tmp_root / COMPLETE_PATH.name).write_text(
            json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_root.replace(OUTPUT_ROOT)
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    try:
        ctx = validate_inputs(require_output_absent=True)
    except Exception as exc:
        print("SERS I0 integrated orchestration preflight: FAIL")
        print(" -", exc)
        print("Runtime network calls:", 0)
        print("Runtime LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        print("Fresh Reserve C authorized:", False)
        return 2

    report = ctx["r2_report"]
    print("SERS I0 integrated orchestration preflight: PASS")
    print("R2 freeze ID:", ctx["r2_manifest"]["freeze_id"])
    print("R2 report ID:", report["report_id"])
    print(
        "Primary remaining candidate:",
        report["portfolio_decision"]["primary_remaining_candidate_hypothesis_id"],
    )
    print("I0 code commit:", ctx["i0_code_commit"])
    print("Scientific reassessment performed:", False)
    print("Runtime network calls:", 0)
    print("Runtime LLM calls:", 0)
    print("R1 executed:", False)
    print("Fresh Reserve C consumed:", False)
    print("Fresh Reserve C authorized:", False)
    print("Automatic next stage authorized:", False)
    if args.preflight:
        return 0

    handoff = build_handoff(ctx)
    marker = {
        "schema_version": "sers-i0-integrated-orchestration-complete-v1",
        "orchestration_id": handoff["orchestration_id"],
        "orchestration_sha256": handoff["orchestration_sha256"],
        "complete": True,
        "input_lineage_verified": True,
        "scientific_reassessment_performed": False,
        "r1_executed": False,
        "fresh_reserve_c_readiness_assessed": False,
        "fresh_reserve_c_authorized": False,
        "fresh_reserve_c_consumed": False,
        "automatic_next_stage_authorized": False,
        "stop": True,
    }
    try:
        _atomic_write_output(handoff, marker)
    except Exception as exc:
        print("SERS I0 integrated orchestration execution: FAIL")
        print(" - output write:", exc)
        return 2

    print("SERS I0 integrated orchestration execution: PASS")
    print("Orchestration ID:", handoff["orchestration_id"])
    print("Orchestration SHA256:", handoff["orchestration_sha256"])
    print(
        "Primary remaining candidate:",
        handoff["frozen_r2_portfolio_decision"][
            "primary_remaining_candidate_hypothesis_id"
        ],
    )
    print("Scientific reassessment performed:", False)
    print("Runtime network calls:", 0)
    print("Runtime LLM calls:", 0)
    print("Hypothesis rewrites:", 0)
    print("R1 executed:", False)
    print("Fresh Reserve C readiness assessed:", False)
    print("Fresh Reserve C authorized:", False)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    print("STOP:", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
