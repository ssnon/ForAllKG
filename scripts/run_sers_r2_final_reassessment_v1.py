from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "feat/SERS-targeted-retrieval-live-dev"
SPEC_PATH = ROOT / "dac_her/sers_r2_final_reassessment_spec_v1.json"
R0_ADJ_ROOT = ROOT / "evaluation/sers_novelty_gap/r0_manual_scientific_adjudication_v1"
R0_ADJ_PATH = R0_ADJ_ROOT / "adjudication.json"
R0_REVIEW_PATH = R0_ADJ_ROOT / "SCIENTIFIC_REVIEW.md"
R0_FREEZE_ROOT = ROOT / "evaluation/sers_novelty_gap/r0_manual_scientific_adjudication_freeze_v1"
R0_FREEZE_MANIFEST = R0_FREEZE_ROOT / "freeze_manifest.json"
R0_FREEZE_READY = R0_FREEZE_ROOT / "FREEZE_READY.json"
OUTPUT_ROOT = ROOT / "evaluation/sers_novelty_gap/r2_final_reassessment_run_v1"
REPORT_PATH = OUTPUT_ROOT / "r2_report.json"
COMPLETE_PATH = OUTPUT_ROOT / "R2_COMPLETE.json"

H1 = "direction_aware_trend_hypothesis:ad13dac8334238124899"
H2 = "direction_aware_trend_hypothesis:8507f8cadfc46d8d80de"
H3 = "direction_aware_trend_hypothesis:1cf889e57332402d88c9"


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


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


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_spec() -> dict[str, Any]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    payload = dict(spec)
    spec_id = payload.pop("spec_id", None)
    spec_sha = payload.pop("spec_sha256", None)
    recomputed = sha256_bytes(canonical(payload).encode("utf-8"))
    expected_id = "sers_r2_final_reassessment_spec_v1:" + recomputed[:20]
    if spec_sha != recomputed or spec_id != expected_id:
        raise ValueError("R2 spec hash/ID mismatch")
    return spec


def _assert_r0_outcomes(adjudication: dict[str, Any]) -> None:
    rows = {row["hypothesis_id"]: row for row in adjudication["r0_outcomes"]}
    expected = {
        H1: ("pass_original_to_r2", "directly_covered"),
        H2: ("pass_through_frozen", None),
        H3: ("pass_original_to_r2", "relational_gap_remains"),
    }
    if set(rows) != set(expected):
        raise ValueError("R0 hypothesis set mismatch")
    for hid, (route, state) in expected.items():
        row = rows[hid]
        if row.get("route") != route:
            raise ValueError(f"R0 route mismatch:{hid}")
        if row.get("evidence_state") != state:
            raise ValueError(f"R0 evidence state mismatch:{hid}")
        if row.get("r1_authorized") is not False:
            raise ValueError(f"R1 unexpectedly authorized:{hid}")
        if row.get("max_refinements_authorized") != 0:
            raise ValueError(f"R1 refinement allowance changed:{hid}")
        if row.get("r2_required") is not True:
            raise ValueError(f"R2 requirement missing:{hid}")


def validate_inputs(*, require_output_absent: bool) -> dict[str, Any]:
    issues: list[str] = []
    branch = git_text("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        issues.append(f"unexpected branch:{branch}")

    tracked_dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0
    )
    if tracked_dirty:
        issues.append("tracked working tree/index is not clean")

    for path in [SPEC_PATH, R0_ADJ_PATH, R0_REVIEW_PATH, R0_FREEZE_MANIFEST, R0_FREEZE_READY]:
        if not path.is_file():
            issues.append(f"required file missing:{rel(path)}")

    if require_output_absent and OUTPUT_ROOT.exists():
        issues.append("R2 output root already exists")

    if issues:
        raise ValueError("; ".join(issues))

    base_verify = subprocess.run(
        [sys.executable, "-m", "scripts.verify_sers_r0_manual_scientific_adjudication_freeze_v1"],
        cwd=ROOT,
        text=True,
    )
    if base_verify.returncode != 0:
        raise ValueError("R0 scientific adjudication freeze verifier failed")

    head = git_text("rev-parse", "HEAD")
    for path in [SPEC_PATH, R0_FREEZE_MANIFEST, R0_FREEZE_READY]:
        rp = rel(path)
        if not tracked_at("HEAD", rp):
            raise ValueError(f"required tracked input absent from HEAD:{rp}")
        if git_bytes_at("HEAD", rp) != path.read_bytes():
            raise ValueError(f"working file differs from tracked HEAD:{rp}")

    manifest = json.loads(R0_FREEZE_MANIFEST.read_text(encoding="utf-8"))
    ready = json.loads(R0_FREEZE_READY.read_text(encoding="utf-8"))
    if ready.get("ready") is not True or ready.get("stop") is not True:
        raise ValueError("R0 freeze ready/STOP boundary invalid")
    if ready.get("freeze_id") != manifest.get("freeze_id"):
        raise ValueError("R0 freeze ready/manifest ID mismatch")
    if ready.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("R0 freeze ready/manifest SHA mismatch")

    guards = {
        "r1_authorized_for_any_hypothesis": False,
        "r2_started": False,
        "fresh_reserve_c_consumed": False,
        "fresh_reserve_c_authorized": False,
        "automatic_next_stage_authorized": False,
        "hypothesis_rewrite_called": False,
    }
    for key, expected in guards.items():
        if manifest.get(key) is not expected:
            raise ValueError(f"R0 freeze guard mismatch:{key}")
    if manifest.get("r0_scientific_adjudication_complete") is not True:
        raise ValueError("R0 adjudication completion flag missing")
    if manifest.get("stop_after_freeze") is not True:
        raise ValueError("R0 stop_after_freeze flag missing")

    source_commit = manifest.get("source_adjudication_commit")
    if not isinstance(source_commit, str) or len(source_commit) < 12:
        raise ValueError("R0 source adjudication commit missing")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, head],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise ValueError("R0 source adjudication commit is not an ancestor of HEAD")

    # Critical repair guard: the source commit must actually contain the scientific
    # adjudication payload, not merely point to ignored local files.
    for path in [R0_ADJ_PATH, R0_REVIEW_PATH]:
        rp = rel(path)
        if not tracked_at(source_commit, rp):
            raise ValueError(f"R0 source commit does not contain adjudication payload:{rp}")
        expected_sha = manifest.get("critical_file_sha256", {}).get(rp)
        if not isinstance(expected_sha, str):
            raise ValueError(f"R0 freeze missing critical hash:{rp}")
        if sha256_bytes(git_bytes_at(source_commit, rp)) != expected_sha:
            raise ValueError(f"R0 source-commit payload hash mismatch:{rp}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"R0 working payload hash mismatch:{rp}")

    adjudication = json.loads(R0_ADJ_PATH.read_text(encoding="utf-8"))
    _assert_r0_outcomes(adjudication)
    stage = adjudication.get("stage_boundary", {})
    if stage.get("r1_authorized_for_any_hypothesis") is not False:
        raise ValueError("R0 adjudication authorizes R1")
    if stage.get("r2_started") is not False:
        raise ValueError("R0 adjudication already records R2 started")
    if stage.get("fresh_reserve_c_authorized") is not False:
        raise ValueError("R0 adjudication authorizes Reserve C")

    spec = load_spec()
    rules = spec["source_rules"]
    required_false_rules = [
        "hypothesis_rewrite_allowed",
        "external_prior_art_can_be_positive_premise",
        "literature_absence_claimed",
        "fresh_reserve_c_authorized",
        "automatic_next_stage_authorized",
    ]
    for key in required_false_rules:
        if rules.get(key) is not False:
            raise ValueError(f"R2 spec guard changed:{key}")
    for key in [
        "r0_freeze_must_be_git_tracked",
        "r0_source_adjudication_must_be_git_reproducible",
        "r1_must_not_have_executed",
    ]:
        if rules.get(key) is not True:
            raise ValueError(f"R2 spec requirement missing:{key}")

    freeze_commit = git_text("log", "-1", "--format=%H", "--", rel(R0_FREEZE_MANIFEST))
    if not freeze_commit:
        raise ValueError("cannot resolve git commit containing R0 freeze manifest")
    r2_code_commit = git_text("log", "-1", "--format=%H", "--", rel(SPEC_PATH))
    if not r2_code_commit:
        raise ValueError("cannot resolve git commit containing R2 spec")
    for path in [SPEC_PATH, ROOT / "scripts/run_sers_r2_final_reassessment_v1.py", ROOT / "scripts/verify_sers_r2_final_reassessment_v1.py"]:
        rp = rel(path)
        if not tracked_at(r2_code_commit, rp):
            raise ValueError(f"R2 code commit missing critical implementation:{rp}")

    return {
        "head": head,
        "r2_code_commit": r2_code_commit,
        "branch": branch,
        "spec": spec,
        "r0_manifest": manifest,
        "r0_ready": ready,
        "r0_adjudication": adjudication,
        "r0_freeze_commit": freeze_commit,
    }


def build_report(ctx: dict[str, Any]) -> dict[str, Any]:
    spec = ctx["spec"]
    manifest = ctx["r0_manifest"]
    payload = {
        "schema_version": "sers-r2-final-reassessment-report-v1",
        "source_lineage": {
            "source_r0_freeze_id": manifest["freeze_id"],
            "source_r0_manifest_sha256": manifest["manifest_sha256"],
            "source_r0_freeze_commit": ctx["r0_freeze_commit"],
            "source_r0_adjudication_commit": manifest["source_adjudication_commit"],
            "source_r2_code_commit": ctx["r2_code_commit"],
            "source_r2_spec_id": spec["spec_id"],
            "source_r2_spec_sha256": spec["spec_sha256"],
        },
        "reviewer": spec["reviewer"],
        "primary_source_records": spec["primary_source_records"],
        "hypothesis_decisions": spec["hypothesis_decisions"],
        "portfolio_decision": spec["portfolio_decision"],
        "epistemic_guards": {
            "r1_executed": False,
            "hypothesis_rewrite_called": False,
            "external_prior_art_used_as_positive_premise": False,
            "literature_absence_claimed": False,
            "runtime_llm_calls": 0,
            "runtime_network_calls": 0,
            "i0_started": False,
            "fresh_reserve_c_consumed": False,
            "fresh_reserve_c_authorized": False,
            "automatic_next_stage_authorized": False,
            "stop_after_r2": True,
        },
    }
    report_sha = sha256_bytes(canonical(payload).encode("utf-8"))
    report = dict(payload)
    report["report_id"] = "sers_r2_final_reassessment_report_v1:" + report_sha[:20]
    report["report_sha256"] = report_sha
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        ctx = validate_inputs(require_output_absent=True)
    except Exception as exc:
        print("SERS R2 final reassessment preflight: FAIL")
        print(" -", exc)
        print("Runtime network calls:", 0)
        print("Runtime LLM calls:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("SERS R2 final reassessment preflight: PASS")
    print("R0 freeze ID:", ctx["r0_manifest"]["freeze_id"])
    print("R0 freeze commit:", ctx["r0_freeze_commit"])
    print("R0 source adjudication commit:", ctx["r0_manifest"]["source_adjudication_commit"])
    print("R2 spec ID:", ctx["spec"]["spec_id"])
    print("R1 executed:", False)
    print("Runtime network calls:", 0)
    print("Runtime LLM calls:", 0)
    print("Fresh Reserve C consumed:", False)
    if args.preflight:
        return 0

    report = build_report(ctx)
    marker = {
        "schema_version": "sers-r2-final-reassessment-complete-v1",
        "report_id": report["report_id"],
        "report_sha256": report["report_sha256"],
        "complete": True,
        "r1_executed": False,
        "hypothesis_rewrite_called": False,
        "i0_started": False,
        "fresh_reserve_c_consumed": False,
        "automatic_next_stage_authorized": False,
        "stop": True,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    COMPLETE_PATH.write_text(json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("SERS R2 final reassessment execution: PASS")
    print("Report ID:", report["report_id"])
    print("Report SHA256:", report["report_sha256"])
    for row in report["hypothesis_decisions"]:
        print(row["hypothesis_id"], "=>", row["candidate_disposition"])
    print("Primary remaining candidate:", report["portfolio_decision"]["primary_remaining_candidate_hypothesis_id"])
    print("R1 executed:", False)
    print("Hypothesis rewrites:", 0)
    print("I0 started:", False)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    print("STOP:", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
