from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ADJUDICATION_PATH = (
    ROOT
    / "evaluation/sers_novelty_gap/r0_manual_scientific_adjudication_v1"
    / "adjudication.json"
)
T1_FREEZE_PATH = (
    ROOT
    / "evaluation/sers_novelty_gap/t1_live_targeted_retrieval_freeze_v2"
    / "freeze_manifest.json"
)
GAP_PLAN_PATH = (
    ROOT
    / "evaluation/sers_novelty_gap/t1_frozen_input_bundle_v1"
    / "novelty_gap_plan.json"
)

EXPECTED_BRANCH = "feat/SERS-targeted-retrieval-live-dev"
EXPECTED_R0_2_COMMIT = "22867c720073bb69323a6801151415bd22c187ed"
EXPECTED_PLAN_ID = "novelty_gap_plan:9c484bce48aefd4cd948"
EXPECTED_PLAN_SHA256 = (
    "935054ac6c7e4cf72a22ee4bbd0b53ae82eb60ea36eb524d56b3f7bd60029f67"
)
EXPECTED_T1_RUN_ID = "sers_targeted_retrieval_t1_live_v2:9a3c03bc59085c0af5fe"
EXPECTED_T1_FREEZE_ID = (
    "sers_targeted_retrieval_t1_final_freeze_v2:d7d8919a7c1c819f57c3"
)
EXPECTED_T1_MANIFEST_SHA256 = (
    "d7d8919a7c1c819f57c38e2b07abb84a9dcc8b93579f160f1f9ebee884361201"
)
R0_2_FROZEN_FILES = (
    "dac_her/r0_contracts.py",
    "dac_her/r0_runtime.py",
    "tests/test_r0_runtime.py",
    "tests/test_r0_sers_regression.py",
)

H1 = "direction_aware_trend_hypothesis:ad13dac8334238124899"
H2 = "direction_aware_trend_hypothesis:8507f8cadfc46d8d80de"
H3 = "direction_aware_trend_hypothesis:1cf889e57332402d88c9"
C_H1_A = "external_novelty_claim:99ed0af7161d694818f6"
C_H1_B = "external_novelty_claim:a735f559d97b4208dca3"
C_H3_A = "external_novelty_claim:4579feb9177160cd54f2"
C_H3_B = "external_novelty_claim:778e935b3b9fb4cf5be2"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_adjudication(data: dict) -> list[str]:
    issues: list[str] = []

    if data.get("schema_version") != "sers-r0-manual-scientific-adjudication-v1":
        issues.append("unexpected adjudication schema")

    payload = dict(data)
    adjudication_id = payload.pop("adjudication_id", None)
    adjudication_sha256 = payload.pop("adjudication_sha256", None)
    recomputed = _sha256_json(payload)
    if adjudication_sha256 != recomputed:
        issues.append("adjudication_sha256 mismatch")
    if adjudication_id != f"sers_r0_manual_scientific_adjudication_v1:{recomputed[:20]}":
        issues.append("adjudication_id mismatch")

    reviewer = data.get("reviewer", {})
    if reviewer.get("mode") != "llm_scientific_adjudication_with_primary_source_verification":
        issues.append("reviewer mode mismatch")
    if reviewer.get("model") != "GPT-5.6 Sol":
        issues.append("reviewer model mismatch")
    if reviewer.get("human_scientist_reviewer_present") is not False:
        issues.append("human scientist reviewer must remain explicitly false")
    if reviewer.get("scientific_reviewer_llm_used") is not True:
        issues.append("LLM scientific reviewer usage must remain explicit")
    if reviewer.get("deterministic_r0_router_llm_calls") != 0:
        issues.append("deterministic R0 router must remain LLM-free")

    lineage = data.get("source_lineage", {})
    expected_lineage = {
        "source_branch": EXPECTED_BRANCH,
        "source_r0_2_commit": EXPECTED_R0_2_COMMIT,
        "gap_plan_id": EXPECTED_PLAN_ID,
        "gap_plan_sha256": EXPECTED_PLAN_SHA256,
        "t1_run_id": EXPECTED_T1_RUN_ID,
        "t1_freeze_id": EXPECTED_T1_FREEZE_ID,
        "t1_manifest_sha256": EXPECTED_T1_MANIFEST_SHA256,
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            issues.append(f"source lineage mismatch:{key}")

    guards = data.get("epistemic_guards", {})
    required_false = [
        "frozen_t1_modified",
        "literature_absence_claimed",
        "external_prior_art_used_as_positive_hypothesis_premise",
        "hypothesis_rewritten",
        "automatic_next_stage_authorized",
        "fresh_reserve_c_consumed",
    ]
    for key in required_false:
        if guards.get(key) is not False:
            issues.append(f"epistemic guard must be false:{key}")
    required_true = [
        "external_lookup_not_part_of_frozen_t1",
        "reviewer_found_external_prior_art_kept_separate",
    ]
    for key in required_true:
        if guards.get(key) is not True:
            issues.append(f"epistemic guard must be true:{key}")

    sources = {s["source_id"]: s for s in data.get("primary_source_records", [])}
    expected_sources = {
        "manual_prior_art:jiang2017_ag_nanomushroom": "10.1038/s41598-017-10262-9",
        "manual_prior_art:rastogi2021_analyte_colocalization": "10.1021/acsami.0c17929",
        "manual_prior_art:ma2020_gap_dependent": "10.1021/acs.jpcc.0c07701",
        "manual_prior_art:wu2017_core_satellite": "10.1038/s41598-017-13577-9",
    }
    if set(sources) != set(expected_sources):
        issues.append("primary source set mismatch")
    else:
        for source_id, doi in expected_sources.items():
            if sources[source_id].get("doi") != doi:
                issues.append(f"primary source DOI mismatch:{source_id}")
    wu = sources.get("manual_prior_art:wu2017_core_satellite", {})
    if wu.get("origin") != "reviewer_found_external_prior_art_not_in_frozen_h3_packet":
        issues.append("H3 reviewer-found prior art provenance must remain explicit")

    claims = {
        (c["hypothesis_id"], c["claim_id"]): c
        for c in data.get("claim_assessments", [])
    }
    expected_claim_status = {
        (H1, C_H1_A): "DIRECT_PRIOR_ART",
        (H1, C_H1_B): "DIRECT_PRIOR_ART",
        (H3, C_H3_A): "PARTIAL_PRIOR_ART",
        (H3, C_H3_B): "COMPONENTS_ONLY",
    }
    if set(claims) != set(expected_claim_status):
        issues.append("claim assessment set mismatch")
    else:
        for key, expected_status in expected_claim_status.items():
            if claims[key].get("status") != expected_status:
                issues.append(f"claim status mismatch:{key[1]}")

    outcomes = {
        o["hypothesis_id"]: o
        for o in data.get("r0_outcomes", [])
    }
    if set(outcomes) != {H1, H2, H3}:
        issues.append("R0 outcome hypothesis set mismatch")
    else:
        expected = {
            H1: (
                "targeted_search_then_refine",
                "directly_covered",
                "pass_original_to_r2",
            ),
            H2: (
                "keep",
                None,
                "pass_through_frozen",
            ),
            H3: (
                "targeted_search_only",
                "relational_gap_remains",
                "pass_original_to_r2",
            ),
        }
        for hypothesis_id, (action, state, route) in expected.items():
            outcome = outcomes[hypothesis_id]
            if outcome.get("source_action") != action:
                issues.append(f"R0 source action mismatch:{hypothesis_id}")
            if outcome.get("evidence_state") != state:
                issues.append(f"R0 evidence state mismatch:{hypothesis_id}")
            if outcome.get("route") != route:
                issues.append(f"R0 route mismatch:{hypothesis_id}")
            if outcome.get("r1_authorized") is not False:
                issues.append(f"R1 unexpectedly authorized:{hypothesis_id}")
            if outcome.get("max_refinements_authorized") != 0:
                issues.append(f"nonzero refinement authorization:{hypothesis_id}")
            if outcome.get("r2_required") is not True:
                issues.append(f"R2 must remain required:{hypothesis_id}")

    boundary = data.get("stage_boundary", {})
    if boundary.get("r0_scientific_adjudication_complete") is not True:
        issues.append("R0 scientific adjudication must be complete")
    if boundary.get("r1_authorized_for_any_hypothesis") is not False:
        issues.append("R1 global authorization must be false")
    if boundary.get("r2_started") is not False:
        issues.append("R2 must not be started")
    if boundary.get("integration_started") is not False:
        issues.append("integration must not be started")
    if boundary.get("fresh_reserve_c_authorized") is not False:
        issues.append("Reserve C authorization must be false")
    if boundary.get("stop_after_freeze") is not True:
        issues.append("freeze boundary must STOP")

    return issues


def main() -> int:
    issues: list[str] = []

    if not ADJUDICATION_PATH.is_file():
        issues.append("adjudication artifact missing")
    if not T1_FREEZE_PATH.is_file():
        issues.append("T1 freeze manifest missing")
    if not GAP_PLAN_PATH.is_file():
        issues.append("frozen novelty gap plan missing")

    if issues:
        print("SERS R0 manual scientific adjudication verification: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2

    data = _load_json(ADJUDICATION_PATH)
    issues.extend(validate_adjudication(data))

    t1 = _load_json(T1_FREEZE_PATH)
    if t1.get("freeze_id") != EXPECTED_T1_FREEZE_ID:
        issues.append("T1 freeze ID mismatch")
    if t1.get("manifest_sha256") != EXPECTED_T1_MANIFEST_SHA256:
        issues.append("T1 manifest SHA mismatch")
    if t1.get("v2_run_id") != EXPECTED_T1_RUN_ID:
        issues.append("T1 run ID mismatch")
    if t1.get("v2_outcome") != "SERS_T1_LIVE_TARGETED_RETRIEVAL_V2_MECHANICAL_PASS":
        issues.append("T1 mechanical outcome mismatch")
    if t1.get("scientific_novelty_reassessed") is not False:
        issues.append("T1 scientific novelty must remain unreassessed")
    if t1.get("ranker_called") is not False:
        issues.append("T1 ranker_called changed")
    if t1.get("claim_reviewer_called") is not False:
        issues.append("T1 claim_reviewer_called changed")
    if t1.get("hypothesis_rewrite_called") is not False:
        issues.append("T1 hypothesis rewrite changed")
    if t1.get("fresh_reserve_c_consumed") is not False:
        issues.append("T1 records Reserve C consumption")
    if t1.get("automatic_next_stage_authorized") is not False:
        issues.append("T1 automatic next-stage flag changed")

    gap_plan = _load_json(GAP_PLAN_PATH)
    if gap_plan.get("plan_id") != EXPECTED_PLAN_ID:
        issues.append("frozen gap plan ID mismatch")
    if gap_plan.get("plan_sha256") != EXPECTED_PLAN_SHA256:
        issues.append("frozen gap plan SHA mismatch")
    actions = {
        gap["hypothesis_id"]: gap["action"]
        for gap in gap_plan.get("gaps", [])
    }
    if actions.get(H1) != "targeted_search_then_refine":
        issues.append("H1 frozen action mismatch")
    if actions.get(H2) != "keep":
        issues.append("H2 frozen action mismatch")
    if actions.get(H3) != "targeted_search_only":
        issues.append("H3 frozen action mismatch")

    try:
        branch = _git("branch", "--show-current")
        if branch != EXPECTED_BRANCH:
            issues.append(f"unexpected branch:{branch}")
        ancestor_check = subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_R0_2_COMMIT, "HEAD"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ancestor_check.returncode != 0:
            issues.append("R0.2 source commit is not an ancestor of HEAD")
        else:
            r0_diff = subprocess.run(
                [
                    "git",
                    "diff",
                    "--quiet",
                    f"{EXPECTED_R0_2_COMMIT}..HEAD",
                    "--",
                    *R0_2_FROZEN_FILES,
                ],
                cwd=ROOT,
            )
            if r0_diff.returncode != 0:
                issues.append("R0.2 frozen implementation files changed after source commit")
    except (OSError, subprocess.CalledProcessError) as exc:
        issues.append(f"git lineage verification failed:{exc}")

    if issues:
        print("SERS R0 manual scientific adjudication verification: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Deterministic R0 router LLM calls:", 0)
        print("Hypothesis rewrites:", 0)
        print("Fresh Reserve C consumed:", False)
        return 2

    print("SERS R0 manual scientific adjudication verification: PASS")
    print("Adjudication ID:", data["adjudication_id"])
    print("Adjudication SHA256:", data["adjudication_sha256"])
    print("Source R0.2 commit:", EXPECTED_R0_2_COMMIT)
    print("H1 state:", "directly_covered")
    print("H1 route:", "pass_original_to_r2")
    print("H1 R1 authorized:", False)
    print("H2 route:", "pass_through_frozen")
    print("H3 state:", "relational_gap_remains")
    print("H3 route:", "pass_original_to_r2")
    print("H3 R1 authorized:", False)
    print("Scientific reviewer LLM used:", True)
    print("Human scientist reviewer present:", False)
    print("Deterministic R0 router LLM calls:", 0)
    print("Hypothesis rewrites:", 0)
    print("Fresh Reserve C consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
