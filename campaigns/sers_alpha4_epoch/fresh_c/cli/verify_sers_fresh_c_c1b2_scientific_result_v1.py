from __future__ import annotations

import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_reviewer_contract_v1 import (
    FreshCFinalAdjudication,
    FreshCPaperReview,
)
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_scientific_adjudication_v1 import (
    DEFAULT_RUN_DIR,
    build_target_boundaries,
    canonical_json_sha256,
    load_object,
    validate_corpus_metadata,
    validate_final_against_reviews,
    validate_frozen_lineage,
    validate_review_grounding,
)

def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def main():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    lineage = validate_frozen_lineage(root)
    targets = build_target_boundaries(lineage["r2_report"])
    records = validate_corpus_metadata(root, parse_pages=True)

    run_dir = root / DEFAULT_RUN_DIR
    if (run_dir / "C1B2_SCIENTIFIC_ADJUDICATION_FAILED.json").exists():
        raise RuntimeError("C1B.2 failed epoch exists; success verification forbidden")
    start = load_object(run_dir / "C1B2_SCIENTIFIC_READ_STARTED.json")
    run = load_object(run_dir / "run_manifest.json")
    complete = load_object(run_dir / "C1B2_SCIENTIFIC_ADJUDICATION_COMPLETE.json")
    final_record = load_object(run_dir / "final_adjudication.json")

    if start.get("same_epoch_rerun_allowed") is not False:
        raise ValueError("C1B.2 start marker lost one-shot semantics")
    if run.get("source_identity_count") != 25:
        raise ValueError("C1B.2 run source count drifted")
    if run.get("paper_review_calls") != 25:
        raise ValueError("C1B.2 paper-review call count drifted")
    if run.get("final_adjudication_calls") != 1:
        raise ValueError("C1B.2 final call count drifted")
    if run.get("scientific_llm_call_attempts") != 26:
        raise ValueError("C1B.2 scientific LLM call count drifted")
    if run.get("scientific_network_call_attempts") != 26:
        raise ValueError("C1B.2 scientific network call count drifted")
    if run.get("all_25_papers_processed") is not True:
        raise ValueError("C1B.2 did not process all 25")
    if run.get("fresh_c_scientific_text_read_performed") is not True:
        raise ValueError("C1B.2 scientific-read state missing")
    if run.get("scientific_adjudication_performed") is not True:
        raise ValueError("C1B.2 adjudication state missing")
    if run.get("same_epoch_rerun_allowed") is not False:
        raise ValueError("C1B.2 run lost no-rerun state")
    if run.get("automatic_post_c1b2_transition_allowed") is not False:
        raise ValueError("C1B.2 auto-transition drifted")
    if run.get("target_boundaries_sha256") != canonical_json_sha256(targets):
        raise ValueError("C1B.2 target boundaries drifted")

    review_objects = []
    expected_review_records = run.get("paper_review_records")
    if not isinstance(expected_review_records, list) or len(expected_review_records) != 25:
        raise ValueError("C1B.2 review record manifest drifted")

    for record, manifest_row in zip(records, expected_review_records):
        if manifest_row["reserve_index"] != record["reserve_index"]:
            raise ValueError("C1B.2 review manifest order drifted")
        path = root / manifest_row["record_path"]
        payload = load_object(path)
        tmp = dict(payload)
        stored = tmp.pop("record_sha256")
        if stored != canonical_json_sha256(tmp):
            raise ValueError("C1B.2 paper-review record SHA drifted")
        review = FreshCPaperReview.model_validate(payload["review"])
        validate_review_grounding(
            review,
            expected_record=record,
            pages_manifest=record["pages_manifest"],
        )
        if canonical_json_sha256(review.model_dump(mode="json")) != payload["review_sha256"]:
            raise ValueError("C1B.2 paper-review payload SHA drifted")
        review_objects.append(review)

    tmp = dict(final_record)
    stored_final = tmp.pop("record_sha256")
    if stored_final != canonical_json_sha256(tmp):
        raise ValueError("C1B.2 final record SHA drifted")
    final = FreshCFinalAdjudication.model_validate(final_record["adjudication"])
    validate_final_against_reviews(final, review_objects)
    if final_record.get("preservation_does_not_establish_absence_or_novelty") is not True:
        raise ValueError("C1B.2 scoped-preservation disclaimer missing")

    tmp_run = dict(run)
    run_id = tmp_run.pop("run_id")
    run_sha = tmp_run.pop("run_sha256")
    if run_sha != canonical_json_sha256(tmp_run):
        raise ValueError("C1B.2 run SHA drifted")
    if run_id != "sers_fresh_c_c1b2_scientific_adjudication_run_v1:" + run_sha[:20]:
        raise ValueError("C1B.2 run ID drifted")
    if complete.get("run_id") != run_id or complete.get("run_sha256") != run_sha:
        raise ValueError("C1B.2 COMPLETE marker drifted")

    print("Fresh-C C1B.2 scientific-adjudication result verifier")
    print(f"Run ID: {run_id}")
    print(f"Run SHA256: {run_sha}")
    print("Paper reviews: 25/25")
    print("Final adjudication: VALID")
    print(f"H1 Fresh-C verdict: {final.h1_fresh_c_verdict}")
    print("H2 remains REJECT_AS_FORMULATED: True")
    print(f"H3 Fresh-C verdict: {final.h3_fresh_c_verdict}")
    print("All evidence page/quote grounding: VALID")
    print("Scientific LLM calls: 26")
    print("Scientific network calls: 26")
    print("External literature used: False")
    print("Count thresholds used: False")
    print("Hypothesis rewrite/upgrade: False/False")
    print("Same-epoch rerun allowed: False")
    print("Automatic post-C1B.2 transition: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
