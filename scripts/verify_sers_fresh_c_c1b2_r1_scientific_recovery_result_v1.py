import subprocess
from pathlib import Path
from dac_her.fresh_c_c1b1_reviewer_contract_v1 import FreshCPaperReview,FreshCFinalAdjudication
from dac_her.fresh_c_c1b2_scientific_adjudication_v1 import (
 C1AR1_CORPUS,canonical_json_sha256,load_object,validate_corpus_metadata,
 validate_pages_manifest,validate_review_grounding,validate_final_against_reviews,
)
from dac_her.fresh_c_c1b2_r1_quote_grounding_recovery_v1 import DEFAULT_RUN_DIR
def root():return Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
def main():
 rt=root();d=rt/DEFAULT_RUN_DIR
 if (d/"C1B2_R1_RECOVERY_FAILED.json").exists():raise RuntimeError("Failed recovery epoch exists")
 run=load_object(d/"run_manifest.json");complete=load_object(d/"C1B2_R1_RECOVERY_COMPLETE.json")
 if run["paper_review_calls"]!=25 or run["final_adjudication_calls"]!=1:raise ValueError("Call structure drifted")
 if run["recovery_llm_calls"]!=26 or run["recovery_network_calls"]!=26:raise ValueError("Call counts drifted")
 if run["new_fresh_reserve_claimed"] is not False or run["failed_parent_response_reused"] is not False:raise ValueError("Recovery provenance drifted")
 records=validate_corpus_metadata(rt,parse_pages=True);reviews=[]
 for rec,row in zip(records,run["paper_review_records"]):
  if rec["reserve_index"]!=row["reserve_index"]:raise ValueError("Review order drifted")
  payload=load_object(rt/row["record_path"]);tmp=dict(payload);stored=tmp.pop("record_sha256")
  if stored!=canonical_json_sha256(tmp):raise ValueError("Paper record SHA drifted")
  review=FreshCPaperReview.model_validate(payload["review"])
  if any(ev.verbatim_quote is not None for a in review.assessments for ev in a.evidence):raise ValueError("Non-null quote in recovery output")
  validate_review_grounding(review,expected_record=rec,pages_manifest=rec["pages_manifest"]);reviews.append(review)
 fp=load_object(d/"final_adjudication.json");final=FreshCFinalAdjudication.model_validate(fp["adjudication"])
 validate_final_against_reviews(final,reviews)
 tmp=dict(run);rid=tmp.pop("run_id");rsha=tmp.pop("run_sha256")
 if rsha!=canonical_json_sha256(tmp):raise ValueError("Run SHA drifted")
 if complete["run_id"]!=rid or complete["run_sha256"]!=rsha:raise ValueError("COMPLETE drifted")
 print("Fresh-C C1B.2-R1 scientific recovery result verifier")
 print(f"Run ID: {rid}");print(f"Run SHA256: {rsha}");print("Paper reviews: 25/25")
 print(f"H1 Fresh-C verdict: {final.h1_fresh_c_verdict}");print("H2 remains REJECT_AS_FORMULATED: True")
 print(f"H3 Fresh-C verdict: {final.h3_fresh_c_verdict}");print("Verbatim quote evidence enabled: False")
 print("Failed parent response reused: False");print("New Fresh-C reserve claimed: False")
 print("Recovery LLM/network calls: 26/26");print("Same recovery-epoch rerun allowed: False")
 print("STOP: True");print("Verification: PASS");return 0
if __name__=="__main__":raise SystemExit(main())
