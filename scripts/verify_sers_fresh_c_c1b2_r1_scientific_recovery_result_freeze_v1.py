import subprocess
from pathlib import Path
from dac_her.fresh_c_c1b2_scientific_adjudication_v1 import canonical_json_sha256,load_object
from dac_her.fresh_c_c1b2_r1_quote_grounding_recovery_v1 import DEFAULT_RESULT_FREEZE_DIR
from scripts.verify_sers_fresh_c_c1b2_r1_scientific_recovery_result_v1 import main as verify
def root():return Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
def main():
 rt=root();verify();m=load_object(rt/DEFAULT_RESULT_FREEZE_DIR/"freeze_manifest.json");r=load_object(rt/DEFAULT_RESULT_FREEZE_DIR/"FREEZE_READY.json")
 tmp=dict(m);stored=tmp.pop("manifest_sha256")
 if stored!=canonical_json_sha256(tmp):raise ValueError("Result freeze SHA drifted")
 if r["freeze_id"]!=m["freeze_id"] or r["manifest_sha256"]!=stored:raise ValueError("FREEZE_READY drifted")
 for k in ("new_fresh_reserve_claimed","failed_parent_response_reused","verbatim_quote_evidence_enabled",
           "external_literature_used","hypothesis_rewrite_performed","hypothesis_upgrade_performed",
           "same_recovery_epoch_rerun_allowed","automatic_next_stage_authorized"):
  if m[k] is not False:raise ValueError(f"Safety field drifted: {k}")
 if m["stop"] is not True:raise ValueError("STOP drifted")
 print("Fresh-C C1B.2-R1 recovery result freeze verifier");print(f"Freeze ID: {m['freeze_id']}")
 print(f"Manifest SHA256: {stored}");print(f"H1 Fresh-C verdict: {m['h1_fresh_c_verdict']}")
 print(f"H3 Fresh-C verdict: {m['h3_fresh_c_verdict']}");print("New Fresh-C reserve claimed: False")
 print("Failed parent response reused: False");print("Verbatim quote evidence enabled: False")
 print("Automatic next stage authorized: False");print("STOP: True");print("Verification: PASS");return 0
if __name__=="__main__":raise SystemExit(main())
