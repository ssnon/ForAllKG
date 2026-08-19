import subprocess
from pathlib import Path
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_scientific_adjudication_v1 import atomic_json,canonical_json_sha256,load_object
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_r1_quote_grounding_recovery_v1 import DEFAULT_RUN_DIR,DEFAULT_RESULT_FREEZE_DIR
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_c1b2_r1_scientific_recovery_result_v1 import main as verify
def root():return Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
def main():
 rt=root();verify();run=load_object(rt/DEFAULT_RUN_DIR/"run_manifest.json");final=load_object(rt/DEFAULT_RUN_DIR/"final_adjudication.json")["adjudication"]
 body={"schema_version":"sers-fresh-c-c1b2-r1-scientific-recovery-result-freeze-v1",
       "source_run_id":run["run_id"],"source_run_sha256":run["run_sha256"],
       "h1_fresh_c_verdict":final["h1_fresh_c_verdict"],"h2_terminal_state":final["h2_terminal_state"],
       "h2_resurrected":False,"h3_fresh_c_verdict":final["h3_fresh_c_verdict"],
       "recovery_llm_calls":26,"recovery_network_calls":26,
       "recovery_after_parent_validation_failure":True,"new_fresh_reserve_claimed":False,
       "failed_parent_response_reused":False,"verbatim_quote_evidence_enabled":False,
       "external_literature_used":False,"hypothesis_rewrite_performed":False,
       "hypothesis_upgrade_performed":False,"same_recovery_epoch_rerun_allowed":False,
       "automatic_next_stage_authorized":False,"stop":True}
 ident=canonical_json_sha256(body);body["freeze_id"]="sers_fresh_c_c1b2_r1_scientific_recovery_result_freeze_v1:"+ident[:20]
 tmp=dict(body);body["manifest_sha256"]=canonical_json_sha256(tmp);out=rt/DEFAULT_RESULT_FREEZE_DIR
 if out.exists():raise FileExistsError("Recovery result freeze exists")
 atomic_json(out/"freeze_manifest.json",body);atomic_json(out/"FREEZE_READY.json",
  {"freeze_id":body["freeze_id"],"manifest_sha256":body["manifest_sha256"],"automatic_next_stage_authorized":False,"stop":True})
 print("Fresh-C C1B.2-R1 scientific recovery result freeze");print(f"Freeze ID: {body['freeze_id']}")
 print(f"Manifest SHA256: {body['manifest_sha256']}");print(f"H1 Fresh-C verdict: {body['h1_fresh_c_verdict']}")
 print(f"H3 Fresh-C verdict: {body['h3_fresh_c_verdict']}");print("New Fresh-C reserve claimed: False")
 print("Automatic next stage authorized: False");print("STOP: True");return 0
if __name__=="__main__":raise SystemExit(main())
