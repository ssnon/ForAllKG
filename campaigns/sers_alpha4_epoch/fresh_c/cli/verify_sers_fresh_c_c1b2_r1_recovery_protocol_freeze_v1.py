import hashlib, subprocess
from pathlib import Path
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_scientific_adjudication_v1 import canonical_json_sha256, load_object, sha256_file
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_r1_quote_grounding_recovery_v1 import (
 DEFAULT_PROTOCOL_PATH,DEFAULT_SCHEMA_QUALIFICATION_DIR,DEFAULT_PROTOCOL_FREEZE_DIR,
 validate_protocol,validate_parent_failure_state,validate_schema_qualification,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.freeze_sers_fresh_c_c1b2_r1_recovery_protocol_v1 import CRITICAL
def root():return Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
def main():
 rt=root();p=validate_protocol(rt/DEFAULT_PROTOCOL_PATH);validate_parent_failure_state(rt)
 qp=rt/DEFAULT_SCHEMA_QUALIFICATION_DIR/"qualification_result.json";q=load_object(qp);validate_schema_qualification(q)
 m=load_object(rt/DEFAULT_PROTOCOL_FREEZE_DIR/"freeze_manifest.json");r=load_object(rt/DEFAULT_PROTOCOL_FREEZE_DIR/"FREEZE_READY.json")
 tmp=dict(m);stored=tmp.pop("manifest_sha256")
 if stored!=canonical_json_sha256(tmp):raise ValueError("Recovery freeze SHA drifted")
 if m["protocol_id"]!=p["protocol_id"] or m["protocol_sha256"]!=p["protocol_sha256"]:raise ValueError("Protocol binding drifted")
 if m["schema_qualification_file_sha256"]!=sha256_file(qp):raise ValueError("Qualification file SHA drifted")
 source=m["source_code_commit"]
 for rel in CRITICAL:
  committed=subprocess.check_output(["git","show",f"{source}:{rel}"],cwd=rt);sha=hashlib.sha256(committed).hexdigest()
  if m["critical_component_sha256"].get(rel)!=sha:raise ValueError(f"Frozen component drifted: {rel}")
  if hashlib.sha256((rt/rel).read_bytes()).hexdigest()!=sha:raise ValueError(f"Current component drifted: {rel}")
 if r["freeze_id"]!=m["freeze_id"] or r["manifest_sha256"]!=stored:raise ValueError("FREEZE_READY drifted")
 if m["recovery_execution_ready"] is not True or m["recovery_execution_authorized"] is not False:raise ValueError("Recovery readiness drifted")
 if m["additional_scientific_text_read_during_freeze"] is not False:raise ValueError("Freeze read science")
 if m["network_calls_during_freeze"]!=0 or m["llm_calls_during_freeze"]!=0:raise ValueError("Freeze used network/LLM")
 if m["stop"] is not True:raise ValueError("STOP drifted")
 print("Fresh-C C1B.2-R1 recovery protocol freeze verifier")
 print(f"Freeze ID: {m['freeze_id']}");print(f"Manifest SHA256: {stored}")
 print("Parent failed C1B.2 state: CURRENT");print("Quote-null recovery schema: CURRENT")
 print("Recovery components: CURRENT");print("Additional scientific read during verification: False")
 print("Network/LLM calls during verification: 0/0");print("Recovery execution ready: True")
 print("Recovery execution authorized: False");print("STOP: True");print("Verification: PASS");return 0
if __name__=="__main__":raise SystemExit(main())
