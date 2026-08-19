import hashlib, subprocess
from pathlib import Path
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_scientific_adjudication_v1 import atomic_json, canonical_json_sha256, load_object, sha256_file
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_r1_quote_grounding_recovery_v1 import (
    DEFAULT_PROTOCOL_PATH,DEFAULT_SCHEMA_QUALIFICATION_DIR,DEFAULT_PROTOCOL_FREEZE_DIR,
    validate_protocol,validate_parent_failure_state,validate_schema_qualification,
)
CRITICAL=(
"campaigns/sers_alpha4_epoch/fresh_c/fresh_c_c1b2_r1_quote_grounding_recovery_v1.py",
"dac_her/sers_fresh_c_c1b2_r1_recovery_protocol_v1.json",
"campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b2_r1_recovery_protocol_v1.py",
"campaigns/sers_alpha4_epoch/fresh_c/cli/run_sers_fresh_c_c1b2_r1_quote_grounding_recovery_v1.py",
"campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_c1b2_r1_recovery_protocol_v1.py",
"campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b2_r1_recovery_protocol_freeze_v1.py",
"campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b2_r1_scientific_recovery_result_v1.py",
"campaigns/sers_alpha4_epoch/fresh_c/cli/freeze_sers_fresh_c_c1b2_r1_scientific_recovery_result_v1.py",
"campaigns/sers_alpha4_epoch/fresh_c/cli/verify_sers_fresh_c_c1b2_r1_scientific_recovery_result_freeze_v1.py",
"tests/test_sers_fresh_c_c1b2_r1_quote_grounding_recovery_v1.py",
)
def root(): return Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
def main():
 rt=root()
 if subprocess.run(["git","diff","--quiet","--"],cwd=rt).returncode: raise RuntimeError("Tracked worktree dirty")
 if subprocess.run(["git","diff","--cached","--quiet","--"],cwd=rt).returncode: raise RuntimeError("Index dirty")
 p=validate_protocol(rt/DEFAULT_PROTOCOL_PATH); validate_parent_failure_state(rt)
 qp=rt/DEFAULT_SCHEMA_QUALIFICATION_DIR/"qualification_result.json"; q=load_object(qp)
 validate_schema_qualification(q)
 if q["protocol_id"]!=p["protocol_id"] or q["protocol_sha256"]!=p["protocol_sha256"]: raise ValueError("Qualification binding drifted")
 source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=rt,text=True).strip(); hashes={}
 for rel in CRITICAL:
  committed=subprocess.check_output(["git","show",f"{source}:{rel}"],cwd=rt)
  sha=hashlib.sha256(committed).hexdigest()
  if hashlib.sha256((rt/rel).read_bytes()).hexdigest()!=sha: raise RuntimeError(f"Component drifted: {rel}")
  hashes[rel]=sha
 body={"schema_version":"sers-fresh-c-c1b2-r1-recovery-protocol-freeze-v1",
       "protocol_id":p["protocol_id"],"protocol_sha256":p["protocol_sha256"],
       "parent_protocol_freeze_id":p["parent_protocol_freeze_id"],
       "parent_protocol_freeze_sha256":p["parent_protocol_freeze_sha256"],
       "source_code_commit":source,"critical_component_sha256":hashes,
       "schema_qualification_id":q["qualification_id"],
       "schema_qualification_sha256":q["qualification_sha256"],
       "schema_qualification_file_sha256":sha256_file(qp),
       "recovery_schema_adapter_id":p["recovery_schema_adapter_id"],
       "paper_recovery_transport_schema_sha256":p["paper_recovery_transport_schema_sha256"],
       "final_recovery_transport_schema_sha256":p["final_recovery_transport_schema_sha256"],
       "additional_scientific_text_read_during_freeze":False,
       "scientific_adjudication_performed_during_freeze":False,
       "network_calls_during_freeze":0,"llm_calls_during_freeze":0,
       "recovery_execution_ready":True,"recovery_execution_authorized":False,
       "automatic_post_recovery_transition_allowed":False,"stop":True}
 ident=canonical_json_sha256(body); body["freeze_id"]="sers_fresh_c_c1b2_r1_recovery_protocol_freeze_v1:"+ident[:20]
 tmp=dict(body); body["manifest_sha256"]=canonical_json_sha256(tmp)
 out=rt/DEFAULT_PROTOCOL_FREEZE_DIR
 if out.exists(): raise FileExistsError("Recovery protocol freeze exists")
 atomic_json(out/"freeze_manifest.json",body)
 atomic_json(out/"FREEZE_READY.json",{"freeze_id":body["freeze_id"],"manifest_sha256":body["manifest_sha256"],
                                      "recovery_execution_ready":True,"recovery_execution_authorized":False,"stop":True})
 print("Fresh-C C1B.2-R1 recovery protocol freeze")
 print(f"Freeze ID: {body['freeze_id']}"); print(f"Manifest SHA256: {body['manifest_sha256']}")
 print(f"Source code commit: {source}"); print("Quote-null recovery schema qualified: True")
 print("Additional scientific read during freeze: False"); print("Network/LLM calls during freeze: 0/0")
 print("Recovery execution ready: True"); print("Recovery execution authorized: False"); print("STOP: True")
 return 0
if __name__=="__main__": raise SystemExit(main())
