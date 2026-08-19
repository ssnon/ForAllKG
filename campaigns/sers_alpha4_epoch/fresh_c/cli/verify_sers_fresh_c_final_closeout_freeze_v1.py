import hashlib
import subprocess

from campaigns.sers_alpha4_epoch.fresh_c.sers_fresh_c_final_closeout_v1 import (
    DEFAULT_CLOSEOUT_DIR,
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    canonical_json_sha256,
    git_root,
    load_object,
    validate_protocol,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.freeze_sers_fresh_c_final_closeout_v1 import CRITICAL_COMPONENTS
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_final_closeout_v1 import main as verify_closeout

def main():
    root = git_root()
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    verify_closeout()
    closeout = load_object(root / DEFAULT_CLOSEOUT_DIR / "closeout_manifest.json")
    m = load_object(root / DEFAULT_FREEZE_DIR / "freeze_manifest.json")
    ready = load_object(root / DEFAULT_FREEZE_DIR / "FREEZE_READY.json")

    tmp = dict(m)
    stored = tmp.pop("manifest_sha256")
    if stored != canonical_json_sha256(tmp):
        raise ValueError("Final closeout freeze SHA drifted")
    if m["source_closeout_id"] != closeout["closeout_id"]:
        raise ValueError("Final closeout freeze source ID drifted")
    if m["source_closeout_sha256"] != closeout["closeout_sha256"]:
        raise ValueError("Final closeout freeze source SHA drifted")
    if m["protocol_id"] != p["protocol_id"]:
        raise ValueError("Final closeout freeze protocol ID drifted")
    if m["protocol_sha256"] != p["protocol_sha256"]:
        raise ValueError("Final closeout freeze protocol SHA drifted")

    source = m["source_code_commit"]
    for rel in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source}:{rel}"], cwd=root
        )
        sha = hashlib.sha256(committed).hexdigest()
        if m["critical_component_sha256"].get(rel) != sha:
            raise ValueError(f"Frozen closeout component mismatch: {rel}")
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Current closeout component drifted: {rel}")

    if ready["freeze_id"] != m["freeze_id"]:
        raise ValueError("Closeout FREEZE_READY ID drifted")
    if ready["manifest_sha256"] != stored:
        raise ValueError("Closeout FREEZE_READY SHA drifted")
    if m.get("campaign_closed") is not True:
        raise ValueError("Frozen campaign not closed")
    if m.get("network_calls_during_freeze") != 0:
        raise ValueError("Closeout freeze unexpectedly used network")
    if m.get("llm_calls_during_freeze") != 0:
        raise ValueError("Closeout freeze unexpectedly used LLM")
    if m.get("automatic_next_stage_authorized") is not False:
        raise ValueError("Frozen closeout authorizes next stage")
    if m.get("stop") is not True:
        raise ValueError("Frozen closeout STOP drifted")

    print("SERS Fresh-C final closeout freeze verifier")
    print(f"Freeze ID: {m['freeze_id']}")
    print(f"Manifest SHA256: {stored}")
    print(f"Source closeout ID: {m['source_closeout_id']}")
    print(f"H1 final state: {m['final_h1_state']}")
    print(f"H2 final state: {m['final_h2_state']}")
    print(f"H3 final state: {m['final_h3_state']}")
    print("Campaign closed: True")
    print("Automatic next stage authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
