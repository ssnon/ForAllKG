from campaigns.sers_alpha4_epoch.fresh_c.sers_fresh_c_final_closeout_v1 import (
    DEFAULT_CLOSEOUT_DIR,
    DEFAULT_PROTOCOL_PATH,
    canonical_json_sha256,
    git_root,
    load_object,
    validate_final_scientific_state,
    validate_protocol,
)

def main():
    root = git_root()
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    validate_final_scientific_state(root)
    closeout = load_object(root / DEFAULT_CLOSEOUT_DIR / "closeout_manifest.json")
    complete = load_object(root / DEFAULT_CLOSEOUT_DIR / "CLOSEOUT_COMPLETE.json")

    tmp = dict(closeout)
    closeout_id = tmp.pop("closeout_id")
    stored = tmp.pop("closeout_sha256")
    if stored != canonical_json_sha256(tmp):
        raise ValueError("Closeout SHA drifted")
    if closeout_id != "sers_fresh_c_final_closeout_v1:" + stored[:20]:
        raise ValueError("Closeout ID drifted")
    if closeout["protocol_id"] != p["protocol_id"]:
        raise ValueError("Closeout protocol ID drifted")
    if closeout["protocol_sha256"] != p["protocol_sha256"]:
        raise ValueError("Closeout protocol SHA drifted")
    if complete["closeout_id"] != closeout_id:
        raise ValueError("CLOSEOUT_COMPLETE ID drifted")
    if complete["closeout_sha256"] != stored:
        raise ValueError("CLOSEOUT_COMPLETE SHA drifted")
    if closeout.get("campaign_closed") is not True:
        raise ValueError("Campaign not closed")
    if closeout.get("automatic_next_stage_authorized") is not False:
        raise ValueError("Closeout unexpectedly authorizes next stage")
    if closeout.get("stop") is not True:
        raise ValueError("Closeout STOP drifted")

    print("SERS Fresh-C final closeout verifier")
    print(f"Closeout ID: {closeout_id}")
    print(f"Closeout SHA256: {stored}")
    print(f"H1 final state: {closeout['final_scientific_state']['H1']}")
    print(f"H2 final state: {closeout['final_scientific_state']['H2']}")
    print(f"H3 final state: {closeout['final_scientific_state']['H3']}")
    print("Accepted scientific outputs: 26")
    print("C1B.2 scientific call attempts: 27")
    print("Campaign closed: True")
    print("Automatic next stage authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
