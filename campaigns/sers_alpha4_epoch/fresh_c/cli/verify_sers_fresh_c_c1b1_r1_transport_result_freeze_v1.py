import subprocess
from pathlib import Path
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b1_r1_transport_qualification_v1 import (
    DEFAULT_RESULT_FREEZE_DIR,
    canonical_json_sha256,
    load_object,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_c1b1_r1_transport_result_v1 import main as verify_result

def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def main():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_result()
    m = load_object(root / DEFAULT_RESULT_FREEZE_DIR / "freeze_manifest.json")
    ready = load_object(root / DEFAULT_RESULT_FREEZE_DIR / "FREEZE_READY.json")
    tmp = dict(m)
    stored = tmp.pop("manifest_sha256")
    if stored != canonical_json_sha256(tmp):
        raise ValueError("Transport result freeze SHA drifted")
    if ready["freeze_id"] != m["freeze_id"] or ready["manifest_sha256"] != stored:
        raise ValueError("Transport result FREEZE_READY mismatch")
    if m.get("catalog_membership_verified") is not True:
        raise ValueError("Catalog membership not frozen")
    if m.get("structured_json_schema_call_passed") is not True:
        raise ValueError("Structured qualification not frozen")
    for key in (
        "fresh_c_scientific_text_semantic_read_performed",
        "scientific_adjudication_performed",
        "c1b2_authorized",
    ):
        if m.get(key) is not False:
            raise ValueError(f"Safety field drifted: {key}")
    if m.get("stop") is not True:
        raise ValueError("STOP drifted")
    print("Fresh-C C1B.1-R1 transport result freeze verifier")
    print(f"Freeze ID: {m['freeze_id']}")
    print(f"Manifest SHA256: {stored}")
    print(f"Corrected transport: {m['corrected_transport_semantics']}")
    print(f"Requested model: {m['requested_model']}")
    print(f"Served model: {m['served_model']}")
    print("Catalog membership verified: True")
    print("Structured JSON-schema call passed: True")
    print("Fresh-C scientific text read: False")
    print("Scientific adjudication: False")
    print("C1B.2 authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
