import hashlib
import subprocess

from dac_her.sers_fresh_c_final_closeout_v1 import (
    DEFAULT_CLOSEOUT_DIR,
    DEFAULT_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    atomic_json,
    canonical_json_sha256,
    git_root,
    load_object,
    validate_protocol,
)
from scripts.verify_sers_fresh_c_final_closeout_v1 import main as verify_closeout

CRITICAL_COMPONENTS = (
    "dac_her/sers_fresh_c_final_closeout_v1.py",
    "dac_her/sers_fresh_c_final_closeout_protocol_v1.json",
    "scripts/verify_sers_fresh_c_final_closeout_protocol_v1.py",
    "scripts/build_sers_fresh_c_final_closeout_v1.py",
    "scripts/verify_sers_fresh_c_final_closeout_v1.py",
    "scripts/freeze_sers_fresh_c_final_closeout_v1.py",
    "scripts/verify_sers_fresh_c_final_closeout_freeze_v1.py",
    "tests/test_sers_fresh_c_final_closeout_v1.py",
)

def main():
    root = git_root()
    if subprocess.run(["git", "diff", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Tracked worktree dirty; refuse final closeout freeze")
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--"], cwd=root).returncode:
        raise RuntimeError("Index dirty; refuse final closeout freeze")

    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    verify_closeout()
    closeout = load_object(root / DEFAULT_CLOSEOUT_DIR / "closeout_manifest.json")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()

    hashes = {}
    for rel in CRITICAL_COMPONENTS:
        committed = subprocess.check_output(
            ["git", "show", f"{source_commit}:{rel}"], cwd=root
        )
        sha = hashlib.sha256(committed).hexdigest()
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"Closeout component drifted: {rel}")
        hashes[rel] = sha

    body = {
        "schema_version": "sers-fresh-c-final-closeout-freeze-v1",
        "protocol_id": p["protocol_id"],
        "protocol_sha256": p["protocol_sha256"],
        "source_closeout_id": closeout["closeout_id"],
        "source_closeout_sha256": closeout["closeout_sha256"],
        "source_code_commit": source_commit,
        "critical_component_sha256": hashes,
        "final_h1_state": p["final_h1_state"],
        "final_h2_state": p["final_h2_state"],
        "final_h3_state": p["final_h3_state"],
        "campaign_closed": True,
        "network_calls_during_freeze": 0,
        "llm_calls_during_freeze": 0,
        "scientific_text_read_during_freeze": False,
        "scientific_adjudication_during_freeze": False,
        "automatic_next_stage_authorized": False,
        "stop": True,
    }
    ident = canonical_json_sha256(body)
    body["freeze_id"] = (
        "sers_fresh_c_final_closeout_freeze_v1:" + ident[:20]
    )
    tmp = dict(body)
    body["manifest_sha256"] = canonical_json_sha256(tmp)

    out = root / DEFAULT_FREEZE_DIR
    if out.exists():
        raise FileExistsError("Final closeout freeze directory already exists")
    atomic_json(out / "freeze_manifest.json", body)
    atomic_json(out / "FREEZE_READY.json", {
        "freeze_id": body["freeze_id"],
        "manifest_sha256": body["manifest_sha256"],
        "campaign_closed": True,
        "automatic_next_stage_authorized": False,
        "stop": True,
    })

    print("SERS Fresh-C final closeout freeze")
    print(f"Freeze ID: {body['freeze_id']}")
    print(f"Manifest SHA256: {body['manifest_sha256']}")
    print(f"Source closeout ID: {body['source_closeout_id']}")
    print(f"Source code commit: {source_commit}")
    print("Campaign closed: True")
    print("Network/LLM calls during freeze: 0/0")
    print("Automatic next stage authorized: False")
    print("STOP: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
