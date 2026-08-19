from __future__ import annotations

import subprocess
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b2_scientific_adjudication_v1 import (
    DEFAULT_RESULT_FREEZE_DIR,
    canonical_json_sha256,
    load_object,
)
from campaigns.sers_alpha4_epoch.fresh_c.cli.verify_sers_fresh_c_c1b2_scientific_result_v1 import main as verify_result

def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def main():
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    verify_result()
    m = load_object(root / DEFAULT_RESULT_FREEZE_DIR / "freeze_manifest.json")
    ready = load_object(root / DEFAULT_RESULT_FREEZE_DIR / "FREEZE_READY.json")
    tmp = dict(m)
    stored = tmp.pop("manifest_sha256")
    if stored != canonical_json_sha256(tmp):
        raise ValueError("C1B.2 result freeze SHA drifted")
    if ready["freeze_id"] != m["freeze_id"] or ready["manifest_sha256"] != stored:
        raise ValueError("C1B.2 result FREEZE_READY drifted")
    if m.get("source_identity_count") != 25:
        raise ValueError("C1B.2 frozen source count drifted")
    if m.get("scientific_llm_calls") != 26 or m.get("scientific_network_calls") != 26:
        raise ValueError("C1B.2 frozen call counts drifted")
    if m.get("fresh_c_scientific_text_read_performed") is not True:
        raise ValueError("C1B.2 frozen read state drifted")
    if m.get("scientific_adjudication_performed") is not True:
        raise ValueError("C1B.2 frozen adjudication state drifted")
    if m.get("h2_terminal_state") != "REJECT_AS_FORMULATED":
        raise ValueError("C1B.2 H2 terminal state drifted")
    if m.get("h2_resurrected") is not False:
        raise ValueError("C1B.2 H2 resurrection drifted")
    for key in (
        "external_literature_used",
        "count_threshold_used",
        "hypothesis_rewrite_performed",
        "hypothesis_upgrade_performed",
        "same_epoch_rerun_allowed",
        "automatic_next_stage_authorized",
    ):
        if m.get(key) is not False:
            raise ValueError(f"C1B.2 frozen safety field drifted: {key}")
    if m.get("preservation_does_not_establish_absence_or_novelty") is not True:
        raise ValueError("C1B.2 scoped-preservation disclaimer drifted")
    if m.get("stop") is not True:
        raise ValueError("C1B.2 result freeze STOP drifted")

    print("Fresh-C C1B.2 scientific-adjudication result freeze verifier")
    print(f"Freeze ID: {m['freeze_id']}")
    print(f"Manifest SHA256: {stored}")
    print(f"Source run ID: {m['source_run_id']}")
    print(f"H1 Fresh-C verdict: {m['h1_fresh_c_verdict']}")
    print("H2 remains REJECT_AS_FORMULATED: True")
    print(f"H3 Fresh-C verdict: {m['h3_fresh_c_verdict']}")
    print("Paper reviews frozen: 25")
    print("Scientific LLM/network calls: 26/26")
    print("External literature used: False")
    print("Hypothesis rewrite/upgrade: False/False")
    print("Same-epoch rerun allowed: False")
    print("Automatic next stage authorized: False")
    print("STOP: True")
    print("Verification: PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
