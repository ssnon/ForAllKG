import subprocess
from pathlib import Path

from dac_her.sers_fresh_c_final_closeout_v1 import (
    DEFAULT_CLOSEOUT_DIR,
    DEFAULT_PROTOCOL_PATH,
    UPSTREAM_LINEAGE,
    atomic_json,
    canonical_json_sha256,
    git_root,
    validate_final_scientific_state,
    validate_protocol,
)

def main():
    root = git_root()
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    state = validate_final_scientific_state(root)
    out = root / DEFAULT_CLOSEOUT_DIR
    if out.exists():
        raise FileExistsError("Final closeout directory already exists")

    body = {
        "schema_version": "sers-fresh-c-final-closeout-v1",
        "protocol_id": p["protocol_id"],
        "protocol_sha256": p["protocol_sha256"],
        "final_scientific_commit": p["final_scientific_commit"],
        "closeout_source_head": state["current_head"],
        "final_run_id": p["final_run_id"],
        "final_run_sha256": p["final_run_sha256"],
        "final_result_freeze_id": p["final_result_freeze_id"],
        "final_result_freeze_sha256": p["final_result_freeze_sha256"],
        "final_scientific_state": {
            "H1": p["final_h1_state"],
            "H2": p["final_h2_state"],
            "H3": p["final_h3_state"],
        },
        "scientific_accounting": {
            "accepted_paper_reviews": 25,
            "accepted_final_adjudications": 1,
            "accepted_scientific_outputs": 26,
            "original_failed_c1b2_scientific_call_attempts": 1,
            "recovery_scientific_call_attempts": 26,
            "total_c1b2_scientific_call_attempts": 27,
        },
        "failed_epoch_lineage": {
            "parent_protocol_freeze_id": (
                "sers_fresh_c_c1b2_scientific_protocol_freeze_v1:"
                "cd9065ffee576865bd09"
            ),
            "parent_protocol_freeze_sha256": (
                "01bde9481335febe4ddec8a18405a31736e400700e10cbdb3e6b240f6e740202"
            ),
            "completed_parent_reviews": 0,
            "parent_scientific_call_attempts": 1,
            "same_parent_epoch_rerun_allowed": False,
            "failure_restored_freshness": False,
        },
        "recovery_lineage": {
            "recovery_protocol_freeze_id": (
                "sers_fresh_c_c1b2_r1_recovery_protocol_freeze_v1:"
                "a788b204dd503ab64f5e"
            ),
            "recovery_protocol_freeze_sha256": (
                "698e619a5931f4c251e4aa1feed65cdd29f4b7924301432cf4909f4fd30044b3"
            ),
            "failed_parent_response_reused": False,
            "new_fresh_reserve_claimed": False,
            "verbatim_quote_evidence_enabled": False,
        },
        "epistemic_guards": {
            "external_literature_used_during_c1b2": False,
            "count_threshold_used": False,
            "hypothesis_rewrite_performed": False,
            "hypothesis_upgrade_performed": False,
            "h2_resurrected": False,
            "h1_preservation_is_fresh_c_scoped_only": True,
            "h3_erosion_is_fresh_c_scoped_only": True,
            "literature_wide_novelty_claim_authorized": False,
        },
        "upstream_lineage": UPSTREAM_LINEAGE,
        "closeout_execution": {
            "network_calls": 0,
            "llm_calls": 0,
            "scientific_text_read": False,
            "scientific_adjudication": False,
        },
        "campaign_closed": True,
        "automatic_next_stage_authorized": False,
        "stop": True,
    }
    body["closeout_sha256"] = canonical_json_sha256(body)
    body["closeout_id"] = (
        "sers_fresh_c_final_closeout_v1:" + body["closeout_sha256"][:20]
    )
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "closeout_manifest.json", body)
    atomic_json(out / "CLOSEOUT_COMPLETE.json", {
        "closeout_id": body["closeout_id"],
        "closeout_sha256": body["closeout_sha256"],
        "campaign_closed": True,
        "automatic_next_stage_authorized": False,
        "stop": True,
    })

    print("SERS Fresh-C final closeout")
    print(f"Closeout ID: {body['closeout_id']}")
    print(f"Closeout SHA256: {body['closeout_sha256']}")
    print(f"H1 final state: {body['final_scientific_state']['H1']}")
    print(f"H2 final state: {body['final_scientific_state']['H2']}")
    print(f"H3 final state: {body['final_scientific_state']['H3']}")
    print("Accepted scientific outputs: 26")
    print("C1B.2 scientific call attempts: 27")
    print("Closeout network/LLM calls: 0/0")
    print("Campaign closed: True")
    print("Automatic next stage authorized: False")
    print("STOP: True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
