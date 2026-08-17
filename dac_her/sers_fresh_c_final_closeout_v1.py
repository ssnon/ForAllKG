from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


STAGE = "FINAL_CLOSEOUT"
SEMANTICS_ID = "sers_fresh_c_final_closeout_v1"
PROTOCOL_PREFIX = "sers_fresh_c_final_closeout_protocol_v1"

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_final_closeout_protocol_v1.json"
)
DEFAULT_CLOSEOUT_DIR = Path(
    "evaluation/sers_fresh_c/final_closeout_v1"
)
DEFAULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/final_closeout_freeze_v1"
)

FINAL_SCIENTIFIC_COMMIT = "2d09162a4da0590fe7e91a754bad3e4f49c41646"

FINAL_RUN_PATH = Path(
    "evaluation/sers_fresh_c/"
    "c1b2_r1_scientific_recovery_run_v1/run_manifest.json"
)
FINAL_ADJUDICATION_PATH = Path(
    "evaluation/sers_fresh_c/"
    "c1b2_r1_scientific_recovery_run_v1/final_adjudication.json"
)
FINAL_RESULT_FREEZE_PATH = Path(
    "evaluation/sers_fresh_c/"
    "c1b2_r1_scientific_recovery_result_freeze_v1/freeze_manifest.json"
)
PARENT_FAILED_START_PATH = Path(
    "evaluation/sers_fresh_c/"
    "c1b2_scientific_adjudication_run_v1/"
    "C1B2_SCIENTIFIC_READ_STARTED.json"
)
PARENT_FAILED_MARKER_PATH = Path(
    "evaluation/sers_fresh_c/"
    "c1b2_scientific_adjudication_run_v1/"
    "C1B2_SCIENTIFIC_ADJUDICATION_FAILED.json"
)
RECOVERY_PROTOCOL_FREEZE_PATH = Path(
    "evaluation/sers_fresh_c/"
    "c1b2_r1_recovery_protocol_freeze_v1/freeze_manifest.json"
)

EXPECTED_FINAL_RUN_ID = (
    "sers_fresh_c_c1b2_r1_scientific_recovery_run_v1:"
    "f48db30f437c4555ac47"
)
EXPECTED_FINAL_RUN_SHA256 = (
    "f48db30f437c4555ac47f47764a587bea4c55b0ba168c6c8fe09d1ffb875f476"
)
EXPECTED_FINAL_RESULT_FREEZE_ID = (
    "sers_fresh_c_c1b2_r1_scientific_recovery_result_freeze_v1:"
    "76b31a3c265fd75ed891"
)
EXPECTED_FINAL_RESULT_FREEZE_SHA256 = (
    "55fff4474e3e4b84479652d7d711a97d6deb2d1f6518375c9a13266c8c84c383"
)
EXPECTED_RECOVERY_PROTOCOL_FREEZE_ID = (
    "sers_fresh_c_c1b2_r1_recovery_protocol_freeze_v1:"
    "a788b204dd503ab64f5e"
)
EXPECTED_RECOVERY_PROTOCOL_FREEZE_SHA256 = (
    "698e619a5931f4c251e4aa1feed65cdd29f4b7924301432cf4909f4fd30044b3"
)
EXPECTED_PARENT_C1B2_FREEZE_ID = (
    "sers_fresh_c_c1b2_scientific_protocol_freeze_v1:"
    "cd9065ffee576865bd09"
)
EXPECTED_PARENT_C1B2_FREEZE_SHA256 = (
    "01bde9481335febe4ddec8a18405a31736e400700e10cbdb3e6b240f6e740202"
)

FINAL_H1 = "FRESH_C_PRESERVES_PRE_C_BOUNDED_EXTENSION"
FINAL_H2 = "REJECT_AS_FORMULATED"
FINAL_H3 = "FRESH_C_ERODES_PRE_C_RELATIONAL_GAP"

UPSTREAM_LINEAGE = {
    "r2_final_reassessment": {
        "freeze_id": (
            "sers_r2_final_reassessment_freeze_v1:"
            "aa2f75aa46fb82284db0"
        ),
        "freeze_sha256": (
            "aa2f75aa46fb82284db04c5031b40227b4b2bdd12f4044ccdb8acfa84a1d3831"
        ),
    },
    "i0_integration": {
        "freeze_id": (
            "sers_i0_integrated_orchestration_freeze_v1:"
            "11a5fc254379f718a679"
        ),
        "freeze_sha256": (
            "11a5fc254379f718a679cc8b61c168a704979d86e94ccb11617e2fa8e9d48a62"
        ),
    },
    "fresh_c_content_acquisition": {
        "freeze_id": (
            "sers_fresh_c_content_acquisition_result_freeze_v1:"
            "afc55cfdc78819827cde"
        ),
        "freeze_sha256": (
            "c0686755c472dd936f5e58a3bea9599eb32b259b884a2fe51d65b427976fbf84"
        ),
    },
    "c1a_r1_materialization": {
        "freeze_id": (
            "sers_fresh_c_c1a_r1_recovery_result_freeze_v1:"
            "87b7254c15c383ccaa4c"
        ),
        "freeze_sha256": (
            "b7da11545050a4562457e65ec2da80e8015482b9e3757a911a86a305e26187c5"
        ),
    },
    "c1b0_input_contract": {
        "freeze_id": (
            "sers_fresh_c_c1b0_contract_result_freeze_v1:"
            "accdc6f461b02c13dfc0"
        ),
        "freeze_sha256": (
            "a89f6149177431a1db00d8798657b2504e455937926e7ed7fc75813d9c07fb40"
        ),
    },
    "c1b1_reviewer_contract": {
        "freeze_id": (
            "sers_fresh_c_c1b1_reviewer_protocol_freeze_v1:"
            "30fc8ea1d36ec3503c21"
        ),
        "freeze_sha256": (
            "31ba6d47570935bcb9809bebe8b727b65bd4aab5831106ad96f5eb8b3cb650be"
        ),
    },
    "c1b1_r1_transport": {
        "freeze_id": (
            "sers_fresh_c_c1b1_r1_transport_result_freeze_v1:"
            "dbbdc7f8c7f08310c9a2"
        ),
        "freeze_sha256": (
            "56678366f7749c94f5d1e6b3fdc6aecb72a3f253905563319028fc39351b3aaa"
        ),
    },
}


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def protocol_expected_id(payload: dict[str, Any]) -> str:
    tmp = dict(payload)
    tmp.pop("protocol_id", None)
    tmp.pop("protocol_sha256", None)
    return PROTOCOL_PREFIX + ":" + canonical_json_sha256(tmp)[:20]


def validate_protocol(path: Path) -> dict[str, Any]:
    p = load_object(path)
    if p.get("schema_version") != "sers-fresh-c-final-closeout-protocol-v1":
        raise ValueError("Closeout protocol schema mismatch")
    if p.get("stage") != STAGE or p.get("semantics_id") != SEMANTICS_ID:
        raise ValueError("Closeout protocol stage/semantics mismatch")
    if p.get("protocol_id") != protocol_expected_id(p):
        raise ValueError("Closeout protocol ID mismatch")
    tmp = dict(p)
    stored = tmp.pop("protocol_sha256", None)
    if stored != canonical_json_sha256(tmp):
        raise ValueError("Closeout protocol SHA mismatch")

    exact = {
        "final_scientific_commit": FINAL_SCIENTIFIC_COMMIT,
        "final_run_id": EXPECTED_FINAL_RUN_ID,
        "final_run_sha256": EXPECTED_FINAL_RUN_SHA256,
        "final_result_freeze_id": EXPECTED_FINAL_RESULT_FREEZE_ID,
        "final_result_freeze_sha256": EXPECTED_FINAL_RESULT_FREEZE_SHA256,
        "final_h1_state": FINAL_H1,
        "final_h2_state": FINAL_H2,
        "final_h3_state": FINAL_H3,
        "accepted_scientific_paper_reviews": 25,
        "accepted_final_adjudications": 1,
        "accepted_scientific_outputs": 26,
        "original_failed_c1b2_scientific_call_attempts": 1,
        "recovery_scientific_call_attempts": 26,
        "total_c1b2_scientific_call_attempts": 27,
        "external_literature_used_during_c1b2": False,
        "count_threshold_used": False,
        "hypothesis_rewrite_performed": False,
        "hypothesis_upgrade_performed": False,
        "h2_resurrected": False,
        "new_fresh_reserve_claimed_in_recovery": False,
        "failed_parent_response_reused": False,
        "verbatim_quote_evidence_enabled_in_recovery": False,
        "automatic_next_stage_authorized": False,
        "network_calls_during_closeout": 0,
        "llm_calls_during_closeout": 0,
        "scientific_text_read_during_closeout": False,
        "scientific_adjudication_during_closeout": False,
        "stop_after_closeout_freeze": True,
    }
    for key, expected in exact.items():
        if p.get(key) != expected:
            raise ValueError(f"Closeout protocol field drifted: {key}")
    if p.get("upstream_lineage") != UPSTREAM_LINEAGE:
        raise ValueError("Closeout upstream lineage drifted")
    return p


def git_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )


def validate_final_scientific_state(root: Path) -> dict[str, Any]:
    run = load_object(root / FINAL_RUN_PATH)
    final_record = load_object(root / FINAL_ADJUDICATION_PATH)
    final_freeze = load_object(root / FINAL_RESULT_FREEZE_PATH)
    parent_started = load_object(root / PARENT_FAILED_START_PATH)
    parent_failed = load_object(root / PARENT_FAILED_MARKER_PATH)
    recovery_freeze = load_object(root / RECOVERY_PROTOCOL_FREEZE_PATH)

    if run.get("run_id") != EXPECTED_FINAL_RUN_ID:
        raise ValueError("Final recovery run ID drifted")
    if run.get("run_sha256") != EXPECTED_FINAL_RUN_SHA256:
        raise ValueError("Final recovery run SHA drifted")
    if run.get("paper_review_calls") != 25:
        raise ValueError("Final recovery paper-review count drifted")
    if run.get("final_adjudication_calls") != 1:
        raise ValueError("Final recovery final-adjudication count drifted")
    if run.get("recovery_llm_calls") != 26:
        raise ValueError("Final recovery LLM count drifted")
    if run.get("recovery_network_calls") != 26:
        raise ValueError("Final recovery network count drifted")
    if run.get("new_fresh_reserve_claimed") is not False:
        raise ValueError("Recovery incorrectly claims a new reserve")
    if run.get("failed_parent_response_reused") is not False:
        raise ValueError("Recovery reused failed parent response")
    if run.get("verbatim_quote_evidence_enabled") is not False:
        raise ValueError("Recovery enabled verbatim quote evidence")

    adj = final_record.get("adjudication")
    if not isinstance(adj, dict):
        raise ValueError("Final adjudication payload missing")
    if adj.get("h1_fresh_c_verdict") != FINAL_H1:
        raise ValueError("Final H1 state drifted")
    if adj.get("h2_terminal_state") != FINAL_H2:
        raise ValueError("Final H2 state drifted")
    if adj.get("h2_resurrected") is not False:
        raise ValueError("H2 resurrection drifted")
    if adj.get("h3_fresh_c_verdict") != FINAL_H3:
        raise ValueError("Final H3 state drifted")

    if final_freeze.get("freeze_id") != EXPECTED_FINAL_RESULT_FREEZE_ID:
        raise ValueError("Final result freeze ID drifted")
    if final_freeze.get("manifest_sha256") != EXPECTED_FINAL_RESULT_FREEZE_SHA256:
        raise ValueError("Final result freeze SHA drifted")
    if final_freeze.get("h1_fresh_c_verdict") != FINAL_H1:
        raise ValueError("Frozen H1 state drifted")
    if final_freeze.get("h2_terminal_state") != FINAL_H2:
        raise ValueError("Frozen H2 state drifted")
    if final_freeze.get("h3_fresh_c_verdict") != FINAL_H3:
        raise ValueError("Frozen H3 state drifted")
    if final_freeze.get("automatic_next_stage_authorized") is not False:
        raise ValueError("Final result freeze unexpectedly authorizes next stage")

    if parent_started.get("same_epoch_rerun_allowed") is not False:
        raise ValueError("Parent failed epoch rerun policy drifted")
    if parent_failed.get("completed_paper_reviews") != 0:
        raise ValueError("Parent failed epoch completed-review count drifted")
    if parent_failed.get("scientific_llm_call_attempts") != 1:
        raise ValueError("Parent failed LLM attempt count drifted")
    if parent_failed.get("scientific_network_call_attempts") != 1:
        raise ValueError("Parent failed network attempt count drifted")
    if parent_failed.get("failure_restores_freshness") is not False:
        raise ValueError("Parent failure freshness policy drifted")

    if recovery_freeze.get("freeze_id") != EXPECTED_RECOVERY_PROTOCOL_FREEZE_ID:
        raise ValueError("Recovery protocol freeze ID drifted")
    if recovery_freeze.get("manifest_sha256") != EXPECTED_RECOVERY_PROTOCOL_FREEZE_SHA256:
        raise ValueError("Recovery protocol freeze SHA drifted")

    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if current_commit != FINAL_SCIENTIFIC_COMMIT:
        # Closeout implementation will necessarily be committed later, so callers
        # that run after that commit should verify ancestry rather than equality.
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", FINAL_SCIENTIFIC_COMMIT, current_commit],
            cwd=root,
        ).returncode == 0
        if not ancestor:
            raise ValueError("Final scientific commit is not an ancestor of current HEAD")

    return {
        "run": run,
        "final_adjudication": adj,
        "final_result_freeze": final_freeze,
        "parent_failed": parent_failed,
        "recovery_protocol_freeze": recovery_freeze,
        "current_head": current_commit,
    }
