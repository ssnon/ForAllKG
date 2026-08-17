from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from dac_her.fresh_c_c1b0_contract_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RUN_DIR,
    canonical_json_sha256,
    load_object,
    validate_c1ar1_lineage,
    validate_protocol,
    validate_r2_lineage,
)
from scripts.verify_sers_fresh_c_c1b0_contract_v1_protocol_freeze import (
    main as verify_protocol_freeze,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _preflight(root: Path):
    verify_protocol_freeze()
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    r2 = validate_r2_lineage(root)
    c1a = validate_c1ar1_lineage(root, hash_text_files=True)
    run_dir = root / DEFAULT_RUN_DIR
    run_dir_empty = not run_dir.exists() or not any(run_dir.iterdir())
    freeze = load_object(root / DEFAULT_PROTOCOL_FREEZE_DIR / "freeze_manifest.json")
    return {
        "protocol_id": p["protocol_id"],
        "protocol_sha256": p["protocol_sha256"],
        "protocol_freeze_id": freeze["freeze_id"],
        "r2_report_id": r2["r2_report_id"],
        "r2_freeze_id": r2["r2_freeze_id"],
        "scientific_target_hypothesis_ids": r2["scientific_target_hypothesis_ids"],
        "terminal_rejected_hypothesis_ids": r2["terminal_rejected_hypothesis_ids"],
        "primary_remaining_candidate_hypothesis_id": r2[
            "primary_remaining_candidate_hypothesis_id"
        ],
        "source_identity_count": c1a["source_identity_count"],
        "direct_original_count": c1a["direct_original_count"],
        "structurally_repaired_derivative_count": c1a[
            "structurally_repaired_derivative_count"
        ],
        "repaired_reserve_indexes": c1a["repaired_reserve_indexes"],
        "all_25_materialized_text_hashes_current": True,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "network_calls_during_preflight": 0,
        "llm_calls_during_preflight": 0,
        "c1b0_run_dir_empty": run_dir_empty,
        "c1b1_authorized": False,
        "authorized": False,
        "stop": True,
    }


def preflight() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight(root)
    if not state["c1b0_run_dir_empty"]:
        raise RuntimeError("C1B.0 audit run directory is not empty.")
    print("Fresh-C C1B.0 guarded input-contract preflight")
    for key, value in state.items():
        print(f"{key}: {value}")
    print("Preflight: PASS")
    return 0


def execute() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight(root)
    if not state["c1b0_run_dir_empty"]:
        raise RuntimeError("C1B.0 audit epoch already exists.")

    run_dir = root / DEFAULT_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=False)

    r2 = validate_r2_lineage(root)
    c1a = validate_c1ar1_lineage(root, hash_text_files=True)
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)

    contract = {
        "schema_version": "sers-fresh-c-c1b0-input-contract-v1",
        "protocol_id": p["protocol_id"],
        "protocol_sha256": p["protocol_sha256"],
        "r2": r2,
        "fresh_c_corpus": c1a,
        "hypothesis_transition_contract": {
            "direction_aware_trend_hypothesis:ad13dac8334238124899": {
                "r2_state": "KEEP_BOUNDED_EXTENSION",
                "c1b1_target": True,
                "may_preserve": True,
                "may_downgrade": True,
                "may_reject": True,
                "may_upgrade": False,
                "may_refine_or_rewrite": False,
            },
            "direction_aware_trend_hypothesis:8507f8cadfc46d8d80de": {
                "r2_state": "REJECT_AS_FORMULATED",
                "c1b1_target": False,
                "terminal_rejected": True,
                "may_resurrect": False,
                "may_refine_or_rewrite": False,
            },
            "direction_aware_trend_hypothesis:1cf889e57332402d88c9": {
                "r2_state": "KEEP_RELATIONAL_GAP_CANDIDATE",
                "c1b1_target": True,
                "primary_remaining_candidate": True,
                "may_preserve": True,
                "may_downgrade": True,
                "may_reject": True,
                "may_upgrade": False,
                "may_refine_or_rewrite": False,
            },
        },
        "future_c1b1_evidence_contract": {
            "all_25_papers_must_be_processed": True,
            "cherry_pick_allowed": False,
            "external_literature_lookup_allowed": False,
            "count_thresholds_can_establish_novelty_or_absence": False,
            "negative_absence_inference_from_any_single_paper_allowed": False,
            "repaired_reserve_14_positive_evidence_allowed": True,
            "repaired_reserve_14_absence_inference_allowed": False,
            "repaired_reserve_14_completeness_claim_allowed": False,
            "fresh_c_scope_only_no_literature_wide_novelty_claim": True,
            "fresh_c_evidence_may_be_positive_prior_art_premise_for_generation": False,
        },
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "network_calls": 0,
        "llm_calls": 0,
        "c1b1_authorized": False,
        "stop": True,
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    contract["contract_id"] = (
        "sers_fresh_c_c1b0_input_contract_v1:" + contract["contract_sha256"][:20]
    )
    _atomic(run_dir / "input_contract.json", contract)
    _atomic(run_dir / "C1B0_AUDIT_COMPLETE.json", {
        "contract_id": contract["contract_id"],
        "contract_sha256": contract["contract_sha256"],
        "source_identity_count": 25,
        "fresh_c_scientific_text_semantic_read_performed": False,
        "scientific_adjudication_performed": False,
        "network_calls": 0,
        "llm_calls": 0,
        "c1b1_authorized": False,
        "stop": True,
    })

    print("Fresh-C C1B.0 input-contract audit complete")
    print(f"Contract ID: {contract['contract_id']}")
    print(f"Contract SHA256: {contract['contract_sha256']}")
    print("Scientific targets: H1,H3")
    print("H2 terminal rejected: True")
    print("Exact materialized corpus: 25/25")
    print("Structurally repaired reserve indexes: [14]")
    print("Fresh-C scientific text semantic read performed: False")
    print("Scientific adjudication performed: False")
    print("Network calls: 0")
    print("LLM calls: 0")
    print("C1B.1 authorized: False")
    print("STOP: True")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute-contract-audit", action="store_true")
    args = parser.parse_args()
    return preflight() if args.preflight else execute()


if __name__ == "__main__":
    raise SystemExit(main())
