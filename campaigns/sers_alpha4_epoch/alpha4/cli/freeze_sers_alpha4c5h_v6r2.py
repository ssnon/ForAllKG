from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f2_reserve import (
    validate_blind_split,
    validate_pool_manifest,
)
from campaigns.sers_alpha4_epoch.alpha4.alpha4c5h_freeze import (
    ALPHA4C5H_CONFIRMATION_PROTOCOL_SEMANTICS_ID,
    ALPHA4C5H_FREEZE_SEMANTICS_ID,
    EXPECTED_5E_PROTOCOL_ID,
    EXPECTED_ACTIVE_PRE_FREEZE_TREND_SEMANTICS_ID,
    EXPECTED_SPLIT_ID,
    EXPECTED_SPLIT_SEMANTIC_SHA256,
    EXPECTED_TREND_SEMANTICS_ID,
    EXPECTED_V6R2_FILE_SHA256,
    KNOWN_FROZEN_BUILDER_SHA256,
    find_reserve_b_paper_ids,
    find_scalar_values,
    find_string_in_json_files,
    hash_inventory,
    make_confirmation_protocol_id,
    make_freeze_id,
    read_json,
    scientific_code_inventory,
    semantic_sha256,
    sha256_file,
    write_json,
)


ROOT = Path.cwd()

DEFAULT_POOL = Path(
    "evaluation/sers_alpha4c5f2/pool_v1/pool_manifest.json"
)
DEFAULT_SPLIT = Path(
    "evaluation/sers_alpha4c5f2/pool_v1/blind_split.json"
)
DEFAULT_V6R2_SUMMARY = Path(
    "evaluation/sers_alpha4c5g2r2/dev_v1/summary.json"
)
DEFAULT_OUTPUT = Path(
    "evaluation/sers_alpha4c5h/freeze_v1"
)
RESERVE_A_PASS = Path(
    "evaluation/sers_alpha4c5f2/"
    "sers_alpha4c5f2_reserve_a_v1/CAMPAIGN_PASS.json"
)


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze accepted Development Trend v6r2 scientific code "
            "and create a sealed Reserve-B confirmation protocol. "
            "Does not register, prepare, inspect, or consume Reserve B."
        )
    )
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        default=DEFAULT_POOL,
    )
    parser.add_argument(
        "--blind-split",
        type=Path,
        default=DEFAULT_SPLIT,
    )
    parser.add_argument(
        "--v6r2-summary",
        type=Path,
        default=DEFAULT_V6R2_SUMMARY,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--confirm-freeze-after-development-pass",
        action="store_true",
    )
    return parser.parse_args()


def _assert_exact_v6r2_pass(summary: dict[str, Any]) -> None:
    expected = {
        "development_only": True,
        "paper_count": 53,
        "current_semantics_id":
            EXPECTED_ACTIVE_PRE_FREEZE_TREND_SEMANTICS_ID,
        "candidate_semantics_id":
            EXPECTED_TREND_SEMANTICS_ID,
        "current_evidence_count": 9,
        "candidate_evidence_count": 15,
        "added_evidence_count": 6,
        "removed_evidence_count": 0,
        "candidate_local_override_allowed_count": 1,
        "candidate_local_override_emitted_count": 1,
        "global_method_context_mutation_count": 0,
        "unresolved_2nm_8nm_pair_emitted_count": 0,
        "candidate_structural_gate": True,
        "passes_candidate_regression": True,
        "reserve_a_used": False,
        "reserve_b_used": False,
        "reserve_b_remains_sealed": True,
        "llm_calls": 0,
        "active_registry_modified": False,
        "scientific_semantics_activated": False,
        "count_thresholds_used_for_acceptance": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(
                f"v6r2 PASS invariant drift for {key}: "
                f"{summary.get(key)!r} != {value!r}"
            )

    expected_claims = {
        "claim_gap_dependent_ef",
        "claim_gap_dependent_sers",
        "claim_gap_enhancement_trend",
        "claim_gap_size_sers_intensity",
        "obs_gap_dependent_enhancement",
    }
    if set(summary.get("added_claim_ids", [])) != expected_claims:
        raise RuntimeError(
            "v6r2 exact added-claim set drifted."
        )

    numeric_sets = {
        frozenset(row)
        for row in summary.get(
            "added_numeric_measurement_sets",
            [],
        )
    }
    expected_numeric = {
        frozenset(
            {
                "meas_ef_nanobox_gap_15p6_1135",
                "meas_ef_nanobox_gap_1p2_1135",
            }
        )
    }
    if numeric_sets != expected_numeric:
        raise RuntimeError(
            "v6r2 exact added numeric set drifted."
        )

    pass_conditions = dict(
        summary.get("pass_conditions", {})
    )
    if not pass_conditions or not all(
        value is True
        for value in pass_conditions.values()
    ):
        raise RuntimeError(
            "v6r2 pass_conditions are not all true."
        )


def _validate_split_semantics(
    *,
    pool_path: Path,
    split_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # The historical alpha4c.5f.2 validator is the source of truth for split
    # identity. It validates the internal semantic SHA and recomputes the
    # deterministic ID-only split from the pool.
    pool = validate_pool_manifest(
        root=ROOT,
        pool_path=pool_path,
        verify_source_manifest=False,
    )
    split = read_json(split_path)
    split = validate_blind_split(
        pool=pool,
        split=split,
    )

    if split.get("split_id") != EXPECTED_SPLIT_ID:
        raise RuntimeError(
            "Blind split ID drift: "
            f"{split.get('split_id')!r} != {EXPECTED_SPLIT_ID!r}"
        )
    if (
        split.get("split_sha256")
        != EXPECTED_SPLIT_SEMANTIC_SHA256
    ):
        raise RuntimeError(
            "Blind split semantic SHA drift: "
            f"{split.get('split_sha256')!r} != "
            f"{EXPECTED_SPLIT_SEMANTIC_SHA256!r}"
        )
    if split.get(
        "reserve_b_sealed_for_future_confirmation"
    ) is not True:
        raise RuntimeError(
            "Reserve B is not sealed in the frozen split."
        )
    return pool, split


def _assert_reserve_b_not_consumed() -> None:
    evaluation_root = ROOT / "evaluation"

    suspicious_true = []
    pass_markers = []
    if evaluation_root.exists():
        for path in evaluation_root.rglob("*.json"):
            path_text = str(path).casefold()
            if "reserve_b" not in path_text and "reserve-b" not in path_text:
                continue
            if path.name == "CAMPAIGN_PASS.json":
                pass_markers.append(str(path))
            try:
                value = read_json(path)
            except Exception:
                continue

            consumed = find_scalar_values(
                value,
                key_predicate=lambda key: (
                    "consumed" in key
                    or "executed" in key
                ),
            )
            for key_path, scalar in consumed:
                if scalar is True:
                    suspicious_true.append(
                        f"{path}:{key_path}"
                    )

    if pass_markers or suspicious_true:
        raise RuntimeError(
            "Reserve B appears already consumed/executed. "
            f"PASS markers={pass_markers}, "
            f"consumed flags={suspicious_true}"
        )


def main() -> int:
    args = parse_args()
    if not args.confirm_freeze_after_development_pass:
        raise SystemExit(
            "--confirm-freeze-after-development-pass is required."
        )

    pool_path = rooted(args.pool_manifest)
    split_path = rooted(args.blind_split)
    summary_path = rooted(args.v6r2_summary)
    output_dir = rooted(args.output_dir)

    if output_dir.exists():
        raise SystemExit(
            f"Refusing existing freeze directory: {output_dir}"
        )
    if not pool_path.exists():
        raise RuntimeError(
            f"Pool manifest missing: {pool_path}"
        )
    if not split_path.exists():
        raise RuntimeError(
            f"Blind split missing: {split_path}"
        )
    if not summary_path.exists():
        raise RuntimeError(
            f"v6r2 Development summary missing: {summary_path}"
        )
    if not rooted(RESERVE_A_PASS).exists():
        raise RuntimeError(
            "Historical Reserve-A PASS marker is missing."
        )

    v6r2 = read_json(summary_path)
    _assert_exact_v6r2_pass(v6r2)

    pool, split = _validate_split_semantics(
        pool_path=pool_path,
        split_path=split_path,
    )
    _assert_reserve_b_not_consumed()

    reserve_b_ids, reserve_b_source_path = (
        find_reserve_b_paper_ids(split)
    )
    if len(reserve_b_ids) != 25:
        raise RuntimeError(
            "Reserve B must contain exactly 25 paper IDs."
        )

    for rel, expected in EXPECTED_V6R2_FILE_SHA256.items():
        path = ROOT / rel
        if not path.exists():
            raise RuntimeError(
                f"Required v6r2 implementation missing: {rel}"
            )
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"v6r2 implementation drift for {rel}: "
                f"{observed} != {expected}"
            )

    for rel, expected in KNOWN_FROZEN_BUILDER_SHA256.items():
        path = ROOT / rel
        if not path.exists():
            raise RuntimeError(
                f"Frozen scientific builder missing: {rel}"
            )
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"Scientific builder drift for {rel}: "
                f"{observed} != {expected}"
            )

    protocol_files = []
    for search_root in (
        ROOT / "configs",
        ROOT / "evaluation",
    ):
        protocol_files.extend(
            find_string_in_json_files(
                search_root,
                EXPECTED_5E_PROTOCOL_ID,
            )
        )
    protocol_files = sorted(set(protocol_files))
    if not protocol_files:
        raise RuntimeError(
            "Frozen alpha4c.5e evaluation protocol ID was not found "
            "under configs/ or evaluation/: "
            + EXPECTED_5E_PROTOCOL_ID
        )

    code_paths = scientific_code_inventory(ROOT)
    if not code_paths:
        raise RuntimeError(
            "Scientific code inventory is empty."
        )
    code_hashes = hash_inventory(
        ROOT,
        code_paths,
    )

    bound_5e_files = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in protocol_files
    }

    pool_raw_sha = sha256_file(pool_path)
    split_raw_sha = sha256_file(split_path)

    freeze_payload = {
        "freeze_semantics_id":
            ALPHA4C5H_FREEZE_SEMANTICS_ID,
        "trend_semantics_id":
            EXPECTED_TREND_SEMANTICS_ID,
        "development_regression": {
            "path": str(summary_path.relative_to(ROOT)),
            "sha256": sha256_file(summary_path),
            "semantic_sha256": semantic_sha256(v6r2),
            "passes_candidate_regression": True,
        },
        "blind_split": {
            "pool_manifest_path":
                str(pool_path.relative_to(ROOT)),
            "pool_manifest_semantic_sha256":
                pool.get("manifest_sha256"),
            "pool_manifest_raw_file_sha256":
                pool_raw_sha,
            "path": str(split_path.relative_to(ROOT)),
            "split_id": EXPECTED_SPLIT_ID,
            "semantic_sha256":
                EXPECTED_SPLIT_SEMANTIC_SHA256,
            "raw_file_sha256": split_raw_sha,
            "validation": (
                "alpha4c5f2.validate_pool_manifest + "
                "alpha4c5f2.validate_blind_split"
            ),
        },
        "acceptance_protocol": {
            "protocol_id": EXPECTED_5E_PROTOCOL_ID,
            "bound_files": bound_5e_files,
        },
        "scientific_code_sha256": code_hashes,
        "scientific_code_file_count": len(code_hashes),
        "rules": {
            "scientific_semantics_frozen": True,
            "acceptance_semantics_frozen": True,
            "count_thresholds_for_acceptance": False,
            "reserve_a_reuse_for_scientific_evaluation": False,
            "reserve_b_seen_or_consumed_at_freeze": False,
            "reserve_b_may_be_consumed_only_after_guarded_readiness": True,
            "reserve_b_may_be_consumed_at_most_once": True,
            "reserve_b_result_must_not_be_used_to_tune_v6r2": True,
        },
    }
    freeze_id = make_freeze_id(freeze_payload)
    freeze_manifest = {
        "freeze_id": freeze_id,
        **freeze_payload,
    }

    protocol_payload = {
        "protocol_semantics_id":
            ALPHA4C5H_CONFIRMATION_PROTOCOL_SEMANTICS_ID,
        "freeze_id": freeze_id,
        "trend_semantics_id":
            EXPECTED_TREND_SEMANTICS_ID,
        "acceptance_protocol_id":
            EXPECTED_5E_PROTOCOL_ID,
        "blind_split_id": EXPECTED_SPLIT_ID,
        "blind_split_semantic_sha256":
            EXPECTED_SPLIT_SEMANTIC_SHA256,
        "blind_split_raw_file_sha256":
            split_raw_sha,
        "reserve_partition": "B",
        "reserve_b_paper_ids": reserve_b_ids,
        "reserve_b_paper_count": 25,
        "reserve_b_source_path": reserve_b_source_path,
        "execution_policy": {
            "development_closed": True,
            "scientific_semantics_changes_after_freeze": False,
            "acceptance_semantics_changes_after_freeze": False,
            "readiness_required_before_consumption": True,
            "freeze_revalidation_required_immediately_before_consumption": True,
            "consumption_is_irreversible": True,
            "one_shot_confirmation": True,
            "no_count_thresholds": True,
            "reserve_b_failure_does_not_authorize_tuning": True,
        },
        "planned_scientific_sequence": [
            "canonical_readiness_guard",
            "guarded_reserve_b_consumption",
            "evidence_projection_and_corpus",
            "measurement_result_identity",
            "metric_definition",
            "comparison",
            "trend_v6r2",
            "trend_precision",
            "cross_context_if_local_trend_exists",
            "explorer_hypothesis_context",
            "alpha4c5a_capability_mapping",
            "alpha4c5b_trend_aware_input",
            "direction_aware_maker",
            "alpha4c5e_final_evaluation",
        ],
    }
    protocol_id = make_confirmation_protocol_id(
        protocol_payload
    )
    confirmation_protocol = {
        "confirmation_protocol_id": protocol_id,
        **protocol_payload,
    }

    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "freeze_manifest.json",
        freeze_manifest,
    )
    write_json(
        output_dir / "reserve_b_confirmation_protocol.json",
        confirmation_protocol,
    )
    write_json(
        output_dir / "freeze_status.json",
        {
            "freeze_id": freeze_id,
            "confirmation_protocol_id": protocol_id,
            "reserve_b_registered_for_confirmation": True,
            "reserve_b_consumed": False,
            "reserve_b_scientific_artifacts_inspected": False,
            "readiness_prepared": False,
            "consumption_marker_written": False,
            "llm_calls": 0,
        },
    )

    print(
        "alpha4c.5h Trend v6r2 Freeze & Reserve-B "
        "Confirmation Protocol: COMPLETE"
    )
    print("Freeze ID:", freeze_id)
    print(
        "Confirmation protocol ID:",
        protocol_id,
    )
    print("Trend semantics:", EXPECTED_TREND_SEMANTICS_ID)
    print(
        "Blind split semantic SHA256:",
        EXPECTED_SPLIT_SEMANTIC_SHA256,
    )
    print(
        "Blind split raw file SHA256:",
        split_raw_sha,
    )
    print("Scientific code files frozen:", len(code_hashes))
    print("5e protocol:", EXPECTED_5E_PROTOCOL_ID)
    print("Reserve B papers:", len(reserve_b_ids))
    print("Reserve B consumed:", False)
    print("Readiness prepared:", False)
    print("Scientific semantics frozen:", True)
    print("Acceptance semantics frozen:", True)
    print("Count thresholds:", False)
    print("LLM calls:", 0)
    print(
        "Freeze manifest:",
        output_dir / "freeze_manifest.json",
    )
    print(
        "Confirmation protocol:",
        output_dir
        / "reserve_b_confirmation_protocol.json",
    )
    print(
        "NEXT: build/run alpha4c.5h.1 Reserve-B readiness "
        "against this exact freeze. Do NOT consume Reserve B yet."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
