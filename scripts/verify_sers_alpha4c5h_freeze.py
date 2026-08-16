from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.alpha4c5f2_reserve import (
    validate_blind_split,
    validate_pool_manifest,
)
from dac_her.alpha4c5h_freeze import (
    EXPECTED_SPLIT_SEMANTIC_SHA256,
    EXPECTED_TREND_SEMANTICS_ID,
    read_json,
    sha256_file,
    verify_hash_inventory,
)


ROOT = Path.cwd()
DEFAULT_FREEZE = Path(
    "evaluation/sers_alpha4c5h/freeze_v1/freeze_manifest.json"
)
DEFAULT_PROTOCOL = Path(
    "evaluation/sers_alpha4c5h/freeze_v1/"
    "reserve_b_confirmation_protocol.json"
)
DEFAULT_STATUS = Path(
    "evaluation/sers_alpha4c5h/freeze_v1/freeze_status.json"
)


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only alpha4c.5h freeze verification. "
            "Does not prepare or consume Reserve B."
        )
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=DEFAULT_FREEZE,
    )
    parser.add_argument(
        "--confirmation-protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=DEFAULT_STATUS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    freeze_path = rooted(args.freeze_manifest)
    protocol_path = rooted(args.confirmation_protocol)
    status_path = rooted(args.status)

    freeze = read_json(freeze_path)
    protocol = read_json(protocol_path)
    status = read_json(status_path)

    if freeze.get("trend_semantics_id") != EXPECTED_TREND_SEMANTICS_ID:
        raise RuntimeError("Trend freeze semantics drift.")
    if protocol.get("trend_semantics_id") != EXPECTED_TREND_SEMANTICS_ID:
        raise RuntimeError("Protocol Trend semantics drift.")
    if (
        protocol.get("freeze_id")
        != freeze.get("freeze_id")
    ):
        raise RuntimeError("Protocol/freeze ID binding mismatch.")
    if (
        protocol.get("blind_split_semantic_sha256")
        != EXPECTED_SPLIT_SEMANTIC_SHA256
    ):
        raise RuntimeError("Protocol split semantic SHA drift.")
    if int(protocol.get("reserve_b_paper_count", -1)) != 25:
        raise RuntimeError("Reserve B paper count drift.")
    if len(set(protocol.get("reserve_b_paper_ids", []))) != 25:
        raise RuntimeError("Reserve B IDs are not 25 unique papers.")

    if status.get("reserve_b_consumed") is not False:
        raise RuntimeError("Reserve B status is already consumed.")
    if status.get("consumption_marker_written") is not False:
        raise RuntimeError("Reserve B consumption marker already written.")

    split_info = freeze["blind_split"]
    pool_path = ROOT / split_info["pool_manifest_path"]
    split_path = ROOT / split_info["path"]

    pool = validate_pool_manifest(
        root=ROOT,
        pool_path=pool_path,
        verify_source_manifest=False,
    )
    split = validate_blind_split(
        pool=pool,
        split=read_json(split_path),
    )
    if (
        split.get("split_sha256")
        != EXPECTED_SPLIT_SEMANTIC_SHA256
    ):
        raise RuntimeError(
            "Current split semantic SHA no longer matches freeze."
        )

    drift = verify_hash_inventory(
        ROOT,
        freeze["scientific_code_sha256"],
    )

    observed_pool_raw = sha256_file(pool_path)
    expected_pool_raw = split_info[
        "pool_manifest_raw_file_sha256"
    ]
    if observed_pool_raw != expected_pool_raw:
        drift.append(
            {
                "path": str(pool_path.relative_to(ROOT)),
                "issue": "pool_raw_file_sha256_mismatch",
                "expected": expected_pool_raw,
                "observed": observed_pool_raw,
            }
        )

    observed_split_raw = sha256_file(split_path)
    expected_split_raw = split_info[
        "raw_file_sha256"
    ]
    if observed_split_raw != expected_split_raw:
        drift.append(
            {
                "path": str(split_path.relative_to(ROOT)),
                "issue": "split_raw_file_sha256_mismatch",
                "expected": expected_split_raw,
                "observed": observed_split_raw,
            }
        )

    if (
        protocol.get("blind_split_raw_file_sha256")
        != expected_split_raw
    ):
        raise RuntimeError(
            "Confirmation protocol raw split SHA binding drift."
        )

    for rel, expected in (
        freeze["acceptance_protocol"]["bound_files"].items()
    ):
        path = ROOT / rel
        if not path.exists():
            drift.append(
                {
                    "path": rel,
                    "issue": "bound_acceptance_file_missing",
                    "expected": expected,
                    "observed": "",
                }
            )
            continue
        observed = sha256_file(path)
        if observed != expected:
            drift.append(
                {
                    "path": rel,
                    "issue": "bound_acceptance_file_drift",
                    "expected": expected,
                    "observed": observed,
                }
            )

    if drift:
        raise RuntimeError(
            "alpha4c.5h freeze verification FAILED:\n"
            + "\n".join(
                f"- {row['path']}: {row['issue']} "
                f"expected={row['expected']} "
                f"observed={row['observed']}"
                for row in drift
            )
        )

    print("alpha4c.5h Freeze Verification: PASS")
    print("Freeze ID:", freeze["freeze_id"])
    print(
        "Confirmation protocol ID:",
        protocol["confirmation_protocol_id"],
    )
    print(
        "Blind split semantic SHA256:",
        EXPECTED_SPLIT_SEMANTIC_SHA256,
    )
    print(
        "Blind split raw file SHA256:",
        expected_split_raw,
    )
    print(
        "Scientific code files verified:",
        len(freeze["scientific_code_sha256"]),
    )
    print("Reserve B papers:", 25)
    print("Reserve B consumed:", False)
    print("Consumption marker written:", False)
    print("Scientific code drift:", 0)
    print("Acceptance protocol drift:", 0)
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
