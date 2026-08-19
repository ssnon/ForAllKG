from __future__ import annotations

import argparse
import json
from pathlib import Path

from campaigns.sers_alpha4_epoch.alpha4.alpha4c5f2_reserve import (
    load_pool_and_split,
)
from dac_her.hypothesis_trend_evaluation import (
    load_protocol,
    make_reserve_manifest,
    verify_protocol_integrity,
)


ROOT = Path.cwd()
DEFAULT_5E = Path(
    "configs/heldout/"
    "sers_alpha4c5e_trend_hypothesis_evaluation_protocol.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register alpha4c.5f.2 Reserve A under the already-frozen "
            "alpha4c.5e Trend-Hypothesis evaluation protocol. "
            "Reserve B remains sealed."
        )
    )
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--blind-split", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_5E)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reserve-id",
        default="sers_alpha4c5e_reserve_alpha4c5f2_a_v1",
    )
    parser.add_argument("--confirm-unseen", action="store_true")
    parser.add_argument(
        "--confirm-not-inspected-for-alpha4c5e",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_unseen:
        raise SystemExit("--confirm-unseen is required.")
    if not args.confirm_not_inspected_for_alpha4c5e:
        raise SystemExit(
            "--confirm-not-inspected-for-alpha4c5e is required."
        )

    pool_path = (
        args.pool_manifest
        if args.pool_manifest.is_absolute()
        else ROOT / args.pool_manifest
    )
    split_path = (
        args.blind_split
        if args.blind_split.is_absolute()
        else ROOT / args.blind_split
    )
    protocol_path = (
        args.protocol
        if args.protocol.is_absolute()
        else ROOT / args.protocol
    )
    output_path = (
        args.output
        if args.output.is_absolute()
        else ROOT / args.output
    )
    if output_path.exists():
        raise SystemExit(
            f"Refusing to replace existing reserve manifest: "
            f"{output_path}"
        )

    _, split = load_pool_and_split(
        root=ROOT,
        pool_path=pool_path,
        split_path=split_path,
        verify_source_manifest=True,
    )
    protocol = load_protocol(protocol_path)
    issues = verify_protocol_integrity(protocol, root=ROOT)
    if issues:
        raise SystemExit(
            "Frozen alpha4c.5e protocol integrity failed:\n- "
            + "\n- ".join(issues)
        )

    reserve = make_reserve_manifest(
        protocol,
        reserve_id=args.reserve_id,
        paper_ids=list(split["reserve_a"]),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            reserve.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("alpha4c.5f.2 Reserve A registration: PASS")
    print("5e protocol:", protocol.protocol_id)
    print("Reserve ID:", reserve.reserve_id)
    print("Manifest ID:", reserve.manifest_id)
    print("Manifest SHA256:", reserve.manifest_sha256)
    print("Papers:", len(reserve.paper_ids))
    print("Declared unseen:", True)
    print("Reserve consumed:", False)
    print("Reserve B registered:", False)
    print("LLM calls:", 0)
    print("Saved:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
