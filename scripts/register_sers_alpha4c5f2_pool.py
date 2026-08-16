from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.alpha4c5f2_reserve import (
    atomic_json,
    make_blind_split,
    make_pool_manifest,
    read_json,
)


ROOT = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register the new 103-paper SERS alpha4c.5f.2 candidate pool "
            "and create an ID-only deterministic 53/25/25 blind split. "
            "No scientific result is consumed and no LLM is called."
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-new-pool", action="store_true")
    parser.add_argument(
        "--confirm-not-inspected-for-alpha4c-reserve-selection",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_new_pool:
        raise SystemExit("--confirm-new-pool is required.")
    if not args.confirm_not_inspected_for_alpha4c_reserve_selection:
        raise SystemExit(
            "--confirm-not-inspected-for-alpha4c-reserve-selection "
            "is required."
        )

    source_path = (
        args.source_manifest
        if args.source_manifest.is_absolute()
        else ROOT / args.source_manifest
    )
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )
    pool_path = output_dir / "pool_manifest.json"
    split_path = output_dir / "blind_split.json"
    if pool_path.exists() or split_path.exists():
        raise SystemExit(
            "Refusing to replace existing alpha4c.5f.2 pool/split."
        )

    source = read_json(source_path)
    pool = make_pool_manifest(
        source_manifest_path=args.source_manifest,
        source_manifest=source,
    )
    split = make_blind_split(pool)

    atomic_json(pool_path, pool)
    atomic_json(split_path, split)

    print("alpha4c.5f.2 new SERS pool registration: PASS")
    print("Pool semantics:", pool["semantics_id"])
    print("Split semantics:", split["semantics_id"])
    print("Papers:", split["paper_count"])
    print("Development:", split["development_count"])
    print("Reserve A:", split["reserve_a_count"])
    print("Reserve B:", split["reserve_b_count"])
    print("Split inputs: paper_id only")
    print("Scientific fields used:", False)
    print("LLM calls:", 0)
    print("Reserve A consumed:", False)
    print("Reserve B consumed:", False)
    print("Reserve B sealed:", True)
    print("Pool:", pool_path)
    print("Split:", split_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
