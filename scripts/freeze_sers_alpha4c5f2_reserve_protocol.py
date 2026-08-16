from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.alpha4c5f2_reserve import (
    DEFAULT_5E_PROTOCOL_PATH,
    DEFAULT_LEGACY_5F_PROTOCOL_PATH,
    make_5f2_protocol,
)


ROOT = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a new alpha4c.5f.2 Reserve-A E2E protocol only after "
            "the 103-paper pool/split, 5e reserve manifest, and 5f.1 "
            "canonical readiness lock already exist."
        )
    )
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--blind-split", type=Path, required=True)
    parser.add_argument("--reserve-manifest", type=Path, required=True)
    parser.add_argument("--readiness-lock", type=Path, required=True)
    parser.add_argument(
        "--evaluation-protocol",
        type=Path,
        default=DEFAULT_5E_PROTOCOL_PATH,
    )
    parser.add_argument(
        "--legacy-5f-protocol",
        type=Path,
        default=DEFAULT_LEGACY_5F_PROTOCOL_PATH,
    )
    parser.add_argument(
        "--campaign-id",
        default="sers_alpha4c5f2_reserve_a_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    output_path = _rooted(args.output)
    if output_path.exists():
        raise SystemExit(
            f"Refusing to replace frozen protocol: {output_path}"
        )

    protocol = make_5f2_protocol(
        root=ROOT,
        campaign_id=args.campaign_id,
        pool_path=args.pool_manifest,
        split_path=args.blind_split,
        reserve_manifest_path=args.reserve_manifest,
        readiness_lock_path=_rooted(args.readiness_lock),
        evaluation_protocol_path=_rooted(
            args.evaluation_protocol
        ),
        legacy_5f_protocol_path=_rooted(
            args.legacy_5f_protocol
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            protocol.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("alpha4c.5f.2 protocol freeze: PASS")
    print("Protocol ID:", protocol.protocol_id)
    print("Protocol SHA256:", protocol.protocol_sha256)
    print("Campaign:", protocol.campaign_id)
    print("Partition:", protocol.reserve_partition)
    print("Papers:", len(protocol.reserve_paper_ids))
    print(
        "Canonical readiness lock:",
        protocol.canonical_readiness_lock_payload_sha256,
    )
    print("Direct consumption marker write allowed:", False)
    print("Reserve consumed:", False)
    print("LLM calls:", 0)
    print("Saved:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
