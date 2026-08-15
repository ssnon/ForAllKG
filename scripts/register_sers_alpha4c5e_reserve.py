from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.hypothesis_trend_evaluation import (
    load_protocol,
    make_reserve_manifest,
    verify_protocol_integrity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register an unseen alpha4c.5e reserve paper set before "
            "Trend/Hypothesis evaluation. This performs no extraction or "
            "LLM calls."
        )
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--reserve-id", required=True)
    parser.add_argument(
        "--paper-id",
        action="append",
        required=True,
        dest="paper_ids",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--confirm-unseen",
        action="store_true",
        help=(
            "Confirm these papers were not used to tune alpha4c.5e "
            "Trend-to-Hypothesis semantics."
        ),
    )
    parser.add_argument(
        "--confirm-not-inspected-for-alpha4c5e",
        action="store_true",
        help=(
            "Confirm the reserve contents were not inspected for the "
            "Trend/Hypothesis acceptance rules before registration."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_unseen:
        raise SystemExit("--confirm-unseen is required")
    if not args.confirm_not_inspected_for_alpha4c5e:
        raise SystemExit(
            "--confirm-not-inspected-for-alpha4c5e is required"
        )

    protocol = load_protocol(args.protocol)
    drift = verify_protocol_integrity(
        protocol,
        root=Path.cwd(),
    )
    if drift:
        raise SystemExit(
            "Frozen 5e implementation drifted; do not register reserve:\n"
            + "\n".join(drift)
        )

    manifest = make_reserve_manifest(
        protocol,
        reserve_id=args.reserve_id,
        paper_ids=args.paper_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != text:
            raise SystemExit(
                "Refusing to replace a different reserve manifest."
            )
    else:
        args.output.write_text(text, encoding="utf-8")

    print("alpha4c.5e reserve registration")
    print("Reserve ID:", manifest.reserve_id)
    print("Manifest ID:", manifest.manifest_id)
    print("Manifest SHA256:", manifest.manifest_sha256)
    print("Paper IDs:", ", ".join(manifest.paper_ids))
    print("Acceptance rules already frozen: True")
    print("Reserve consumed by registration: False")
    print("LLM calls: 0")
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
