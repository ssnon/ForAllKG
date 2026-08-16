from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.alpha4c5h1_reserve_b import (
    DEFAULT_5E_PROTOCOL,
    DEFAULT_5H_CONFIRMATION_PROTOCOL,
    DEFAULT_5H_FREEZE_MANIFEST,
    make_h1_protocol,
    raw_json_bytes,
    sha256_bytes,
)
from dac_her.hypothesis_trend_evaluation import (
    load_protocol,
    make_reserve_manifest,
    verify_protocol_integrity,
)


ROOT = Path.cwd()
DEFAULT_READINESS = Path(
    "evaluation/sers_alpha4c5h1/reserve_b_v1/control/"
    "canonical_readiness_lock.json"
)
DEFAULT_COMPAT = Path(
    "evaluation/sers_alpha4c5h1/dev_compat_v1/summary.json"
)
DEFAULT_RESERVE_MANIFEST = Path(
    "evaluation/sers_alpha4c5h1/reserve_b_v1/control/"
    "reserve_b_5e_manifest.json"
)
DEFAULT_PROTOCOL = Path(
    "evaluation/sers_alpha4c5h1/reserve_b_v1/control/"
    "execution_protocol.json"
)


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Freeze alpha4c.5h.1 Reserve-B execution protocol after "
            "DEV downstream compatibility and canonical readiness are locked. "
            "Creates the 5e Reserve-B registration manifest but does not "
            "consume Reserve B."
        )
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=DEFAULT_5H_FREEZE_MANIFEST,
    )
    parser.add_argument(
        "--confirmation-protocol",
        type=Path,
        default=DEFAULT_5H_CONFIRMATION_PROTOCOL,
    )
    parser.add_argument(
        "--readiness-lock",
        type=Path,
        default=DEFAULT_READINESS,
    )
    parser.add_argument(
        "--dev-compatibility",
        type=Path,
        default=DEFAULT_COMPAT,
    )
    parser.add_argument(
        "--evaluation-protocol",
        type=Path,
        default=DEFAULT_5E_PROTOCOL,
    )
    parser.add_argument(
        "--reserve-manifest-output",
        type=Path,
        default=DEFAULT_RESERVE_MANIFEST,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    parser.add_argument(
        "--confirm-unseen",
        action="store_true",
    )
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

    freeze_path = rooted(args.freeze_manifest)
    confirmation_path = rooted(args.confirmation_protocol)
    readiness_path = rooted(args.readiness_lock)
    compat_path = rooted(args.dev_compatibility)
    evaluation_path = rooted(args.evaluation_protocol)
    reserve_path = rooted(args.reserve_manifest_output)
    output_path = rooted(args.output)

    for path, label in (
        (freeze_path, "5h freeze"),
        (confirmation_path, "5h confirmation"),
        (readiness_path, "readiness lock"),
        (compat_path, "DEV compatibility"),
        (evaluation_path, "5e protocol"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} missing: {path}")
    if reserve_path.exists():
        raise SystemExit(
            f"Refusing existing Reserve-B manifest: {reserve_path}"
        )
    if output_path.exists():
        raise SystemExit(
            f"Refusing existing execution protocol: {output_path}"
        )

    confirmation = json.loads(
        confirmation_path.read_text(encoding="utf-8")
    )
    reserve_ids = sorted(
        str(value)
        for value in confirmation["reserve_b_paper_ids"]
    )

    evaluation = load_protocol(evaluation_path)
    issues = verify_protocol_integrity(
        evaluation,
        root=ROOT,
    )
    if issues:
        raise SystemExit(
            "Frozen 5e protocol integrity failed:\n- "
            + "\n- ".join(issues)
        )

    reserve = make_reserve_manifest(
        evaluation,
        reserve_id="sers_alpha4c5e_reserve_alpha4c5h1_b_v1",
        paper_ids=reserve_ids,
    )
    reserve_payload = reserve.model_dump(mode="json")
    reserve_bytes = raw_json_bytes(reserve_payload)
    reserve_file_sha = sha256_bytes(reserve_bytes)

    protocol = make_h1_protocol(
        root=ROOT,
        freeze_manifest_path=freeze_path,
        confirmation_protocol_path=confirmation_path,
        readiness_lock_path=readiness_path,
        development_compatibility_path=compat_path,
        evaluation_protocol_path=evaluation_path,
        reserve_manifest_path=reserve_path,
        reserve_manifest=reserve,
        reserve_manifest_file_sha256=reserve_file_sha,
    )

    reserve_path.parent.mkdir(parents=True, exist_ok=True)
    reserve_path.write_bytes(reserve_bytes)
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

    print("alpha4c.5h.1 execution protocol freeze: PASS")
    print("Protocol ID:", protocol.protocol_id)
    print("Protocol SHA256:", protocol.protocol_sha256)
    print("Campaign:", protocol.campaign_id)
    print("Partition:", protocol.reserve_partition)
    print("Papers:", len(protocol.reserve_paper_ids))
    print(
        "5h freeze ID:",
        protocol.five_h_freeze_id,
    )
    print(
        "Readiness lock SHA256:",
        protocol.canonical_readiness_lock_payload_sha256,
    )
    print("Trend semantics:", protocol.trend_semantics_id)
    print("Precision semantics:", protocol.precision_semantics_id)
    print("5e Reserve manifest ID:", protocol.reserve_manifest_id)
    print("Execution components:", len(
        protocol.execution_component_sha256
    ))
    print("Reserve B consumed:", False)
    print("Count thresholds:", False)
    print("LLM calls:", 0)
    print("Reserve manifest:", reserve_path)
    print("Execution protocol:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
