from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.alpha4c5f2_readiness import (
    audit_sers_alpha4c5f2_canonical_readiness,
    prepare_sers_alpha4c5f2_readiness_lock,
)
from dac_her.alpha4c5h1_reserve_b import (
    DEFAULT_5H_CONFIRMATION_PROTOCOL,
    DEFAULT_5H_FREEZE_MANIFEST,
    load_and_verify_5h_binding,
)


ROOT = Path.cwd()
DEFAULT_OUTPUT = Path(
    "evaluation/sers_alpha4c5h1/reserve_b_v1/control/"
    "canonical_readiness_lock.json"
)


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "alpha4c.5h.1 Reserve-B canonical readiness gate. "
            "Structural/provenance only; zero scientific value disclosure "
            "and zero LLM calls."
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--confirm-canonical-refreeze",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    freeze_path = rooted(args.freeze_manifest)
    confirmation_path = rooted(args.confirmation_protocol)
    output_path = rooted(args.output)

    _freeze, confirmation = load_and_verify_5h_binding(
        root=ROOT,
        freeze_manifest_path=freeze_path,
        confirmation_protocol_path=confirmation_path,
    )
    paper_ids = sorted(
        str(value)
        for value in confirmation["reserve_b_paper_ids"]
    )

    audits = [
        audit_sers_alpha4c5f2_canonical_readiness(
            paper_id,
            detailed=False,
        )
        for paper_id in paper_ids
    ]
    not_ready = [
        row for row in audits if row["ready"] is not True
    ]
    blocked = [
        row
        for row in not_ready
        if row["refreeze_eligible"] is not True
    ]

    print("alpha4c.5h.1 Reserve-B readiness")
    print("5h freeze ID:", _freeze["freeze_id"])
    print(
        "Confirmation protocol ID:",
        confirmation["confirmation_protocol_id"],
    )
    print("Papers:", len(paper_ids))
    print("Ready:", len(paper_ids) - len(not_ready))
    print("Not ready:", len(not_ready))
    print("Non-refreeze-eligible:", len(blocked))
    print("Scientific value disclosure:", False)
    print("Reserve B consumed:", False)
    print("LLM calls:", 0)

    for row in not_ready:
        print(
            " -",
            row["paper_id"],
            "issues=",
            row["canonical"]["readiness_issues"],
            "refreeze_eligible=",
            row["refreeze_eligible"],
        )

    if args.preflight:
        if not_ready:
            print("Readiness preflight: BLOCKED")
            return 2
        print("Readiness preflight: PASS")
        return 0

    if output_path.exists():
        raise SystemExit(
            f"Refusing existing readiness lock: {output_path}"
        )
    if not args.confirm_canonical_refreeze:
        raise SystemExit(
            "--confirm-canonical-refreeze is required with --prepare."
        )
    if blocked:
        raise SystemExit(
            "Non-refreeze-eligible readiness failures: "
            + ", ".join(row["paper_id"] for row in blocked)
        )

    lock = prepare_sers_alpha4c5f2_readiness_lock(
        paper_ids=paper_ids,
        output_path=output_path,
        allow_refreeze=True,
        source_label=str(confirmation_path.relative_to(ROOT)),
    )
    print("Readiness preparation: PASS")
    print("Lock semantics:", lock["semantics_id"])
    print("Lock SHA256:", lock["lock_sha256"])
    print("All ready:", lock["all_ready"])
    print("Scientific value disclosure:", False)
    print("New extraction LLM calls:", 0)
    print("Reserve B consumed:", False)
    print("Saved:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
