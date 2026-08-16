from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.alpha4c5f_reserve import sha256_file
from dac_her.alpha4c5h2_postmortem import (
    DEFAULT_POSTMORTEM_MANIFEST,
    DEFAULT_SOURCE_PROTOCOL,
    atomic_json,
    build_postmortem_manifest,
    load_postmortem_manifest,
    verify_postmortem_manifest,
)


ROOT = Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the consumed alpha4c.5h.1 Reserve-B validation "
            "failure as an immutable, descriptive alpha4c.5h.2 "
            "postmortem manifest. This command does not rerun science, "
            "does not call an LLM, and does not mutate the closed 5h.1 "
            "campaign directory."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_SOURCE_PROTOCOL,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_POSTMORTEM_MANIFEST,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--confirm-terminal-reserve-b-failure",
        action="store_true",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _print_summary(manifest) -> None:
    print("alpha4c.5h.2 Reserve-B postmortem")
    print("Postmortem ID:", manifest.postmortem_id)
    print("Postmortem SHA256:", manifest.postmortem_sha256)
    print("Source campaign:", manifest.source_campaign_id)
    print("Partition:", manifest.source_partition)
    print("Papers:", manifest.paper_count)
    print("Reserve B consumed:", manifest.reserve_consumed)
    print("Campaign terminal state:", manifest.campaign_terminal_state)
    print("Failure stage:", manifest.failure_stage)
    print(
        "Observed error codes:",
        ", ".join(manifest.observed_error_codes),
    )
    print("Maker repairs:", manifest.maker_repair_attempts)
    print("Maker validation errors:", manifest.maker_validation_errors)
    print("Maker validation warnings:", manifest.maker_validation_warnings)
    print("Campaign closed:", manifest.campaign_closed)
    print("Rerun allowed:", manifest.rerun_allowed)
    print(
        "Reserve-B failure authorizes tuning:",
        manifest.reserve_b_failure_authorizes_tuning,
    )
    print(
        "Scientific transformation performed:",
        manifest.scientific_transformation_performed,
    )
    print("Scientific values printed:", manifest.scientific_values_printed)
    print("LLM calls:", manifest.llm_calls)


def main() -> int:
    args = parse_args()
    protocol_path = _resolve(args.protocol)
    output_path = _resolve(args.output)

    # The postmortem must live outside the closed 5h.1 evaluation root.
    closed_root = (ROOT / "evaluation/sers_alpha4c5h1/reserve_b_v1").resolve()
    try:
        output_path.resolve().relative_to(closed_root)
    except ValueError:
        pass
    else:
        raise SystemExit(
            "Refusing to write alpha4c.5h.2 inside the closed "
            "alpha4c.5h.1 Reserve-B campaign directory."
        )

    manifest = build_postmortem_manifest(
        root=ROOT,
        protocol_path=protocol_path,
    )
    _print_summary(manifest)

    if args.preflight:
        if output_path.exists():
            print(
                "Existing postmortem manifest:",
                output_path,
            )
            existing = load_postmortem_manifest(output_path)
            issues = verify_postmortem_manifest(
                root=ROOT,
                manifest=existing,
            )
            if issues:
                print("Existing postmortem verification: FAIL")
                for issue in issues:
                    print(" -", issue)
                return 2
            print("Existing postmortem verification: PASS")
        else:
            print("Output already exists: False")
        print("Postmortem preflight: PASS")
        print("Write performed: False")
        return 0

    if not args.confirm_terminal_reserve_b_failure:
        raise SystemExit(
            "--confirm-terminal-reserve-b-failure is required for "
            "the one-time postmortem freeze."
        )
    if output_path.exists():
        raise SystemExit(
            "Postmortem manifest already exists; refusing overwrite: "
            f"{output_path}"
        )

    atomic_json(
        output_path,
        manifest.model_dump(mode="json"),
    )
    observed = load_postmortem_manifest(output_path)
    if observed.model_dump(mode="json") != manifest.model_dump(mode="json"):
        raise RuntimeError("Post-write postmortem reload mismatch.")

    issues = verify_postmortem_manifest(
        root=ROOT,
        manifest=observed,
    )
    if issues:
        raise RuntimeError(
            "Post-write postmortem verification failed:\n- "
            + "\n- ".join(issues)
        )

    print("Postmortem freeze: PASS")
    print("Manifest:", output_path)
    print("Manifest file SHA256:", sha256_file(output_path))
    print("Closed 5h.1 campaign modified: False")
    print("Scientific semantics modified: False")
    print("Reserve B rerun: False")
    print("LLM calls: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
