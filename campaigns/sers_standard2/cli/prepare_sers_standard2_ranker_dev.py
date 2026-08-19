from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_standard2.ranker_dev_validation import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_SPEC_ROOT,
    atomic_json,
    atomic_text,
    build_spec,
    canonical_json,
    verify_spec,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    args = parser.parse_args()

    diagnostic_root = args.diagnostic_root.expanduser().resolve()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )

    if output_root.exists():
        print("ranker-only DEV spec preflight: FAIL")
        print(" - output root exists:", output_root)
        print("Network calls:", 0)
        return 2

    try:
        spec = build_spec(
            repo_root=ROOT,
            diagnostic_root=diagnostic_root,
        )
    except Exception as exc:
        print("ranker-only DEV spec preflight: FAIL")
        print(" -", f"{type(exc).__name__}: {exc}")
        print("Network calls:", 0)
        return 2

    print("SERS Ranker-only DEV Specification")
    print("Prospective spec ID:", spec["spec_id"])
    print("Prospective spec SHA256:", spec["spec_sha256"])
    print("Canonical packet:", spec["source_canonical_packet_id"])
    print("Canonical works:", spec["canonical_work_count"])
    print("Claims:", spec["claim_count"])
    print("Core claims:", spec["core_claim_count"])
    print("Domain profile:", spec["ranker"]["domain_profile_id"])
    print("Embedding model:", spec["ranker"]["embed_model"])
    print(
        "Model fingerprint:",
        spec["ranker"]["model_behavior_fingerprint"]["sentinel_sha256"],
    )
    print(
        "Top-N:",
        spec["ranker"]["max_ranked_works_per_claim"],
    )
    print("LLM calls:", 0)
    print("Network calls:", 0)
    print("Novelty verdict:", False)
    print("Scientific relevance auto-pass:", False)

    if args.preflight:
        print("Preflight: PASS")
        print("Write performed: False")
        return 0

    output_root.mkdir(parents=True, exist_ok=False)
    spec_path = output_root / "ranker_spec.json"
    query_snapshot_path = output_root / "frozen_query_plan.json"
    pass_path = output_root / "SPEC_FREEZE_PASS.json"

    try:
        source_query_path = (
            diagnostic_root
            / Path(
                "evaluation/sers_alpha4c5k/dev_e2e_v2/"
                "external_novelty.claims_queries.json"
            )
        )
        atomic_text(
            query_snapshot_path,
            source_query_path.read_text(encoding="utf-8"),
        )
        atomic_json(spec_path, spec)

        issues, verified = verify_spec(
            repo_root=ROOT,
            diagnostic_root=diagnostic_root,
            spec_path=spec_path,
        )
        if issues:
            raise RuntimeError(
                "spec verification failed:\n- "
                + "\n- ".join(issues)
            )

        atomic_json(
            pass_path,
            {
                "status": "spec_freeze_pass",
                "spec_id": verified["spec_id"],
                "spec_sha256": verified["spec_sha256"],
                "source_query_plan_file_sha256":
                    verified["source_query_plan_file_sha256"],
                "source_canonical_packet_sha256":
                    verified["source_canonical_packet_sha256"],
                "network_calls": 0,
                "llm_calls": 0,
            },
        )
    except Exception:
        for path in (
            pass_path,
            query_snapshot_path,
            spec_path,
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            output_root.rmdir()
        except OSError:
            pass
        raise

    print("ranker-only DEV specification: FROZEN")
    print("Spec ID:", spec["spec_id"])
    print("Network calls:", 0)
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
