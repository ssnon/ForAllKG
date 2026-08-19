from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_standard2.claim_review_dev_validation import (
    DEFAULT_SPEC_ROOT,
    atomic_json,
    build_spec,
    verify_spec,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-abstract-chars", type=int, default=1400)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    args = parser.parse_args()

    if not args.run:
        parser.error("--run is required for specification freeze")

    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )
    if output_root.exists():
        print("claim-review-only DEV spec freeze: FAIL")
        print(" - output root exists:", output_root)
        return 2

    try:
        spec = build_spec(
            repo_root=ROOT,
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            max_abstract_chars=args.max_abstract_chars,
        )
    except Exception as exc:
        print("claim-review-only DEV spec freeze: FAIL")
        print(" -", f"{type(exc).__name__}: {exc}")
        print("Literature network calls:", 0)
        print("LLM calls:", 0)
        return 2

    print("SERS Claim-review-only DEV Specification")
    print("Prospective spec ID:", spec["spec_id"])
    print("Prospective spec SHA256:", spec["spec_sha256"])
    print("Source ranker run:", spec["source_ranker_run_id"])
    print("Canonical works:", spec["canonical_work_count"])
    print("Claims:", spec["claim_count"])
    print("Core claims:", spec["core_claim_count"])
    print("Frozen top-N:", 8)
    print("Review model:", spec["review_backend"]["model"])
    print("API key env:", spec["review_backend"]["api_key_env"])
    print("API key configured:", True)
    print("Base URL:", spec["review_backend"]["base_url"])
    print("Instructor mode:", spec["review_backend"]["instructor_mode"])
    print("Temperature:", spec["review_backend"]["temperature"])
    print("Parse retries:", spec["review_backend"]["parse_retries"])
    print("Literature network calls:", 0)
    print("LLM calls:", 0)
    print("Hypothesis-level novelty verdict:", False)

    output_root.mkdir(parents=True, exist_ok=False)
    spec_path = output_root / "claim_review_spec.json"
    marker_path = output_root / "SPEC_FREEZE_PASS.json"
    try:
        atomic_json(spec_path, spec)
        issues, verified = verify_spec(
            repo_root=ROOT,
            spec_path=spec_path,
        )
        if issues:
            raise RuntimeError(
                "spec verification failed:\n- "
                + "\n- ".join(issues)
            )
        atomic_json(
            marker_path,
            {
                "status": "spec_freeze_pass",
                "spec_id": verified["spec_id"],
                "spec_sha256": verified["spec_sha256"],
                "source_ranker_run_id":
                    verified["source_ranker_run_id"],
                "literature_network_calls": 0,
                "llm_calls": 0,
                "hypothesis_level_novelty_status_computed": False,
            },
        )
    except Exception:
        for path in (marker_path, spec_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            output_root.rmdir()
        except OSError:
            pass
        raise

    print("claim-review-only DEV specification: FROZEN")
    print("Spec ID:", spec["spec_id"])
    print("Literature network calls:", 0)
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
