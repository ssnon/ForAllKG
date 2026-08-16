from __future__ import annotations

import argparse
import os
from pathlib import Path

from dac_her.prior_art_review_audit import prior_art_review_audit_scope
from dac_her.standard2_claim_review_dev_validation import (
    candidate_set_from_ranker_row,
    compile_drafts,
    load_inputs,
    read_json,
    read_jsonl,
    render_human_audit,
    reviewer_input_from_candidates,
    scan_output_for_secrets,
    structural_checks,
)
from dac_her.standard2_claim_review_dev_validation_v2 import (
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    atomic_json,
    atomic_text,
    create_backend,
    report_from_results,
    verify_spec,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--confirm-one-shot-claim-review-dev-v2",
        action="store_true",
    )
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    args = parser.parse_args()

    if not args.run:
        parser.error("--run is required")
    if not args.confirm_one_shot_claim_review_dev_v2:
        parser.error(
            "--confirm-one-shot-claim-review-dev-v2 is required"
        )

    spec_root = (
        args.spec_root
        if args.spec_root.is_absolute()
        else ROOT / args.spec_root
    )
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )

    issues, spec = verify_spec(
        repo_root=ROOT,
        spec_path=spec_root / "claim_review_spec_v2.json",
    )
    if issues:
        print("claim-review-only DEV v2 preflight: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Literature network calls:", 0)
        print("LLM calls:", 0)
        return 2

    marker_path = spec_root / "SPEC_FREEZE_PASS.json"
    if not marker_path.is_file():
        print("claim-review-only DEV v2 preflight: FAIL")
        print(" - SPEC_FREEZE_PASS missing")
        return 2
    marker = read_json(marker_path)
    if (
        marker.get("status") != "spec_freeze_pass"
        or marker.get("spec_id") != spec.get("spec_id")
    ):
        print("claim-review-only DEV v2 preflight: FAIL")
        print(" - spec freeze marker mismatch")
        return 2

    if output_root.exists():
        print("claim-review-only DEV v2 preflight: FAIL")
        print(" - output root exists:", output_root)
        print("Automatic rerun:", False)
        return 2

    plan, packet, _ranker_spec, ranker_report = load_inputs(ROOT)

    print("SERS Claim-review-only DEV v2 One-Shot Validation")
    print("Spec ID:", spec["spec_id"])
    print("Parent v1 run:", spec["parent_v1_run_id"])
    print("Source ranker run:", spec["source_ranker_run_id"])
    print("Claims:", spec["claim_count"])
    print("Core claims:", spec["core_claim_count"])
    print("Frozen top-N:", 8)
    print("Review model:", spec["review_backend"]["model"])
    print("Relation-nucleus hardening:", True)
    print("Compiler changed:", False)
    print("Ranker recomputed:", False)
    print("Literature retrieval:", False)
    print("Claim decomposition:", False)
    print("Hypothesis-level novelty verdict:", False)
    print("Expected logical LLM review calls:", 12)
    print("Automatic rerun:", False)

    output_root.mkdir(parents=True, exist_ok=False)
    audit_path = output_root / "prior_art_review_calls.jsonl"
    telemetry_path = output_root / "llm_telemetry.jsonl"

    atomic_json(
        output_root / "RUN_STARTED.json",
        {
            "status": "run_started",
            "spec_id": spec["spec_id"],
            "parent_v1_run_id": spec["parent_v1_run_id"],
            "source_ranker_run_id": spec["source_ranker_run_id"],
            "expected_logical_review_calls": 12,
            "automatic_rerun_authorized": False,
            "hypothesis_level_novelty_status_computed": False,
        },
    )

    previous_audit_env = os.getenv(
        "GRAPHAGENTS_PRIOR_ART_REVIEW_AUDIT_PATH"
    )
    os.environ[
        "GRAPHAGENTS_PRIOR_ART_REVIEW_AUDIT_PATH"
    ] = str(audit_path)

    drafts = {}
    logical_calls = 0
    backend = create_backend(
        spec=spec,
        telemetry_path=telemetry_path,
    )

    try:
        claims = {
            claim.claim_id: claim
            for group in plan.claims
            for claim in group.claims
        }

        with prior_art_review_audit_scope(
            assessment_kind="claim_review_only_dev_v2_relation_nucleus",
            source_portfolio_id=plan.source_portfolio_id,
            query_plan_id=plan.plan_id,
            prior_art_packet_id=packet.packet_id,
            source_ranker_run_id=spec["source_ranker_run_id"],
            claim_review_spec_id=spec["spec_id"],
            parent_v1_run_id=spec["parent_v1_run_id"],
        ):
            for index, row in enumerate(
                ranker_report["claim_reports"],
                start=1,
            ):
                claim_id = str(row["claim_id"])
                claim = claims[claim_id]
                candidates = candidate_set_from_ranker_row(row)
                reviewer_input = reviewer_input_from_candidates(
                    packet=packet,
                    candidates=candidates,
                )

                print(
                    f"[{index}/12] review "
                    f"{claim.importance} {claim.kind} "
                    f"{claim.claim_id}"
                )
                draft = backend.review_claim(
                    claim,
                    reviewer_input,
                )
                logical_calls += 1
                drafts[claim_id] = draft
                print(
                    "    returned matches:",
                    len(draft.matches),
                )

        reviews = compile_drafts(
            spec=spec,
            plan=plan,
            packet=packet,
            ranker_report=ranker_report,
            drafts=drafts,
        )

        audit_rows = read_jsonl(audit_path)
        telemetry_rows = read_jsonl(telemetry_path)

        checks, diagnostics = structural_checks(
            spec=spec,
            plan=plan,
            packet=packet,
            ranker_report=ranker_report,
            drafts=drafts,
            reviews=reviews,
            logical_review_calls=logical_calls,
            audit_rows=audit_rows,
            prompt_records=backend.prompt_records,
        )

        prompt_manifest = {
            "schema_version":
                "claim-review-prompt-manifest-v2",
            "relation_nucleus_hardening": True,
            "review_prompt_sha256":
                spec["review_backend"]["review_prompt_sha256"],
            "prompts": [
                {
                    "name": record.name,
                    "prompt_sha256": record.prompt_sha256,
                }
                for record in backend.prompt_records
            ],
            "full_prompt_text_persisted": False,
        }
        atomic_json(
            output_root / "prompt_manifest.json",
            prompt_manifest,
        )

        secret_values = [
            os.getenv(str(spec["review_backend"]["api_key_env"]))
            or "",
            os.getenv("OPENAI_API_KEY") or "",
            os.getenv("OPENROUTER_API_KEY") or "",
            os.getenv("SEMANTIC_SCHOLAR_API_KEY") or "",
            os.getenv("OPENALEX_API_KEY") or "",
        ]
        secret_scan_pass, offending_files = scan_output_for_secrets(
            output_root=output_root,
            secret_values=secret_values,
        )
        if not secret_scan_pass:
            for rel in offending_files:
                path = output_root / rel
                if path.is_file():
                    path.unlink()
            atomic_json(
                output_root / "SECRET_SCAN_FAIL.json",
                {
                    "status": "secret_scan_fail",
                    "offending_files_removed": offending_files,
                    "automatic_rerun_authorized": False,
                },
            )
            raise RuntimeError(
                "secret scan failed; offending persisted files "
                "were removed"
            )

        report = report_from_results(
            spec=spec,
            reviews=reviews,
            drafts=drafts,
            checks=checks,
            diagnostics=diagnostics,
            prompt_records=backend.prompt_records,
            audit_rows=audit_rows,
            telemetry_rows=telemetry_rows,
            secret_scan_pass=secret_scan_pass,
        )

        atomic_json(
            output_root / "claim_review_report_v2.json",
            report,
        )
        atomic_text(
            output_root / "human_relationship_audit_v2.md",
            render_human_audit(
                reviews=reviews,
                drafts=drafts,
                ranker_report=ranker_report,
            ),
        )

        if report["structural_outcome"] != (
            "CLAIM_REVIEW_V2_STRUCTURAL_DEV_PASS"
        ):
            atomic_json(
                output_root / "STRUCTURAL_FAIL.json",
                {
                    "status": "structural_fail",
                    "run_id": report["run_id"],
                    "run_sha256": report["run_sha256"],
                    "automatic_rerun_authorized": False,
                    "automatic_next_stage_authorized": False,
                },
            )
            print()
            print("claim-review-only DEV v2: STRUCTURAL FAIL")
            print("Run ID:", report["run_id"])
            print("Checks:", report["checks"])
            print("Diagnostics:", report["diagnostics"])
            print("Automatic rerun:", False)
            return 2

        atomic_json(
            output_root / "STRUCTURAL_PASS.json",
            {
                "status": "structural_pass",
                "run_id": report["run_id"],
                "run_sha256": report["run_sha256"],
                "scientific_relationship_outcome":
                    "MANUAL_REVIEW_REQUIRED",
                "hypothesis_level_novelty_status_computed": False,
                "automatic_next_stage_authorized": False,
                "fresh_reserve_consumed": False,
            },
        )
        try:
            (output_root / "RUN_STARTED.json").unlink()
        except FileNotFoundError:
            pass

        print()
        print("claim-review-only DEV v2: COMPLETE")
        print("Run ID:", report["run_id"])
        print(
            "Structural outcome:",
            report["structural_outcome"],
        )
        print(
            "Scientific relationship outcome:",
            report["scientific_relationship_outcome"],
        )
        print(
            "Compiled status counts:",
            report["diagnostics"]["compiled_status_counts"],
        )
        print(
            "Compiled relationship counts:",
            report["diagnostics"]["compiled_relationship_counts"],
        )
        print(
            "Core NO_DIRECT_MATCH_FOUND:",
            report["diagnostics"][
                "core_no_direct_match_claim_ids"
            ],
        )
        print(
            "Core INSUFFICIENT_METADATA:",
            report["diagnostics"][
                "core_insufficient_metadata_claim_ids"
            ],
        )
        print("Logical LLM review calls:", report["logical_review_calls"])
        print("Literature network calls:", 0)
        print("Ranker recomputed:", False)
        print("Compiler changed:", False)
        print("Hypothesis-level novelty verdict:", False)
        print("Automatic next stage authorized:", False)
        print(
            "Human audit:",
            output_root / "human_relationship_audit_v2.md",
        )
        return 0

    except Exception as exc:
        atomic_json(
            output_root / "RUN_INTERRUPTED.json",
            {
                "status": "run_interrupted",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "completed_logical_review_calls": logical_calls,
                "automatic_rerun_authorized": False,
                "automatic_next_stage_authorized": False,
                "hypothesis_level_novelty_status_computed": False,
            },
        )
        print()
        print("claim-review-only DEV v2: INTERRUPTED")
        print(" -", f"{type(exc).__name__}: {exc}")
        print("Completed logical review calls:", logical_calls)
        print("Automatic rerun:", False)
        return 2

    finally:
        if previous_audit_env is None:
            os.environ.pop(
                "GRAPHAGENTS_PRIOR_ART_REVIEW_AUDIT_PATH",
                None,
            )
        else:
            os.environ[
                "GRAPHAGENTS_PRIOR_ART_REVIEW_AUDIT_PATH"
            ] = previous_audit_env


if __name__ == "__main__":
    raise SystemExit(main())
