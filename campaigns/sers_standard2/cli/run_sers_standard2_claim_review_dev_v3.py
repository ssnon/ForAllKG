from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from dac_her.prior_art_review_audit import prior_art_review_audit_scope
from campaigns.sers_standard2.claim_review_dev_validation import (
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
from campaigns.sers_standard2.claim_review_dev_validation_v3 import (
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    atomic_json,
    atomic_text,
    create_backend,
    prompt_allowed_id_check,
    validate_draft_work_ids,
    verify_spec,
    sha256_json,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--confirm-one-shot-claim-review-dev-v3",
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
    if not args.confirm_one_shot_claim_review_dev_v3:
        parser.error(
            "--confirm-one-shot-claim-review-dev-v3 is required"
        )

    spec_root = (
        args.spec_root if args.spec_root.is_absolute()
        else ROOT / args.spec_root
    )
    output_root = (
        args.output_root if args.output_root.is_absolute()
        else ROOT / args.output_root
    )

    issues, spec = verify_spec(
        repo_root=ROOT,
        spec_path=spec_root / "claim_review_spec_v3.json",
    )
    if issues:
        print("claim-review-only DEV v3 preflight: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2

    marker = read_json(spec_root / "SPEC_FREEZE_PASS.json")
    if (
        marker.get("status") != "spec_freeze_pass"
        or marker.get("spec_id") != spec.get("spec_id")
    ):
        print("claim-review-only DEV v3 preflight: FAIL")
        print(" - spec freeze marker mismatch")
        return 2

    if output_root.exists():
        print("claim-review-only DEV v3 preflight: FAIL")
        print(" - output root exists:", output_root)
        print("Automatic rerun:", False)
        return 2

    plan, packet, _ranker_spec, ranker_report = load_inputs(ROOT)

    print("SERS Claim-review-only DEV v3 One-Shot Validation")
    print("Spec ID:", spec["spec_id"])
    print("Parent v2 failed run:", spec["parent_v2_failed_run_id"])
    print("Claims:", 12)
    print("Frozen top-N:", 8)
    print("Relation-nucleus hardening:", True)
    print("Exact work-ID copy contract:", True)
    print("Fail-fast invalid work ID:", True)
    print("Compiler changed:", False)
    print("Ranker recomputed:", False)
    print("Hypothesis-level novelty verdict:", False)
    print("Automatic rerun:", False)

    output_root.mkdir(parents=True, exist_ok=False)
    audit_path = output_root / "prior_art_review_calls.jsonl"
    telemetry_path = output_root / "llm_telemetry.jsonl"
    atomic_json(
        output_root / "RUN_STARTED.json",
        {
            "status": "run_started",
            "spec_id": spec["spec_id"],
            "parent_v2_failed_run_id":
                spec["parent_v2_failed_run_id"],
            "expected_logical_review_calls": 12,
            "automatic_rerun_authorized": False,
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
            assessment_kind=
                "claim_review_only_dev_v3_relation_nucleus_work_id",
            source_portfolio_id=plan.source_portfolio_id,
            query_plan_id=plan.plan_id,
            prior_art_packet_id=packet.packet_id,
            source_ranker_run_id=spec["source_ranker_run_id"],
            claim_review_spec_id=spec["spec_id"],
            parent_v2_failed_run_id=spec["parent_v2_failed_run_id"],
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
                allowed = [
                    work.work_id
                    for work in candidates.ranked_works
                ]

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

                # Provenance failure is terminal for this one-shot.
                validate_draft_work_ids(
                    claim_id=claim_id,
                    draft=draft,
                    allowed_work_ids=allowed,
                )
                drafts[claim_id] = draft
                print(
                    "    returned matches:",
                    len(draft.matches),
                    "| work-ID contract: PASS",
                )

        prompt_ok, prompt_issues = prompt_allowed_id_check(
            prompt_records=backend.prompt_records,
            ranker_report=ranker_report,
        )
        if not prompt_ok:
            raise RuntimeError(
                "Captured prompt ALLOWED_WORK_IDS verification failed: "
                + "; ".join(prompt_issues)
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
        checks = dict(checks)
        checks["allowed_work_id_block_verified_in_every_prompt"] = (
            prompt_ok
        )
        checks["reviewer_match_ids_unique_per_claim"] = all(
            len([m.work_id for m in draft.matches])
            == len({m.work_id for m in draft.matches})
            for draft in drafts.values()
        )
        checks["reviewer_match_count_not_exceed_candidates"] = all(
            len(drafts[str(row["claim_id"])].matches)
            <= len(row["top_ranked_works"])
            for row in ranker_report["claim_reports"]
        )

        secret_values = [
            os.getenv(str(spec["review_backend"]["api_key_env"]))
            or "",
            os.getenv("OPENAI_API_KEY") or "",
            os.getenv("OPENROUTER_API_KEY") or "",
            os.getenv("SEMANTIC_SCHOLAR_API_KEY") or "",
            os.getenv("OPENALEX_API_KEY") or "",
        ]
        secret_scan_pass, offending = scan_output_for_secrets(
            output_root=output_root,
            secret_values=secret_values,
        )
        if not secret_scan_pass:
            for rel in offending:
                path = output_root / rel
                if path.is_file():
                    path.unlink()
            raise RuntimeError(
                "secret scan failed; offending persisted files removed"
            )

        structural_pass = all(checks.values()) and secret_scan_pass
        status_counts = dict(
            sorted(Counter(r.status for r in reviews).items())
        )
        relationship_counts = dict(
            sorted(
                Counter(
                    m.relationship
                    for r in reviews
                    for m in r.matches
                ).items()
            )
        )

        body = {
            "schema_version":
                "sers-standard2-claim-review-only-dev-run-v3",
            "semantics_id":
                "sers_standard2_claim_review_relation_nucleus_work_id_v3",
            "source_spec_id": spec["spec_id"],
            "source_spec_sha256": spec["spec_sha256"],
            "parent_v2_failed_run_id":
                spec["parent_v2_failed_run_id"],
            "source_ranker_run_id": spec["source_ranker_run_id"],
            "structural_outcome": (
                "CLAIM_REVIEW_V3_STRUCTURAL_DEV_PASS"
                if structural_pass
                else "CLAIM_REVIEW_V3_STRUCTURAL_DEV_FAIL"
            ),
            "scientific_relationship_outcome":
                "MANUAL_REVIEW_REQUIRED",
            "checks": checks,
            "secret_scan_pass": secret_scan_pass,
            "diagnostics": {
                **diagnostics,
                "compiled_status_counts": status_counts,
                "compiled_relationship_counts": relationship_counts,
                "prompt_allowed_id_issues": prompt_issues,
            },
            "claim_reviews": [
                review.model_dump(mode="json")
                for review in reviews
            ],
            "raw_review_drafts": {
                cid: draft.model_dump(mode="json")
                for cid, draft in sorted(drafts.items())
            },
            "logical_review_calls": logical_calls,
            "successful_prior_art_audit_rows": len(audit_rows),
            "telemetry_row_count": len(telemetry_rows),
            "literature_network_calls": 0,
            "ranker_recomputed": False,
            "compiler_changed_from_v2": False,
            "invalid_id_guess_mapping_used": False,
            "case_specific_expected_statuses_used": False,
            "hypothesis_level_novelty_status_computed": False,
            "automatic_next_stage_authorized": False,
            "fresh_reserve_consumed": False,
        }
        body["run_sha256"] = sha256_json(body)
        body["run_id"] = (
            "sers_standard2_claim_review_only_dev_run_v3:"
            + body["run_sha256"][:20]
        )

        atomic_json(
            output_root / "claim_review_report_v3.json",
            body,
        )
        atomic_text(
            output_root / "human_relationship_audit_v3.md",
            render_human_audit(
                reviews=reviews,
                drafts=drafts,
                ranker_report=ranker_report,
            ),
        )

        if not structural_pass:
            atomic_json(
                output_root / "STRUCTURAL_FAIL.json",
                {
                    "status": "structural_fail",
                    "run_id": body["run_id"],
                    "automatic_rerun_authorized": False,
                    "automatic_next_stage_authorized": False,
                },
            )
            print("claim-review-only DEV v3: STRUCTURAL FAIL")
            print("Checks:", checks)
            return 2

        atomic_json(
            output_root / "STRUCTURAL_PASS.json",
            {
                "status": "structural_pass",
                "run_id": body["run_id"],
                "run_sha256": body["run_sha256"],
                "scientific_relationship_outcome":
                    "MANUAL_REVIEW_REQUIRED",
                "automatic_next_stage_authorized": False,
                "fresh_reserve_consumed": False,
            },
        )
        try:
            (output_root / "RUN_STARTED.json").unlink()
        except FileNotFoundError:
            pass

        print()
        print("claim-review-only DEV v3: COMPLETE")
        print("Run ID:", body["run_id"])
        print("Structural outcome:", body["structural_outcome"])
        print(
            "Compiled status counts:",
            status_counts,
        )
        print(
            "Compiled relationship counts:",
            relationship_counts,
        )
        print("Logical review calls:", logical_calls)
        print("Literature network calls:", 0)
        print("Ranker recomputed:", False)
        print("Compiler changed:", False)
        print("Invalid-ID guess mapping:", False)
        print("Hypothesis-level novelty verdict:", False)
        print("Automatic next stage authorized:", False)
        print(
            "Human audit:",
            output_root / "human_relationship_audit_v3.md",
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
        print("claim-review-only DEV v3: INTERRUPTED")
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
