from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from dac_her.fresh_c_c1b1_reviewer_contract_v1 import (
    FINAL_ADJUDICATOR_SYSTEM_PROMPT,
    PAPER_REVIEW_SYSTEM_PROMPT,
    FreshCFinalAdjudication,
    FreshCPaperReview,
    H1,
    H3,
)
from dac_her.fresh_c_c1b2_scientific_adjudication_v1 import (
    DEFAULT_PROTOCOL_FREEZE_DIR,
    DEFAULT_PROTOCOL_PATH,
    DEFAULT_RUN_DIR,
    DEFAULT_SCHEMA_QUALIFICATION_DIR,
    atomic_json,
    build_final_prompt,
    build_target_boundaries,
    canonical_json_sha256,
    format_paper_prompt,
    load_object,
    openai_strict_transport_schema,
    review_payload_sha,
    schema_qualification_valid,
    validate_corpus_metadata,
    validate_final_against_reviews,
    validate_frozen_lineage,
    validate_pages_manifest,
    validate_protocol,
    validate_review_grounding,
    validate_runtime_env,
)

def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

def _client(p: dict[str, Any], env: dict[str, Any]) -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=env["base_url"],
        timeout=300.0,
        max_retries=0,
    )

def _call_structured(
    *,
    client: OpenAI,
    p: dict[str, Any],
    model_cls,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
):
    response = client.chat.completions.create(
        model=p["reviewer_model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        seed=p["deterministic_seed"],
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": model_cls.__name__,
                "strict": True,
                "schema": openai_strict_transport_schema(model_cls),
            },
        },
        extra_body={
            "reasoning": {
                "effort": p["reasoning_effort"],
                "exclude": p["reasoning_exclude"],
            },
            "provider": {
                "only": p["provider_only"],
                "allow_fallbacks": p["provider_allow_fallbacks"],
                "require_parameters": p["provider_require_parameters"],
                "data_collection": p["provider_data_collection"],
            },
        },
    )
    if response.model != p["reviewer_model"]:
        raise RuntimeError(
            f"Served model drifted: requested={p['reviewer_model']} "
            f"served={response.model}"
        )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Structured response content is empty")
    parsed = model_cls.model_validate_json(content)
    usage = response.usage
    usage_row = {
        "requested_model": p["reviewer_model"],
        "served_model": response.model,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "finish_reason": response.choices[0].finish_reason,
    }
    return parsed, usage_row

def synthetic_schema_qualification() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    validate_frozen_lineage(root)
    env = validate_runtime_env()

    out = root / DEFAULT_SCHEMA_QUALIFICATION_DIR
    if out.exists():
        raise FileExistsError(
            "C1B.2 exact-schema qualification directory already exists; "
            "do not overwrite a prior qualification result"
        )

    client = _client(p, env)

    paper_prompt = (
        "Synthetic schema qualification only. No Fresh-C text or real scientific "
        "hypothesis content is present. Return a valid paper review for reserve 1 "
        "with canonical_id='doi:synthetic', materialization_mode='DIRECT_ORIGINAL'. "
        "Assess H1 and H3 exactly once each as IRRELEVANT with short synthetic "
        "rationales and no evidence. All guard flags must be false."
    )
    paper, paper_usage = _call_structured(
        client=client,
        p=p,
        model_cls=FreshCPaperReview,
        system_prompt=PAPER_REVIEW_SYSTEM_PROMPT,
        user_prompt=paper_prompt,
        max_tokens=p["paper_review_max_tokens"],
    )

    final_prompt = (
        "Synthetic schema qualification only. No Fresh-C text, literature, or "
        "real scientific hypothesis content is present. Return H1 and H3 as "
        "FRESH_C_INCONCLUSIVE, keep H2 terminally rejected, provide short "
        "synthetic rationales, no supporting_evidence, all_25_papers_processed=true, "
        "and every guard flag false."
    )
    final, final_usage = _call_structured(
        client=client,
        p=p,
        model_cls=FreshCFinalAdjudication,
        system_prompt=FINAL_ADJUDICATOR_SYSTEM_PROMPT,
        user_prompt=final_prompt,
        max_tokens=p["final_adjudication_max_tokens"],
    )

    # Exact deterministic checks that the synthetic payload did not smuggle science.
    if paper.canonical_id != "doi:synthetic":
        raise RuntimeError("Synthetic paper schema qualification canonical ID drifted")
    if {a.hypothesis_id for a in paper.assessments} != {H1, H3}:
        raise RuntimeError("Synthetic paper schema qualification target set drifted")
    if any(a.relation_label != "IRRELEVANT" for a in paper.assessments):
        raise RuntimeError("Synthetic paper schema qualification relation drifted")
    if final.h1_fresh_c_verdict != "FRESH_C_INCONCLUSIVE":
        raise RuntimeError("Synthetic final H1 verdict drifted")
    if final.h3_fresh_c_verdict != "FRESH_C_INCONCLUSIVE":
        raise RuntimeError("Synthetic final H3 verdict drifted")

    payload = {
        "schema_version": "sers-fresh-c-c1b2-exact-schema-qualification-v1",
        "protocol_id": p["protocol_id"],
        "protocol_sha256": p["protocol_sha256"],
        "requested_model": p["reviewer_model"],
        "served_model_paper": paper_usage["served_model"],
        "served_model_final": final_usage["served_model"],
        "paper_schema_passed": True,
        "final_schema_passed": True,
        "paper_review_schema_sha256": p["paper_review_schema_sha256"],
        "final_adjudication_schema_sha256": p["final_adjudication_schema_sha256"],
        "transport_schema_adapter_id": p["transport_schema_adapter_id"],
        "paper_review_transport_schema_sha256": p["paper_review_transport_schema_sha256"],
        "final_adjudication_transport_schema_sha256": p["final_adjudication_transport_schema_sha256"],
        "paper_review_prompt_sha256": p["paper_review_system_prompt_sha256"],
        "final_adjudicator_prompt_sha256": p["final_adjudicator_system_prompt_sha256"],
        "paper_usage": paper_usage,
        "final_usage": final_usage,
        "network_calls": 2,
        "llm_calls": 2,
        "fresh_c_scientific_text_used": False,
        "scientific_hypothesis_text_used": False,
        "scientific_adjudication_performed": False,
        "c1b2_live_authorized": False,
        "stop": True,
    }
    payload["qualification_sha256"] = canonical_json_sha256(payload)
    payload["qualification_id"] = (
        "sers_fresh_c_c1b2_exact_schema_qualification_v1:"
        + payload["qualification_sha256"][:20]
    )
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "qualification_result.json", payload)
    schema_qualification_valid(payload)

    print("Fresh-C C1B.2 exact-schema synthetic qualification")
    print(f"Qualification ID: {payload['qualification_id']}")
    print(f"Qualification SHA256: {payload['qualification_sha256']}")
    print(f"Requested/served paper model: {p['reviewer_model']} / {paper_usage['served_model']}")
    print(f"Requested/served final model: {p['reviewer_model']} / {final_usage['served_model']}")
    print("Exact FreshCPaperReview schema passed: True")
    print("Exact FreshCFinalAdjudication schema passed: True")
    print("Network calls: 2")
    print("LLM calls: 2")
    print("Fresh-C scientific text used: False")
    print("Scientific hypothesis text used: False")
    print("Scientific adjudication performed: False")
    print("C1B.2 live authorized: False")
    print("STOP: True")
    return 0

def _preflight_state(root: Path) -> dict[str, Any]:
    from scripts.verify_sers_fresh_c_c1b2_scientific_protocol_freeze_v1 import (
        main as verify_protocol_freeze,
    )
    verify_protocol_freeze()
    p = validate_protocol(root / DEFAULT_PROTOCOL_PATH)
    lineage = validate_frozen_lineage(root)
    env = validate_runtime_env()
    targets = build_target_boundaries(lineage["r2_report"])
    # Metadata/hash validation only. pages.json scientific text is NOT parsed here.
    records = validate_corpus_metadata(root, parse_pages=False)

    run_dir = root / DEFAULT_RUN_DIR
    run_absent = not run_dir.exists()
    if not run_absent:
        raise RuntimeError("C1B.2 run directory already exists")
    return {
        "protocol": p,
        "env": env,
        "targets": targets,
        "records": records,
        "run_absent": run_absent,
    }

def preflight() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight_state(root)
    p = state["protocol"]
    print("Fresh-C C1B.2 guarded scientific-adjudication preflight")
    print(f"Protocol ID: {p['protocol_id']}")
    print(f"Protocol freeze current: True")
    print(f"Reviewer model: {state['env']['reviewer_model']}")
    print(f"Base URL: {state['env']['base_url']}")
    print("Exact frozen scientific targets: 2 (H1/H3)")
    print("H2 terminal rejected: True")
    print("Exact frozen Fresh-C papers: 25/25")
    print("All materialized text/pages hashes: CURRENT")
    print("Paper order: 1..25")
    print("Expected scientific LLM/network calls: 26/26")
    print("Scientific page text parsed during preflight: False")
    print("Fresh-C scientific reviewer read performed: False")
    print("Scientific adjudication performed: False")
    print("One-shot marker not written: True")
    print("Live execution ready: True")
    print("Live execution authorized: False")
    print("Automatic post-C1B.2 transition: False")
    print("STOP: True")
    print("Preflight: PASS")
    return 0

def live_execute() -> int:
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    state = _preflight_state(root)
    p = state["protocol"]
    targets = state["targets"]
    records = state["records"]
    env = state["env"]

    run_dir = root / DEFAULT_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=False)
    reviews_dir = run_dir / "paper_reviews"
    reviews_dir.mkdir(parents=True, exist_ok=False)
    start_marker = run_dir / "C1B2_SCIENTIFIC_READ_STARTED.json"
    fail_marker = run_dir / "C1B2_SCIENTIFIC_ADJUDICATION_FAILED.json"

    # IRREVERSIBLE SCIENTIFIC-READ BOUNDARY:
    # write immediately before any pages.json scientific text is parsed.
    atomic_json(start_marker, {
        "schema_version": "sers-fresh-c-c1b2-scientific-read-started-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": p["protocol_id"],
        "protocol_sha256": p["protocol_sha256"],
        "fresh_reserve_c_already_consumed": True,
        "this_is_new_reserve_c_consumption": False,
        "scientific_read_irreversible_for_this_epoch": True,
        "same_epoch_rerun_allowed": False,
        "failure_restores_freshness": False,
        "failure_authorizes_tuning_on_fresh_c": False,
        "paper_review_order": list(range(1, 26)),
        "planned_paper_review_calls": 25,
        "planned_final_adjudication_calls": 1,
        "external_literature_lookup_allowed": False,
        "hypothesis_rewrite_allowed": False,
        "automatic_post_c1b2_transition_allowed": False,
        "stop": True,
    })

    network_attempts = 0
    llm_attempts = 0
    completed_indices: list[int] = []
    review_objects: list[FreshCPaperReview] = []
    review_records: list[dict[str, Any]] = []

    client = _client(p, env)

    try:
        for record in records:
            # First semantic/source-text parse occurs only AFTER the marker.
            corpus = load_object(root / (
                "evaluation/sers_fresh_c/"
                "c1a_r1_recovery_run_v1/recovered_corpus_manifest.json"
            ))
            source_row = next(
                row for row in corpus["records"]
                if row["reserve_index"] == record["reserve_index"]
            )
            pages_manifest = validate_pages_manifest(root, source_row)
            prompt = format_paper_prompt(
                target_boundaries=targets,
                reserve_index=record["reserve_index"],
                canonical_id=record["canonical_id"],
                materialization_mode=record["materialization_mode"],
                pages_manifest=pages_manifest,
            )
            network_attempts += 1
            llm_attempts += 1
            review, usage = _call_structured(
                client=client,
                p=p,
                model_cls=FreshCPaperReview,
                system_prompt=PAPER_REVIEW_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=p["paper_review_max_tokens"],
            )
            validate_review_grounding(
                review,
                expected_record=record,
                pages_manifest=pages_manifest,
            )
            review_sha = review_payload_sha(review)
            payload = {
                "schema_version": "sers-fresh-c-c1b2-paper-review-record-v1",
                "reserve_index": record["reserve_index"],
                "canonical_id": record["canonical_id"],
                "materialization_mode": record["materialization_mode"],
                "pages_manifest_sha256": record["pages_manifest_sha256"],
                "review_sha256": review_sha,
                "review": review.model_dump(mode="json"),
                "usage": usage,
                "fresh_c_scientific_text_read": True,
                "external_literature_used": False,
            }
            payload["record_sha256"] = canonical_json_sha256(payload)
            path = reviews_dir / f"reserve_c_{record['reserve_index']:03d}.json"
            atomic_json(path, payload)
            completed_indices.append(record["reserve_index"])
            review_objects.append(review)
            review_records.append({
                "reserve_index": record["reserve_index"],
                "canonical_id": record["canonical_id"],
                "record_path": str(path.relative_to(root)),
                "record_sha256": payload["record_sha256"],
                "review_sha256": review_sha,
            })
            print(
                f"[C1B.2] paper {record['reserve_index']:02d}/25 "
                f"reviewed | {record['canonical_id']}"
            )

        if completed_indices != list(range(1, 26)):
            raise RuntimeError("C1B.2 did not complete exact reserve order 1..25")

        final_prompt = build_final_prompt(
            target_boundaries=targets,
            reviews=review_objects,
        )
        network_attempts += 1
        llm_attempts += 1
        final, final_usage = _call_structured(
            client=client,
            p=p,
            model_cls=FreshCFinalAdjudication,
            system_prompt=FINAL_ADJUDICATOR_SYSTEM_PROMPT,
            user_prompt=final_prompt,
            max_tokens=p["final_adjudication_max_tokens"],
        )
        validate_final_against_reviews(final, review_objects)

        final_payload = {
            "schema_version": "sers-fresh-c-c1b2-final-adjudication-record-v1",
            "adjudication": final.model_dump(mode="json"),
            "usage": final_usage,
            "fresh_c_scope_only": True,
            "preservation_does_not_establish_absence_or_novelty": True,
            "external_literature_used": False,
            "count_threshold_used": False,
            "hypothesis_rewrite_performed": False,
            "hypothesis_upgrade_performed": False,
        }
        final_payload["record_sha256"] = canonical_json_sha256(final_payload)
        atomic_json(run_dir / "final_adjudication.json", final_payload)

        run_manifest = {
            "schema_version": "sers-fresh-c-c1b2-scientific-adjudication-run-v1",
            "protocol_id": p["protocol_id"],
            "protocol_sha256": p["protocol_sha256"],
            "reviewer_model": p["reviewer_model"],
            "target_boundaries_sha256": canonical_json_sha256(targets),
            "source_identity_count": 25,
            "paper_review_records": review_records,
            "final_adjudication_record_sha256": final_payload["record_sha256"],
            "paper_review_calls": 25,
            "final_adjudication_calls": 1,
            "scientific_llm_call_attempts": llm_attempts,
            "scientific_network_call_attempts": network_attempts,
            "all_25_papers_processed": True,
            "fresh_c_scientific_text_read_performed": True,
            "scientific_adjudication_performed": True,
            "fresh_reserve_c_already_consumed": True,
            "this_is_new_reserve_c_consumption": False,
            "external_literature_used": False,
            "count_threshold_used": False,
            "hypothesis_rewrite_performed": False,
            "hypothesis_upgrade_performed": False,
            "h2_resurrected": False,
            "same_epoch_rerun_allowed": False,
            "automatic_post_c1b2_transition_allowed": False,
            "stop": True,
        }
        run_manifest["run_sha256"] = canonical_json_sha256(run_manifest)
        run_manifest["run_id"] = (
            "sers_fresh_c_c1b2_scientific_adjudication_run_v1:"
            + run_manifest["run_sha256"][:20]
        )
        atomic_json(run_dir / "run_manifest.json", run_manifest)
        atomic_json(run_dir / "C1B2_SCIENTIFIC_ADJUDICATION_COMPLETE.json", {
            "run_id": run_manifest["run_id"],
            "run_sha256": run_manifest["run_sha256"],
            "complete": True,
            "all_25_papers_processed": True,
            "fresh_c_scientific_text_read_performed": True,
            "scientific_adjudication_performed": True,
            "same_epoch_rerun_allowed": False,
            "automatic_post_c1b2_transition_allowed": False,
            "stop": True,
        })

        print("Fresh-C C1B.2 one-shot scientific adjudication complete")
        print(f"Run ID: {run_manifest['run_id']}")
        print(f"Run SHA256: {run_manifest['run_sha256']}")
        print("Paper reviews completed: 25/25")
        print("Final adjudication completed: True")
        print(f"Scientific LLM calls: {llm_attempts}")
        print(f"Scientific network calls: {network_attempts}")
        print("Fresh-C scientific text read performed: True")
        print("Scientific adjudication performed: True")
        print("Same-epoch rerun allowed: False")
        print("Automatic post-C1B.2 transition: False")
        print("STOP: True")
        return 0

    except Exception as exc:
        atomic_json(fail_marker, {
            "schema_version": "sers-fresh-c-c1b2-scientific-adjudication-failed-v1",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_id": p["protocol_id"],
            "protocol_sha256": p["protocol_sha256"],
            "error_type": type(exc).__name__,
            "error_summary": str(exc)[:1000],
            "completed_reserve_indexes": completed_indices,
            "completed_paper_reviews": len(completed_indices),
            "scientific_llm_call_attempts": llm_attempts,
            "scientific_network_call_attempts": network_attempts,
            "fresh_c_scientific_text_read_performed": True,
            "scientific_adjudication_complete": False,
            "fresh_reserve_c_already_consumed": True,
            "same_epoch_rerun_allowed": False,
            "failure_restores_freshness": False,
            "failure_authorizes_tuning_on_fresh_c": False,
            "automatic_post_c1b2_transition_allowed": False,
            "stop": True,
        })
        print("Fresh-C C1B.2 one-shot scientific adjudication: FAILED")
        print(f"Error type: {type(exc).__name__}")
        print(f"Completed paper reviews: {len(completed_indices)}/25")
        print(f"Scientific LLM call attempts: {llm_attempts}")
        print(f"Scientific network call attempts: {network_attempts}")
        print("Fresh-C scientific text read performed: True")
        print("Same-epoch rerun allowed: False")
        print("Failure restores freshness: False")
        print("Failure authorizes tuning on Fresh-C: False")
        print("STOP: True")
        raise

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-schema-qualification", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument(
        "--confirm-one-shot-scientific-adjudication",
        action="store_true",
    )
    args = parser.parse_args()

    if args.synthetic_schema_qualification:
        return synthetic_schema_qualification()
    if args.preflight:
        return preflight()
    return live_execute()

if __name__ == "__main__":
    raise SystemExit(main())
