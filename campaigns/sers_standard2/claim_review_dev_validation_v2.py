from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from dac_her.domains import get_domain_profile
from dac_her.external_novelty_contracts import (
    ClaimPriorArtReview,
    ClaimPriorArtReviewDraft,
    ExternalNoveltyPolicy,
)
from dac_her.external_novelty_llm import (
    InstructorOpenAICompatibleExternalNoveltyBackend,
    _REVIEW_SYSTEM,
)
from dac_her.prior_art_review_audit import prior_art_review_audit_scope
from campaigns.sers_standard2.claim_review_dev_validation import (
    CANONICAL_ROOT,
    RANKER_RUN_ROOT,
    RANKER_SPEC_ROOT,
    EXPECTED_CLAIM_COUNT,
    EXPECTED_CORE_CLAIM_COUNT,
    EXPECTED_TOPN,
    atomic_json,
    atomic_text,
    candidate_set_from_ranker_row,
    canonical_json,
    claim_map_from_plan,
    compile_drafts,
    load_inputs,
    read_json,
    read_jsonl,
    render_human_audit,
    reviewer_input_from_candidates,
    scan_output_for_secrets,
    sha256_file,
    sha256_json,
    structural_checks,
)


SEMANTICS_ID = "sers_standard2_claim_review_relation_nucleus_v2"
DOMAIN_PROFILE_ID = "sers_au_ag"

PARENT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "claim_review_only_dev_spec_v1"
)
PARENT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "claim_review_only_dev_run_v1"
)

DEFAULT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "claim_review_only_dev_spec_v2"
)
DEFAULT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "claim_review_only_dev_run_v2"
)

EXPECTED_PARENT_SPEC_ID = (
    "sers_standard2_claim_review_only_dev_spec:"
    "74e0cabc2ba8940d115a"
)
EXPECTED_PARENT_RUN_ID = (
    "sers_standard2_claim_review_only_dev_run:"
    "80cd3649574b3769b73c"
)

SOURCE_FILES_TO_FREEZE = (
    Path("dac_her/external_novelty_llm.py"),
    Path("dac_her/prior_art_matching.py"),
    Path("dac_her/external_novelty_contracts.py"),
    Path("dac_her/prior_art_review_audit.py"),
    Path("dac_her/llm_telemetry.py"),
    Path("dac_her/domain_profile.py"),
    Path("dac_her/domains/registry.py"),
    Path('domains/sers/profile.py'),
)

REQUIRED_PROMPT_SENTINELS = (
    "RELATION-NUCLEUS RULES:",
    "A thematically neighboring relation is not, by itself, PARTIAL_PRIOR_ART.",
    "For context_condition claims, PARTIAL_PRIOR_ART requires a comparison across contexts",
    "For distinctive_prediction claims, PARTIAL_PRIOR_ART requires the same dependent relation or contrast",
    "For mediator claims, PARTIAL_PRIOR_ART requires evidence linking the proposed mediator",
    "For moderator_interaction or descriptor_interaction claims, PARTIAL_PRIOR_ART requires an interaction",
)


def source_hashes(repo_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for rel in SOURCE_FILES_TO_FREEZE:
        path = repo_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(rel)] = sha256_file(path)
    return result


def validate_hardened_prompt() -> None:
    for sentinel in REQUIRED_PROMPT_SENTINELS:
        if sentinel not in _REVIEW_SYSTEM:
            raise ValueError(
                "Relation-nucleus reviewer prompt sentinel missing: "
                + sentinel
            )
    forbidden = (
        "PARTIAL_PRIOR_ART: an ABSTRACT-BACKED record establishes "
        "a substantial subset or a closely neighboring relation"
    )
    if forbidden in _REVIEW_SYSTEM:
        raise ValueError(
            "Legacy broad PARTIAL_PRIOR_ART definition still present."
        )


def load_parent_lineage(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_spec_path = (
        repo_root / PARENT_SPEC_ROOT / "claim_review_spec.json"
    )
    parent_report_path = (
        repo_root / PARENT_RUN_ROOT / "claim_review_report.json"
    )
    parent_marker_path = (
        repo_root / PARENT_RUN_ROOT / "STRUCTURAL_PASS.json"
    )
    for path in (
        parent_spec_path,
        parent_report_path,
        parent_marker_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    parent_spec = read_json(parent_spec_path)
    parent_report = read_json(parent_report_path)
    parent_marker = read_json(parent_marker_path)

    if parent_spec.get("spec_id") != EXPECTED_PARENT_SPEC_ID:
        raise ValueError("Unexpected parent claim-review spec ID.")
    if parent_report.get("run_id") != EXPECTED_PARENT_RUN_ID:
        raise ValueError("Unexpected parent claim-review run ID.")
    if parent_report.get("structural_outcome") != (
        "CLAIM_REVIEW_STRUCTURAL_DEV_PASS"
    ):
        raise ValueError("Parent claim-review run is not structural PASS.")
    if parent_report.get("scientific_relationship_outcome") != (
        "MANUAL_REVIEW_REQUIRED"
    ):
        raise ValueError(
            "Parent scientific relationship outcome drift."
        )
    if parent_marker.get("status") != "structural_pass":
        raise ValueError("Parent STRUCTURAL_PASS marker mismatch.")
    if parent_marker.get("run_id") != EXPECTED_PARENT_RUN_ID:
        raise ValueError("Parent STRUCTURAL_PASS run ID mismatch.")
    if parent_report.get(
        "hypothesis_level_novelty_status_computed"
    ) is not False:
        raise ValueError(
            "Parent unexpectedly computed hypothesis novelty."
        )
    if parent_report.get("fresh_reserve_consumed") is not False:
        raise ValueError("Parent unexpectedly consumed Fresh Reserve.")

    return parent_spec, parent_report


def _resolved_base_url(explicit: str | None) -> str | None:
    value = explicit or os.getenv("OPENAI_BASE_URL") or None
    if value is None:
        return None
    return str(value).strip() or None


def build_spec(
    *,
    repo_root: Path,
    model: str,
    api_key_env: str,
    base_url: str | None,
    instructor_mode: str,
    temperature: float,
    parse_retries: int,
    timeout: float,
    max_abstract_chars: int,
) -> dict[str, Any]:
    validate_hardened_prompt()
    parent_spec, parent_report = load_parent_lineage(repo_root)
    plan, packet, ranker_spec, ranker_report = load_inputs(repo_root)

    model = str(model).strip()
    api_key_env = str(api_key_env).strip()
    if not model:
        raise ValueError("--model must not be blank.")
    if not api_key_env:
        raise ValueError("--api-key-env must not be blank.")
    if not os.getenv(api_key_env):
        raise RuntimeError(
            f"No API key configured in environment variable "
            f"{api_key_env!r}. Secret value is not persisted."
        )
    if importlib.util.find_spec("openai") is None:
        raise RuntimeError("Python package 'openai' is not installed.")
    if importlib.util.find_spec("instructor") is None:
        raise RuntimeError(
            "Python package 'instructor' is not installed."
        )

    profile = get_domain_profile(DOMAIN_PROFILE_ID)
    if profile.profile_id != DOMAIN_PROFILE_ID:
        raise ValueError("SERS domain-profile resolution mismatch.")

    policy = ExternalNoveltyPolicy()
    if policy.max_ranked_works_per_claim != EXPECTED_TOPN:
        raise ValueError("Production policy top-N drift.")

    claim_rows = []
    for row in ranker_report["claim_reports"]:
        claim_rows.append(
            {
                "hypothesis_id": row["hypothesis_id"],
                "claim_id": row["claim_id"],
                "claim_rank": row["claim_rank"],
                "importance": row["importance"],
                "kind": row["kind"],
                "claim_text_sha256": hashlib.sha256(
                    row["claim_text"].encode("utf-8")
                ).hexdigest(),
                "ranked_work_ids": [
                    work["work_id"]
                    for work in row["top_ranked_works"]
                ],
                "ranked_input_sha256": sha256_json(
                    row["top_ranked_works"]
                ),
            }
        )

    review_prompt_sha256 = hashlib.sha256(
        _REVIEW_SYSTEM.encode("utf-8")
    ).hexdigest()

    body: dict[str, Any] = {
        "schema_version":
            "sers-standard2-claim-review-only-dev-spec-v2",
        "semantics_id": SEMANTICS_ID,
        "parent_v1_spec_id": parent_spec["spec_id"],
        "parent_v1_spec_sha256": parent_spec["spec_sha256"],
        "parent_v1_run_id": parent_report["run_id"],
        "parent_v1_run_sha256": parent_report["run_sha256"],
        "source_ranker_spec_id": ranker_spec["spec_id"],
        "source_ranker_spec_sha256": ranker_spec["spec_sha256"],
        "source_ranker_spec_file_sha256": sha256_file(
            repo_root / RANKER_SPEC_ROOT / "ranker_spec.json"
        ),
        "source_ranker_run_id": ranker_report["run_id"],
        "source_ranker_run_sha256": ranker_report["run_sha256"],
        "source_ranker_report_file_sha256": sha256_file(
            repo_root / RANKER_RUN_ROOT / "ranker_report.json"
        ),
        "source_query_plan_id": plan.plan_id,
        "source_query_plan_sha256": plan.plan_sha256,
        "source_query_plan_file_sha256": sha256_file(
            repo_root / RANKER_SPEC_ROOT / "frozen_query_plan.json"
        ),
        "source_canonical_packet_id": packet.packet_id,
        "source_canonical_packet_sha256": packet.packet_sha256,
        "source_canonical_packet_file_sha256": sha256_file(
            repo_root / CANONICAL_ROOT / "canonical_prior_art_v2.json"
        ),
        "canonical_work_count": len(packet.works),
        "claim_count": len(claim_rows),
        "core_claim_count": sum(
            row["importance"] == "core"
            for row in claim_rows
        ),
        "claims": claim_rows,
        "review_backend": {
            "class":
                "dac_her.external_novelty_llm."
                "InstructorOpenAICompatibleExternalNoveltyBackend",
            "model": model,
            "api_key_env": api_key_env,
            "api_key_configured": True,
            "base_url": _resolved_base_url(base_url),
            "instructor_mode": str(instructor_mode).upper(),
            "temperature": float(temperature),
            "parse_retries": int(parse_retries),
            "timeout_seconds": float(timeout),
            "max_abstract_chars": int(max_abstract_chars),
            "capture_prompts": True,
            "review_prompt_sha256": review_prompt_sha256,
            "relation_nucleus_hardening": True,
        },
        "compiler": {
            "class":
                "dac_her.prior_art_matching.ClaimPriorArtCompiler",
            "domain_profile_id": profile.profile_id,
            "policy": policy.model_dump(mode="json"),
            "changed_from_parent_v1": False,
        },
        "controlled_change": {
            "only_intended_scientific_semantic_change":
                "reviewer relation-nucleus classification contract",
            "ranker_changed": False,
            "canonical_packet_changed": False,
            "claim_set_changed": False,
            "topn_changed": False,
            "compiler_thresholds_changed": False,
            "hypothesis_novelty_logic_changed": False,
        },
        "source_hashes": source_hashes(repo_root),
        "validation_policy": {
            "reuse_frozen_ranker_topn_without_reranking": True,
            "require_all_12_logical_review_calls": True,
            "require_all_reviewer_work_ids_within_frozen_topn": True,
            "require_compiler_unknown_work_ids_empty": True,
            "require_strong_compiled_matches_abstract_backed": True,
            "require_compiled_review_claim_identity_exact": True,
            "hypothesis_level_novelty_status_forbidden": True,
            "automatic_next_stage_authorization_forbidden": True,
            "case_specific_expected_statuses_forbidden": True,
        },
        "epistemic_policy": {
            "bounded_title_abstract_evidence_only": True,
            "outside_knowledge_forbidden": True,
            "literature_wide_novelty_claim_forbidden": True,
            "no_direct_match_means_bounded_ranked_evidence_only": True,
            "external_prior_art_not_positive_premise": True,
            "scientific_relationship_outcome":
                "MANUAL_REVIEW_REQUIRED",
            "fresh_reserve_consumed": False,
        },
        "network_calls_during_spec_freeze": 0,
        "llm_calls_during_spec_freeze": 0,
    }
    body["spec_sha256"] = sha256_json(body)
    body["spec_id"] = (
        "sers_standard2_claim_review_only_dev_spec_v2:"
        + body["spec_sha256"][:20]
    )
    return body


def verify_spec(
    *,
    repo_root: Path,
    spec_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    if not spec_path.is_file():
        return ["spec missing"], {}

    stored = read_json(spec_path)
    issues: list[str] = []

    body = dict(stored)
    spec_id = body.pop("spec_id", None)
    spec_sha = body.pop("spec_sha256", None)
    observed = sha256_json(body)
    if spec_sha != observed:
        issues.append("spec SHA mismatch")
    if spec_id != (
        "sers_standard2_claim_review_only_dev_spec_v2:"
        + observed[:20]
    ):
        issues.append("spec ID mismatch")

    try:
        validate_hardened_prompt()
        parent_spec, parent_report = load_parent_lineage(repo_root)
        plan, packet, ranker_spec, ranker_report = load_inputs(repo_root)

        if stored.get("parent_v1_spec_id") != parent_spec["spec_id"]:
            issues.append("parent v1 spec ID drift")
        if stored.get("parent_v1_run_id") != parent_report["run_id"]:
            issues.append("parent v1 run ID drift")
        if stored.get("source_ranker_spec_id") != ranker_spec["spec_id"]:
            issues.append("frozen ranker spec ID drift")
        if stored.get("source_ranker_run_id") != ranker_report["run_id"]:
            issues.append("frozen ranker run ID drift")
        if stored.get("source_query_plan_id") != plan.plan_id:
            issues.append("frozen query plan ID drift")
        if stored.get("source_canonical_packet_id") != packet.packet_id:
            issues.append("frozen canonical packet ID drift")
        if stored.get("source_hashes") != source_hashes(repo_root):
            issues.append("reviewer/compiler source hash drift")

        prompt_sha = hashlib.sha256(
            _REVIEW_SYSTEM.encode("utf-8")
        ).hexdigest()
        if stored.get("review_backend", {}).get(
            "review_prompt_sha256"
        ) != prompt_sha:
            issues.append("review prompt SHA drift")

        controlled = stored.get("controlled_change", {})
        expected_false = (
            "ranker_changed",
            "canonical_packet_changed",
            "claim_set_changed",
            "topn_changed",
            "compiler_thresholds_changed",
            "hypothesis_novelty_logic_changed",
        )
        for key in expected_false:
            if controlled.get(key) is not False:
                issues.append(f"controlled-change violation: {key}")
    except Exception as exc:
        issues.append(
            f"input/source verification failed: "
            f"{type(exc).__name__}: {exc}"
        )

    backend = stored.get("review_backend", {})
    api_key_env = str(backend.get("api_key_env") or "")
    if not api_key_env or not os.getenv(api_key_env):
        issues.append(
            "configured API key environment variable is unavailable"
        )

    return sorted(set(issues)), stored


def create_backend(
    *,
    spec: Mapping[str, Any],
    telemetry_path: Path,
) -> InstructorOpenAICompatibleExternalNoveltyBackend:
    cfg = spec["review_backend"]
    return InstructorOpenAICompatibleExternalNoveltyBackend(
        model=str(cfg["model"]),
        api_key_env=str(cfg["api_key_env"]),
        base_url=cfg.get("base_url"),
        instructor_mode=str(cfg["instructor_mode"]),
        temperature=float(cfg["temperature"]),
        parse_retries=int(cfg["parse_retries"]),
        timeout=float(cfg["timeout_seconds"]),
        capture_prompts=True,
        max_abstract_chars=int(cfg["max_abstract_chars"]),
        telemetry_path=str(telemetry_path),
        telemetry_context={
            "pipeline": "sers_claim_review_only_dev_v2",
            "semantics_id": SEMANTICS_ID,
            "source_ranker_run_id": spec["source_ranker_run_id"],
            "source_claim_review_spec_id": spec["spec_id"],
            "parent_v1_run_id": spec["parent_v1_run_id"],
        },
    )


def report_from_results(
    *,
    spec: Mapping[str, Any],
    reviews: list[ClaimPriorArtReview],
    drafts: Mapping[str, ClaimPriorArtReviewDraft],
    checks: Mapping[str, bool],
    diagnostics: Mapping[str, Any],
    prompt_records: list[Any],
    audit_rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]],
    secret_scan_pass: bool,
) -> dict[str, Any]:
    structural_pass = all(checks.values()) and secret_scan_pass

    body: dict[str, Any] = {
        "schema_version":
            "sers-standard2-claim-review-only-dev-run-v2",
        "semantics_id": SEMANTICS_ID,
        "source_spec_id": spec["spec_id"],
        "source_spec_sha256": spec["spec_sha256"],
        "parent_v1_run_id": spec["parent_v1_run_id"],
        "parent_v1_run_sha256": spec["parent_v1_run_sha256"],
        "source_ranker_run_id": spec["source_ranker_run_id"],
        "source_ranker_run_sha256": spec["source_ranker_run_sha256"],
        "source_canonical_packet_id":
            spec["source_canonical_packet_id"],
        "structural_outcome": (
            "CLAIM_REVIEW_V2_STRUCTURAL_DEV_PASS"
            if structural_pass
            else "CLAIM_REVIEW_V2_STRUCTURAL_DEV_FAIL"
        ),
        "scientific_relationship_outcome":
            "MANUAL_REVIEW_REQUIRED",
        "checks": dict(checks),
        "secret_scan_pass": bool(secret_scan_pass),
        "diagnostics": dict(diagnostics),
        "claim_reviews": [
            review.model_dump(mode="json")
            for review in reviews
        ],
        "raw_review_drafts": {
            claim_id: draft.model_dump(mode="json")
            for claim_id, draft in sorted(drafts.items())
        },
        "prompt_manifest": [
            {
                "name": str(record.name),
                "prompt_sha256": str(record.prompt_sha256),
            }
            for record in prompt_records
        ],
        "logical_review_calls": len(prompt_records),
        "successful_prior_art_audit_rows": len(audit_rows),
        "telemetry_row_count": len(telemetry_rows),
        "network_scope":
            "LLM_PROVIDER_ONLY_NO_LITERATURE_RETRIEVAL",
        "literature_network_calls": 0,
        "ranker_recomputed": False,
        "canonicalization_recomputed": False,
        "claim_decomposition_recomputed": False,
        "compiler_changed_from_v1": False,
        "case_specific_expected_statuses_used": False,
        "hypothesis_level_novelty_status_computed": False,
        "automatic_next_stage_authorized": False,
        "fresh_reserve_consumed": False,
    }
    body["run_sha256"] = sha256_json(body)
    body["run_id"] = (
        "sers_standard2_claim_review_only_dev_run_v2:"
        + body["run_sha256"][:20]
    )
    return body


def offline_recompile_from_report(
    *,
    repo_root: Path,
    spec: Mapping[str, Any],
    stored_report: Mapping[str, Any],
) -> tuple[list[ClaimPriorArtReview], dict[str, ClaimPriorArtReviewDraft]]:
    plan, packet, _ranker_spec, ranker_report = load_inputs(repo_root)
    raw = stored_report.get("raw_review_drafts")
    if not isinstance(raw, dict):
        raise ValueError("Stored report lacks raw_review_drafts.")
    drafts = {
        str(claim_id): ClaimPriorArtReviewDraft.model_validate(value)
        for claim_id, value in raw.items()
    }
    reviews = compile_drafts(
        spec=spec,
        plan=plan,
        packet=packet,
        ranker_report=ranker_report,
        drafts=drafts,
    )
    return reviews, drafts
