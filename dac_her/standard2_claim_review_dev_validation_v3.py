from __future__ import annotations

import hashlib
import importlib.util
import os
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
from dac_her.standard2_claim_review_dev_validation import (
    CANONICAL_ROOT,
    RANKER_RUN_ROOT,
    RANKER_SPEC_ROOT,
    EXPECTED_CLAIM_COUNT,
    EXPECTED_CORE_CLAIM_COUNT,
    EXPECTED_TOPN,
    atomic_json,
    atomic_text,
    canonical_json,
    compile_drafts,
    load_inputs,
    read_json,
    reviewer_input_from_candidates,
    sha256_file,
    sha256_json,
)
from dac_her.standard2_claim_review_dev_validation_v2 import (
    source_hashes,
)


SEMANTICS_ID = "sers_standard2_claim_review_relation_nucleus_work_id_v3"
DOMAIN_PROFILE_ID = "sers_au_ag"

PARENT_V1_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "claim_review_only_dev_spec_v1"
)
PARENT_V2_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "claim_review_only_dev_spec_v2"
)
PARENT_V2_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "claim_review_only_dev_run_v2"
)

DEFAULT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "claim_review_only_dev_spec_v3"
)
DEFAULT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "claim_review_only_dev_run_v3"
)

EXPECTED_PARENT_V2_SPEC_ID = (
    "sers_standard2_claim_review_only_dev_spec_v2:"
    "704d2bb8feab605833aa"
)
EXPECTED_PARENT_V2_RUN_ID = (
    "sers_standard2_claim_review_only_dev_run_v2:"
    "6c72819158ab463d28d4"
)

REQUIRED_PROMPT_SENTINELS = (
    "RELATION-NUCLEUS RULES:",
    "WORK-ID COPY CONTRACT:",
    "Every returned work_id MUST be copied byte-for-byte",
    "Return at most one match per allowed work_id.",
    "If you cannot copy the exact supplied ID for a record, OMIT that record.",
)


def validate_hardened_prompt() -> None:
    for sentinel in REQUIRED_PROMPT_SENTINELS:
        if sentinel not in _REVIEW_SYSTEM:
            raise ValueError(
                "v3 reviewer prompt sentinel missing: " + sentinel
            )


def load_parent_v2_failure(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_path = (
        repo_root / PARENT_V2_SPEC_ROOT / "claim_review_spec_v2.json"
    )
    report_path = (
        repo_root / PARENT_V2_RUN_ROOT / "claim_review_report_v2.json"
    )
    fail_path = (
        repo_root / PARENT_V2_RUN_ROOT / "STRUCTURAL_FAIL.json"
    )
    for path in (spec_path, report_path, fail_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    spec = read_json(spec_path)
    report = read_json(report_path)
    fail = read_json(fail_path)

    if spec.get("spec_id") != EXPECTED_PARENT_V2_SPEC_ID:
        raise ValueError("Unexpected parent v2 spec ID.")
    if report.get("run_id") != EXPECTED_PARENT_V2_RUN_ID:
        raise ValueError("Unexpected parent v2 run ID.")
    if report.get("structural_outcome") != (
        "CLAIM_REVIEW_V2_STRUCTURAL_DEV_FAIL"
    ):
        raise ValueError("Parent v2 is not the expected structural FAIL.")
    if fail.get("status") != "structural_fail":
        raise ValueError("Parent v2 STRUCTURAL_FAIL marker mismatch.")
    if fail.get("run_id") != EXPECTED_PARENT_V2_RUN_ID:
        raise ValueError("Parent v2 fail marker run ID mismatch.")

    checks = report.get("checks", {})
    if checks.get("reviewer_work_ids_within_frozen_topn") is not False:
        raise ValueError(
            "Parent v2 did not fail reviewer-work-ID containment."
        )
    if checks.get("compiler_unknown_work_ids_empty") is not False:
        raise ValueError(
            "Parent v2 did not record compiler unknown work IDs."
        )
    if report.get("hypothesis_level_novelty_status_computed") is not False:
        raise ValueError(
            "Parent v2 unexpectedly computed hypothesis novelty."
        )
    if report.get("fresh_reserve_consumed") is not False:
        raise ValueError("Parent v2 unexpectedly consumed Fresh Reserve.")
    return spec, report


def _resolved_base_url(explicit: str | None) -> str | None:
    value = explicit or os.getenv("OPENAI_BASE_URL") or None
    if value is None:
        return None
    return str(value).strip() or None


def validate_draft_work_ids(
    *,
    claim_id: str,
    draft: ClaimPriorArtReviewDraft,
    allowed_work_ids: list[str],
) -> None:
    allowed = set(allowed_work_ids)
    returned = [match.work_id for match in draft.matches]
    unknown = sorted({work_id for work_id in returned if work_id not in allowed})
    duplicates = sorted(
        {
            work_id
            for work_id in returned
            if returned.count(work_id) > 1
        }
    )
    if unknown or duplicates or len(returned) > len(allowed_work_ids):
        parts = [
            f"claim_id={claim_id}",
            f"returned={len(returned)}",
            f"allowed={len(allowed_work_ids)}",
        ]
        if unknown:
            parts.append(f"unknown={unknown}")
        if duplicates:
            parts.append(f"duplicates={duplicates}")
        raise RuntimeError(
            "Reviewer work-ID copy contract violation: "
            + "; ".join(parts)
        )


def prompt_allowed_id_check(
    *,
    prompt_records: list[Any],
    ranker_report: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    expected_by_name = {}
    for row in ranker_report["claim_reports"]:
        expected_by_name[
            "review_" + str(row["claim_id"])
        ] = [
            str(work["work_id"])
            for work in row["top_ranked_works"]
        ]

    issues: list[str] = []
    if len(prompt_records) != len(expected_by_name):
        issues.append(
            f"prompt count {len(prompt_records)} != {len(expected_by_name)}"
        )

    for record in prompt_records:
        name = str(getattr(record, "name", ""))
        user_prompt = str(getattr(record, "user_prompt", ""))
        expected = expected_by_name.get(name)
        if expected is None:
            issues.append(f"unexpected prompt record: {name}")
            continue
        if "ALLOWED_WORK_IDS\n================" not in user_prompt:
            issues.append(f"missing ALLOWED_WORK_IDS block: {name}")
            continue
        for work_id in expected:
            if user_prompt.count(work_id) < 2:
                # One occurrence in the candidate record and one in the
                # explicit ALLOWED_WORK_IDS block.
                issues.append(
                    f"allowed ID not repeated in prompt {name}: {work_id}"
                )
    return not issues, issues


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
    parent_v2_spec, parent_v2_report = load_parent_v2_failure(repo_root)
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
        raise RuntimeError("Python package 'instructor' is not installed.")

    profile = get_domain_profile(DOMAIN_PROFILE_ID)
    if profile.profile_id != DOMAIN_PROFILE_ID:
        raise ValueError("SERS domain-profile resolution mismatch.")

    policy = ExternalNoveltyPolicy()
    if policy.max_ranked_works_per_claim != EXPECTED_TOPN:
        raise ValueError("Production policy top-N drift.")

    claims = []
    for row in ranker_report["claim_reports"]:
        claims.append(
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

    body: dict[str, Any] = {
        "schema_version":
            "sers-standard2-claim-review-only-dev-spec-v3",
        "semantics_id": SEMANTICS_ID,
        "parent_v2_spec_id": parent_v2_spec["spec_id"],
        "parent_v2_spec_sha256": parent_v2_spec["spec_sha256"],
        "parent_v2_failed_run_id": parent_v2_report["run_id"],
        "parent_v2_failed_run_sha256": parent_v2_report["run_sha256"],
        "source_ranker_spec_id": ranker_spec["spec_id"],
        "source_ranker_spec_sha256": ranker_spec["spec_sha256"],
        "source_ranker_run_id": ranker_report["run_id"],
        "source_ranker_run_sha256": ranker_report["run_sha256"],
        "source_query_plan_id": plan.plan_id,
        "source_query_plan_sha256": plan.plan_sha256,
        "source_canonical_packet_id": packet.packet_id,
        "source_canonical_packet_sha256": packet.packet_sha256,
        "canonical_work_count": len(packet.works),
        "claim_count": len(claims),
        "core_claim_count": sum(
            row["importance"] == "core" for row in claims
        ),
        "claims": claims,
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
            "review_prompt_sha256": hashlib.sha256(
                _REVIEW_SYSTEM.encode("utf-8")
            ).hexdigest(),
            "relation_nucleus_hardening": True,
            "work_id_copy_contract_hardening": True,
        },
        "compiler": {
            "class":
                "dac_her.prior_art_matching.ClaimPriorArtCompiler",
            "domain_profile_id": profile.profile_id,
            "policy": policy.model_dump(mode="json"),
            "changed_from_v2": False,
        },
        "controlled_change": {
            "only_intended_change":
                "reviewer exact work-ID copy/provenance contract",
            "relation_nucleus_semantics_changed_from_v2": False,
            "ranker_changed": False,
            "canonical_packet_changed": False,
            "claim_set_changed": False,
            "topn_changed": False,
            "compiler_thresholds_changed": False,
            "hypothesis_novelty_logic_changed": False,
            "case_specific_expected_statuses_used": False,
            "invalid_id_guess_mapping_used": False,
        },
        "source_hashes": source_hashes(repo_root),
        "validation_policy": {
            "reuse_frozen_ranker_topn_without_reranking": True,
            "require_all_12_logical_review_calls": True,
            "require_all_reviewer_work_ids_within_frozen_topn": True,
            "require_unique_reviewer_work_ids_per_claim": True,
            "require_reviewer_match_count_not_exceed_candidates": True,
            "require_allowed_work_id_block_in_every_prompt": True,
            "require_compiler_unknown_work_ids_empty": True,
            "require_strong_compiled_matches_abstract_backed": True,
            "hypothesis_level_novelty_status_forbidden": True,
            "automatic_next_stage_authorization_forbidden": True,
        },
        "epistemic_policy": {
            "bounded_title_abstract_evidence_only": True,
            "outside_knowledge_forbidden": True,
            "literature_wide_novelty_claim_forbidden": True,
            "invalid_work_id_guess_mapping_forbidden": True,
            "no_direct_match_means_bounded_ranked_evidence_only": True,
            "scientific_relationship_outcome":
                "MANUAL_REVIEW_REQUIRED",
            "fresh_reserve_consumed": False,
        },
        "network_calls_during_spec_freeze": 0,
        "llm_calls_during_spec_freeze": 0,
    }
    body["spec_sha256"] = sha256_json(body)
    body["spec_id"] = (
        "sers_standard2_claim_review_only_dev_spec_v3:"
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
        "sers_standard2_claim_review_only_dev_spec_v3:"
        + observed[:20]
    ):
        issues.append("spec ID mismatch")

    try:
        validate_hardened_prompt()
        parent_v2_spec, parent_v2_report = load_parent_v2_failure(
            repo_root
        )
        plan, packet, ranker_spec, ranker_report = load_inputs(repo_root)
        if stored.get("parent_v2_spec_id") != parent_v2_spec["spec_id"]:
            issues.append("parent v2 spec ID drift")
        if stored.get("parent_v2_failed_run_id") != parent_v2_report["run_id"]:
            issues.append("parent v2 failed run ID drift")
        if stored.get("source_ranker_spec_id") != ranker_spec["spec_id"]:
            issues.append("ranker spec ID drift")
        if stored.get("source_ranker_run_id") != ranker_report["run_id"]:
            issues.append("ranker run ID drift")
        if stored.get("source_query_plan_id") != plan.plan_id:
            issues.append("query plan ID drift")
        if stored.get("source_canonical_packet_id") != packet.packet_id:
            issues.append("canonical packet ID drift")
        if stored.get("source_hashes") != source_hashes(repo_root):
            issues.append("source hash drift")
        prompt_sha = hashlib.sha256(
            _REVIEW_SYSTEM.encode("utf-8")
        ).hexdigest()
        if stored.get("review_backend", {}).get(
            "review_prompt_sha256"
        ) != prompt_sha:
            issues.append("review prompt SHA drift")
    except Exception as exc:
        issues.append(
            f"input/source verification failed: "
            f"{type(exc).__name__}: {exc}"
        )

    api_key_env = str(
        stored.get("review_backend", {}).get("api_key_env") or ""
    )
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
            "pipeline": "sers_claim_review_only_dev_v3",
            "semantics_id": SEMANTICS_ID,
            "source_ranker_run_id": spec["source_ranker_run_id"],
            "claim_review_spec_id": spec["spec_id"],
            "parent_v2_failed_run_id":
                spec["parent_v2_failed_run_id"],
        },
    )


def offline_recompile_from_report(
    *,
    repo_root: Path,
    spec: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[list[ClaimPriorArtReview], dict[str, ClaimPriorArtReviewDraft]]:
    plan, packet, _ranker_spec, ranker_report = load_inputs(repo_root)
    raw = report.get("raw_review_drafts")
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
