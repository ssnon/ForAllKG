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
    ClaimPriorArtCandidateSet,
    ClaimPriorArtReview,
    ClaimPriorArtReviewDraft,
    ExternalNoveltyPolicy,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtPacket,
    RankedPriorArtWork,
)
from dac_her.external_novelty_llm import (
    InstructorOpenAICompatibleExternalNoveltyBackend,
)
from dac_her.prior_art_matching import ClaimPriorArtCompiler
from dac_her.prior_art_review_audit import prior_art_review_audit_scope


SEMANTICS_ID = "sers_standard2_claim_review_only_dev_v1"
DOMAIN_PROFILE_ID = "sers_au_ag"

RANKER_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "ranker_only_dev_spec_v1"
)
RANKER_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "ranker_only_dev_run_v1"
)
CANONICAL_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "canonicalization_only_dev_recheck_v2"
)

DEFAULT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "claim_review_only_dev_spec_v1"
)
DEFAULT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "claim_review_only_dev_run_v1"
)

EXPECTED_RANKER_SPEC_ID = (
    "sers_standard2_ranker_only_dev_spec:"
    "bfc3e58cee5d3169227a"
)
EXPECTED_RANKER_RUN_ID = (
    "sers_standard2_ranker_only_dev_run:"
    "28b2c16ed3c9befb6bc0"
)
EXPECTED_CANONICAL_WORK_COUNT = 430
EXPECTED_CLAIM_COUNT = 12
EXPECTED_CORE_CLAIM_COUNT = 10
EXPECTED_TOPN = 8

SOURCE_FILES_TO_FREEZE = (
    Path("dac_her/external_novelty_llm.py"),
    Path("dac_her/prior_art_matching.py"),
    Path("dac_her/external_novelty_contracts.py"),
    Path("dac_her/prior_art_review_audit.py"),
    Path("dac_her/llm_telemetry.py"),
    Path("dac_her/domain_profile.py"),
    Path("dac_her/domains/registry.py"),
    Path("dac_her/domains/sers_au_ag.py"),
)


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSONL object: {path}")
        rows.append(value)
    return rows


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def source_hashes(repo_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for rel in SOURCE_FILES_TO_FREEZE:
        path = repo_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(rel)] = sha256_file(path)
    return result


def load_inputs(
    repo_root: Path,
) -> tuple[
    LiteratureQueryPlan,
    PriorArtPacket,
    dict[str, Any],
    dict[str, Any],
]:
    spec_path = repo_root / RANKER_SPEC_ROOT / "ranker_spec.json"
    query_path = repo_root / RANKER_SPEC_ROOT / "frozen_query_plan.json"
    ranker_report_path = repo_root / RANKER_RUN_ROOT / "ranker_report.json"
    ranker_marker_path = repo_root / RANKER_RUN_ROOT / "MECHANICAL_PASS.json"
    packet_path = repo_root / CANONICAL_ROOT / "canonical_prior_art_v2.json"

    for path in (
        spec_path,
        query_path,
        ranker_report_path,
        ranker_marker_path,
        packet_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    ranker_spec = read_json(spec_path)
    ranker_report = read_json(ranker_report_path)
    marker = read_json(ranker_marker_path)
    plan = LiteratureQueryPlan.model_validate_json(
        query_path.read_text(encoding="utf-8")
    )
    packet = PriorArtPacket.model_validate_json(
        packet_path.read_text(encoding="utf-8")
    )

    if ranker_spec.get("spec_id") != EXPECTED_RANKER_SPEC_ID:
        raise ValueError("Unexpected frozen ranker spec ID.")
    if ranker_report.get("run_id") != EXPECTED_RANKER_RUN_ID:
        raise ValueError("Unexpected frozen ranker run ID.")
    if ranker_report.get("mechanical_outcome") != "RANKER_MECHANICAL_DEV_PASS":
        raise ValueError("Frozen ranker mechanical outcome is not PASS.")
    if ranker_report.get("scientific_relevance_outcome") != "MANUAL_REVIEW_REQUIRED":
        raise ValueError("Frozen ranker scientific relevance status drift.")
    if marker.get("status") != "mechanical_pass":
        raise ValueError("MECHANICAL_PASS marker missing PASS status.")
    if marker.get("run_id") != EXPECTED_RANKER_RUN_ID:
        raise ValueError("MECHANICAL_PASS ranker run ID mismatch.")
    if marker.get("automatic_claim_level_review_authorized") is not False:
        raise ValueError("Ranker run unexpectedly authorized automatic claim review.")

    summary = ranker_report.get("summary", {})
    if summary.get("canonical_work_count") != EXPECTED_CANONICAL_WORK_COUNT:
        raise ValueError("Frozen ranker canonical work count drift.")
    if summary.get("claim_count") != EXPECTED_CLAIM_COUNT:
        raise ValueError("Frozen ranker claim count drift.")
    if summary.get("core_claim_count") != EXPECTED_CORE_CLAIM_COUNT:
        raise ValueError("Frozen ranker core claim count drift.")
    if summary.get("topn") != EXPECTED_TOPN:
        raise ValueError("Frozen ranker top-N drift.")

    if packet.packet_id != ranker_report.get("source_canonical_packet_id"):
        raise ValueError("Ranker/canonical packet ID mismatch.")
    if packet.packet_sha256 != ranker_report.get("source_canonical_packet_sha256"):
        raise ValueError("Ranker/canonical packet SHA mismatch.")
    if packet.source_query_plan_id != plan.plan_id:
        raise ValueError("Canonical packet/query-plan lineage mismatch.")

    packet_body = packet.model_dump(mode="json")
    packet_sha = packet_body.pop("packet_sha256")
    if sha256_json(packet_body) != packet_sha:
        raise ValueError("Canonical packet internal SHA mismatch.")

    claim_reports = ranker_report.get("claim_reports")
    if not isinstance(claim_reports, list):
        raise ValueError("Frozen ranker report lacks claim_reports.")
    if len(claim_reports) != EXPECTED_CLAIM_COUNT:
        raise ValueError("Frozen ranker report claim row count drift.")

    claim_map = {
        claim.claim_id: claim
        for group in plan.claims
        for claim in group.claims
    }
    if len(claim_map) != EXPECTED_CLAIM_COUNT:
        raise ValueError("Frozen query-plan claim count/uniqueness drift.")

    seen_claim_ids: set[str] = set()
    for row in claim_reports:
        claim_id = str(row.get("claim_id") or "")
        if claim_id not in claim_map:
            raise ValueError(
                f"Ranker report contains unknown claim_id: {claim_id}"
            )
        if claim_id in seen_claim_ids:
            raise ValueError(
                f"Ranker report contains duplicate claim_id: {claim_id}"
            )
        seen_claim_ids.add(claim_id)
        ranked = row.get("top_ranked_works")
        if not isinstance(ranked, list):
            raise ValueError(
                f"Ranker row lacks top_ranked_works: {claim_id}"
            )
        if len(ranked) > EXPECTED_TOPN:
            raise ValueError(
                f"Ranker row exceeds frozen top-N: {claim_id}"
            )
        ids = [str(work.get("work_id") or "") for work in ranked]
        if any(not work_id for work_id in ids):
            raise ValueError(f"Blank ranked work ID: {claim_id}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate ranked work ID: {claim_id}")

    return plan, packet, ranker_spec, ranker_report


def _resolved_base_url(explicit: str | None) -> str | None:
    value = explicit or os.getenv("OPENAI_BASE_URL") or None
    return str(value).strip() or None if value is not None else None


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
    plan, packet, ranker_spec, ranker_report = load_inputs(repo_root)

    model = str(model).strip()
    if not model:
        raise ValueError("--model must not be blank.")
    api_key_env = str(api_key_env).strip()
    if not api_key_env:
        raise ValueError("--api-key-env must not be blank.")
    if not os.getenv(api_key_env):
        raise RuntimeError(
            f"No API key configured in environment variable {api_key_env!r}. "
            "The secret value is not persisted."
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

    body: dict[str, Any] = {
        "schema_version":
            "sers-standard2-claim-review-only-dev-spec-v1",
        "semantics_id": SEMANTICS_ID,
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
        },
        "compiler": {
            "class":
                "dac_her.prior_art_matching.ClaimPriorArtCompiler",
            "domain_profile_id": profile.profile_id,
            "policy": policy.model_dump(mode="json"),
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
        "sers_standard2_claim_review_only_dev_spec:"
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
        "sers_standard2_claim_review_only_dev_spec:"
        + observed[:20]
    ):
        issues.append("spec ID mismatch")

    try:
        plan, packet, ranker_spec, ranker_report = load_inputs(repo_root)
        if stored.get("source_ranker_spec_id") != ranker_spec.get("spec_id"):
            issues.append("frozen ranker spec ID drift")
        if stored.get("source_ranker_run_id") != ranker_report.get("run_id"):
            issues.append("frozen ranker run ID drift")
        if stored.get("source_query_plan_id") != plan.plan_id:
            issues.append("frozen query plan ID drift")
        if stored.get("source_canonical_packet_id") != packet.packet_id:
            issues.append("frozen canonical packet ID drift")
        if stored.get("source_hashes") != source_hashes(repo_root):
            issues.append("reviewer/compiler source hash drift")
        if stored.get("source_ranker_spec_file_sha256") != sha256_file(
            repo_root / RANKER_SPEC_ROOT / "ranker_spec.json"
        ):
            issues.append("ranker spec file SHA drift")
        if stored.get("source_ranker_report_file_sha256") != sha256_file(
            repo_root / RANKER_RUN_ROOT / "ranker_report.json"
        ):
            issues.append("ranker report file SHA drift")
        if stored.get("source_query_plan_file_sha256") != sha256_file(
            repo_root / RANKER_SPEC_ROOT / "frozen_query_plan.json"
        ):
            issues.append("query plan file SHA drift")
        if stored.get("source_canonical_packet_file_sha256") != sha256_file(
            repo_root / CANONICAL_ROOT / "canonical_prior_art_v2.json"
        ):
            issues.append("canonical packet file SHA drift")
    except Exception as exc:
        issues.append(
            f"input/source verification failed: {type(exc).__name__}: {exc}"
        )

    backend = stored.get("review_backend", {})
    api_key_env = str(backend.get("api_key_env") or "")
    if not api_key_env or not os.getenv(api_key_env):
        issues.append(
            "configured API key environment variable is unavailable"
        )

    return sorted(set(issues)), stored


def claim_map_from_plan(
    plan: LiteratureQueryPlan,
) -> dict[str, NoveltyClaim]:
    return {
        claim.claim_id: claim
        for group in plan.claims
        for claim in group.claims
    }


def candidate_set_from_ranker_row(
    row: Mapping[str, Any],
) -> ClaimPriorArtCandidateSet:
    ranked = [
        RankedPriorArtWork(
            work_id=str(work["work_id"]),
            relevance_score=float(work["relevance_score"]),
            semantic_similarity=float(work["semantic_similarity"]),
            lexical_coverage=float(work["lexical_coverage"]),
            reaction_domain_relevance=float(
                work["reaction_domain_relevance"]
            ),
            catalyst_scope_relevance=float(
                work["catalyst_scope_relevance"]
            ),
            abstract_available=bool(work["abstract_available"]),
        )
        for work in row["top_ranked_works"]
    ]
    return ClaimPriorArtCandidateSet(
        hypothesis_id=str(row["hypothesis_id"]),
        claim_id=str(row["claim_id"]),
        ranked_works=ranked,
    )


def reviewer_input_from_candidates(
    *,
    packet: PriorArtPacket,
    candidates: ClaimPriorArtCandidateSet,
) -> list[dict[str, Any]]:
    work_index = {
        work.work_id: work
        for work in packet.works
    }
    result = []
    for ranked in candidates.ranked_works:
        try:
            work = work_index[ranked.work_id]
        except KeyError as exc:
            raise ValueError(
                f"Ranked work missing from canonical packet: {ranked.work_id}"
            ) from exc
        result.append(
            {
                "work_id": work.work_id,
                "title": work.title,
                "year": work.year,
                "doi": work.doi,
                "abstract": work.abstract,
                "semantic_similarity": ranked.semantic_similarity,
                "lexical_coverage": ranked.lexical_coverage,
                "reaction_domain_relevance":
                    ranked.reaction_domain_relevance,
                "catalyst_scope_relevance":
                    ranked.catalyst_scope_relevance,
                "relevance_score": ranked.relevance_score,
            }
        )
    return result


def make_compiler(spec: Mapping[str, Any]) -> ClaimPriorArtCompiler:
    policy = ExternalNoveltyPolicy.model_validate(
        spec["compiler"]["policy"]
    )
    profile = get_domain_profile(
        str(spec["compiler"]["domain_profile_id"])
    )
    return ClaimPriorArtCompiler(
        min_match_confidence=policy.min_match_confidence,
        direct_match_confidence=policy.direct_match_confidence,
        require_abstract_for_strong_match=(
            policy.require_abstract_for_strong_match
        ),
        require_abstract_for_partial_match=(
            policy.require_abstract_for_partial_match
        ),
        min_reaction_domain_for_conflict=(
            policy.min_reaction_domain_for_conflict
        ),
        min_catalyst_scope_for_conflict=(
            policy.min_catalyst_scope_for_conflict
        ),
        domain_profile=profile,
    )


def compile_drafts(
    *,
    spec: Mapping[str, Any],
    plan: LiteratureQueryPlan,
    packet: PriorArtPacket,
    ranker_report: Mapping[str, Any],
    drafts: Mapping[str, ClaimPriorArtReviewDraft],
) -> list[ClaimPriorArtReview]:
    claims = claim_map_from_plan(plan)
    compiler = make_compiler(spec)
    reviews: list[ClaimPriorArtReview] = []

    for row in ranker_report["claim_reports"]:
        claim_id = str(row["claim_id"])
        claim = claims[claim_id]
        candidates = candidate_set_from_ranker_row(row)
        draft = drafts[claim_id]
        review = compiler.compile(
            claim,
            candidates,
            draft,
            packet,
            plan,
        )
        reviews.append(review)

    return reviews


def structural_checks(
    *,
    spec: Mapping[str, Any],
    plan: LiteratureQueryPlan,
    packet: PriorArtPacket,
    ranker_report: Mapping[str, Any],
    drafts: Mapping[str, ClaimPriorArtReviewDraft],
    reviews: list[ClaimPriorArtReview],
    logical_review_calls: int,
    audit_rows: list[dict[str, Any]],
    prompt_records: list[Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    claims = claim_map_from_plan(plan)
    frozen_rows = {
        str(row["claim_id"]): row
        for row in ranker_report["claim_reports"]
    }
    packet_index = {
        work.work_id: work
        for work in packet.works
    }

    reviewer_ids_within_topn = True
    unknown_draft_ids: dict[str, list[str]] = {}
    for claim_id, draft in drafts.items():
        allowed = {
            str(row["work_id"])
            for row in frozen_rows[claim_id]["top_ranked_works"]
        }
        unknown = sorted(
            {
                match.work_id
                for match in draft.matches
                if match.work_id not in allowed
            }
        )
        if unknown:
            reviewer_ids_within_topn = False
            unknown_draft_ids[claim_id] = unknown

    compiled_identity_exact = all(
        review.claim_id in claims
        and review.hypothesis_id
        == claims[review.claim_id].hypothesis_id
        and review.claim_text
        == claims[review.claim_id].text
        and review.importance
        == claims[review.claim_id].importance
        for review in reviews
    )

    compiler_unknown_empty = all(
        not review.reviewer_unknown_work_ids
        for review in reviews
    )

    strong_relationships = {
        "DIRECT_PRIOR_ART",
        "PARTIAL_PRIOR_ART",
        "CONFLICTING_PRIOR_ART",
    }
    strong_abstract_backed = all(
        packet_index[match.work_id].abstract is not None
        for review in reviews
        for match in review.matches
        if match.relationship in strong_relationships
    )

    topn_identity_preserved = True
    for row in ranker_report["claim_reports"]:
        reconstructed = candidate_set_from_ranker_row(row)
        expected_ids = [
            str(work["work_id"])
            for work in row["top_ranked_works"]
        ]
        observed_ids = [
            work.work_id
            for work in reconstructed.ranked_works
        ]
        if expected_ids != observed_ids:
            topn_identity_preserved = False
            break

    audit_claim_ids = [
        str(row.get("claim_id") or "")
        for row in audit_rows
        if row.get("record_type") == "prior_art_review_call"
    ]
    prompt_names = [
        str(getattr(record, "name", ""))
        for record in prompt_records
    ]

    checks = {
        "frozen_ranker_topn_reused_without_reranking": True,
        "all_12_logical_review_calls_completed":
            logical_review_calls == EXPECTED_CLAIM_COUNT,
        "all_12_successful_audit_rows_present":
            len(audit_claim_ids) == EXPECTED_CLAIM_COUNT,
        "audit_claim_ids_exactly_once":
            sorted(audit_claim_ids) == sorted(claims),
        "captured_prompt_count_exact":
            len(prompt_names) == EXPECTED_CLAIM_COUNT,
        "draft_count_exact":
            len(drafts) == EXPECTED_CLAIM_COUNT,
        "compiled_review_count_exact":
            len(reviews) == EXPECTED_CLAIM_COUNT,
        "reviewer_work_ids_within_frozen_topn":
            reviewer_ids_within_topn,
        "compiler_unknown_work_ids_empty":
            compiler_unknown_empty,
        "strong_compiled_matches_abstract_backed":
            strong_abstract_backed,
        "compiled_review_claim_identity_exact":
            compiled_identity_exact,
        "frozen_topn_identity_preserved":
            topn_identity_preserved,
        "hypothesis_level_novelty_status_not_computed": True,
        "automatic_next_stage_not_authorized": True,
    }

    diagnostics = {
        "unknown_draft_work_ids": unknown_draft_ids,
        "compiled_status_counts": dict(
            sorted(
                Counter(review.status for review in reviews).items()
            )
        ),
        "compiled_relationship_counts": dict(
            sorted(
                Counter(
                    match.relationship
                    for review in reviews
                    for match in review.matches
                ).items()
            )
        ),
        "core_no_direct_match_claim_ids": [
            review.claim_id
            for review in reviews
            if (
                review.importance == "core"
                and review.status == "NO_DIRECT_MATCH_FOUND"
            )
        ],
        "core_insufficient_metadata_claim_ids": [
            review.claim_id
            for review in reviews
            if (
                review.importance == "core"
                and review.status == "INSUFFICIENT_METADATA"
            )
        ],
        "reviewer_omitted_all_matches_claim_ids": [
            claim_id
            for claim_id, draft in drafts.items()
            if not draft.matches
        ],
    }
    return checks, diagnostics


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
            "pipeline": "sers_claim_review_only_dev",
            "semantics_id": SEMANTICS_ID,
            "source_ranker_run_id": spec["source_ranker_run_id"],
            "source_claim_review_spec_id": spec["spec_id"],
        },
    )


def scan_output_for_secrets(
    *,
    output_root: Path,
    secret_values: list[str],
) -> tuple[bool, list[str]]:
    secrets = [
        value
        for value in secret_values
        if isinstance(value, str) and len(value) >= 8
    ]
    if not secrets:
        return True, []

    offenders: list[str] = []
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(secret in text for secret in secrets):
            offenders.append(str(path.relative_to(output_root)))
    return not offenders, sorted(offenders)


def render_human_audit(
    *,
    reviews: list[ClaimPriorArtReview],
    drafts: Mapping[str, ClaimPriorArtReviewDraft],
    ranker_report: Mapping[str, Any],
) -> str:
    ranker_rows = {
        str(row["claim_id"]): row
        for row in ranker_report["claim_reports"]
    }
    lines = [
        "# SERS Claim-level Prior-Art Review-only DEV Audit",
        "",
        "- Hypothesis-level novelty verdict: **NOT COMPUTED**",
        "- Relationship/scientific outcome: **MANUAL_REVIEW_REQUIRED**",
        "- Reviewer evidence scope: frozen ranker top-8 title/abstract metadata only",
        "",
        (
            "`NO_DIRECT_MATCH_FOUND` below means only that the bounded "
            "frozen ranked evidence did not yield a compiled direct/partial/"
            "component match; it is not literature-wide absence evidence."
        ),
        "",
    ]

    for index, review in enumerate(reviews, start=1):
        draft = drafts[review.claim_id]
        ranker_row = ranker_rows[review.claim_id]
        lines.extend(
            [
                (
                    f"## Claim {index} — {review.importance} — "
                    f"{review.status}"
                ),
                "",
                review.claim_text,
                "",
                (
                    f"Frozen candidates: "
                    f"{len(ranker_row['top_ranked_works'])} | "
                    f"Reviewer matches returned: {len(draft.matches)} | "
                    f"Compiled matches: {len(review.matches)}"
                ),
                "",
                f"Compiled interpretation: {review.interpretation}",
                "",
            ]
        )
        if review.reason_codes:
            lines.append(
                "Reason codes: `"
                + "`, `".join(review.reason_codes)
                + "`"
            )
            lines.append("")

        for match in review.matches:
            lines.extend(
                [
                    (
                        f"### {match.relationship} — "
                        f"{match.title}"
                    ),
                    "",
                    (
                        f"`confidence={match.confidence:.3f}` | "
                        f"`ranker_relevance={match.relevance_score:.4f}` | "
                        f"`semantic={match.semantic_similarity:.4f}` | "
                        f"`lexical={match.lexical_coverage:.4f}` | "
                        f"`domain={match.reaction_domain_relevance:.4f}` | "
                        f"`scope={match.catalyst_scope_relevance:.4f}` | "
                        f"`abstract={match.abstract_available}`"
                    ),
                    "",
                    f"Rationale: {match.rationale}",
                    "",
                ]
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def report_from_results(
    *,
    spec: Mapping[str, Any],
    reviews: list[ClaimPriorArtReview],
    drafts: Mapping[str, ClaimPriorArtReviewDraft],
    ranker_report: Mapping[str, Any],
    checks: Mapping[str, bool],
    diagnostics: Mapping[str, Any],
    prompt_records: list[Any],
    audit_rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]],
    secret_scan_pass: bool,
) -> dict[str, Any]:
    structural_pass = all(checks.values()) and secret_scan_pass

    prompt_manifest = [
        {
            "name": str(record.name),
            "prompt_sha256": str(record.prompt_sha256),
        }
        for record in prompt_records
    ]

    body: dict[str, Any] = {
        "schema_version":
            "sers-standard2-claim-review-only-dev-run-v1",
        "semantics_id": SEMANTICS_ID,
        "source_spec_id": spec["spec_id"],
        "source_spec_sha256": spec["spec_sha256"],
        "source_ranker_run_id": spec["source_ranker_run_id"],
        "source_ranker_run_sha256": spec["source_ranker_run_sha256"],
        "source_canonical_packet_id":
            spec["source_canonical_packet_id"],
        "structural_outcome": (
            "CLAIM_REVIEW_STRUCTURAL_DEV_PASS"
            if structural_pass
            else "CLAIM_REVIEW_STRUCTURAL_DEV_FAIL"
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
        "prompt_manifest": prompt_manifest,
        "logical_review_calls": len(prompt_records),
        "successful_prior_art_audit_rows": len(audit_rows),
        "telemetry_row_count": len(telemetry_rows),
        "network_scope":
            "LLM_PROVIDER_ONLY_NO_LITERATURE_RETRIEVAL",
        "literature_network_calls": 0,
        "ranker_recomputed": False,
        "canonicalization_recomputed": False,
        "claim_decomposition_recomputed": False,
        "hypothesis_level_novelty_status_computed": False,
        "automatic_next_stage_authorized": False,
        "fresh_reserve_consumed": False,
    }
    body["run_sha256"] = sha256_json(body)
    body["run_id"] = (
        "sers_standard2_claim_review_only_dev_run:"
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
