from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
)
from dac_her.literature_provider_plan import (
    LiteratureProviderPlan,
    build_literature_providers,
    resolve_literature_provider_plan,
)
from dac_her.literature_retrieval import (
    canonicalize_prior_art_packet,
)


SPEC_SEMANTICS_ID = "sers_standard2_canonicalization_dev_spec_v1"
RUN_SEMANTICS_ID = "sers_standard2_canonicalization_dev_run_v1"

EXPECTED_PROVIDER_MODE = "STANDARD_2_PROVIDER"
EXPECTED_ACTIVE_PROVIDERS = ["openalex", "crossref"]
EXPECTED_PROVIDER_PLAN_ID = "literature_provider_plan:9d6d3c9161d16be3f6fd"
EXPECTED_PROVIDER_PLAN_SHA256 = (
    "9d6d3c9161d16be3f6fd997f5b05c1dc892a8f3db165a19fb3f2410067519164"
)

EXPECTED_PREREQUISITE_SPEC_ID = (
    "sers_standard2_devwide_coverage_spec:b001c8198e62a84fb1a0"
)
EXPECTED_PREREQUISITE_RUN_ID = (
    "sers_standard2_devwide_coverage_run:5c39416d1e672cefe564"
)
EXPECTED_PREREQUISITE_OUTCOME = "STANDARD2_DEVWIDE_OPERATIONAL_PASS"

DEFAULT_DIAGNOSTIC_ROOT = Path.home() / "GraphAgentsDAC"
DEFAULT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "canonicalization_only_dev_spec_v1"
)
DEFAULT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "canonicalization_only_dev_run_v1"
)
PREREQUISITE_RUN = Path(
    "evaluation/sers_provider_reliability/"
    "standard2_devwide_coverage_run_v1/coverage_run.json"
)
BASELINE_QUERY_PLAN = Path(
    "evaluation/sers_alpha4c5k/dev_e2e_v2/"
    "external_novelty.claims_queries.json"
)

RESULT_LIMIT = 12
EXPECTED_QUERY_COUNT = 27
EXPECTED_PROVIDER_COUNT = 2
EXPECTED_LOGICAL_EXECUTIONS = 54

SOURCE_FILES_TO_FREEZE = (
    Path("dac_her/literature_retrieval.py"),
    Path("dac_her/literature_provider_plan.py"),
    Path("dac_her/external_novelty_contracts.py"),
)


def canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _norm_doi(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("https://doi.org/"):
        text = text[len("https://doi.org/") :]
    if text.startswith("doi:"):
        text = text[4:]
    return text or None


_SUPPLEMENTARY_DOI_RE = re.compile(r"\.s\d+$", re.I)


def _doi_family(value: Any) -> str | None:
    doi = _norm_doi(value)
    if not doi:
        return None
    return _SUPPLEMENTARY_DOI_RE.sub("", doi)


def _norm_title(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", text)
    return " ".join(text.split())


def title_cross_doi_collision_groups(
    works: list[PriorArtWork],
) -> list[dict[str, Any]]:
    """Find title-fallback groups that contain multiple distinct DOI families.

    The production canonicalizer merges exact normalized titles after DOI-family
    merging. Such a group is therefore a fail-closed diagnostic because it can
    represent an over-merge of distinct DOI-bearing records.
    """
    groups: dict[str, list[PriorArtWork]] = defaultdict(list)
    for work in works:
        title = _norm_title(work.title)
        if len(title) >= 20:
            groups[title].append(work)

    collisions: list[dict[str, Any]] = []
    for title, rows in groups.items():
        families = sorted(
            {
                family
                for family in (_doi_family(row.doi) for row in rows)
                if family
            }
        )
        if len(families) > 1:
            collisions.append(
                {
                    "normalized_title_sha256":
                        hashlib.sha256(
                            title.encode("utf-8")
                        ).hexdigest(),
                    "raw_record_count": len(rows),
                    "distinct_doi_family_count": len(families),
                    "doi_family_hashes": [
                        hashlib.sha256(
                            family.encode("utf-8")
                        ).hexdigest()
                        for family in families
                    ],
                }
            )
    return sorted(
        collisions,
        key=lambda row: row["normalized_title_sha256"],
    )


def _frozen_source_hashes(root: Path) -> dict[str, str]:
    result = {}
    for rel in SOURCE_FILES_TO_FREEZE:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(rel)] = sha256_file(path)
    return result


def resolve_expected_plan() -> LiteratureProviderPlan:
    plan = resolve_literature_provider_plan()
    if plan.mode != EXPECTED_PROVIDER_MODE:
        raise ValueError(
            f"Expected {EXPECTED_PROVIDER_MODE}, observed {plan.mode}."
        )
    if plan.active_providers != EXPECTED_ACTIVE_PROVIDERS:
        raise ValueError(
            f"Expected active providers {EXPECTED_ACTIVE_PROVIDERS}, "
            f"observed {plan.active_providers}."
        )
    if plan.plan_id != EXPECTED_PROVIDER_PLAN_ID:
        raise ValueError(
            f"Provider plan ID drift: {plan.plan_id}"
        )
    if plan.plan_sha256 != EXPECTED_PROVIDER_PLAN_SHA256:
        raise ValueError(
            f"Provider plan SHA drift: {plan.plan_sha256}"
        )
    return plan


def _validate_prerequisite(root: Path) -> dict[str, Any]:
    path = root / PREREQUISITE_RUN
    if not path.is_file():
        raise FileNotFoundError(path)
    run = read_json(path)
    if run.get("source_spec_id") != EXPECTED_PREREQUISITE_SPEC_ID:
        raise ValueError(
            "Prerequisite DEV-wide coverage spec ID mismatch."
        )
    if run.get("run_id") != EXPECTED_PREREQUISITE_RUN_ID:
        raise ValueError(
            "Prerequisite DEV-wide coverage run ID mismatch."
        )
    if run.get("operational_outcome") != EXPECTED_PREREQUISITE_OUTCOME:
        raise ValueError(
            "Prerequisite DEV-wide coverage outcome is not PASS."
        )
    if run.get("successful_logical_execution_count") != 54:
        raise ValueError(
            "Prerequisite DEV-wide coverage was not 54/54."
        )
    return {
        "run_id": run["run_id"],
        "run_sha256": run["run_sha256"],
        "operational_outcome": run["operational_outcome"],
    }


def build_spec(
    *,
    repo_root: Path,
    diagnostic_root: Path,
) -> dict[str, Any]:
    provider_plan = resolve_expected_plan()
    prerequisite = _validate_prerequisite(repo_root)

    query_plan_path = diagnostic_root / BASELINE_QUERY_PLAN
    if not query_plan_path.is_file():
        raise FileNotFoundError(query_plan_path)
    query_plan = LiteratureQueryPlan.model_validate_json(
        query_plan_path.read_text(encoding="utf-8")
    )
    if len(query_plan.queries) != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUERY_COUNT} baseline queries, "
            f"observed {len(query_plan.queries)}."
        )

    body: dict[str, Any] = {
        "schema_version":
            "sers-standard2-canonicalization-dev-spec-v1",
        "semantics_id":
            SPEC_SEMANTICS_ID,
        "provider_plan":
            provider_plan.model_dump(mode="json"),
        "source_hashes":
            _frozen_source_hashes(repo_root),
        "prerequisite_devwide_coverage":
            prerequisite,
        "baseline_query_plan_path":
            str(BASELINE_QUERY_PLAN),
        "baseline_query_plan_sha256":
            sha256_file(query_plan_path),
        "source_query_plan_id":
            query_plan.plan_id,
        "source_portfolio_id":
            query_plan.source_portfolio_id,
        "query_count":
            len(query_plan.queries),
        "provider_count":
            len(provider_plan.active_providers),
        "logical_execution_count":
            len(query_plan.queries)
            * len(provider_plan.active_providers),
        "result_limit_per_query_provider":
            RESULT_LIMIT,
        "canonicalizer":
            "dac_her.literature_retrieval.canonicalize_prior_art_packet",
        "validation_policy": {
            "require_all_provider_query_executions_successful": True,
            "require_deterministic_recanonicalization": True,
            "require_unique_canonical_work_ids": True,
            "require_global_query_provenance_preserved": True,
            "require_global_provider_provenance_preserved": True,
            "require_claim_provenance_subset_of_query_plan": True,
            "fail_on_exact_title_multiple_doi_families": True,
            "require_secret_scan_pass": True,
        },
        "epistemic_policy": {
            "purpose":
                "canonicalization_identity_deduplication_and_provenance_validation_only",
            "ranker_used": False,
            "llm_used": False,
            "claim_review_used": False,
            "novelty_status_change_authorized": False,
            "scientific_result_interpretation": False,
            "fresh_reserve_consumed": False,
            "automatic_next_stage_authorized": False,
        },
        "execution_policy": {
            "one_shot_network_retrieval": True,
            "automatic_rerun_authorized": False,
            "persist_raw_prior_art_packet": True,
            "persist_canonical_prior_art_packet": True,
            "canonical_packet_may_feed_later_dev_ranker_validation": True,
        },
    }
    body["spec_sha256"] = sha256_json(body)
    body["spec_id"] = (
        "sers_standard2_canonicalization_dev_spec:"
        + body["spec_sha256"][:20]
    )
    return body


def verify_spec(
    *,
    repo_root: Path,
    diagnostic_root: Path,
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
        "sers_standard2_canonicalization_dev_spec:"
        + observed[:20]
    ):
        issues.append("spec ID mismatch")

    try:
        recomputed = build_spec(
            repo_root=repo_root,
            diagnostic_root=diagnostic_root,
        )
        if canonical_json(recomputed) != canonical_json(stored):
            issues.append(
                "deterministic spec/source/provider recomputation mismatch"
            )
    except Exception as exc:
        issues.append(
            f"spec recomputation failed: {type(exc).__name__}: {exc}"
        )

    return sorted(set(issues)), stored


def _make_raw_packet(
    *,
    plan: LiteratureQueryPlan,
    providers: list[Any],
    result_limit: int,
) -> PriorArtPacket:
    raw_works: list[PriorArtWork] = []
    executions: list[QueryExecution] = []

    for query in plan.queries:
        for provider in providers:
            started = time.perf_counter()
            try:
                rows = provider.search(
                    query,
                    limit=result_limit,
                )
                elapsed = time.perf_counter() - started
                raw_works.extend(rows)
                executions.append(
                    QueryExecution(
                        query_id=query.query_id,
                        provider=provider.provider_name,
                        success=True,
                        result_count=len(rows),
                        elapsed_seconds=elapsed,
                    )
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                executions.append(
                    QueryExecution(
                        query_id=query.query_id,
                        provider=provider.provider_name,
                        success=False,
                        result_count=0,
                        elapsed_seconds=elapsed,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    searched_at = datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    body = {
        "schema_version": "prior-art-packet-v1",
        "packet_id": "raw_prior_art_packet:pending",
        "source_portfolio_id": plan.source_portfolio_id,
        "source_query_plan_id": plan.plan_id,
        "searched_at_utc": searched_at,
        "providers_requested": [
            provider.provider_name
            for provider in providers
        ],
        "works": [
            work.model_dump(mode="json")
            for work in raw_works
        ],
        "executions": [
            row.model_dump(mode="json")
            for row in executions
        ],
        "raw_work_count": len(raw_works),
        "canonical_work_count": len(raw_works),
        "deduplicated_work_count": 0,
        "supplementary_records_collapsed": 0,
        "epistemic_usage":
            "prior_art_only_not_positive_premise",
    }
    body_for_id = dict(body)
    body_for_id["packet_id"] = None
    digest = sha256_json(body_for_id)
    body["packet_id"] = (
        "raw_prior_art_packet:" + digest[:20]
    )
    return PriorArtPacket(
        **body,
        packet_sha256=sha256_json(body),
    )


def _secret_scan(
    serialized_values: list[str],
) -> tuple[bool, list[str]]:
    secret_envs = (
        "OPENALEX_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
    )
    found: list[str] = []
    for env_name in secret_envs:
        secret = str(os.getenv(env_name) or "").strip()
        if not secret:
            continue
        if any(secret in text for text in serialized_values):
            found.append(env_name)
    return (not found), found


def _global_provenance(
    works: list[PriorArtWork],
) -> dict[str, set[str]]:
    return {
        "providers": {
            provider
            for work in works
            for provider in work.providers
        },
        "query_ids": {
            query_id
            for work in works
            for query_id in work.retrieval_query_ids
        },
        "claim_ids": {
            claim_id
            for work in works
            for claim_id in work.retrieval_claim_ids
        },
    }


def build_validation_report(
    *,
    spec: Mapping[str, Any],
    query_plan: LiteratureQueryPlan,
    raw_packet: PriorArtPacket,
    canonical_packet: PriorArtPacket,
    canonical_packet_repeat: PriorArtPacket,
) -> dict[str, Any]:
    raw_works = list(raw_packet.works)
    canonical_works = list(canonical_packet.works)

    valid_query_ids = {
        row.query_id
        for row in query_plan.queries
    }
    valid_claim_ids = {
        claim.claim_id
        for decomposition in query_plan.claims
        for claim in decomposition.claims
    }

    raw_provenance = _global_provenance(raw_works)
    canonical_provenance = _global_provenance(
        canonical_works
    )

    invalid_query_ids = sorted(
        canonical_provenance["query_ids"]
        - valid_query_ids
    )
    invalid_claim_ids = sorted(
        canonical_provenance["claim_ids"]
        - valid_claim_ids
    )

    title_collisions = (
        title_cross_doi_collision_groups(
            raw_works
        )
    )

    canonical_ids = [
        row.work_id
        for row in canonical_works
    ]
    unique_canonical_ids = (
        len(canonical_ids)
        == len(set(canonical_ids))
    )

    deterministic = (
        canonical_json(canonical_packet)
        == canonical_json(canonical_packet_repeat)
    )

    provider_query_successes = sum(
        bool(row.success)
        for row in raw_packet.executions
    )
    all_exec_success = (
        provider_query_successes
        == int(spec["logical_execution_count"])
    )

    raw_count = len(raw_works)
    canonical_count = len(canonical_works)
    counts_sane = (
        canonical_count <= raw_count
        and canonical_packet.raw_work_count == raw_count
        and canonical_packet.canonical_work_count
        == canonical_count
        and canonical_packet.deduplicated_work_count
        == raw_count - canonical_count
    )

    raw_abstract_count = sum(
        bool(row.abstract)
        for row in raw_works
    )
    canonical_abstract_count = sum(
        bool(row.abstract)
        for row in canonical_works
    )

    multi_provider_count = sum(
        len(set(row.providers)) >= 2
        for row in canonical_works
    )
    multi_query_count = sum(
        len(set(row.retrieval_query_ids)) >= 2
        for row in canonical_works
    )
    multi_claim_count = sum(
        len(set(row.retrieval_claim_ids)) >= 2
        for row in canonical_works
    )

    canonical_by_query = Counter()
    raw_by_query = Counter()
    for work in raw_works:
        for query_id in set(
            work.retrieval_query_ids
        ):
            raw_by_query[query_id] += 1
    for work in canonical_works:
        for query_id in set(
            work.retrieval_query_ids
        ):
            canonical_by_query[query_id] += 1

    lost_nonempty_queries = sorted(
        query_id
        for query_id, count
        in raw_by_query.items()
        if count > 0
        and canonical_by_query.get(
            query_id, 0
        ) == 0
    )

    secret_scan_pass, secret_env_names = (
        _secret_scan(
            [
                canonical_json(raw_packet),
                canonical_json(canonical_packet),
            ]
        )
    )

    checks = {
        "all_provider_query_executions_successful":
            all_exec_success,
        "canonicalization_deterministic":
            deterministic,
        "canonical_counts_sane":
            counts_sane,
        "canonical_work_ids_unique":
            unique_canonical_ids,
        "global_query_provenance_preserved":
            (
                raw_provenance["query_ids"]
                == canonical_provenance["query_ids"]
                and not lost_nonempty_queries
            ),
        "global_provider_provenance_preserved":
            (
                raw_provenance["providers"]
                == canonical_provenance["providers"]
            ),
        "claim_provenance_valid":
            not invalid_claim_ids,
        "query_provenance_valid":
            not invalid_query_ids,
        "no_exact_title_multiple_doi_family_collision":
            not title_collisions,
        "secret_scan_pass":
            secret_scan_pass,
    }
    passed = all(checks.values())

    body: dict[str, Any] = {
        "schema_version":
            "sers-standard2-canonicalization-dev-run-v1",
        "semantics_id":
            RUN_SEMANTICS_ID,
        "source_spec_id":
            spec["spec_id"],
        "source_spec_sha256":
            spec["spec_sha256"],
        "source_query_plan_id":
            query_plan.plan_id,
        "raw_packet_id":
            raw_packet.packet_id,
        "raw_packet_sha256":
            raw_packet.packet_sha256,
        "canonical_packet_id":
            canonical_packet.packet_id,
        "canonical_packet_sha256":
            canonical_packet.packet_sha256,
        "outcome":
            (
                "CANONICALIZATION_DEV_PASS"
                if passed
                else "CANONICALIZATION_DEV_FAIL"
            ),
        "checks":
            checks,
        "counts": {
            "logical_execution_count":
                len(raw_packet.executions),
            "successful_logical_execution_count":
                provider_query_successes,
            "raw_work_count":
                raw_count,
            "canonical_work_count":
                canonical_count,
            "deduplicated_work_count":
                raw_count - canonical_count,
            "deduplication_fraction":
                (
                    (raw_count - canonical_count)
                    / raw_count
                    if raw_count
                    else 0.0
                ),
            "supplementary_records_collapsed":
                canonical_packet.supplementary_records_collapsed,
            "raw_abstract_work_count":
                raw_abstract_count,
            "canonical_abstract_work_count":
                canonical_abstract_count,
            "canonical_multi_provider_work_count":
                multi_provider_count,
            "canonical_multi_query_work_count":
                multi_query_count,
            "canonical_multi_claim_work_count":
                multi_claim_count,
            "title_cross_doi_collision_group_count":
                len(title_collisions),
        },
        "provenance_diagnostics": {
            "raw_provider_set":
                sorted(raw_provenance["providers"]),
            "canonical_provider_set":
                sorted(
                    canonical_provenance["providers"]
                ),
            "raw_query_id_count":
                len(raw_provenance["query_ids"]),
            "canonical_query_id_count":
                len(
                    canonical_provenance["query_ids"]
                ),
            "raw_claim_id_count":
                len(raw_provenance["claim_ids"]),
            "canonical_claim_id_count":
                len(
                    canonical_provenance["claim_ids"]
                ),
            "invalid_canonical_query_ids":
                invalid_query_ids,
            "invalid_canonical_claim_ids":
                invalid_claim_ids,
            "lost_nonempty_query_ids":
                lost_nonempty_queries,
        },
        "title_cross_doi_collision_groups":
            title_collisions,
        "secret_scan": {
            "passed":
                secret_scan_pass,
            "matched_secret_env_names":
                secret_env_names,
            "secret_values_persisted":
                not secret_scan_pass,
        },
        "ranker_used":
            False,
        "llm_calls":
            0,
        "claim_review_used":
            False,
        "scientific_result_interpretation":
            False,
        "novelty_status_change_authorized":
            False,
        "fresh_reserve_consumed":
            False,
        "automatic_next_stage_authorized":
            False,
        "canonical_packet_eligible_for_dev_ranker_validation":
            passed,
    }
    body["run_sha256"] = sha256_json(body)
    body["run_id"] = (
        "sers_standard2_canonicalization_dev_run:"
        + body["run_sha256"][:20]
    )
    return body


def run_validation(
    *,
    repo_root: Path,
    diagnostic_root: Path,
    spec: Mapping[str, Any],
) -> tuple[
    PriorArtPacket,
    PriorArtPacket,
    dict[str, Any],
]:
    query_plan_path = (
        diagnostic_root
        / Path(
            str(spec["baseline_query_plan_path"])
        )
    )
    query_plan = LiteratureQueryPlan.model_validate_json(
        query_plan_path.read_text(encoding="utf-8")
    )
    if (
        sha256_file(query_plan_path)
        != spec["baseline_query_plan_sha256"]
    ):
        raise RuntimeError(
            "Frozen query-plan SHA drift."
        )
    if (
        _frozen_source_hashes(repo_root)
        != spec["source_hashes"]
    ):
        raise RuntimeError(
            "Frozen canonicalization source hash drift."
        )

    plan = LiteratureProviderPlan.model_validate(
        spec["provider_plan"]
    )
    providers = build_literature_providers(plan)

    raw_packet = _make_raw_packet(
        plan=query_plan,
        providers=providers,
        result_limit=int(
            spec["result_limit_per_query_provider"]
        ),
    )
    canonical_packet = (
        canonicalize_prior_art_packet(
            raw_packet
        )
    )
    canonical_packet_repeat = (
        canonicalize_prior_art_packet(
            raw_packet
        )
    )
    report = build_validation_report(
        spec=spec,
        query_plan=query_plan,
        raw_packet=raw_packet,
        canonical_packet=canonical_packet,
        canonical_packet_repeat=
            canonical_packet_repeat,
    )
    return (
        raw_packet,
        canonical_packet,
        report,
    )


def verify_run(
    *,
    spec: Mapping[str, Any],
    raw_packet_path: Path,
    canonical_packet_path: Path,
    report_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    if not raw_packet_path.is_file():
        issues.append("raw prior-art packet missing")
    if not canonical_packet_path.is_file():
        issues.append(
            "canonical prior-art packet missing"
        )
    if not report_path.is_file():
        issues.append(
            "canonicalization report missing"
        )
    if issues:
        return sorted(set(issues)), {}

    raw_packet = PriorArtPacket.model_validate_json(
        raw_packet_path.read_text(encoding="utf-8")
    )
    canonical_packet = (
        PriorArtPacket.model_validate_json(
            canonical_packet_path.read_text(
                encoding="utf-8"
            )
        )
    )
    report = read_json(report_path)

    body = dict(report)
    run_id = body.pop("run_id", None)
    run_sha = body.pop("run_sha256", None)
    observed = sha256_json(body)
    if run_sha != observed:
        issues.append("report SHA mismatch")
    if run_id != (
        "sers_standard2_canonicalization_dev_run:"
        + observed[:20]
    ):
        issues.append("report ID mismatch")

    if report.get("source_spec_id") != spec.get(
        "spec_id"
    ):
        issues.append("run/spec ID mismatch")
    if report.get(
        "source_spec_sha256"
    ) != spec.get("spec_sha256"):
        issues.append("run/spec SHA mismatch")

    raw_body = raw_packet.model_dump(mode="json")
    raw_sha = raw_body.pop("packet_sha256")
    if sha256_json(raw_body) != raw_sha:
        issues.append("raw packet SHA mismatch")

    canonical_body = canonical_packet.model_dump(
        mode="json"
    )
    canonical_sha = canonical_body.pop(
        "packet_sha256"
    )
    if sha256_json(canonical_body) != canonical_sha:
        issues.append(
            "canonical packet SHA mismatch"
        )

    recanonical = canonicalize_prior_art_packet(
        raw_packet
    )
    if (
        canonical_json(recanonical)
        != canonical_json(canonical_packet)
    ):
        issues.append(
            "offline recanonicalization mismatch"
        )

    if report.get("outcome") != (
        "CANONICALIZATION_DEV_PASS"
    ):
        issues.append(
            "canonicalization outcome is not PASS"
        )
    if report.get(
        "canonical_packet_eligible_for_dev_ranker_validation"
    ) is not True:
        issues.append(
            "canonical packet not eligible for ranker validation"
        )
    if report.get("ranker_used") is not False:
        issues.append("ranker-use violation")
    if report.get("llm_calls") != 0:
        issues.append("unexpected LLM calls")
    if report.get(
        "scientific_result_interpretation"
    ) is not False:
        issues.append(
            "scientific interpretation violation"
        )
    if report.get(
        "novelty_status_change_authorized"
    ) is not False:
        issues.append(
            "novelty-status authorization violation"
        )
    if report.get(
        "fresh_reserve_consumed"
    ) is not False:
        issues.append(
            "fresh-reserve consumption violation"
        )

    return sorted(set(issues)), report
