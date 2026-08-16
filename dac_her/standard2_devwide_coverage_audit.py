from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from dac_her.external_novelty_contracts import LiteratureQuery, LiteratureQueryPlan
from dac_her.literature_provider_plan import (
    LiteratureProviderPlan,
    build_literature_providers,
    resolve_literature_provider_plan,
)
from dac_her.provider_failure_taxonomy import classify_failure


SPEC_SEMANTICS_ID = "sers_standard2_devwide_coverage_spec_v1"
RUN_SEMANTICS_ID = "sers_standard2_devwide_coverage_run_v1"

EXPECTED_PROVIDER_MODE = "STANDARD_2_PROVIDER"
EXPECTED_ACTIVE_PROVIDERS = ["openalex", "crossref"]
EXPECTED_PROVIDER_PLAN_ID = "literature_provider_plan:9d6d3c9161d16be3f6fd"
EXPECTED_PROVIDER_PLAN_SHA256 = (
    "9d6d3c9161d16be3f6fd997f5b05c1dc892a8f3db165a19fb3f2410067519164"
)

DEFAULT_DIAGNOSTIC_ROOT = Path.home() / "GraphAgentsDAC"
DEFAULT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "standard2_devwide_coverage_spec_v1"
)
DEFAULT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "standard2_devwide_coverage_run_v1"
)
BASELINE_QUERY_PLAN = Path(
    "evaluation/sers_alpha4c5k/dev_e2e_v2/"
    "external_novelty.claims_queries.json"
)

RESULT_LIMIT = 12
EXPECTED_QUERY_COUNT = 27
EXPECTED_PROVIDER_COUNT = 2
EXPECTED_LOGICAL_EXECUTIONS = EXPECTED_QUERY_COUNT * EXPECTED_PROVIDER_COUNT


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


def _query_sha(row: LiteratureQuery) -> str:
    return sha256_json(
        {
            "query_id": row.query_id,
            "hypothesis_id": row.hypothesis_id,
            "claim_id": row.claim_id,
            "query_kind": row.query_kind,
            "query_text": row.query_text,
        }
    )


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


def build_spec(
    *,
    diagnostic_root: Path,
) -> dict[str, Any]:
    provider_plan = resolve_expected_plan()
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

    frozen_queries = []
    for row in query_plan.queries:
        frozen_queries.append(
            {
                "query_id": row.query_id,
                "hypothesis_id": row.hypothesis_id,
                "claim_id": row.claim_id,
                "query_kind": row.query_kind,
                "query_text": row.query_text,
                "query_sha256": _query_sha(row),
            }
        )

    body: dict[str, Any] = {
        "schema_version":
            "sers-standard2-devwide-coverage-spec-v1",
        "semantics_id":
            SPEC_SEMANTICS_ID,
        "provider_plan":
            provider_plan.model_dump(mode="json"),
        "baseline_query_plan_sha256":
            sha256_file(query_plan_path),
        "query_count":
            len(frozen_queries),
        "provider_count":
            len(provider_plan.active_providers),
        "logical_execution_count":
            len(frozen_queries)
            * len(provider_plan.active_providers),
        "result_limit_per_query_provider":
            RESULT_LIMIT,
        "queries":
            frozen_queries,
        "epistemic_policy": {
            "purpose":
                "provider_metadata_coverage_characterization_only",
            "scientific_result_interpretation":
                False,
            "novelty_status_change_authorized":
                False,
            "ranker_use":
                False,
            "llm_use":
                False,
            "paper_titles_persisted":
                False,
            "raw_abstract_text_persisted":
                False,
            "automatic_full_dev_rerun_authorized":
                False,
            "automatic_provider_policy_change_authorized":
                False,
        },
        "execution_policy": {
            "one_shot":
                True,
            "automatic_rerun_authorized":
                False,
        },
    }
    body["spec_sha256"] = sha256_json(body)
    body["spec_id"] = (
        "sers_standard2_devwide_coverage_spec:"
        + body["spec_sha256"][:20]
    )
    return body


def verify_spec(
    *,
    diagnostic_root: Path,
    spec_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    if not spec_path.is_file():
        return ["spec missing"], {}

    stored = read_json(spec_path)
    issues = []

    body = dict(stored)
    spec_id = body.pop("spec_id", None)
    spec_sha = body.pop("spec_sha256", None)
    observed = sha256_json(body)
    if spec_sha != observed:
        issues.append("spec SHA mismatch")
    if spec_id != (
        "sers_standard2_devwide_coverage_spec:"
        + observed[:20]
    ):
        issues.append("spec ID mismatch")

    try:
        recomputed = build_spec(diagnostic_root=diagnostic_root)
        if canonical_json(recomputed) != canonical_json(stored):
            issues.append(
                "deterministic spec recomputation mismatch"
            )
    except Exception as exc:
        issues.append(
            f"spec recomputation failed: {type(exc).__name__}: {exc}"
        )

    return sorted(set(issues)), stored


def _work_fingerprint(work: Any) -> str:
    # Prefer the repository's canonical work_id. It is deterministic and
    # normally DOI-derived when a DOI is available. Hash again so this audit
    # persists no titles/provider IDs/DOIs.
    return hashlib.sha256(
        str(work.work_id).encode("utf-8")
    ).hexdigest()


def _failure_row(exc: Exception) -> dict[str, Any]:
    text = f"{type(exc).__name__}: {exc}"
    classified = classify_failure(text)
    return {
        "failure_category":
            classified["category"],
        "http_status":
            classified["http_status"],
        "error_text_sha256":
            classified["error_text_sha256"],
    }


def run_audit(
    *,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    plan = LiteratureProviderPlan.model_validate(
        spec["provider_plan"]
    )
    providers = build_literature_providers(plan)

    executions = []
    provider_work_ids: dict[str, set[str]] = {
        name: set()
        for name in plan.active_providers
    }
    provider_abstract_ids: dict[str, set[str]] = {
        name: set()
        for name in plan.active_providers
    }
    provider_query_abstract_counts: dict[
        str, dict[str, int]
    ] = {
        name: {}
        for name in plan.active_providers
    }

    for provider in providers:
        for frozen in spec["queries"]:
            query = LiteratureQuery(
                query_id=frozen["query_id"],
                hypothesis_id=frozen["hypothesis_id"],
                claim_id=frozen["claim_id"],
                query_kind=frozen["query_kind"],
                query_text=frozen["query_text"],
            )

            try:
                works = provider.search(
                    query,
                    limit=int(
                        spec[
                            "result_limit_per_query_provider"
                        ]
                    ),
                )
                success = True
                failure = {
                    "failure_category": None,
                    "http_status": None,
                    "error_text_sha256": None,
                }
                fingerprints = [
                    _work_fingerprint(work)
                    for work in works
                ]
                abstract_fingerprints = [
                    _work_fingerprint(work)
                    for work in works
                    if bool(work.abstract)
                ]
                provider_work_ids[
                    provider.provider_name
                ].update(fingerprints)
                provider_abstract_ids[
                    provider.provider_name
                ].update(abstract_fingerprints)
                provider_query_abstract_counts[
                    provider.provider_name
                ][frozen["query_id"]] = len(
                    abstract_fingerprints
                )
                result_count = len(works)
                abstract_count = len(
                    abstract_fingerprints
                )
            except Exception as exc:
                success = False
                failure = _failure_row(exc)
                result_count = 0
                abstract_count = 0
                provider_query_abstract_counts[
                    provider.provider_name
                ][frozen["query_id"]] = 0

            executions.append(
                {
                    "provider":
                        provider.provider_name,
                    "query_id":
                        frozen["query_id"],
                    "query_kind":
                        frozen["query_kind"],
                    "hypothesis_id":
                        frozen["hypothesis_id"],
                    "query_sha256":
                        frozen["query_sha256"],
                    "success":
                        success,
                    "result_count":
                        result_count,
                    "abstract_result_count":
                        abstract_count,
                    **failure,
                }
            )

    provider_summary: dict[str, dict[str, Any]] = {}
    for provider_name in plan.active_providers:
        rows = [
            row
            for row in executions
            if row["provider"] == provider_name
        ]
        raw_results = sum(
            int(row["result_count"])
            for row in rows
        )
        raw_abstracts = sum(
            int(row["abstract_result_count"])
            for row in rows
        )
        successful = sum(
            bool(row["success"])
            for row in rows
        )
        queries_with_abstract = sum(
            int(row["abstract_result_count"]) > 0
            for row in rows
        )
        provider_summary[provider_name] = {
            "logical_execution_count":
                len(rows),
            "successful_logical_execution_count":
                successful,
            "failed_logical_execution_count":
                len(rows) - successful,
            "raw_result_count":
                raw_results,
            "raw_abstract_result_count":
                raw_abstracts,
            "raw_abstract_fraction":
                (
                    raw_abstracts / raw_results
                    if raw_results
                    else None
                ),
            "unique_work_fingerprint_count":
                len(
                    provider_work_ids[
                        provider_name
                    ]
                ),
            "unique_abstract_work_fingerprint_count":
                len(
                    provider_abstract_ids[
                        provider_name
                    ]
                ),
            "queries_with_at_least_one_abstract":
                queries_with_abstract,
            "query_abstract_coverage_fraction":
                (
                    queries_with_abstract
                    / len(rows)
                    if rows
                    else None
                ),
            "failure_category_counts":
                dict(
                    sorted(
                        Counter(
                            row[
                                "failure_category"
                            ]
                            for row in rows
                            if row[
                                "failure_category"
                            ]
                            is not None
                        ).items()
                    )
                ),
        }

    openalex_ids = provider_work_ids.get(
        "openalex", set()
    )
    crossref_ids = provider_work_ids.get(
        "crossref", set()
    )
    overlap = openalex_ids & crossref_ids
    union = openalex_ids | crossref_ids

    openalex_abs = provider_abstract_ids.get(
        "openalex", set()
    )
    crossref_abs = provider_abstract_ids.get(
        "crossref", set()
    )
    combined_abstract_ids = (
        openalex_abs | crossref_abs
    )

    combined_query_abstract_coverage = {}
    for frozen in spec["queries"]:
        query_id = frozen["query_id"]
        combined_query_abstract_coverage[
            query_id
        ] = (
            provider_query_abstract_counts[
                "openalex"
            ].get(query_id, 0)
            + provider_query_abstract_counts[
                "crossref"
            ].get(query_id, 0)
        )

    queries_with_any_abstract = sum(
        count > 0
        for count in combined_query_abstract_coverage.values()
    )
    queries_with_three_or_more_abstracts = sum(
        count >= 3
        for count in combined_query_abstract_coverage.values()
    )

    success_count = sum(
        bool(row["success"])
        for row in executions
    )
    if success_count == int(
        spec["logical_execution_count"]
    ):
        operational_outcome = (
            "STANDARD2_DEVWIDE_OPERATIONAL_PASS"
        )
    elif success_count == 0:
        operational_outcome = (
            "STANDARD2_DEVWIDE_OPERATIONAL_FAIL_ALL"
        )
    else:
        operational_outcome = (
            "STANDARD2_DEVWIDE_OPERATIONAL_PARTIAL"
        )

    if (
        provider_summary["openalex"][
            "queries_with_at_least_one_abstract"
        ]
        > provider_summary["crossref"][
            "queries_with_at_least_one_abstract"
        ]
    ):
        abstract_complementarity = (
            "OPENALEX_IMPROVES_QUERY_LEVEL_ABSTRACT_COVERAGE"
        )
    elif (
        len(combined_abstract_ids)
        > max(
            len(openalex_abs),
            len(crossref_abs),
        )
    ):
        abstract_complementarity = (
            "COMPLEMENTARY_ABSTRACT_COVERAGE_WITHOUT_QUERY_COUNT_GAIN"
        )
    else:
        abstract_complementarity = (
            "NO_DEMONSTRATED_ABSTRACT_COMPLEMENTARITY"
        )

    body: dict[str, Any] = {
        "schema_version":
            "sers-standard2-devwide-coverage-run-v1",
        "semantics_id":
            RUN_SEMANTICS_ID,
        "source_spec_id":
            spec["spec_id"],
        "source_spec_sha256":
            spec["spec_sha256"],
        "provider_plan_id":
            plan.plan_id,
        "provider_plan_sha256":
            plan.plan_sha256,
        "provider_mode":
            plan.mode,
        "active_providers":
            list(plan.active_providers),
        "logical_execution_count":
            len(executions),
        "successful_logical_execution_count":
            success_count,
        "operational_outcome":
            operational_outcome,
        "provider_summary":
            provider_summary,
        "cross_provider_unique_work_overlap": {
            "openalex_unique_work_count":
                len(openalex_ids),
            "crossref_unique_work_count":
                len(crossref_ids),
            "overlap_work_fingerprint_count":
                len(overlap),
            "union_work_fingerprint_count":
                len(union),
            "jaccard_overlap":
                (
                    len(overlap) / len(union)
                    if union
                    else None
                ),
        },
        "combined_abstract_coverage": {
            "unique_abstract_work_fingerprint_count":
                len(combined_abstract_ids),
            "queries_with_at_least_one_abstract":
                queries_with_any_abstract,
            "query_coverage_fraction":
                (
                    queries_with_any_abstract
                    / int(spec["query_count"])
                ),
            "queries_with_three_or_more_raw_abstract_results":
                queries_with_three_or_more_abstracts,
            "three_or_more_query_fraction":
                (
                    queries_with_three_or_more_abstracts
                    / int(spec["query_count"])
                ),
        },
        "abstract_complementarity":
            abstract_complementarity,
        "executions":
            executions,
        "paper_titles_persisted":
            False,
        "raw_abstract_text_persisted":
            False,
        "scientific_result_interpretation":
            False,
        "novelty_status_change_authorized":
            False,
        "ranker_used":
            False,
        "llm_calls":
            0,
        "full_dev_rerun_authorized":
            False,
        "automatic_provider_policy_change_authorized":
            False,
    }
    body["run_sha256"] = sha256_json(body)
    body["run_id"] = (
        "sers_standard2_devwide_coverage_run:"
        + body["run_sha256"][:20]
    )
    return body


def verify_run(
    *,
    spec: Mapping[str, Any],
    run_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    if not run_path.is_file():
        return ["run missing"], {}

    stored = read_json(run_path)
    issues = []

    body = dict(stored)
    run_id = body.pop("run_id", None)
    run_sha = body.pop("run_sha256", None)
    observed = sha256_json(body)
    if run_sha != observed:
        issues.append("run SHA mismatch")
    if run_id != (
        "sers_standard2_devwide_coverage_run:"
        + observed[:20]
    ):
        issues.append("run ID mismatch")
    if stored.get("source_spec_id") != spec.get("spec_id"):
        issues.append("run/spec ID mismatch")
    if stored.get("source_spec_sha256") != spec.get("spec_sha256"):
        issues.append("run/spec SHA mismatch")
    if stored.get("logical_execution_count") != EXPECTED_LOGICAL_EXECUTIONS:
        issues.append("logical execution count mismatch")
    if stored.get("paper_titles_persisted") is not False:
        issues.append("title persistence violation")
    if stored.get("raw_abstract_text_persisted") is not False:
        issues.append("abstract-text persistence violation")
    if stored.get("scientific_result_interpretation") is not False:
        issues.append("scientific interpretation violation")
    if stored.get("ranker_used") is not False:
        issues.append("ranker-use violation")
    if stored.get("llm_calls") != 0:
        issues.append("unexpected LLM calls")
    if stored.get("full_dev_rerun_authorized") is not False:
        issues.append("unexpected full DEV authorization")

    return sorted(set(issues)), stored
