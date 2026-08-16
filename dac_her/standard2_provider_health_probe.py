from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from dac_her.external_novelty_contracts import LiteratureQuery, LiteratureQueryPlan
from dac_her.literature_provider_plan import (
    LiteratureProviderPlan,
    build_literature_providers,
    resolve_literature_provider_plan,
)
from dac_her.provider_failure_taxonomy import classify_failure


SPEC_SEMANTICS_ID = "sers_standard2_provider_health_probe_spec_v1"
RUN_SEMANTICS_ID = "sers_standard2_provider_health_probe_run_v1"

EXPECTED_PROVIDER_MODE = "STANDARD_2_PROVIDER"
EXPECTED_ACTIVE_PROVIDERS = ["openalex", "crossref"]
EXPECTED_PROVIDER_PLAN_ID = "literature_provider_plan:9d6d3c9161d16be3f6fd"
EXPECTED_PROVIDER_PLAN_SHA256 = (
    "9d6d3c9161d16be3f6fd997f5b05c1dc892a8f3db165a19fb3f2410067519164"
)

DEFAULT_DIAGNOSTIC_ROOT = Path.home() / "GraphAgentsDAC"
DEFAULT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "standard2_health_probe_spec_v1"
)
DEFAULT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "standard2_health_probe_run_v1"
)
BASELINE_QUERY_PLAN = Path(
    "evaluation/sers_alpha4c5k/dev_e2e_v2/"
    "external_novelty.claims_queries.json"
)

EXPECTED_HYPOTHESIS_COUNT = 3
RESULT_LIMIT = 1
SELECTION_POLICY = (
    "one_unique_hypothesis_composite_query_per_baseline_hypothesis"
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


def _query_sha(row: LiteratureQuery) -> str:
    return sha256_json(
        {
            "query_id": row.query_id,
            "hypothesis_id": row.hypothesis_id,
            "query_kind": row.query_kind,
            "query_text": row.query_text,
        }
    )


def _resolve_expected_plan() -> LiteratureProviderPlan:
    plan = resolve_literature_provider_plan()
    if plan.mode != EXPECTED_PROVIDER_MODE:
        raise ValueError(
            f"Expected provider mode {EXPECTED_PROVIDER_MODE}, observed {plan.mode}."
        )
    if plan.active_providers != EXPECTED_ACTIVE_PROVIDERS:
        raise ValueError(
            "Expected active providers "
            f"{EXPECTED_ACTIVE_PROVIDERS}, observed {plan.active_providers}."
        )
    if plan.plan_id != EXPECTED_PROVIDER_PLAN_ID:
        raise ValueError(
            f"Provider plan ID drift: {plan.plan_id}"
        )
    if plan.plan_sha256 != EXPECTED_PROVIDER_PLAN_SHA256:
        raise ValueError(
            f"Provider plan SHA drift: {plan.plan_sha256}"
        )
    if plan.semantic_scholar_api_key_configured:
        raise ValueError(
            "This probe is specifically for STANDARD_2_PROVIDER with S2 absent."
        )
    return plan


def build_spec(
    *,
    diagnostic_root: Path,
) -> dict[str, Any]:
    provider_plan = _resolve_expected_plan()
    query_plan_path = diagnostic_root / BASELINE_QUERY_PLAN
    if not query_plan_path.is_file():
        raise FileNotFoundError(query_plan_path)

    query_plan = LiteratureQueryPlan.model_validate_json(
        query_plan_path.read_text(encoding="utf-8")
    )
    hypothesis_ids = sorted(
        {row.hypothesis_id for row in query_plan.queries}
    )
    if len(hypothesis_ids) != EXPECTED_HYPOTHESIS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_HYPOTHESIS_COUNT} baseline hypotheses, "
            f"observed {len(hypothesis_ids)}."
        )

    selected = []
    for hypothesis_id in hypothesis_ids:
        rows = [
            row for row in query_plan.queries
            if row.hypothesis_id == hypothesis_id
            and row.query_kind == "hypothesis_composite"
        ]
        if len(rows) != 1:
            raise ValueError(
                "Expected exactly one hypothesis_composite query for "
                f"{hypothesis_id}; observed {len(rows)}."
            )
        row = rows[0]
        selected.append(
            {
                "query_id": row.query_id,
                "hypothesis_id": row.hypothesis_id,
                "query_kind": row.query_kind,
                "query_text": row.query_text,
                "query_sha256": _query_sha(row),
            }
        )

    body: dict[str, Any] = {
        "schema_version": "sers-standard2-provider-health-probe-spec-v1",
        "semantics_id": SPEC_SEMANTICS_ID,
        "provider_plan": provider_plan.model_dump(mode="json"),
        "selection_policy": SELECTION_POLICY,
        "queries": selected,
        "result_limit_per_query": RESULT_LIMIT,
        "expected_hypothesis_count": EXPECTED_HYPOTHESIS_COUNT,
        "expected_provider_count": len(EXPECTED_ACTIVE_PROVIDERS),
        "expected_logical_execution_count": (
            EXPECTED_HYPOTHESIS_COUNT * len(EXPECTED_ACTIVE_PROVIDERS)
        ),
        "baseline_query_plan_sha256": sha256_file(query_plan_path),
        "epistemic_policy": {
            "purpose": "provider_connectivity_and_metadata_health_only",
            "scientific_result_interpretation": False,
            "paper_titles_persisted": False,
            "query_rewrite_allowed": False,
            "ranking_change_allowed": False,
            "novelty_policy_change_allowed": False,
            "full_dev_rerun_auto_authorized": False,
        },
        "execution_policy": {
            "one_shot": True,
            "automatic_rerun_authorized": False,
            "llm_calls": 0,
        },
    }
    body["spec_sha256"] = sha256_json(body)
    body["spec_id"] = (
        "sers_standard2_provider_health_probe_spec:"
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
        "sers_standard2_provider_health_probe_spec:"
        + observed[:20]
    ):
        issues.append("spec ID mismatch")

    try:
        recomputed = build_spec(diagnostic_root=diagnostic_root)
        if canonical_json(recomputed) != canonical_json(stored):
            issues.append("deterministic spec recomputation mismatch")
    except Exception as exc:
        issues.append(
            f"spec recomputation failed: {type(exc).__name__}: {exc}"
        )

    return sorted(set(issues)), stored


def _sanitize_failure(exc: Exception) -> dict[str, Any]:
    text = f"{type(exc).__name__}: {exc}"
    classified = classify_failure(text)
    return {
        "failure_category": classified["category"],
        "http_status": classified["http_status"],
        "error_text_sha256": classified["error_text_sha256"],
    }


def run_probe(
    *,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    plan = LiteratureProviderPlan.model_validate(
        spec["provider_plan"]
    )
    providers = build_literature_providers(plan)

    executions = []
    for provider in providers:
        for frozen in spec["queries"]:
            query = LiteratureQuery(
                query_id=frozen["query_id"],
                hypothesis_id=frozen["hypothesis_id"],
                claim_id=None,
                query_kind=frozen["query_kind"],
                query_text=frozen["query_text"],
            )
            try:
                rows = provider.search(
                    query,
                    limit=int(spec["result_limit_per_query"]),
                )
                success = True
                failure = {
                    "failure_category": None,
                    "http_status": None,
                    "error_text_sha256": None,
                }
                result_count = len(rows)
                abstract_result_count = sum(
                    bool(row.abstract) for row in rows
                )
            except Exception as exc:
                success = False
                failure = _sanitize_failure(exc)
                result_count = 0
                abstract_result_count = 0

            executions.append(
                {
                    "provider": provider.provider_name,
                    "query_id": frozen["query_id"],
                    "query_sha256": frozen["query_sha256"],
                    "hypothesis_id": frozen["hypothesis_id"],
                    "success": success,
                    "result_count": result_count,
                    "abstract_result_count": abstract_result_count,
                    **failure,
                }
            )

    provider_summary: dict[str, dict[str, Any]] = {}
    for provider_name in EXPECTED_ACTIVE_PROVIDERS:
        rows = [
            row for row in executions
            if row["provider"] == provider_name
        ]
        provider_summary[provider_name] = {
            "logical_execution_count": len(rows),
            "successful_logical_execution_count": sum(
                row["success"] for row in rows
            ),
            "failed_logical_execution_count": sum(
                not row["success"] for row in rows
            ),
            "returned_work_count": sum(
                int(row["result_count"]) for row in rows
            ),
            "returned_abstract_count": sum(
                int(row["abstract_result_count"]) for row in rows
            ),
            "failure_categories": sorted(
                {
                    row["failure_category"]
                    for row in rows
                    if row["failure_category"] is not None
                }
            ),
        }

    success_count = sum(row["success"] for row in executions)
    expected = int(spec["expected_logical_execution_count"])

    if success_count == expected:
        outcome = "STANDARD2_HEALTH_PASS"
    elif success_count == 0:
        outcome = "STANDARD2_HEALTH_FAIL_ALL"
    else:
        outcome = "STANDARD2_HEALTH_PARTIAL"

    body: dict[str, Any] = {
        "schema_version": "sers-standard2-provider-health-probe-run-v1",
        "semantics_id": RUN_SEMANTICS_ID,
        "source_spec_id": spec["spec_id"],
        "source_spec_sha256": spec["spec_sha256"],
        "provider_plan_id": plan.plan_id,
        "provider_plan_sha256": plan.plan_sha256,
        "provider_mode": plan.mode,
        "active_providers": list(plan.active_providers),
        "logical_execution_count": len(executions),
        "successful_logical_execution_count": success_count,
        "outcome": outcome,
        "provider_summary": provider_summary,
        "executions": executions,
        "paper_titles_persisted": False,
        "scientific_result_interpretation": False,
        "full_dev_rerun_authorized": False,
        "automatic_rerun_authorized": False,
        "llm_calls": 0,
    }
    body["run_sha256"] = sha256_json(body)
    body["run_id"] = (
        "sers_standard2_provider_health_probe_run:"
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
        "sers_standard2_provider_health_probe_run:"
        + observed[:20]
    ):
        issues.append("run ID mismatch")
    if stored.get("source_spec_id") != spec.get("spec_id"):
        issues.append("run/spec ID mismatch")
    if stored.get("source_spec_sha256") != spec.get("spec_sha256"):
        issues.append("run/spec SHA mismatch")
    if stored.get("provider_mode") != EXPECTED_PROVIDER_MODE:
        issues.append("provider mode mismatch")
    if stored.get("active_providers") != EXPECTED_ACTIVE_PROVIDERS:
        issues.append("active provider set mismatch")
    if stored.get("paper_titles_persisted") is not False:
        issues.append("title persistence violation")
    if stored.get("scientific_result_interpretation") is not False:
        issues.append("scientific interpretation violation")
    if stored.get("full_dev_rerun_authorized") is not False:
        issues.append("unexpected full DEV authorization")
    if stored.get("llm_calls") != 0:
        issues.append("unexpected LLM calls")

    return sorted(set(issues)), stored
