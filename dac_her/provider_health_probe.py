from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from dac_her.external_novelty_contracts import (
    LiteratureQuery,
    LiteratureQueryPlan,
)
from dac_her.literature_retrieval import (
    SemanticScholarProvider,
)
from dac_her.provider_failure_taxonomy import (
    classify_failure,
)


PROVIDER_HEALTH_SPEC_SEMANTICS_ID = (
    "sers_provider_health_probe_spec_v1"
)
PROVIDER_HEALTH_RUN_SEMANTICS_ID = (
    "sers_provider_health_probe_run_v1"
)

EXPECTED_CLEAN_BRANCH = "feat/SERS-clean-next"
EXPECTED_CLEAN_HEAD = (
    "cdbba2eff2d9f59bfdddd0b28373adf6e0904b00"
)
EXPECTED_PATCHED_RETRIEVAL_BLOB = (
    "e35fcc1626c4db480e8da72d990262b73bc2e919"
)
EXPECTED_PROVIDER_RESILIENCE_SHA256 = (
    "b94319a4126e58a14da18d66db653670fea523421438b4c4715a7382b2773578"
)
EXPECTED_TAXONOMY_ID = (
    "sers_provider_failure_taxonomy:d780d647b35d76af00d6"
)
EXPECTED_TAXONOMY_SHA256 = (
    "d780d647b35d76af00d679a40f16d60e2328581f3ab2146d2b9224601ce6592e"
)

DEFAULT_DIAGNOSTIC_ROOT = (
    Path.home() / "GraphAgentsDAC"
)
DEFAULT_SPEC_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "provider_health_probe_spec_v1"
)
DEFAULT_RUN_ROOT = Path(
    "evaluation/sers_provider_reliability/"
    "provider_health_probe_run_v1"
)
BASELINE_QUERY_PLAN = Path(
    "evaluation/sers_alpha4c5k/dev_e2e_v2/"
    "external_novelty.claims_queries.json"
)
TAXONOMY_AUDIT = Path(
    "evaluation/sers_provider_reliability/"
    "provider_failure_taxonomy_v1/audit.json"
)
HARDENING_MANIFEST = Path(
    "evaluation/sers_provider_reliability/"
    "provider_resilience_hardening_v1/patch_manifest.json"
)

PROVIDER = "semantic_scholar"
SELECTION_POLICY = (
    "one_unique_hypothesis_composite_query_per_baseline_hypothesis"
)
RESULT_LIMIT = 1
EXPECTED_HYPOTHESIS_COUNT = 3


def canonical_json(value: Any) -> str:
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
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )
    return value


def atomic_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_name(
        path.name + ".tmp"
    )
    tmp.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _query_sha(
    *,
    query_id: str,
    hypothesis_id: str,
    query_kind: str,
    query_text: str,
) -> str:
    return sha256_json({
        "query_id": query_id,
        "hypothesis_id":
            hypothesis_id,
        "query_kind":
            query_kind,
        "query_text":
            query_text,
    })


def build_probe_spec(
    *,
    root: Path,
    diagnostic_root: Path,
) -> dict[str, Any]:
    taxonomy_path = (
        root / TAXONOMY_AUDIT
    )
    hardening_path = (
        root / HARDENING_MANIFEST
    )
    query_plan_path = (
        diagnostic_root
        / BASELINE_QUERY_PLAN
    )

    for path in (
        taxonomy_path,
        hardening_path,
        query_plan_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                path
            )

    taxonomy = read_json(
        taxonomy_path
    )
    hardening = read_json(
        hardening_path
    )
    if (
        taxonomy.get("audit_id")
        != EXPECTED_TAXONOMY_ID
        or taxonomy.get(
            "audit_sha256"
        )
        != EXPECTED_TAXONOMY_SHA256
    ):
        raise ValueError(
            "Unexpected provider taxonomy binding."
        )

    conclusion = taxonomy.get(
        "semantic_scholar_conclusion",
        {},
    )
    if (
        conclusion.get(
            "dominant_failure_category"
        )
        != "HTTP_429_RATE_LIMIT"
        or conclusion.get(
            "dominant_failure_fraction"
        )
        != 1.0
        or conclusion.get(
            "generality"
        )
        != "CROSS_HYPOTHESIS_PROVIDER_FAILURE"
    ):
        raise ValueError(
            "Taxonomy no longer justifies 429 provider health probe."
        )

    if (
        hardening.get(
            "patched_literature_retrieval_git_blob"
        )
        != EXPECTED_PATCHED_RETRIEVAL_BLOB
        or not hardening.get(
            "retry_after_supported"
        )
        or not hardening.get(
            "terminal_retryable_failure_cooldown"
        )
    ):
        raise ValueError(
            "Unexpected provider hardening manifest."
        )

    plan = LiteratureQueryPlan.model_validate_json(
        query_plan_path.read_text(
            encoding="utf-8"
        )
    )

    hypotheses = sorted({
        row.hypothesis_id
        for row in plan.queries
    })
    if len(hypotheses) != (
        EXPECTED_HYPOTHESIS_COUNT
    ):
        raise ValueError(
            "Expected exactly three baseline hypotheses; "
            f"observed {len(hypotheses)}."
        )

    selected = []
    for hypothesis_id in hypotheses:
        candidates = [
            row
            for row in plan.queries
            if (
                row.hypothesis_id
                == hypothesis_id
                and row.query_kind
                == "hypothesis_composite"
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one hypothesis_composite query for "
                f"{hypothesis_id}; observed {len(candidates)}."
            )
        row = candidates[0]
        selected.append({
            "hypothesis_id":
                hypothesis_id,
            "query_id":
                row.query_id,
            "query_kind":
                row.query_kind,
            "query_text":
                row.query_text,
            "query_sha256":
                _query_sha(
                    query_id=
                        row.query_id,
                    hypothesis_id=
                        hypothesis_id,
                    query_kind=
                        row.query_kind,
                    query_text=
                        row.query_text,
                ),
        })

    body: dict[str, Any] = {
        "schema_version":
            "sers-provider-health-probe-spec-v1",
        "semantics_id":
            PROVIDER_HEALTH_SPEC_SEMANTICS_ID,
        "provider":
            PROVIDER,
        "selection_policy":
            SELECTION_POLICY,
        "result_limit_per_query":
            RESULT_LIMIT,
        "expected_hypothesis_count":
            EXPECTED_HYPOTHESIS_COUNT,
        "expected_logical_execution_count":
            len(selected),
        "queries":
            selected,
        "bindings": {
            "taxonomy_audit_id":
                EXPECTED_TAXONOMY_ID,
            "taxonomy_audit_sha256":
                EXPECTED_TAXONOMY_SHA256,
            "hardening_manifest_sha256":
                sha256_file(
                    hardening_path
                ),
            "baseline_query_plan_sha256":
                sha256_file(
                    query_plan_path
                ),
            "patched_literature_retrieval_blob":
                EXPECTED_PATCHED_RETRIEVAL_BLOB,
            "provider_resilience_sha256":
                EXPECTED_PROVIDER_RESILIENCE_SHA256,
        },
        "epistemic_policy": {
            "scientific_results_used":
                False,
            "paper_titles_persisted":
                False,
            "query_rewrite_allowed":
                False,
            "ranking_change_allowed":
                False,
            "novelty_verdict_change_allowed":
                False,
            "purpose":
                "provider reliability validation only",
        },
        "execution_policy": {
            "one_shot":
                True,
            "automatic_rerun_authorized":
                False,
            "provider_count":
                1,
            "logical_execution_count":
                len(selected),
            "llm_calls":
                0,
        },
    }
    body["spec_sha256"] = (
        sha256_json(body)
    )
    body["spec_id"] = (
        "sers_provider_health_probe_spec:"
        + body["spec_sha256"][:20]
    )
    return body


def verify_probe_spec(
    *,
    root: Path,
    diagnostic_root: Path,
    spec_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    if not spec_path.is_file():
        return ["probe spec missing"], {}

    stored = read_json(
        spec_path
    )
    issues = []

    body = dict(stored)
    stored_id = body.pop(
        "spec_id",
        None,
    )
    stored_sha = body.pop(
        "spec_sha256",
        None,
    )
    observed_sha = sha256_json(
        body
    )
    if stored_sha != observed_sha:
        issues.append(
            "probe spec SHA mismatch"
        )
    if stored_id != (
        "sers_provider_health_probe_spec:"
        + observed_sha[:20]
    ):
        issues.append(
            "probe spec ID mismatch"
        )

    try:
        recomputed = build_probe_spec(
            root=root,
            diagnostic_root=
                diagnostic_root,
        )
        if canonical_json(
            recomputed
        ) != canonical_json(
            stored
        ):
            issues.append(
                "deterministic spec recomputation mismatch"
            )
    except Exception as exc:
        issues.append(
            "spec recomputation failed: "
            f"{type(exc).__name__}: {exc}"
        )

    return sorted(set(issues)), stored


def _telemetry_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, int]:
    keys = sorted(
        set(before)
        | set(after)
    )
    return {
        key: int(
            after.get(key, 0)
        )
        - int(
            before.get(key, 0)
        )
        for key in keys
    }


def run_probe(
    *,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    provider = SemanticScholarProvider()

    initial_health = (
        provider.health_snapshot()
    )
    executions = []

    for frozen in spec["queries"]:
        query = LiteratureQuery(
            query_id=
                frozen["query_id"],
            hypothesis_id=
                frozen[
                    "hypothesis_id"
                ],
            claim_id=None,
            query_kind=
                frozen["query_kind"],
            query_text=
                frozen["query_text"],
        )

        before = provider.health_snapshot()
        try:
            works = provider.search(
                query,
                limit=int(
                    spec[
                        "result_limit_per_query"
                    ]
                ),
            )
            success = True
            error = None
            failure = None
            result_count = len(works)
            abstract_count = sum(
                bool(work.abstract)
                for work in works
            )
        except Exception as exc:
            success = False
            error = (
                f"{type(exc).__name__}: {exc}"
            )
            failure = classify_failure(
                error
            )
            result_count = 0
            abstract_count = 0

        after = provider.health_snapshot()
        delta = _telemetry_delta(
            before["telemetry"],
            after["telemetry"],
        )
        executions.append({
            "hypothesis_id":
                frozen[
                    "hypothesis_id"
                ],
            "query_id":
                frozen[
                    "query_id"
                ],
            "query_sha256":
                frozen[
                    "query_sha256"
                ],
            "provider":
                PROVIDER,
            "success":
                success,
            "result_count":
                result_count,
            "abstract_result_count":
                abstract_count,
            "telemetry_delta":
                delta,
            "failure_category":
                (
                    failure[
                        "category"
                    ]
                    if failure
                    else None
                ),
            "http_status":
                (
                    failure[
                        "http_status"
                    ]
                    if failure
                    else None
                ),
            "error_text_sha256":
                (
                    failure[
                        "error_text_sha256"
                    ]
                    if failure
                    else None
                ),
        })

    final_health = (
        provider.health_snapshot()
    )
    success_count = sum(
        row["success"]
        for row in executions
    )
    terminal_429_count = sum(
        (
            not row["success"]
            and row[
                "failure_category"
            ]
            == "HTTP_429_RATE_LIMIT"
        )
        for row in executions
    )
    total_429_events = int(
        final_health["telemetry"][
            "http_429_events"
        ]
    ) - int(
        initial_health["telemetry"][
            "http_429_events"
        ]
    )

    if (
        success_count
        == len(executions)
        and total_429_events == 0
    ):
        outcome = (
            "HEALTH_PASS_NO_429"
        )
    elif (
        success_count
        == len(executions)
    ):
        outcome = (
            "HEALTH_PASS_RECOVERED_429"
        )
    elif (
        terminal_429_count
        == len(executions)
    ):
        outcome = (
            "HEALTH_FAIL_ALL_TERMINAL_429"
        )
    elif terminal_429_count > 0:
        outcome = (
            "HEALTH_PARTIAL_TERMINAL_429"
        )
    else:
        outcome = (
            "HEALTH_FAIL_NON429_OR_MIXED"
        )

    body: dict[str, Any] = {
        "schema_version":
            "sers-provider-health-probe-run-v1",
        "semantics_id":
            PROVIDER_HEALTH_RUN_SEMANTICS_ID,
        "source_spec_id":
            spec["spec_id"],
        "source_spec_sha256":
            spec["spec_sha256"],
        "provider":
            PROVIDER,
        "api_key_configured":
            bool(
                initial_health[
                    "api_key_configured"
                ]
            ),
        "minimum_interval_seconds":
            initial_health[
                "minimum_interval_seconds"
            ],
        "logical_execution_count":
            len(executions),
        "successful_logical_execution_count":
            success_count,
        "terminal_429_logical_failure_count":
            terminal_429_count,
        "total_http_attempt_count":
            sum(
                row[
                    "telemetry_delta"
                ].get(
                    "attempts",
                    0,
                )
                for row in executions
            ),
        "total_429_event_count":
            total_429_events,
        "retry_after_honored_count":
            sum(
                row[
                    "telemetry_delta"
                ].get(
                    "retry_after_honored",
                    0,
                )
                for row in executions
            ),
        "terminal_cooldown_count":
            sum(
                row[
                    "telemetry_delta"
                ].get(
                    "cooldowns_scheduled_after_terminal_failure",
                    0,
                )
                for row in executions
            ),
        "outcome":
            outcome,
        "executions":
            executions,
        "epistemic_usage":
            "provider_health_only_not_scientific_evidence",
        "full_dev_rerun_authorized":
            False,
        "automatic_rerun_authorized":
            False,
        "paper_titles_persisted":
            False,
        "llm_calls":
            0,
    }
    body["run_sha256"] = (
        sha256_json(body)
    )
    body["run_id"] = (
        "sers_provider_health_probe_run:"
        + body["run_sha256"][:20]
    )
    return body


def verify_run(
    *,
    run_path: Path,
    spec: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    if not run_path.is_file():
        return ["probe run missing"], {}

    value = read_json(
        run_path
    )
    issues = []

    body = dict(value)
    run_id = body.pop(
        "run_id",
        None,
    )
    run_sha = body.pop(
        "run_sha256",
        None,
    )
    observed_sha = sha256_json(
        body
    )
    if run_sha != observed_sha:
        issues.append(
            "run SHA mismatch"
        )
    if run_id != (
        "sers_provider_health_probe_run:"
        + observed_sha[:20]
    ):
        issues.append(
            "run ID mismatch"
        )
    if value.get(
        "source_spec_id"
    ) != spec.get(
        "spec_id"
    ):
        issues.append(
            "run/spec ID mismatch"
        )
    if value.get(
        "source_spec_sha256"
    ) != spec.get(
        "spec_sha256"
    ):
        issues.append(
            "run/spec SHA mismatch"
        )
    if value.get(
        "logical_execution_count"
    ) != (
        spec.get(
            "expected_logical_execution_count"
        )
    ):
        issues.append(
            "logical execution count mismatch"
        )
    if value.get(
        "paper_titles_persisted"
    ) is not False:
        issues.append(
            "scientific title persistence violation"
        )
    if value.get(
        "full_dev_rerun_authorized"
    ) is not False:
        issues.append(
            "unexpected full DEV authorization"
        )
    if value.get(
        "llm_calls"
    ) != 0:
        issues.append(
            "unexpected LLM calls"
        )

    return sorted(set(issues)), value
